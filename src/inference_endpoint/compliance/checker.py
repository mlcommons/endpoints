# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compliance checks for Edge-Agentic (BFCL v4) submissions.

A submission is a run's report directory containing the resolved ``config.yaml``
plus the scorer outputs (``accuracy/accuracy_results.json`` for the accuracy run,
``scores.json`` for the agentic performance run). The checker compares those
artifacts against a
registered ruleset (default: ``mlperf-edge-current`` / ``qwen3.6-27b``):

* config-lock — deterministic, single-stream settings the rules require;
* accuracy gate — ``score >= factor x reference`` from the model's
  ``accuracy_target_settings`` (3% one-sided, factor 0.97);
* run validity — the agentic performance run must have 0 dropped turns.

Server-side launch flags (``--reasoning off``, ``--ctx-size 32768``) are not
present in the client artifacts and cannot be auto-verified; they are surfaced as
manual attestations rather than pass/fail checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config.ruleset_base import BenchmarkSuiteRuleset
from ..config.ruleset_registry import get_ruleset
from ..config.schema import BenchmarkConfig
from ..evaluation.accuracy_results import (
    find_accuracy_breakdown as _find_accuracy_score,
)
from ..evaluation.accuracy_results import (
    find_accuracy_entry as _find_accuracy_entry,
)
from ..evaluation.accuracy_results import (
    to_float as _to_float,
)
from ..evaluation.bfcl_v4_metrics import (
    ACCURACY_METRIC_KEYS as _ACCURACY_METRIC_KEYS,
)


@dataclass(frozen=True)
class Check:
    """One pass/fail compliance check."""

    name: str
    passed: bool
    detail: str


@dataclass
class ComplianceReport:
    """Aggregate result of all checks run against a submission."""

    report_dir: str
    ruleset: str
    model: str
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name, passed, detail))

    def render(self) -> str:
        lines = [
            f"Compliance report: {self.report_dir}",
            f"  ruleset={self.ruleset} model={self.model}",
            "",
        ]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        for n in self.notes:
            lines.append(f"  [MANUAL] {n}")
        lines.append("")
        lines.append(f"  OVERALL: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _get(d: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def check_config_lock(config: dict[str, Any]) -> list[Check]:
    """Validate deterministic + single-stream settings from the resolved config."""
    checks: list[Check] = []

    temperature = _get(config, "model_params", "temperature")
    checks.append(
        Check(
            "temperature==0",
            temperature == 0,
            f"model_params.temperature={temperature}",
        )
    )

    # Client seed is optional in the offline accuracy config; the dataloader seed
    # is always present. Require any seed that is set to be 42.
    model_seed = _get(config, "model_params", "seed")
    dl_seed = _get(config, "settings", "runtime", "dataloader_random_seed")
    seeds = {"model_params.seed": model_seed, "dataloader_random_seed": dl_seed}
    set_seeds = {k: v for k, v in seeds.items() if v is not None}
    if not set_seeds:
        # No seed anywhere: report the specific failure rather than an all-None
        # dict that reads like "seed != 42".
        seeds_ok = False
        detail = "no seed set (expected seed==42)"
    else:
        offending = {k: v for k, v in set_seeds.items() if v != 42}
        seeds_ok = not offending
        detail = f"seeds={set_seeds}" + (
            f" (expected 42; offending: {offending})" if offending else ""
        )
    checks.append(Check("seed==42", seeds_ok, detail))

    workers = _get(config, "settings", "client", "num_workers")
    conns = _get(config, "settings", "client", "max_connections")
    single_stream = workers == 1 and conns == 1
    detail = f"num_workers={workers} max_connections={conns}"
    target_concurrency = _get(config, "settings", "load_pattern", "target_concurrency")
    if target_concurrency is not None:
        single_stream = single_stream and target_concurrency == 1
        detail += f" target_concurrency={target_concurrency}"
    checks.append(Check("single_stream", single_stream, detail))

    return checks


def _resolve_model(ruleset: BenchmarkSuiteRuleset, model_name: str) -> Any:
    benchmark_rulesets = getattr(ruleset, "benchmark_rulesets", {})
    for model in benchmark_rulesets:
        if model.name == model_name:
            return model
    raise KeyError(
        f"Model '{model_name}' not found in ruleset. "
        f"Available: {[m.name for m in benchmark_rulesets]}"
    )


# The breakdown key for the headline accuracy. Only BFCL stores it in the block;
# every other scorer keeps it on the entry's scalar ``score`` (see
# accuracy_results.find_accuracy_entry), so this is the one metric the gate falls
# back to the entry score for.
_OVERALL_ACCURACY_KEY = "overall_accuracy"


def check_accuracy(
    results: dict[str, Any],
    golden: dict[str, float],
    factors: dict[str, tuple[float, ...]],
    min_samples: int | None,
) -> list[Check]:
    """Validate the accuracy gate from an ``accuracy_results.json`` dict."""
    checks: list[Check] = []

    entry = _find_accuracy_entry(results)
    if entry is None:
        return [Check("accuracy_results_present", False, "no accuracy score found")]
    block = entry.get("breakdown")
    if not isinstance(block, dict):
        return [Check("accuracy_results_present", False, "no accuracy score found")]

    applicable_metrics = 0
    for golden_key, result_key in _ACCURACY_METRIC_KEYS.items():
        if golden_key not in golden or golden_key not in factors:
            continue
        applicable_metrics += 1
        measured = _to_float(block.get(result_key))
        if measured is None and result_key == _OVERALL_ACCURACY_KEY:
            # The headline accuracy lives on the entry's scalar ``score``; only
            # BFCL also duplicates it into the breakdown block. Fall back to the
            # entry score (parity with the plot path) so a DeepSeek-R1 submission
            # — which omits ``overall_accuracy`` from the block — is still gated.
            # The block metrics and the deepseek/bfcl headline are percentages in
            # [0, 100]; gpt-oss's fraction scorers aren't gate targets (they'd
            # need the scale homogenization tracked separately).
            measured = _to_float(entry.get("score"))
        if measured is None:
            checks.append(
                Check(f"accuracy:{result_key}", False, "metric missing or non-numeric")
            )
            continue
        factor = factors[golden_key][0]
        threshold = golden[golden_key] * factor
        checks.append(
            Check(
                f"accuracy:{result_key}",
                measured >= threshold,
                f"{measured:.2f} >= {threshold:.2f} (={factor} x {golden[golden_key]})",
            )
        )

    # An accuracy score was found but none of its metrics intersect the model's
    # golden/factor tables: the gate would otherwise silently PASS on zero
    # accuracy checks. Fail explicitly so a misconfigured golden table is visible.
    if applicable_metrics == 0:
        checks.append(
            Check(
                "accuracy_metric_applicable",
                False,
                "no accuracy metric matched the model's golden/factor tables",
            )
        )

    if min_samples is not None:
        raw_total = block.get("total_samples")
        total = _to_float(raw_total)
        checks.append(
            Check(
                "min_sample_count",
                total is not None and total >= min_samples,
                f"total_samples={raw_total} >= {min_samples}",
            )
        )

    return checks


def check_perf_validity(scores: dict[str, Any]) -> list[Check]:
    """Validate run-validity for the agentic performance run (0 dropped turns)."""
    turns = scores.get("turns")
    if not isinstance(turns, dict):
        return [Check("perf_turns_present", False, "no turns block in scores.json")]

    missing = turns.get("missing")
    issued = turns.get("issued")
    observed = turns.get("observed")
    # "expected" is the count of scorable turns, which can be fewer than the
    # turns actually issued (e.g. a dataset with 1007 issued but 1006 scorable).
    # Validate against expected so a complete run isn't failed for that gap.
    expected = turns.get("expected")
    return [
        Check(
            "no_dropped_turns",
            missing == 0,
            f"missing={missing} (issued={issued}, observed={observed})",
        ),
        Check(
            "all_turns_observed",
            expected is not None and observed == expected,
            f"observed={observed} == expected={expected} (issued={issued})",
        ),
    ]


def check_submission(
    report_dir: str | Path,
    ruleset_name: str = "mlperf-edge-current",
    model_name: str = "qwen3.6-27b",
) -> ComplianceReport:
    """Run all applicable compliance checks against a run's report directory."""
    report_dir = Path(report_dir)
    report = ComplianceReport(
        report_dir=str(report_dir), ruleset=ruleset_name, model=model_name
    )

    ruleset = get_ruleset(ruleset_name)
    benchmark_rulesets = getattr(ruleset, "benchmark_rulesets", {})
    model = _resolve_model(ruleset, model_name)
    _, golden = model.golden_accuracy
    factors = model.accuracy_target_settings[0]
    per_model = next(iter(benchmark_rulesets[model].values()))
    min_samples = per_model.min_sample_count_valid

    config_path = report_dir / "config.yaml"
    if not config_path.exists():
        report.add("config_present", False, f"missing {config_path}")
    else:
        # Load through the schema (env-var resolution + validation + discriminated
        # union) rather than a raw yaml.safe_load, so config-lock checks run on the
        # same normalized shape the benchmark actually used.
        try:
            config = BenchmarkConfig.from_yaml_file(config_path).model_dump()
        except (ValidationError, OSError, ValueError) as e:
            report.add("config_valid", False, f"{config_path} failed to load: {e}")
        else:
            report.checks.extend(check_config_lock(config))

    results_path = report_dir / "accuracy" / "accuracy_results.json"
    scores_path = report_dir / "scores.json"

    is_accuracy = False
    if results_path.exists():
        results = json.loads(results_path.read_text())
        if _find_accuracy_score(results) is not None:
            is_accuracy = True
            report.checks.extend(check_accuracy(results, golden, factors, min_samples))

    if scores_path.exists() and "turns" in json.loads(scores_path.read_text()):
        scores = json.loads(scores_path.read_text())
        report.checks.extend(check_perf_validity(scores))
    elif not is_accuracy:
        report.add(
            "scorer_output_present",
            False,
            "no accuracy/accuracy_results.json or performance scores.json found",
        )

    report.notes.append(
        "Verify the server was launched with --reasoning off and "
        "--ctx-size 32768 (not recorded in client artifacts)."
    )
    return report
