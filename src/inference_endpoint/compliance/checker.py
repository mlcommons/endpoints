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
plus the scorer outputs (``results.json`` for the accuracy run, ``scores.json``
for the agentic performance run). The checker compares those artifacts against a
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

import yaml

from ..config.ruleset_base import BenchmarkSuiteRuleset
from ..config.ruleset_registry import get_ruleset

# BFCL accuracy: ruleset golden-metric name -> key in the scorer's score block.
_ACCURACY_METRIC_KEYS = {
    "bfcl_overall_accuracy": "overall_accuracy",
    "bfcl_normalized_accuracy": "normalized_single_turn_score",
}


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


def _to_float(value: Any) -> float | None:
    """Coerce a score value to float, or None if absent/non-numeric.

    ``BFCLv4Scorer.score()`` serializes accuracy metrics as formatted strings
    (e.g. ``"86.23"``) and stores them verbatim in ``results.json``, so the gate
    must coerce before any numeric comparison or ``:.2f`` formatting.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    seeds_ok = bool(set_seeds) and all(v == 42 for v in set_seeds.values())
    checks.append(Check("seed==42", seeds_ok, str(seeds)))

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


def check_accuracy(
    results: dict[str, Any],
    golden: dict[str, float],
    factors: dict[str, tuple[float, ...]],
    min_samples: int | None,
) -> list[Check]:
    """Validate the accuracy gate from a BFCL accuracy ``results.json`` dict."""
    checks: list[Check] = []

    score = _find_accuracy_score(results)
    if score is None:
        return [Check("accuracy_results_present", False, "no accuracy score found")]

    for golden_key, result_key in _ACCURACY_METRIC_KEYS.items():
        if golden_key not in golden or golden_key not in factors:
            continue
        measured = _to_float(score.get(result_key))
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

    if min_samples is not None:
        raw_total = score.get("total_samples")
        total = _to_float(raw_total)
        checks.append(
            Check(
                "min_sample_count",
                total is not None and total >= min_samples,
                f"total_samples={raw_total} >= {min_samples}",
            )
        )

    return checks


def _find_accuracy_score(results: dict[str, Any]) -> dict[str, Any] | None:
    accuracy_scores = results.get("accuracy_scores")
    if not isinstance(accuracy_scores, dict):
        return None
    for entry in accuracy_scores.values():
        score = entry.get("score") if isinstance(entry, dict) else None
        if isinstance(score, dict) and "overall_accuracy" in score:
            return score
    return None


def check_perf_validity(scores: dict[str, Any]) -> list[Check]:
    """Validate run-validity for the agentic performance run (0 dropped turns)."""
    turns = scores.get("turns")
    if not isinstance(turns, dict):
        return [Check("perf_turns_present", False, "no turns block in scores.json")]

    missing = turns.get("missing")
    issued = turns.get("issued")
    observed = turns.get("observed")
    return [
        Check(
            "no_dropped_turns",
            missing == 0,
            f"missing={missing} (issued={issued}, observed={observed})",
        ),
        Check(
            "all_turns_observed",
            issued is not None and observed == issued,
            f"observed={observed} == issued={issued}",
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
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
        report.checks.extend(check_config_lock(config))
    else:
        report.add("config_present", False, f"missing {config_path}")

    results_path = report_dir / "results.json"
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
            "no accuracy results.json or performance scores.json found",
        )

    report.notes.append(
        "Verify the server was launched with --reasoning off and "
        "--ctx-size 32768 (not recorded in client artifacts)."
    )
    return report
