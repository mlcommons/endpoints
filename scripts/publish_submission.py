#!/usr/bin/env python3
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

"""Publish a benchmark run into the MLPerf Inference "endpoints" submission tree.

The MLPerf Inference submission checker (``tools/submission/submission_checker``
in ``mlcommons/inference``, v5.0+) grew an endpoints parser that, instead of
classic LoadGen ``mlperf_log_*`` text logs, reads the artifacts this tool already
emits in each run's ``report_dir``:

  * ``result_summary.json``  (highest priority)  -- perf rollups: latency /
    ttft / tpot percentiles (keys are float-formatted, e.g. ``"99.0"``),
    ``n_samples_issued``, ``duration_ns``, ``git_sha``, ``version``.
  * ``results.json``         -- ``results.total/failed/qps`` and, for accuracy
    runs, the ``accuracy_scores`` list.
  * ``config.yaml``          (lowest priority)   -- run configuration
    (``type``, ``model_params.streaming``, ``settings.runtime.*``,
    ``settings.load_pattern.*``).

So "publishing a log the checker can parse" is purely a matter of copying that
trio into the directory layout the checker walks::

  {division}/{submitter}/results/{system}/{benchmark}/{scenario}/performance/run_1/
  {division}/{submitter}/results/{system}/{benchmark}/{scenario}/accuracy/
  {division}/{submitter}/results/{system}/{benchmark}/{scenario}/measurements.json
  {division}/{submitter}/systems/{system}.json

This script assembles that tree and self-verifies the handful of fields the
endpoints checker reads (primary metric QPS, p99 latency, TTFT/TPOT p99, and the
accuracy score), so you can catch a missing/empty artifact before handing the
tree to the upstream checker.

Usage:
    # Combined run (perf + accuracy in one report_dir):
    python scripts/publish_submission.py \\
        --run results/edge_agentic_full_run \\
        --output submission \\
        --submitter NVIDIA --system AGX_Thor --benchmark qwen3.6-27b

    # Separate perf and accuracy report_dirs:
    python scripts/publish_submission.py \\
        --performance-run results/edge_perf \\
        --accuracy-run results/edge_accuracy \\
        --output submission \\
        --submitter NVIDIA --system AGX_Thor --benchmark qwen3.6-27b

Then run the upstream checker on the output tree::

    python3 -m inference.tools.submission.submission_checker.main \\
        --input submission --version v6.1 --submitter NVIDIA
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Artifacts the endpoints parser reads, as (source-path-in-run-dir, dest-name),
# in priority order. result_summary.json lives under performance/ in the run
# dir; it is still copied flat into the MLPerf performance/run_1 layout.
_RUN_ARTIFACTS = (
    ("performance/result_summary.json", "result_summary.json"),
    ("results.json", "results.json"),
    ("config.yaml", "config.yaml"),
)

# v6.1 ("default") endpoints submission layout, mirroring
# tools/submission/submission_checker/constants.py in mlcommons/inference.
_PERF_SUBDIR = "performance/run_1"
_ACC_SUBDIR = "accuracy"


def _results_root(
    division: str, submitter: str, system: str, benchmark: str, scenario: str
) -> Path:
    return Path(division) / submitter / "results" / system / benchmark / scenario


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _copy_artifacts(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy the parseable trio from ``src_dir`` to ``dst_dir``.

    Returns the list of artifact filenames actually copied.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src_rel, dst_name in _RUN_ARTIFACTS:
        src = src_dir / src_rel
        if src.is_file():
            shutil.copy2(src, dst_dir / dst_name)
            copied.append(dst_name)
    return copied


def _percentile(metric: dict[str, Any], key: str) -> Any:
    """Read a percentile value tolerating int/float key spellings (``"99"`` vs ``"99.0"``).

    Matches on numeric value, so a request for ``"99.5"`` never falls back to the
    ``"99"`` bucket.
    """
    perc = metric.get("percentiles", {}) if isinstance(metric, dict) else {}
    if key in perc:
        return perc[key]
    try:
        want = float(key)
    except (TypeError, ValueError):
        return None
    for k, v in perc.items():
        try:
            if float(k) == want:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _verify_performance(run_dir: Path) -> list[str]:
    """Return human-readable findings for the performance artifacts."""
    findings: list[str] = []
    summary = _load_json(run_dir / "performance" / "result_summary.json")
    results = _load_json(run_dir / "results.json")
    if not summary:
        findings.append("  MISSING: result_summary.json (perf rollups unreadable)")
        return findings

    duration_ns = summary.get("duration_ns")
    n_issued = summary.get("n_samples_issued")
    qps = (results.get("results") or {}).get("qps")
    if not qps and duration_ns and n_issued:
        qps = n_issued / (duration_ns / 1e9)
    findings.append(f"  QPS (primary metric): {qps if qps else 'N/A'}")

    p99 = _percentile(summary.get("latency", {}), "99.0")
    findings.append(
        f"  latency p99: {p99 / 1e6:.2f} ms" if p99 else "  latency p99: N/A (empty)"
    )

    ttft_p99 = _percentile(summary.get("ttft", {}), "99.0")
    tpot_p99 = _percentile(summary.get("tpot", {}), "99.0")
    if ttft_p99:
        findings.append(f"  TTFT p99: {ttft_p99 / 1e6:.2f} ms")
    if tpot_p99:
        findings.append(f"  TPOT p99: {tpot_p99 / 1e6:.2f} ms")
    if not p99:
        findings.append(
            "  WARNING: latency percentiles are empty -- this looks like an "
            "accuracy-only run, not a performance run."
        )
    return findings


def _verify_accuracy(run_dir: Path) -> list[str]:
    findings: list[str] = []
    results = _load_json(run_dir / "results.json")
    scores = results.get("accuracy_scores")
    if not scores:
        findings.append(
            "  MISSING: results.json has no non-null 'accuracy_scores' "
            "(endpoints accuracy check requires this)."
        )
        return findings
    for entry in scores:
        if not isinstance(entry, dict):
            continue
        name = entry.get("dataset_name", "?")
        score = entry.get("score")
        findings.append(f"  accuracy_scores[{name}].score = {score}")
        breakdown = entry.get("breakdown")
        if isinstance(breakdown, dict):
            findings.append(
                f"  accuracy_scores[{name}].breakdown.overall_accuracy = "
                f"{breakdown.get('overall_accuracy')}"
            )
    return findings


_MEASUREMENTS_TEMPLATE = {
    "starting_weights_filename": "TODO: e.g. Qwen3.6-27B-Q4_K_M.gguf",
    "weight_data_types": "TODO: e.g. int4",
    "input_data_types": "TODO: e.g. int32 (token ids)",
    "retraining": "no",
    "weight_transformations": "TODO: e.g. quantization (Q4_K_M)",
}

_SYSTEM_TEMPLATE = {
    "submitter": "TODO",
    "division": "TODO: closed|open",
    "status": "TODO: available|preview|rdi",
    "system_type": "edge",
    "system_name": "TODO",
    "number_of_nodes": "1",
    "host_processor_model_name": "TODO",
    "host_processors_per_node": "1",
    "host_processor_core_count": "TODO",
    "host_memory_capacity": "TODO",
    "accelerator_model_name": "TODO",
    "accelerators_per_node": "1",
    "accelerator_memory_capacity": "TODO",
    "framework": "TODO: e.g. llama.cpp",
    "operating_system": "TODO",
}


def _scaffold_json(path: Path, template: dict[str, Any], force: bool) -> bool:
    """Write a template JSON if absent (or ``force``). Returns True if written."""
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
        f.write("\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_argument_group("run sources")
    src.add_argument(
        "--run",
        type=Path,
        help="Combined report_dir used for BOTH performance and accuracy "
        "(shorthand for setting --performance-run and --accuracy-run to the "
        "same path).",
    )
    src.add_argument(
        "--performance-run",
        type=Path,
        help="report_dir of the performance run (has populated latency "
        "percentiles in result_summary.json).",
    )
    src.add_argument(
        "--accuracy-run",
        type=Path,
        help="report_dir of the accuracy run (has accuracy_scores in results.json).",
    )

    coords = parser.add_argument_group("submission coordinates")
    coords.add_argument("--output", type=Path, default=Path("submission"))
    coords.add_argument(
        "--division", default="closed", choices=["closed", "open", "network"]
    )
    coords.add_argument("--submitter", required=True)
    coords.add_argument("--system", required=True, help="System name (e.g. AGX_Thor).")
    coords.add_argument(
        "--benchmark", required=True, help="MLPerf model/benchmark name."
    )
    coords.add_argument("--scenario", default="SingleStream")
    coords.add_argument(
        "--version",
        default="v6.1",
        help="Checker version (informational; v5.0+ all use the same endpoints layout).",
    )
    coords.add_argument(
        "--scaffold",
        action="store_true",
        help="Also write measurements.json + systems/<system>.json TEMPLATES "
        "(fill in the TODO fields before submitting).",
    )
    coords.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scaffold templates.",
    )
    args = parser.parse_args()

    perf_run = args.performance_run or args.run
    acc_run = args.accuracy_run or args.run
    if perf_run is None and acc_run is None:
        parser.error("provide --run, or --performance-run / --accuracy-run")

    for label, run in (("--performance-run", perf_run), ("--accuracy-run", acc_run)):
        if run is not None and not run.is_dir():
            parser.error(f"{label} path is not a directory: {run}")

    results_root = args.output / _results_root(
        args.division, args.submitter, args.system, args.benchmark, args.scenario
    )

    print(f"Assembling endpoints submission ({args.version}) under: {args.output}\n")

    # Track runs that produced no parseable artifacts so a wrong/empty path
    # fails loudly instead of exiting 0 with an incomplete submission tree.
    empty_runs: list[str] = []

    if perf_run is not None:
        perf_dst = results_root / _PERF_SUBDIR
        copied = _copy_artifacts(perf_run, perf_dst)
        print(f"[performance] {perf_run}  ->  {perf_dst}")
        print(f"             copied: {', '.join(copied) or '(none!)'}")
        if not copied:
            print(
                f"             ERROR: no parseable artifacts "
                f"({', '.join(n for _, n in _RUN_ARTIFACTS)}) found under {perf_run}"
            )
            empty_runs.append(f"--performance-run {perf_run}")
        for line in _verify_performance(perf_run):
            print(line)
        print()

    if acc_run is not None:
        acc_dst = results_root / _ACC_SUBDIR
        copied = _copy_artifacts(acc_run, acc_dst)
        print(f"[accuracy]    {acc_run}  ->  {acc_dst}")
        print(f"             copied: {', '.join(copied) or '(none!)'}")
        if not copied:
            print(
                f"             ERROR: no parseable artifacts "
                f"({', '.join(n for _, n in _RUN_ARTIFACTS)}) found under {acc_run}"
            )
            empty_runs.append(f"--accuracy-run {acc_run}")
        for line in _verify_accuracy(acc_run):
            print(line)
        print()

    if args.scaffold:
        measurements = results_root / "measurements.json"
        system_json = (
            args.output
            / args.division
            / args.submitter
            / "systems"
            / f"{args.system}.json"
        )
        wrote_m = _scaffold_json(measurements, _MEASUREMENTS_TEMPLATE, args.force)
        wrote_s = _scaffold_json(system_json, _SYSTEM_TEMPLATE, args.force)
        print("[scaffold]")
        print(
            f"             measurements.json: "
            f"{'written (fill TODOs)' if wrote_m else 'left as-is (exists)'} -> {measurements}"
        )
        print(
            f"             systems/{args.system}.json: "
            f"{'written (fill TODOs)' if wrote_s else 'left as-is (exists)'} -> {system_json}"
        )
        print()

    if empty_runs:
        print(
            "FAILED: no parseable artifacts were copied for: "
            f"{', '.join(empty_runs)}. The submission tree is incomplete "
            "(check the run path(s) and that the run emitted "
            f"{', '.join(n for _, n in _RUN_ARTIFACTS)})."
        )
        return 1

    print("Done. Validate with the upstream checker:")
    print(
        f"  python3 -m inference.tools.submission.submission_checker.main "
        f"--input {args.output} --version {args.version} --submitter {args.submitter}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
