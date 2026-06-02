#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-score accuracy datasets from an existing benchmark report directory.

Use after `inference-endpoint benchmark from-config` when inference completed but
scoring failed (e.g. LiveCodeBench container was not running). Dataset names must
match the YAML preset suffixes (e.g. `gpqa::deepseek_v4`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_endpoint.dataset_manager.predefined.aime25 import AIME25
from inference_endpoint.dataset_manager.predefined.gpqa import GPQA
from inference_endpoint.dataset_manager.predefined.livecodebench import LiveCodeBench
from inference_endpoint.evaluation.extractor import (
    ABCDExtractor,
    BoxedMathExtractor,
    PythonCodeExtractor,
)
from inference_endpoint.evaluation.scoring import LiveCodeBenchScorer, PassAt1Scorer

DATASET_CACHE = Path("dataset_cache")


def score_gpqa(report_dir: Path) -> tuple[str, float, int]:
    ds = GPQA.load_from_file(DATASET_CACHE / "gpqa/diamond/gpqa_diamond.parquet")
    ds.load()
    name = "gpqa::deepseek_v4"
    scorer = PassAt1Scorer(name, ds, report_dir, extractor=ABCDExtractor)
    score, n_repeats = scorer.score()
    return name, score, n_repeats


def score_aime25(report_dir: Path) -> tuple[str, float, int]:
    ds = AIME25.load_from_file(DATASET_CACHE / "aime25/aime25.parquet")
    ds.load()
    name = "aime25::deepseek_v4"
    scorer = PassAt1Scorer(
        name,
        ds,
        report_dir,
        extractor=BoxedMathExtractor,
        ground_truth_column="answer",
    )
    score, n_repeats = scorer.score()
    return name, score, n_repeats


def score_livecodebench(
    report_dir: Path, lcb_version: str, timeout: int
) -> tuple[str, float, int]:
    ds = LiveCodeBench.load_from_file(
        DATASET_CACHE / f"livecodebench/{lcb_version}/livecodebench_{lcb_version}.parquet"
    )
    ds.load()
    name = "livecodebench::deepseek_v4"
    scorer = LiveCodeBenchScorer(
        name,
        ds,
        report_dir,
        extractor=PythonCodeExtractor,
        lcb_version=lcb_version,
        timeout=timeout,
    )
    score, n_repeats = scorer.score()
    return name, score, n_repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Benchmark report directory (contains events.jsonl and sample_idx_map.json)",
    )
    parser.add_argument(
        "--lcb-version",
        default="release_v6",
        help="LiveCodeBench dataset version tag",
    )
    parser.add_argument(
        "--lcb-timeout",
        type=int,
        default=60,
        help="Per-test timeout for LiveCodeBench evaluation (seconds)",
    )
    parser.add_argument(
        "--skip-lcb",
        action="store_true",
        help="Skip LiveCodeBench (score GPQA and AIME25 only)",
    )
    parser.add_argument(
        "--write-results-json",
        action="store_true",
        help="Write results.json in the same format as benchmark finalize",
    )
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    accuracy_scores: dict[str, dict] = {}

    for label, fn in (
        ("GPQA", lambda: score_gpqa(report_dir)),
        ("AIME25", lambda: score_aime25(report_dir)),
    ):
        name, score, n_repeats = fn()
        print(f"{label} Pass@1 ({n_repeats} repeats): {score:.4f}")
        accuracy_scores[name] = {
            "dataset_name": name,
            "score": score,
            "n_repeats": n_repeats,
        }

    if not args.skip_lcb:
        name, score, n_repeats = score_livecodebench(
            report_dir, args.lcb_version, args.lcb_timeout
        )
        print(f"LiveCodeBench Pass@1 ({n_repeats} repeats): {score:.4f}")
        accuracy_scores[name] = {
            "dataset_name": name,
            "score": score,
            "n_repeats": n_repeats,
        }

    if args.write_results_json:
        out = report_dir / "results.json"
        payload = {"accuracy_scores": accuracy_scores}
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
