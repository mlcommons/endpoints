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

"""CLI entry point for BFCL v4 multi-turn accuracy evaluation.

Runs multi-turn agentic conversations against an OpenAI-compatible endpoint
and scores results using bfcl-eval's multi_turn_checker.

Usage:
    python -m inference_endpoint.evaluation.bfcl_v4_multi_turn_cli \\
        --endpoint http://localhost:8080 \\
        --model Qwen3.6-27B \\
        --subsets multi_turn_base \\
        --report-dir /tmp/bfcl_mt_results

    # Or run all multi-turn subsets:
    python -m inference_endpoint.evaluation.bfcl_v4_multi_turn_cli \\
        --endpoint http://localhost:8080 \\
        --model Qwen3.6-27B \\
        --report-dir /tmp/bfcl_mt_results
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from ..dataset_manager.predefined.bfcl_v4.multi_turn import (
    MULTI_TURN_SUBSETS,
    load_multi_turn_entries,
)
from .bfcl_v4_multi_turn_runner import BFCLMultiTurnRunner
from .bfcl_v4_multi_turn_scorer import BFCLv4MultiTurnScorer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BFCL v4 Multi-Turn Accuracy Evaluation"
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="OpenAI-compatible endpoint URL (e.g., http://localhost:8080)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to send in requests",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        choices=MULTI_TURN_SUBSETS,
        help=f"Subsets to evaluate. Default: all ({', '.join(MULTI_TURN_SUBSETS)})",
    )
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=None,
        help="Percentage (0-100) of entries per subset to evaluate. "
        "Selection is deterministic (first N). Default: all entries.",
    )
    parser.add_argument(
        "--api-key",
        default="not-needed",
        help="API key for endpoint authentication",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for determinism)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None)",
    )
    parser.add_argument(
        "--max-steps-per-turn",
        type=int,
        default=25,
        help="Max steps within a single turn before force-termination (default: 25)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    subsets = args.subsets or MULTI_TURN_SUBSETS

    # Load entries
    logger.info("Loading BFCL v4 multi-turn entries for: %s", subsets)
    entries = load_multi_turn_entries(subsets=subsets)
    logger.info("Loaded %d entries", len(entries))

    if args.sample_pct is not None:
        if not (0 < args.sample_pct <= 100):
            parser.error("--sample-pct must be in (0, 100]")
        by_subset: dict[str, list] = defaultdict(list)
        for entry in entries:
            by_subset[entry.subset].append(entry)
        sampled: list = []
        for subset_name, subset_entries in by_subset.items():
            n = max(1, int(len(subset_entries) * args.sample_pct / 100))
            sampled.extend(subset_entries[:n])
            logger.info(
                "  %s: %d/%d entries (%.1f%%)",
                subset_name,
                n,
                len(subset_entries),
                args.sample_pct,
            )
        entries = sampled
        logger.info(
            "After %.1f%% sampling: %d total entries", args.sample_pct, len(entries)
        )

    if not entries:
        logger.warning(
            "No entries to evaluate for subsets %s (after sampling); nothing to run.",
            subsets,
        )
        return

    # Run conversations
    logger.info(
        "Starting evaluation against %s (model=%s, temperature=%s, seed=%s)",
        args.endpoint,
        args.model,
        args.temperature,
        args.seed,
    )
    t0 = time.time()

    with BFCLMultiTurnRunner(
        endpoint_url=args.endpoint,
        model_name=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        seed=args.seed,
        max_steps_per_turn=args.max_steps_per_turn,
        timeout_s=args.timeout,
    ) as runner:

        def progress(idx: int, total: int, entry_id: str) -> None:
            if idx % 10 == 0 or idx == total - 1:
                logger.info("Progress: %d/%d (%s)", idx + 1, total, entry_id)

        results = runner.run_all(entries, progress_callback=progress)

    elapsed = time.time() - t0
    logger.info("Evaluation complete in %.1f min (%.0f s)", elapsed / 60, elapsed)

    # Score
    scorer = BFCLv4MultiTurnScorer()
    scores = scorer.score(results)

    # Display results
    print("\n" + "=" * 60)
    print("BFCL v4 Multi-Turn Accuracy Results")
    print("=" * 60)
    print(f"\nOverall Multi-Turn Accuracy: {scores['overall_accuracy']}%")
    print("\nPer-Subset Scores:")
    for subset, acc in scores["subset_scores"].items():
        total = sum(1 for r in results if r["subset"] == subset)
        correct = sum(
            1
            for s in scores["per_entry_scores"]
            if s["subset"] == subset and s["valid"]
        )
        print(f"  {subset:<30} {acc}% ({correct}/{total})")

    print(f"\nTotal Samples: {scores['total_samples']}")
    print(f"Total Time: {elapsed:.1f}s ({elapsed / len(entries):.2f}s/entry)")

    # Save results
    if args.report_dir:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "config": {
                "endpoint": args.endpoint,
                "model": args.model,
                "temperature": args.temperature,
                "subsets": subsets,
                "max_steps_per_turn": args.max_steps_per_turn,
            },
            "accuracy_scores": [
                {
                    "dataset_name": "bfcl_v4::multi_turn",
                    "extractor": None,
                    "ground_truth_column": None,
                    "score": scores.get("overall_accuracy"),
                    "unit_samples": scores["total_samples"],
                    "num_repeats": 1,
                    "total_samples": scores["total_samples"],
                    "complete": True,
                    "breakdown": scores,
                },
            ],
            "elapsed_time": elapsed,
        }

        results_path = report_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", results_path)

        # Save detailed per-entry scores
        detail_path = report_dir / "per_entry_scores.json"
        with open(detail_path, "w") as f:
            json.dump(scores["per_entry_scores"], f, indent=2)
        logger.info("Detailed scores saved to %s", detail_path)


if __name__ == "__main__":
    main()
