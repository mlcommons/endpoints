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

"""Out-of-process runner for the MLCommons DeepSeek-R1 accuracy evaluator.

Invoked by ``inference_endpoint.evaluation.scoring.LegacyMLPerfDeepSeekR1Scorer`` via
``uv run --project``. Reads a parquet of per-sample model outputs, runs the
fetched MLCommons ``eval_accuracy.process_dataframe`` per subset (so one
subset's failure - e.g. a missing LiveCodeBench dataset - does not sink the
others), and writes an aggregate metrics JSON:

    {
      "exact_match": <float 0-100>,        # mean per-sample accuracy
      "tokens_per_sample": <float>,         # mean generated-token count
      "num_samples": <int>,                 # total rows scored
      "complete": <bool>,                   # all subsets evaluated cleanly
      "per_dataset": {<subset>: {exact_match, tokens_per_sample, num_samples,
                                 status}}
    }

The evaluator source + its prm800k / LiveCodeBench submodules are fetched by
``setup_eval.sh`` into ``./mlperf_eval/``. See
examples/07_DeepSeekR1_Example/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deepseek_eval_runner")


def _load_evaluator(eval_dir: Path):
    """Import the fetched MLCommons ``eval_accuracy`` module."""
    if not (eval_dir / "eval_accuracy.py").exists():
        raise FileNotFoundError(
            f"eval_accuracy.py not found at {eval_dir}. Run setup_eval.sh first."
        )
    sys.path.insert(0, str(eval_dir))
    import eval_accuracy  # noqa: E402  (path injected above)

    return eval_accuracy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        required=True,
        help="Parquet with columns model_output, ground_truth, dataset, question",
    )
    ap.add_argument("--output", required=True, help="Path to write metrics JSON")
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer path or HF id for output-token counting (DeepSeek-R1)",
    )
    ap.add_argument(
        "--eval-dir",
        default=str(Path(__file__).resolve().parent / "mlperf_eval"),
        help="Directory holding the fetched eval_accuracy.py + submodules",
    )
    ap.add_argument(
        "--external-subsets",
        default="",
        help=(
            "Comma-separated subset ids to tokenize but NOT grade here (the "
            "caller scores them out-of-band, e.g. livecodebench via the "
            "lcb-service container). Each is reported with exact_match=null and "
            "status='external' and excluded from the in-process aggregate."
        ),
    )
    args = ap.parse_args()
    external_subsets = {
        s.strip() for s in args.external_subsets.split(",") if s.strip()
    }

    eval_accuracy = _load_evaluator(Path(args.eval_dir))

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    # tok_model_output_len is a required column for the evaluator and is the
    # basis for tokens_per_sample. Count with the DeepSeek tokenizer so the
    # number matches MLPerf token accounting.
    logger.info("Loading tokenizer: %s", args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    df["tok_model_output_len"] = [
        len(tok.encode(o, add_special_tokens=False))
        for o in df["model_output"].astype(str)
    ]

    per_dataset: dict[str, dict] = {}
    scored_frames: list[pd.DataFrame] = []
    complete = True

    for subset, group in df.groupby("dataset"):
        group = group.copy()
        if str(subset) in external_subsets:
            # Tokenized above (tokens_per_sample stays correct) but graded
            # out-of-band by the caller; do not run the in-process executor.
            per_dataset[str(subset)] = {
                "exact_match": None,
                "tokens_per_sample": float(group["tok_model_output_len"].mean()),
                "num_samples": int(len(group)),
                "status": "external",
            }
            logger.info(
                "subset=%s external (graded out-of-band) n=%d", subset, len(group)
            )
            continue
        try:
            evaluated = eval_accuracy.process_dataframe(group)
            scored_frames.append(evaluated)
            per_dataset[str(subset)] = {
                "exact_match": float(evaluated["prompt_accuracy"].mean()),
                "tokens_per_sample": float(evaluated["tok_model_output_len"].mean()),
                "num_samples": int(len(evaluated)),
                "status": "ok",
            }
            logger.info(
                "subset=%s exact_match=%.4f n=%d",
                subset,
                per_dataset[str(subset)]["exact_match"],
                len(evaluated),
            )
        except Exception as e:  # noqa: BLE001 - isolate per-subset failures
            complete = False
            logger.exception("subset=%s evaluation FAILED: %s", subset, e)
            per_dataset[str(subset)] = {
                "exact_match": None,
                "tokens_per_sample": float(group["tok_model_output_len"].mean()),
                "num_samples": int(len(group)),
                "status": f"failed: {e}",
            }

    if scored_frames:
        all_scored = pd.concat(scored_frames, ignore_index=True)
        overall_exact_match: float | None = float(all_scored["prompt_accuracy"].mean())
        evaluated_samples = int(len(all_scored))
    else:
        overall_exact_match = None
        evaluated_samples = 0

    # mean() is NaN on an empty/all-NaN frame (e.g. a standalone run on an
    # all-failed set); NaN serializes as invalid JSON the scorer can't decode.
    tps_mean = df["tok_model_output_len"].mean() if len(df) else 0.0
    results = {
        "exact_match": overall_exact_match,
        "tokens_per_sample": 0.0 if pd.isna(tps_mean) else float(tps_mean),
        "num_samples": int(len(df)),
        "evaluated_samples": evaluated_samples,
        "complete": complete,
        "per_dataset": per_dataset,
    }

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if not complete:
        logger.warning(
            "One or more subsets failed to evaluate; exact_match covers "
            "%d/%d samples. See per_dataset[*].status.",
            evaluated_samples,
            len(df),
        )


if __name__ == "__main__":
    main()
