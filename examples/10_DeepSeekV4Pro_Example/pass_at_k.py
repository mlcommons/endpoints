#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute pass@1 and pass@k (default k=4) for AIME25, GPQA and LiveCodeBench.

Reads a finished benchmark report directory (``events.jsonl`` +
``sample_idx_map.json``) produced by ``inference-endpoint benchmark from-config``
and reports both pass@1 and pass@k for each accuracy dataset. Requires the run to
have used ``num_repeats >= k`` (this example uses ``num_repeats: 4``).

pass@k uses the unbiased estimator from Chen et al. 2021 (HumanEval):

    pass@k = E_problems [ 1 - C(n - c, k) / C(n, k) ]

where ``n`` is the number of attempts for a problem and ``c`` the number that
passed. pass@1 reduces to the mean per-attempt success rate (matching the
framework's native ``pass_at_1`` scorer).

Usage:
    uv run python examples/10_DeepSeekV4Pro_Example/pass_at_k.py \
        --report-dir results/sglang_deepseek_v4_pro_accuracy \
        --write-json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections import defaultdict
from math import comb
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


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator for one problem with n attempts, c correct."""
    if k > n:
        # Not enough attempts to estimate pass@k for this problem; treat the
        # available attempts as the sample (any-correct) so a short problem does
        # not silently drop out of the average.
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _aggregate_pass_at_k(
    per_problem: dict[int | str, list[bool]], k: int
) -> tuple[float, float, int]:
    """Return (pass@1, pass@k, min_attempts) averaged over problems."""
    if not per_problem:
        return 0.0, 0.0, 0
    p1_terms = []
    pk_terms = []
    min_attempts = min(len(v) for v in per_problem.values())
    for results in per_problem.values():
        n = len(results)
        c = sum(1 for r in results if r)
        p1_terms.append(_pass_at_k(n, c, 1))
        pk_terms.append(_pass_at_k(n, c, k))
    return (
        sum(p1_terms) / len(p1_terms),
        sum(pk_terms) / len(pk_terms),
        min_attempts,
    )


def _per_problem_bools_exact_match(
    scorer: PassAt1Scorer,
) -> dict[int, list[bool]]:
    """Group per-attempt correctness by dataset question index (exact-match scorers)."""
    df = scorer.get_outputs()
    valid = scorer.sample_index_map.keys()
    df = df[df["sample_uuid"].isin(valid)]
    if df.empty:
        raise KeyError(
            f"No COMPLETE events matched sample_idx_map for {scorer.dataset_name}"
        )
    df = df.apply(scorer.match_sample_index, axis=1)

    empirical = df["output"]
    if scorer.extractor is not None:
        empirical = empirical.apply(scorer.extractor.extract)
    empirical = empirical.to_numpy()

    assert scorer.dataset.dataframe is not None
    order = df["sample_index"].to_numpy()
    ground_truths = scorer.dataset.dataframe[scorer.ground_truth_column].to_numpy()[
        order
    ]

    per_problem: dict[int, list[bool]] = defaultdict(list)
    for i, sample_index in enumerate(order):
        passed = scorer.score_single_sample(empirical[i], ground_truths[i]) >= 1.0
        per_problem[int(sample_index)].append(passed)
    return per_problem


def score_exact_match(
    label: str,
    name: str,
    ds,
    report_dir: Path,
    extractor,
    ground_truth_column: str,
    k: int,
) -> dict:
    scorer = PassAt1Scorer(
        name,
        ds,
        report_dir,
        extractor=extractor,
        ground_truth_column=ground_truth_column,
    )
    per_problem = _per_problem_bools_exact_match(scorer)
    p1, pk, min_attempts = _aggregate_pass_at_k(per_problem, k)
    print(
        f"{label}: pass@1={p1:.4f}  pass@{k}={pk:.4f}  "
        f"(problems={len(per_problem)}, min_attempts={min_attempts})"
    )
    return {
        "dataset_name": name,
        "pass_at_1": p1,
        f"pass_at_{k}": pk,
        "n_problems": len(per_problem),
        "min_attempts": min_attempts,
    }


def _lcb_per_problem_bools(
    report_dir: Path, ds, lcb_version: str, timeout: int
) -> dict[str, list[bool]]:
    """Extract code per attempt and evaluate via the isolated lcb_serve subprocess.

    Returns {question_id: [passed_bool, ...]} using lcb_serve's per-question
    results. Runs lcb_serve in a subprocess (same mechanism the scorer uses) so
    LiveCodeBench's sandboxing does not corrupt this process.
    """
    scorer = LiveCodeBenchScorer(
        "livecodebench::deepseek_v4",
        ds,
        report_dir,
        extractor=PythonCodeExtractor,
        lcb_version=lcb_version,
        timeout=timeout,
        lcb_websocket_port=None,
    )
    df = scorer.get_outputs()
    df = df[df["sample_uuid"].isin(scorer.sample_index_map.keys())]
    df = df.apply(scorer.match_sample_index, axis=1)

    assert ds.dataframe is not None
    df["question_id"] = df["sample_index"].apply(
        lambda idx: ds.dataframe.iloc[idx][scorer.question_id_column]
    )
    df["extracted_code"] = df["output"].apply(
        lambda x: PythonCodeExtractor.extract(x, default="# FAILED TO EXTRACT CODE")
    )

    datasets_dir = os.environ.get(
        "LCB_DATASETS_DIR", str(DATASET_CACHE / "livecodebench" / lcb_version)
    )

    worker = (
        "import json,sys,pandas as pd;"
        "from collections import defaultdict;"
        "from inference_endpoint.evaluation.livecodebench.lcb_serve import LCBServe;"
        "df=pd.read_parquet(sys.argv[1]);"
        "df['extracted_code']=df['extracted_code'].fillna('');"
        "cd=defaultdict(list);"
        "[cd[r['question_id']].append(r['extracted_code']) for _,r in df.iterrows()];"
        "srv=LCBServe(version_tag=sys.argv[2], datasets_dir=sys.argv[3]);"
        "res=srv.evaluate(codes_dict=cd, timeout_sec=int(sys.argv[4]));"
        "print('LCB_JSON_START'+json.dumps({q:[bool(b) for b in v] for q,v in res.items()})+'LCB_JSON_END')"
    )

    with tempfile.TemporaryDirectory() as tmp:
        pq = Path(tmp) / f"{uuid.uuid4()}.parquet"
        df[["question_id", "extracted_code"]].to_parquet(pq)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                worker,
                str(pq),
                lcb_version,
                datasets_dir,
                str(timeout),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise RuntimeError("lcb_serve worker failed")
        out = proc.stdout
        payload = out[
            out.index("LCB_JSON_START") + len("LCB_JSON_START") : out.index(
                "LCB_JSON_END"
            )
        ]
        return json.loads(payload)


def score_livecodebench(
    report_dir: Path, lcb_version: str, timeout: int, k: int
) -> dict:
    ds = LiveCodeBench.load_from_file(
        DATASET_CACHE
        / f"livecodebench/{lcb_version}/livecodebench_{lcb_version}.parquet"
    )
    ds.load()
    per_problem = _lcb_per_problem_bools(report_dir, ds, lcb_version, timeout)
    p1, pk, min_attempts = _aggregate_pass_at_k(per_problem, k)
    print(
        f"LiveCodeBench: pass@1={p1:.4f}  pass@{k}={pk:.4f}  "
        f"(problems={len(per_problem)}, min_attempts={min_attempts})"
    )
    return {
        "dataset_name": "livecodebench::deepseek_v4",
        "pass_at_1": p1,
        f"pass_at_{k}": pk,
        "n_problems": len(per_problem),
        "min_attempts": min_attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4, help="k for pass@k (default 4)")
    parser.add_argument("--lcb-version", default="release_v6")
    parser.add_argument("--lcb-timeout", type=int, default=60)
    parser.add_argument("--skip-lcb", action="store_true")
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    results: dict[str, dict] = {}

    gpqa = GPQA.load_from_file(DATASET_CACHE / "gpqa/diamond/gpqa_diamond.parquet")
    gpqa.load()
    results["gpqa::deepseek_v4"] = score_exact_match(
        "GPQA",
        "gpqa::deepseek_v4",
        gpqa,
        report_dir,
        ABCDExtractor,
        "ground_truth",
        args.k,
    )

    aime = AIME25.load_from_file(DATASET_CACHE / "aime25/aime25.parquet")
    aime.load()
    results["aime25::deepseek_v4"] = score_exact_match(
        "AIME25",
        "aime25::deepseek_v4",
        aime,
        report_dir,
        BoxedMathExtractor,
        "answer",
        args.k,
    )

    if not args.skip_lcb:
        results["livecodebench::deepseek_v4"] = score_livecodebench(
            report_dir, args.lcb_version, args.lcb_timeout, args.k
        )

    if args.write_json:
        out = report_dir / "pass_at_k.json"
        out.write_text(json.dumps({"k": args.k, "results": results}, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
