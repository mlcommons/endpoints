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

"""Study 1 — hairball / long-tail bunching analysis.

Given a Phase-0 per-sample Parquet table (``samples.parquet``) plus its
``run_meta.json`` and the raw ``events.jsonl`` for one corpus point, this script:

1. Partitions the windowed region (super_pass >= 1, warmup dropped) into the
   ``drain_tail`` (complete_ns > last_issue_ns) and ``steady`` (complement) sets.
2. Streams ``events.jsonl`` once to recover per-sample output character length
   (an OSL proxy available for ALL completed samples) and, for the tokenize-set
   (the full tail + an equal-size random steady sample), the post-first-chunk
   text needed for a real TPOT number.
3. Tokenizes only the tokenize-set with the point's model tokenizer to get
   ``out_tokens`` and ``tpot_ns = (complete_ns - first_token_ns) / out_tokens``.
4. Reports the tail-vs-steady TPOT comparison (KS, CoV, quantiles), the
   lifetime<->OSL correlation, and the bias of dropping the tail on throughput,
   latency tail (p99/p99.9), and TPOT.

Run (one point at a time)::

    uv run --with pyarrow --with numpy --with scipy --with transformers \
        python scripts/steady_state_tail_analysis.py \
        --point-dir ~/skritch/endpoints-events-jsonl-artifacts/C7168 \
        --tokenizer openai/gpt-oss-120b \
        --out ~/skritch/endpoints-events-jsonl-artifacts/C7168/tail_analysis.json

The heavy per-point results land beside the events (never in the repo). The
findings doc ``docs/design/2026-07-29-hairball-findings.md`` consolidates them.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pyarrow.parquet as pq
from inference_endpoint.core.record import EventRecord, EventType
from inference_endpoint.core.types import TextModelOutput
from scipy import stats

_DECODER = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
_COMPLETE_PREFIX = b'{"event_type":"sample.complete"'


def _percentiles(values: np.ndarray, ps: tuple[float, ...]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    q = np.percentile(values, list(ps))
    return {f"p{p}": float(v) for p, v in zip(ps, q, strict=True)}


def _cov(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    m = float(np.mean(values))
    return float(np.std(values) / m) if m else float("nan")


def stream_outputs(
    events_path: Path,
    tokenize_uuids: set[str],
) -> tuple[dict[str, int], dict[str, str]]:
    """One streaming pass over events.jsonl.

    Returns ``(char_len, tok_text)`` where ``char_len`` maps every completed
    sample uuid to the character length of its full output string (OSL proxy),
    and ``tok_text`` maps each tokenize-set uuid to its post-first-chunk text
    (TPOT denominator input).
    """
    char_len: dict[str, int] = {}
    tok_text: dict[str, str] = {}
    with open(events_path, "rb") as f:
        for line in f:
            if not line.startswith(_COMPLETE_PREFIX):
                continue
            try:
                rec = _DECODER.decode(line)
            except (msgspec.DecodeError, NotImplementedError):
                continue
            data = rec.data
            if not isinstance(data, TextModelOutput):
                continue
            uuid = rec.sample_uuid
            char_len[uuid] = len(str(data))
            if uuid in tokenize_uuids and not data.tool_calls:
                tok_text[uuid] = data.text_after_first_chunk()
    return char_len, tok_text


def tokenize_counts(texts: list[str], tokenizer_name: str) -> list[int]:
    from transformers import AutoTokenizer  # heavy optional dep, load lazily

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    counts: list[int] = []
    # Batch to bound peak memory on large tail sets.
    for i in range(0, len(texts), 1024):
        batch = texts[i : i + 1024]
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        counts.extend(len(ids) for ids in enc)
    return counts


def analyze(point_dir: Path, tokenizer_name: str, seed: int) -> dict[str, Any]:
    meta = json.loads((point_dir / "run_meta.json").read_text())
    last_issue = meta["last_issue_ns"]
    tbl = pq.read_table(point_dir / "samples.parquet")
    uuid = np.array(tbl.column("uuid").to_pylist())
    sp = tbl.column("super_pass").to_numpy()
    complete_ns = tbl.column("complete_ns").to_numpy()
    first_token_ns = tbl.column("first_token_ns").to_numpy()
    lifetime_ns = tbl.column("lifetime_ns").to_numpy().astype(np.float64)

    window = sp >= 1
    is_tail = window & (complete_ns > last_issue)
    is_steady = window & (complete_ns <= last_issue)

    tail_idx = np.flatnonzero(is_tail)
    steady_idx = np.flatnonzero(is_steady)

    result: dict[str, Any] = {
        "point": point_dir.name,
        "N": meta["N"],
        "super_pass_size": meta["super_pass_size"],
        "n_super_passes": int(sp.max()) + 1,
        "n_rows": int(tbl.num_rows),
        "n_window": int(window.sum()),
        "n_tail": int(is_tail.sum()),
        "n_steady": int(is_steady.sum()),
        "tail_frac_of_window": float(is_tail.sum() / window.sum())
        if window.sum()
        else float("nan"),
        "drain_after_last_issue_s": (meta["last_complete_ns"] - last_issue) / 1e9,
    }
    if tail_idx.size == 0 or steady_idx.size == 0:
        result["skipped"] = "empty windowed region (partial_dataset control)"
        return result

    # Equal-size random steady sample for the tokenize-set.
    rng = random.Random(seed)
    steady_sample_idx = np.array(
        sorted(rng.sample(list(steady_idx), min(tail_idx.size, steady_idx.size)))
    )
    tokenize_idx = np.concatenate([tail_idx, steady_sample_idx])
    tokenize_uuids = set(uuid[tokenize_idx].tolist())

    char_len_map, tok_text_map = stream_outputs(
        point_dir / "events.jsonl", tokenize_uuids
    )

    # OSL-proxy char length for ALL completed samples, aligned to the table.
    char_len = np.array([char_len_map.get(u, -1) for u in uuid], dtype=np.float64)
    have_cl = char_len >= 0

    # --- lifetime <-> OSL(char) correlation across ALL completed samples ---
    cl_ok = have_cl & (lifetime_ns > 0)
    if cl_ok.sum() > 2:
        pear = stats.pearsonr(lifetime_ns[cl_ok], char_len[cl_ok])
        spear = stats.spearmanr(lifetime_ns[cl_ok], char_len[cl_ok])
        result["lifetime_vs_charlen_all"] = {
            "n": int(cl_ok.sum()),
            "pearson_r": float(pear.statistic),
            "spearman_r": float(spear.statistic),
            "tail_charlen_median": float(np.median(char_len[is_tail & have_cl])),
            "steady_charlen_median": float(np.median(char_len[is_steady & have_cl])),
        }

    # --- tokenize the tokenize-set -> out_tokens, TPOT ---
    order = list(tokenize_idx)
    texts: list[str] = []
    text_idx: list[int] = []
    for i in order:
        u = uuid[i]
        t = tok_text_map.get(u)
        if t:
            texts.append(t)
            text_idx.append(i)
    counts = tokenize_counts(texts, tokenizer_name)

    out_tok = np.full(tbl.num_rows, -1.0)
    for i, c in zip(text_idx, counts, strict=True):
        out_tok[i] = c
    have_tok = out_tok > 0
    tpot = np.full(tbl.num_rows, np.nan)
    decode_ns = (complete_ns - first_token_ns).astype(np.float64)
    tpot[have_tok] = decode_ns[have_tok] / out_tok[have_tok]

    tail_tok = is_tail & have_tok
    steady_tok = np.zeros(tbl.num_rows, dtype=bool)
    steady_tok[steady_sample_idx] = True
    steady_tok &= have_tok

    tail_tpot = tpot[tail_tok]
    steady_tpot = tpot[steady_tok]

    ks = stats.ks_2samp(tail_tpot, steady_tpot)
    result["tpot"] = {
        "tokenizer": tokenizer_name,
        "n_tail_tok": int(tail_tok.sum()),
        "n_steady_tok": int(steady_tok.sum()),
        "tail_tpot_ms": {
            "mean": float(np.mean(tail_tpot) / 1e6),
            "cov": _cov(tail_tpot),
            **{k: v / 1e6 for k, v in _percentiles(tail_tpot, (50, 90, 99)).items()},
        },
        "steady_tpot_ms": {
            "mean": float(np.mean(steady_tpot) / 1e6),
            "cov": _cov(steady_tpot),
            **{k: v / 1e6 for k, v in _percentiles(steady_tpot, (50, 90, 99)).items()},
        },
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }

    # --- lifetime <-> out_tokens on the tokenized subset ---
    both = have_tok & (lifetime_ns > 0)
    if both.sum() > 2:
        pear = stats.pearsonr(lifetime_ns[both], out_tok[both])
        result["lifetime_vs_outtokens_subset"] = {
            "n": int(both.sum()),
            "pearson_r": float(pear.statistic),
            "tail_outtok_median": float(np.median(out_tok[tail_tok])),
            "steady_outtok_median": float(np.median(out_tok[steady_tok])),
        }

    # --- per-metric bias of dropping the tail ---
    # Latency (all completed samples in window) vs steady-only.
    lat_win = lifetime_ns[window].astype(np.float64)
    lat_steady = lifetime_ns[is_steady].astype(np.float64)

    def _bias(full: np.ndarray, cut: np.ndarray, ps: tuple[float, ...]) -> dict:
        out = {}
        for p in ps:
            fv = float(np.percentile(full, p))
            cv = float(np.percentile(cut, p))
            out[f"p{p}"] = {
                "with_tail": fv,
                "tail_dropped": cv,
                "rel_change_pct": 100.0 * (cv - fv) / fv if fv else float("nan"),
            }
        return out

    result["bias"] = {
        "latency_ns": _bias(lat_win, lat_steady, (99, 99.9)),
    }

    # Throughput: issue-to-issue denominator already excludes drain-tail
    # completions from the numerator convention, but here we report the
    # completion-rate framing the two sets imply.
    span_s = (last_issue - meta["first_issue_ns"]) / 1e9
    result["bias"]["throughput"] = {
        "issue_span_s": span_s,
        "window_completions": int(window.sum()),
        "steady_completions": int(is_steady.sum()),
        # Completions that finish after issuance stops never enter an
        # issue-to-issue throughput numerator regardless of tail-drop.
        "tail_completions_excluded_from_issue_denom": int(is_tail.sum()),
        "completion_rate_over_window_qps": float(window.sum() / span_s),
        "completion_rate_steady_qps": float(is_steady.sum() / span_s),
    }

    # TPOT bias: p50/p99 over the tokenized union (tail+steady sample) vs
    # steady-sample-only. Same-size sets keep this an apples-to-apples shift.
    union_tpot = tpot[have_tok]
    result["bias"]["tpot_ms"] = _bias_ms(union_tpot, steady_tpot, (50, 99))
    return result


def _bias_ms(full: np.ndarray, cut: np.ndarray, ps: tuple[float, ...]) -> dict:
    out = {}
    for p in ps:
        fv = float(np.percentile(full, p)) / 1e6
        cv = float(np.percentile(cut, p)) / 1e6
        out[f"p{p}"] = {
            "with_tail": fv,
            "tail_dropped": cv,
            "rel_change_pct": 100.0 * (cv - fv) / fv if fv else float("nan"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--point-dir", required=True, type=Path)
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer id, e.g. openai/gpt-oss-120b or deepseek-ai/DeepSeek-R1",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", default=1234, type=int)
    args = ap.parse_args()

    result = analyze(args.point_dir.expanduser(), args.tokenizer, args.seed)
    args.out.expanduser().write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
