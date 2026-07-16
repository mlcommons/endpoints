#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc MLPerf early-stopping estimates from a benchmark run's ``events.jsonl``.

Rebuilds the per-sample TTFT / TPOT / latency series exactly the way the metrics
aggregator does (same event stream, same gating, same formulas — see
``async_utils/services/metrics_aggregator/metrics_table.py``) and runs the same
early-stopping math over them (``metrics/early_stopping.py``). This produces ES results
for ANY historical run from its event log, including runs made before the
``settings.early_stopping`` feature existed or with it disabled.

Replicated aggregator rules:
  - a sample participates iff its ``sample.issued`` falls inside the
    start/stop_performance_tracking window (rows are only created while tracking is on);
    completions after stop still count for rows that exist.
  - ttft_ns           = recv_first.ts - issued.ts
  - sample_latency_ns = complete.ts   - issued.ts
  - tpot_ns           = (complete.ts - recv_first.ts) / tokens(text_after_first_chunk)
                        (streaming only; token counts via the raw ``tokenizers`` backend
                        with add_special_tokens=False, identical to
                        ``token_metrics.encode_lengths``). Tool-call samples are skipped
                        (their chat-template tokenization path is not replicated).

usage:
  python scripts/es_from_events.py <events.jsonl>
      [--tokenizer <hf-model-dir-or-tokenizer.json>]   # enables TPOT
      [--percentiles 0.5,0.9,0.95,0.99] [--confidence 0.99]
      [--compare <result_summary.json>]                # cross-check vs in-band blocks
      [--json <out.json>]
"""

from __future__ import annotations

import argparse
import json
import sys

import msgspec.json

from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.metrics.early_stopping import (
    CONFIDENCE,
    PERCENTILES,
    es_percentile_estimate,
)

_loads = msgspec.json.decode

_METRIC_TO_SUMMARY = {
    "ttft_ns": "ttft",
    "tpot_ns": "tpot",
    "sample_latency_ns": "latency",
}
_UNITS = {"ttft_ns": ("ms", 1e6), "tpot_ns": ("ms", 1e6), "sample_latency_ns": ("s", 1e9)}
_TPOT_BATCH = 4096


def extract_tpot_text(data) -> str | None:
    """TPOT denominator text from a ``sample.complete`` data payload.

    Returns None for non-TextModelOutput payloads and for tool-call samples
    (chat-template tokenization is not replicated here); otherwise the
    post-first-chunk text ("" when there is none, e.g. single-chunk output).
    """
    if not (isinstance(data, list) and data and data[0] == "TextModelOutput"):
        return None
    output = data[1] if len(data) > 1 else ""
    reasoning = data[2] if len(data) > 2 else None
    tool_calls = data[3] if len(data) > 3 else None
    if tool_calls:
        return None
    return TextModelOutput(
        output=output or "", reasoning=reasoning
    ).text_after_first_chunk()


def compute_series(path, count_tokens=None, progress=False) -> dict[str, list[float]]:
    """Stream the log and rebuild the {ttft_ns, tpot_ns, sample_latency_ns} series.

    ``count_tokens``: callable(list[str]) -> list[int], or None to skip TPOT.
    """
    ttft: list[float] = []
    latency: list[float] = []
    tpot: list[float] = []
    rows: dict[str, list] = {}  # uuid -> [issued_ns, recv_first_ns]
    tracking = False
    tool_call_skipped = 0
    batch_deltas: list[float] = []
    batch_texts: list[str] = []

    def flush() -> None:
        if batch_texts and count_tokens is not None:
            for delta, cnt in zip(batch_deltas, count_tokens(batch_texts), strict=True):
                if cnt > 0:
                    tpot.append(delta / cnt)
        batch_deltas.clear()
        batch_texts.clear()

    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if progress and i and i % 200_000 == 0:
                print(f"  ...{i} events", file=sys.stderr)
            try:
                e = _loads(line)
            except (ValueError, msgspec.DecodeError):
                continue
            et = e.get("event_type")
            if et == "session.start_performance_tracking":
                tracking = True
            elif et == "session.stop_performance_tracking":
                tracking = False
            elif et == "sample.issued":
                if tracking:
                    rows[e["sample_uuid"]] = [e["timestamp_ns"], None]
            elif et == "sample.recv_first":
                row = rows.get(e["sample_uuid"])
                if row is not None and row[1] is None:
                    row[1] = e["timestamp_ns"]
                    ttft.append(e["timestamp_ns"] - row[0])
            elif et == "sample.complete":
                row = rows.pop(e["sample_uuid"], None)
                if row is None:
                    continue
                ts = e["timestamp_ns"]
                latency.append(ts - row[0])
                if count_tokens is None or row[1] is None:
                    continue
                text = extract_tpot_text(e.get("data"))
                if text is None and isinstance(e.get("data"), list):
                    d = e["data"]
                    if d and d[0] == "TextModelOutput" and len(d) > 3 and d[3]:
                        tool_call_skipped += 1
                if text:
                    batch_deltas.append(ts - row[1])
                    batch_texts.append(text)
                    if len(batch_texts) >= _TPOT_BATCH:
                        flush()
    flush()
    if tool_call_skipped:
        print(
            f"WARN: {tool_call_skipped} tool-call samples skipped for TPOT",
            file=sys.stderr,
        )
    return {"ttft_ns": ttft, "tpot_ns": tpot, "sample_latency_ns": latency}


def es_blocks(values, percentiles, confidence: float) -> list[dict]:
    """Sorted-input ES blocks, same shape as result_summary.json ``early_stopping``."""
    values = sorted(values)
    return [
        es_percentile_estimate(values, p, confidence).as_dict() for p in percentiles
    ]


def _make_counter(tokenizer_path: str):
    from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
        encode_lengths,
        load_reference_backend,
    )

    backend = load_reference_backend(tokenizer_path)
    if backend is None:
        raise SystemExit(f"FATAL: could not load tokenizer backend from {tokenizer_path}")
    return lambda texts: encode_lengths(backend, texts)


def _fmt(v, unit: str, div: float) -> str:
    return "-" if v is None else f"{v / div:,.2f} {unit}"


def _cross_check(summary: dict, series_name: str, values: list, blocks: list) -> None:
    md = summary.get(_METRIC_TO_SUMMARY[series_name]) or {}
    completed = summary.get("n_samples_completed")
    if completed is not None and len(values) != completed:
        # ttft/tpot legitimately count fewer (streaming-only / empty-rest skips);
        # latency should equal n_samples_completed.
        note = " ** MISMATCH **" if series_name == "sample_latency_ns" else " (info)"
        print(f"  XCHECK count: ours={len(values)} n_samples_completed={completed}{note}")
    inband = md.get("early_stopping")
    if not inband:
        return
    if isinstance(inband, dict):  # pre-list single-percentile shape
        inband = [inband]
    ours_by_p = {b["percentile"]: b for b in blocks}
    for ib in inband:
        b = ours_by_p.get(ib["percentile"])
        if b is None:
            continue
        same = all(
            b[k] == ib[k]
            for k in ("n", "estimate", "empirical", "discarded", "sufficient")
        )
        print(
            f"  XCHECK in-band p{ib['percentile'] * 100:g}: "
            + ("EXACT MATCH" if same else f"** MISMATCH ** ours={b} inband={ib}")
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("events", help="path to a run's events.jsonl")
    ap.add_argument("--tokenizer", help="HF model dir (or tokenizer.json) — enables TPOT")
    ap.add_argument(
        "--percentiles",
        default=",".join(str(p) for p in PERCENTILES),
        help="offline-analysis override; the in-band report always uses the standard set",
    )
    ap.add_argument("--confidence", type=float, default=CONFIDENCE)
    ap.add_argument("--compare", help="result_summary.json to cross-check against")
    ap.add_argument("--json", dest="json_out", help="write blocks to this JSON file")
    args = ap.parse_args(argv)

    percentiles = [float(x) for x in args.percentiles.split(",")]
    counter = _make_counter(args.tokenizer) if args.tokenizer else None
    if counter is None:
        print("NOTE: no --tokenizer given -> TPOT skipped", file=sys.stderr)

    series = compute_series(args.events, count_tokens=counter, progress=True)
    summary = json.load(open(args.compare)) if args.compare else None

    result: dict[str, list[dict]] = {}
    rows = []
    for name, values in series.items():
        if not values:
            continue
        blocks = es_blocks(values, percentiles, args.confidence)
        result[_METRIC_TO_SUMMARY[name]] = blocks
        unit, div = _UNITS[name]
        print(f"\n{_METRIC_TO_SUMMARY[name]} (from {name}, n={len(values)}):")
        for b in blocks:
            line = (
                f"  p{b['percentile'] * 100:g}: sufficient={b['sufficient']} "
                f"min_queries={b['min_queries']} discarded={b['discarded']}  "
                f"empirical={_fmt(b['empirical'], unit, div)}  "
                f"estimate={_fmt(b['estimate'], unit, div)}"
            )
            if b["estimate"] is not None and b["empirical"]:
                gap = b["estimate"] - b["empirical"]
                line += f"  gap=+{_fmt(gap, unit, div)} (+{100 * gap / b['empirical']:.2f}%)"
            print(line)
            rows.append((_METRIC_TO_SUMMARY[name], b, unit, div))
        if summary is not None:
            _cross_check(summary, name, sorted(values), blocks)

    print("\n--- markdown ---")
    print("| metric | p | n | empirical | ES-adjusted | gap |")
    print("|---|---|---|---|---|---|")
    for m, b, unit, div in rows:
        if b["estimate"] is not None and b["empirical"]:
            gap = b["estimate"] - b["empirical"]
            g = f"+{_fmt(gap, unit, div)} (+{100 * gap / b['empirical']:.2f}%)"
        else:
            g = "-"
        print(
            f"| **{m.upper()}** | p{b['percentile'] * 100:g} | {b['n']} | "
            f"{_fmt(b['empirical'], unit, div)} | {_fmt(b['estimate'], unit, div)} | {g} |"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nblocks written to {args.json_out}")
    return result


if __name__ == "__main__":
    main()
