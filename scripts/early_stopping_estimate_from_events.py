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

Events are decoded with the product's own typed ``EventRecord`` (the exact wire types
the JSONL writer emitted), so event names, payload shapes, and chunk semantics all come
from ``core/record.py`` / ``core/types.py`` rather than a re-implementation.

usage:
  python scripts/early_stopping_estimate_from_events.py <events.jsonl>
      [--tokenizer <hf-model-dir-or-tokenizer.json>]   # enables TPOT
      [--percentiles 0.5,0.9,0.95,0.99] [--confidence 0.99]
      [--compare <result_summary.json>]                # cross-check vs in-band blocks
      [--json <out.json>]
"""

from __future__ import annotations

import argparse
import json
import sys

import msgspec
import msgspec.json
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    DEFAULT_PERCENTILES,
)
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    encode_lengths,
    load_reference_backend,
)
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.metrics.early_stopping import (
    CONFIDENCE,
    es_percentile_estimate,
    es_targets_from_grid,
    grid_percentile_key,
)
from inference_endpoint.metrics.report import SERIES_TO_SUMMARY_FIELD

# mirror of the JSONL writer: msgspec.json.Encoder(enc_hook=EventType.encode_hook)
_DECODER = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)

# Script-local display units, keyed by result_summary.json field. Latency renders in
# seconds (long-CoT runs sit at 100s+ per sample); the in-band report renders ms.
_UNITS = {
    "ttft": ("ms", 1e6),
    "tpot": ("ms", 1e6),
    "latency": ("s", 1e9),
}
# Texts buffered per tokenizer call: 4096 texts x ~16 KB of CoT output ~= 64 MB peak,
# bounded even on login nodes, while large enough to amortize the batch-encode overhead.
_TPOT_BATCH = 4096


def extract_tpot_text(data) -> str | None:
    """TPOT denominator text from a decoded ``EventRecord.data``.

    None for non-``TextModelOutput`` payloads and for tool-call samples (their
    chat-template tokenization path is not replicated here); otherwise the
    product's own ``TextModelOutput.text_after_first_chunk()`` ("" when there is
    no post-first chunk, e.g. single-chunk or non-streaming output).
    """
    if not isinstance(data, TextModelOutput) or data.tool_calls:
        return None
    return data.text_after_first_chunk()


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
    malformed = 0
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
                rec = _DECODER.decode(line)
            except (msgspec.DecodeError, NotImplementedError):
                # DecodeError covers malformed JSON and wrong-shape records;
                # EventType.decode_hook raises NotImplementedError for a
                # non-string event_type, which msgspec does not wrap.
                malformed += 1
                continue
            et = rec.event_type
            if et is SessionEventType.START_PERFORMANCE_TRACKING:
                tracking = True
            elif et is SessionEventType.STOP_PERFORMANCE_TRACKING:
                tracking = False
            elif et in (
                SampleEventType.ISSUED,
                SampleEventType.RECV_FIRST,
                SampleEventType.COMPLETE,
            ):
                uuid, ts = rec.sample_uuid, rec.timestamp_ns
                if not uuid:
                    malformed += 1
                    continue
                if et is SampleEventType.ISSUED:
                    if not tracking:
                        continue
                    row = rows.get(uuid)
                    if row is not None:
                        # duplicate ISSUED (retry): aggregator parity — keep the
                        # row (recv_first preserved), refresh only the issue ts.
                        row[0] = ts
                    else:
                        rows[uuid] = [ts, None]
                elif et is SampleEventType.RECV_FIRST:
                    row = rows.get(uuid)
                    if row is not None:
                        # aggregator parity: every RECV_FIRST re-fires the ttft
                        # trigger and refreshes the TPOT window start (retried
                        # streaming attempts re-emit their first chunk).
                        row[1] = ts
                        ttft.append(ts - row[0])
                else:  # SampleEventType.COMPLETE
                    row = rows.pop(uuid, None)
                    if row is None:
                        continue
                    latency.append(ts - row[0])
                    if count_tokens is None or row[1] is None:
                        continue
                    text = extract_tpot_text(rec.data)
                    if text is None and isinstance(rec.data, TextModelOutput):
                        tool_call_skipped += 1
                    if text:
                        batch_deltas.append(ts - row[1])
                        batch_texts.append(text)
                        if len(batch_texts) >= _TPOT_BATCH:
                            flush()
    flush()
    if malformed:
        print(
            f"WARN: {malformed} malformed event lines skipped — the series may be "
            "incomplete (truncated/corrupt log?)",
            file=sys.stderr,
        )
    if tool_call_skipped:
        print(
            f"WARN: {tool_call_skipped} tool-call samples skipped for TPOT",
            file=sys.stderr,
        )
    return {"ttft_ns": ttft, "tpot_ns": tpot, "sample_latency_ns": latency}


def es_blocks(sorted_values, percentiles, confidence: float) -> list[dict]:
    """ES blocks over an ascending-sorted series, same shape as ``early_stopping``."""
    return [
        es_percentile_estimate(sorted_values, p, confidence).as_dict()
        for p in percentiles
    ]


def _make_counter(tokenizer_path: str):
    backend = load_reference_backend(tokenizer_path)
    if backend is None:
        raise SystemExit(
            f"FATAL: could not load tokenizer backend from {tokenizer_path}"
        )
    return lambda texts: encode_lengths(backend, texts)


def _fmt(v, unit: str, div: float) -> str:
    return "-" if v is None else f"{v / div:,.2f} {unit}"


def _cross_check(summary: dict, series_name: str, values: list, blocks: list) -> None:
    md = summary.get(SERIES_TO_SUMMARY_FIELD[series_name]) or {}
    completed = summary.get("n_samples_completed")
    if completed is not None and len(values) != completed:
        # ttft/tpot legitimately count fewer (streaming-only / empty-rest skips);
        # latency should equal n_samples_completed.
        note = " ** MISMATCH **" if series_name == "sample_latency_ns" else " (info)"
        print(
            f"  XCHECK count: ours={len(values)} n_samples_completed={completed}{note}"
        )
    inband = md.get("early_stopping_percentiles")
    if not isinstance(inband, dict) or not inband:
        # absent, or an unexpected shape (pre-release artifact) — nothing to compare
        return
    ours = {grid_percentile_key(b["percentile"]): b["estimate"] for b in blocks}
    for key, est in inband.items():
        if key not in ours:
            continue
        same = ours[key] == est
        print(
            f"  XCHECK in-band p{key}: "
            + (
                "EXACT MATCH"
                if same
                else f"** MISMATCH ** ours={ours[key]} inband={est}"
            )
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("events", help="path to a run's events.jsonl")
    ap.add_argument(
        "--tokenizer", help="HF model dir (or tokenizer.json) — enables TPOT"
    )
    ap.add_argument(
        "--percentiles",
        default=",".join(
            str(f) for f in es_targets_from_grid(DEFAULT_PERCENTILES).values()
        ),
        help="offline-analysis override; defaults to the report grid at/above the median",
    )
    ap.add_argument("--confidence", type=float, default=CONFIDENCE)
    ap.add_argument("--compare", help="result_summary.json to cross-check against")
    ap.add_argument("--json", dest="json_out", help="write blocks to this JSON file")
    args = ap.parse_args(argv)

    percentiles = [float(x) for x in args.percentiles.split(",")]
    if not all(0.0 < p < 1.0 for p in percentiles):
        raise SystemExit(f"FATAL: percentiles must be in (0, 1), got {percentiles}")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit(f"FATAL: confidence must be in (0, 1), got {args.confidence}")
    counter = _make_counter(args.tokenizer) if args.tokenizer else None
    if counter is None:
        print("NOTE: no --tokenizer given -> TPOT skipped", file=sys.stderr)

    series = compute_series(args.events, count_tokens=counter, progress=True)
    summary = None
    if args.compare:
        with open(args.compare) as f:
            summary = json.load(f)

    result: dict[str, list[dict]] = {}
    rows = []
    for name, values in series.items():
        if not values:
            continue
        values.sort()
        blocks = es_blocks(values, percentiles, args.confidence)
        result[SERIES_TO_SUMMARY_FIELD[name]] = blocks
        unit, div = _UNITS[SERIES_TO_SUMMARY_FIELD[name]]
        print(f"\n{SERIES_TO_SUMMARY_FIELD[name]} (from {name}, n={len(values)}):")
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
            rows.append((SERIES_TO_SUMMARY_FIELD[name], b, unit, div))
        if summary is not None:
            _cross_check(summary, name, values, blocks)

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
