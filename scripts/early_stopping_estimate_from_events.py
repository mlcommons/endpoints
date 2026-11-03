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
  - tpot_ns           = (complete.ts - recv_first.ts) / tokens after the first chunk
                        (streaming only). The denominator follows the run's token-count
                        basis, read by ``token_count_source_for_run`` from --summary
                        when given, else from the event log's directory
                        (--token-count-source overrides it):
      server:           ``completion_tokens - 1`` from the COMPLETE record's usage
                        (``classify_server_usage``, then LoadGen's rule, as the
                        aggregator). Samples without usable usage, with
                        completion_tokens <= 1 or a non-positive window are skipped.
                        Tool-call samples are included. No tokenizer is needed.
      client:           tokens(text_after_first_chunk) via the raw ``tokenizers``
                        backend with add_special_tokens=False, identical to
                        ``token_metrics.encode_lengths``. Tool-call samples are skipped
                        (their chat-template tokenization path is not replicated).

Events are decoded with the product's own typed ``EventRecord`` (the exact wire types
the JSONL writer emitted), so event names, payload shapes, and chunk semantics all come
from ``core/record.py`` / ``core/types.py`` rather than a re-implementation.

usage:
  python scripts/early_stopping_estimate_from_events.py <events.jsonl>
      [--summary <result_summary.json>]  # the run's summary: enables the in-band
                                         # cross-check, derives the targets from the
                                         # run's own percentile grid, powers --json
      [--json <out.json>]                # write the summary AUGMENTED with
                                         # early_stopping_percentiles maps — exactly the
                                         # shape an ES-enabled run would have produced
      [--token-count-source server|client]  # overrides the run's recorded basis;
                                         # --json refuses one that differs from it
                                         # (it would mislabel the TPOT maps)
      [--tokenizer <hf-model-dir-or-tokenizer.json>]   # enables client-basis TPOT
      [--percentiles 50,90,95,99.9] [--confidence 0.99]      # stdout-analysis overrides
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from inference_endpoint.core.types import TextModelOutput, TokenCountSource
from inference_endpoint.metrics.early_stopping import (
    CONFIDENCE,
    es_percentile_estimate,
    es_targets_from_grid,
)
from inference_endpoint.metrics.report import (
    SERIES_TO_SUMMARY_FIELD,
    place_early_stopping_percentiles,
)
from inference_endpoint.metrics.token_count_basis import (
    TOKEN_COUNT_SOURCE_METAVAR,
    RunBasisError,
    ServerUsage,
    read_server_usage,
    replay_basis,
)

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


def compute_series(
    events_path,
    *,
    count_tokens,
    basis: TokenCountSource,
    progress=False,
) -> dict[str, list[float]]:
    """Stream the log and rebuild the {ttft_ns, tpot_ns, sample_latency_ns} series.

    ``basis`` is the TPOT basis replayed on. SERVER: TPOT from the COMPLETE
    record's usable ``completion_tokens`` (``read_server_usage``, as the
    aggregator); ``count_tokens`` is then unused. Tracked ``TextModelOutput``
    completions without usable usage are counted as missing usage (warned on
    stderr), never tokenized.
    CLIENT: ``count_tokens`` is callable(list[str]) -> list[int], or None to
    skip TPOT.
    """
    server_basis = basis is TokenCountSource.SERVER
    ttft: list[float] = []
    latency: list[float] = []
    tpot: list[float] = []
    rows: dict[str, list] = {}  # uuid -> [issued_ns, recv_first_ns]
    tracking = False
    tool_call_skipped = 0
    missing_usage = 0
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

    with open(events_path, "rb") as f:
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
                    # no row -> the sample was issued outside the tracking window
                    # (e.g. warmup); its events are expected and skipped, same as
                    # the aggregator — not malformed input.
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
                    if server_basis:
                        if isinstance(rec.data, TextModelOutput):
                            usage, n = read_server_usage(
                                rec.prompt_tokens, rec.completion_tokens, rec.data
                            )
                            if usage is ServerUsage.MISSING:
                                missing_usage += 1
                            if (
                                n is not None
                                and n > 1
                                and row[1] is not None
                                and ts - row[1] > 0
                            ):
                                tpot.append((ts - row[1]) / (n - 1))
                        continue
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
    if missing_usage:
        print(
            f"WARN: {missing_usage} tracked samples returned no server token usage; "
            "they are counted as missing (no TPOT), never tokenized",
            file=sys.stderr,
        )
    return {"ttft_ns": ttft, "tpot_ns": tpot, "sample_latency_ns": latency}


def _make_counter(tokenizer_path: str):
    backend = load_reference_backend(tokenizer_path)
    if backend is None:
        raise SystemExit(
            f"FATAL: could not load tokenizer backend from {tokenizer_path}"
        )
    return lambda texts: encode_lengths(backend, texts)


def _fmt(v, unit: str, div: float) -> str:
    return "-" if v is None else f"{v / div:,.2f} {unit}"


def _cross_check(summary: dict, series_name: str, ours: dict) -> None:
    """Compare our {grid_key: estimate} against the run's in-band map, if any."""
    md = summary.get(SERIES_TO_SUMMARY_FIELD[series_name]) or {}
    inband = md.get("early_stopping_percentiles")
    if not isinstance(inband, dict) or not inband:
        # absent, or an unexpected shape (pre-release artifact) — nothing to compare
        return
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
        "--summary",
        help=(
            "the run's result_summary.json: enables the in-band cross-check, derives "
            "targets from the run's own percentile grid, and powers --json"
        ),
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        help=(
            "write the summary augmented with early_stopping_percentiles maps — "
            "exactly the shape an ES-enabled run would produce (requires --summary)"
        ),
    )
    ap.add_argument(
        "--token-count-source",
        type=TokenCountSource,
        metavar=TOKEN_COUNT_SOURCE_METAVAR,
        help=(
            "token-count basis for TPOT (default: the run's own, from --summary or "
            "the performance/result_summary.json beside the event log, else the "
            "config.yaml beside it)"
        ),
    )
    ap.add_argument(
        "--tokenizer",
        help="HF model dir (or tokenizer.json) — enables client-basis TPOT",
    )
    ap.add_argument(
        "--percentiles",
        default=",".join(es_targets_from_grid(DEFAULT_PERCENTILES)),
        help=(
            "stdout-analysis override, grid convention (0-100, e.g. 50,90,99.9); "
            "--json always uses the summary grid"
        ),
    )
    ap.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE,
        help="stdout-analysis override for the confidence level (default: the LoadGen constant 0.99)",
    )
    args = ap.parse_args(argv)

    if args.json_out and not args.summary:
        raise SystemExit(
            "FATAL: --json requires --summary (the augmented output IS the summary)"
        )
    if args.summary and not Path(args.summary).is_file():
        raise SystemExit(f"FATAL: --summary {args.summary} not found")
    values = {float(tok) for tok in args.percentiles.split(",")}
    if not all(0.0 < v < 100.0 for v in values):
        raise SystemExit(
            f"FATAL: percentiles use the grid convention and must be in (0, 100), "
            f"got {sorted(values)}"
        )
    if any(v < 1.0 for v in values):
        # a fraction-style value silently means a sub-1% percentile here — never a
        # legitimate tail-certification target, so reject it with the intended value
        bad = sorted(v for v in values if v < 1.0)
        raise SystemExit(
            f"FATAL: percentiles use the grid convention (0-100); {bad} would "
            f"mean sub-1% percentiles — did you mean {[v * 100 for v in bad]}?"
        )
    # canonical float-style keys, deduped on the parsed value ("99" == "99.0")
    fallback = {str(v): v for v in sorted(values, reverse=True)}
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit(f"FATAL: confidence must be in (0, 1), got {args.confidence}")

    try:
        primary, basis = replay_basis(
            Path(args.events).parent,
            override=args.token_count_source,
            summary_path=None if args.summary is None else Path(args.summary),
        )
    except RunBasisError as exc:
        ap.error(str(exc))
    if primary is None:
        if args.json_out:
            # The summary's own basis is unknown, so the chosen one cannot label it.
            raise SystemExit(
                "FATAL: --json augments the run's summary, whose token-count basis "
                "could not be read; drop --json to replay on the chosen basis"
            )
        # The run's basis is unreadable; the one the user chose stands in.
        primary = basis
    # The summary is labelled with the run's basis: TPOT replayed on the other
    # one is neither cross-checked against it nor written into it.
    off_primary = basis is not primary
    if off_primary and args.json_out:
        raise SystemExit(
            f"FATAL: --json augments the run's summary, which is on the "
            f"{primary.value} token basis; it cannot carry {basis.value}-basis TPOT "
            "estimates (drop --token-count-source or --json)"
        )
    print(f"NOTE: token basis: {basis.value}", file=sys.stderr)
    counter = None
    if basis is TokenCountSource.CLIENT:
        counter = _make_counter(args.tokenizer) if args.tokenizer else None
        if counter is None:
            print("NOTE: no --tokenizer given -> TPOT skipped", file=sys.stderr)

    series = compute_series(
        args.events,
        count_tokens=counter,
        basis=basis,
        progress=True,
    )
    summary = None
    if args.summary:
        with open(args.summary) as f:
            summary = json.load(f)
    augmented = json.loads(json.dumps(summary)) if summary is not None else {}

    print(f"token basis: {basis.value}")
    rows = []
    for name, values in series.items():
        if not values:
            continue
        values.sort()
        field = SERIES_TO_SUMMARY_FIELD.get(name)
        if field is None:
            # a series compute_series learns about before the mapping does —
            # skip loudly rather than crash an analysis run
            print(
                f"WARN: no summary field mapping for series {name!r}; skipped",
                file=sys.stderr,
            )
            continue
        metric = (summary or {}).get(field) or {}
        grid = metric.get("percentiles") or {}
        if grid:
            # the run's own grid: exact key strings, exact (descending) order
            targets = es_targets_from_grid(grid.keys())
        else:
            targets = fallback
        results = {
            key: es_percentile_estimate(values, f, args.confidence)
            for key, f in targets.items()
        }
        esp = {key: r.estimate for key, r in results.items()}

        unit, div = _UNITS[field]
        print(f"\n{field} (from {name}, n={len(values)}):")
        for key, r in results.items():
            line = (
                f"  p{key}: sufficient={r.estimate is not None} "
                f"min_queries={r.min_queries} discarded={r.discarded}  "
                f"empirical={_fmt(r.empirical, unit, div)}  "
                f"estimate={_fmt(r.estimate, unit, div)}"
            )
            if r.estimate is not None and r.empirical is not None:
                gap = r.estimate - r.empirical
                pct = f" (+{100 * gap / r.empirical:.2f}%)" if r.empirical else ""
                line += f"  gap=+{_fmt(gap, unit, div)}{pct}"
            print(line)
            rows.append((field, key, r, unit, div))
        if summary is not None:
            completed = summary.get("n_samples_completed")
            if completed is not None and len(values) != completed:
                note = " ** MISMATCH **" if name == "sample_latency_ns" else " (info)"
                print(
                    f"  XCHECK count: ours={len(values)} "
                    f"n_samples_completed={completed}{note}"
                )
            if off_primary and name == "tpot_ns":
                print(
                    f"  XCHECK in-band: skipped (TPOT replayed on the {basis.value} "
                    f"basis; the run reported {primary.value})"
                )
                continue
            _cross_check(summary, name, dict(esp))
            if metric:  # augment only metrics the run actually reported
                augmented[field] = place_early_stopping_percentiles(
                    augmented[field], esp
                )

    print("\n--- markdown ---")
    print("| metric | p | n | empirical | ES-adjusted | gap |")
    print("|---|---|---|---|---|---|")
    for field, key, r, unit, div in rows:
        if r.estimate is not None and r.empirical is not None:
            gap = r.estimate - r.empirical
            pct = f" (+{100 * gap / r.empirical:.2f}%)" if r.empirical else ""
            g = f"+{_fmt(gap, unit, div)}{pct}"
        else:
            g = "-"
        print(
            f"| **{field.upper()}** | p{key} | {r.n} | {_fmt(r.empirical, unit, div)} | "
            f"{_fmt(r.estimate, unit, div)} | {g} |"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(augmented, f, indent=2)
        print(f"\naugmented summary written to {args.json_out}")
    return augmented


if __name__ == "__main__":
    main()
