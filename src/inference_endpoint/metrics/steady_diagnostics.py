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

"""Run diagnostics for agentic steady state, for either scheduler.

Reads ``events.jsonl`` (and ``lane_plan.json`` when the balanced-lane scheduler
ran) and writes ``steady_diagnostics.json`` with the evidence for the failure
modes the lane scheduler is meant to remove:

- ``window``: counted requests and the measured window.
- ``tail``: requests issued but not counted, and how much of the window ran
  below full concurrency.
- ``drift``: TTFT / TPOT percentiles per third of the window.
- ``mix``: coding share of requests in flight, per minute of the window.
- ``osl``: mean output length of counted vs all requests.
- ``throughput``: counted output tokens per second over the window, and (lanes)
  over the measured phase and over its full-load part (measurement start until
  the measured pool empties), which excludes the end-of-run tail.
- ``coverage`` (lanes): how many times each dataset turn was counted.
- ``fairness`` (lanes): TTFT / TPOT of counted vs uncounted requests completing
  in the same minute. A server that recognises and deprioritises padding shows
  up here as a ratio away from 1.

A request is counted when it was issued between START and STOP of performance
tracking and its ISSUED record is not ``tracked=false``.

Usage: ``python -m inference_endpoint.metrics.steady_diagnostics <report_dir>
[--tokenizer <name-or-path>]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import msgspec

from inference_endpoint.core.record import EventRecord, EventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.load_generator.lane_scheduler import conversation_domain

logger = logging.getLogger(__name__)

_ISSUED = b'"event_type":"sample.issued"'
_FIRST = b'"event_type":"sample.recv_first"'
_COMPLETE = b'"event_type":"sample.complete"'
_START = b'"event_type":"session.start_performance_tracking"'
_STOP = b'"event_type":"session.stop_performance_tracking"'
_PREFIX = 256
_HEAD = 1024
_TS_RE = re.compile(rb'"timestamp_ns":(\d+)')
_UUID_RE = re.compile(rb'"sample_uuid":"([^"]+)"')
_CONV_RE = re.compile(rb'"conversation_id":"([^"]*)"')
_TURN_RE = re.compile(rb'"turn":(\d+)')
_SUFFIX_RE = re.compile(r"__(repeat|unmeasured)_(\d+)$")
_BIN_NS = 60 * 10**9
_FAIRNESS_MIN_PER_BIN = 20


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _rel(a: float | None, b: float | None) -> float | None:
    return None if a is None or b in (None, 0) else 100.0 * (a / b - 1.0)


def _split_conversation(conversation_id: str) -> tuple[str, str, int]:
    """(source id, kind, copy) where kind is repeat or unmeasured."""
    match = _SUFFIX_RE.search(conversation_id)
    if match is None:
        return conversation_id, "repeat", 1
    return conversation_id[: match.start()], match.group(1), int(match.group(2))


def _count_tokens_factory(tokenizer_name: str | None):
    if tokenizer_name is None:
        return None, None
    from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
        BatchTokenizer,
    )
    from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
        extract_tokenization_input,
    )

    tokenizer = BatchTokenizer(tokenizer_name, live_workers=1, n_workers=0)
    tokenizer.__enter__()

    def count(output: TextModelOutput) -> int | None:
        tokenization_input = extract_tokenization_input(output)
        return (
            None
            if tokenization_input is None
            else tokenizer.count_sync(tokenization_input)
        )

    return count, tokenizer


def collect_requests(events_path: Path, tokenizer_name: str | None) -> dict[str, Any]:
    """One pass over the event log: per-request timings, ids and output length."""
    decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
    count, tokenizer = _count_tokens_factory(tokenizer_name)
    requests: dict[str, list[Any]] = {}
    start_ns: int | None = None
    stop_ns: int | None = None
    tracking = False
    try:
        with events_path.open("rb") as events:
            for line in events:
                prefix = line[:_PREFIX]
                if _ISSUED in prefix:
                    head = line[:_HEAD]
                    ts, uuid = _TS_RE.search(head), _UUID_RE.search(head)
                    if ts is None or uuid is None:
                        continue
                    conv = _CONV_RE.search(head)
                    turn = _TURN_RE.search(head)
                    requests[uuid.group(1).decode()] = [
                        int(ts.group(1)),  # issue
                        None,  # first token
                        None,  # complete
                        conv.group(1).decode() if conv else "",
                        int(turn.group(1)) if turn else None,
                        tracking and b'"tracked":false' not in head,  # counted
                        None,  # output tokens
                        False,  # failed / empty
                    ]
                elif _FIRST in prefix:
                    ts, uuid = _TS_RE.search(prefix), _UUID_RE.search(prefix)
                    if ts and uuid:
                        row = requests.get(uuid.group(1).decode())
                        if row is not None and row[1] is None:
                            row[1] = int(ts.group(1))
                elif _COMPLETE in prefix:
                    try:
                        record = decoder.decode(line)
                    except msgspec.DecodeError:
                        continue
                    row = requests.get(record.sample_uuid)
                    if row is None:
                        continue
                    row[2] = record.timestamp_ns
                    if isinstance(record.data, TextModelOutput) and count is not None:
                        try:
                            row[6] = count(record.data)
                        except Exception:  # noqa: BLE001 - one bad turn must not void the run
                            row[6] = None
                    elif not isinstance(record.data, TextModelOutput):
                        row[7] = True
                elif _START in prefix:
                    tracking = True
                    ts = _TS_RE.search(prefix)
                    if ts and start_ns is None:
                        start_ns = int(ts.group(1))
                elif _STOP in prefix:
                    tracking = False
                    ts = _TS_RE.search(prefix)
                    if ts:
                        stop_ns = int(ts.group(1))
    finally:
        if tokenizer is not None:
            tokenizer.__exit__(None, None, None)
    return {"requests": requests, "start_ns": start_ns, "stop_ns": stop_ns}


def _inflight_bins(
    intervals: list[tuple[int, int, bool]], t0: int, t1: int
) -> list[dict[str, float]]:
    """Time-averaged requests in flight per minute in [t0, t1]: all and coding."""
    n_bins = max(1, -(-(t1 - t0) // _BIN_NS))
    total = [0.0] * n_bins
    coding = [0.0] * n_bins
    for issue, complete, is_coding in intervals:
        a, b = max(issue, t0), min(complete, t1)
        if b <= a:
            continue
        first, last = (a - t0) // _BIN_NS, (b - t0 - 1) // _BIN_NS
        for k in range(first, min(last, n_bins - 1) + 1):
            lo = t0 + k * _BIN_NS
            overlap = (min(b, lo + _BIN_NS) - max(a, lo)) / _BIN_NS
            total[k] += overlap
            if is_coding:
                coding[k] += overlap
    width_last = ((t1 - t0) - (n_bins - 1) * _BIN_NS) / _BIN_NS
    out = []
    for k in range(n_bins):
        width = width_last if k == n_bins - 1 else 1.0
        if width <= 0:
            continue
        out.append(
            {
                "minute": k,
                "in_flight": total[k] / width,
                "coding_share": coding[k] / total[k] if total[k] else None,
            }
        )
    return out


def analyse(
    collected: dict[str, Any], lane_plan: dict[str, Any] | None
) -> dict[str, Any]:
    requests = collected["requests"]
    rows = [r for r in requests.values() if r[2] is not None]
    counted = [r for r in rows if r[5]]
    uncounted = [r for r in rows if not r[5]]
    out: dict[str, Any] = {"scheduler": "lanes" if lane_plan else "queue"}
    if not counted:
        out["error"] = "no counted requests"
        return out
    w0 = min(r[0] for r in counted)
    w1 = max(r[2] for r in counted)
    window_s = (w1 - w0) / 1e9
    out["window"] = {
        "counted_requests": len(counted),
        "uncounted_requests": len(uncounted),
        "duration_s": window_s,
        "uncounted_issued_inside_window": sum(1 for r in uncounted if w0 <= r[0] <= w1),
        "uncounted_issued_after_window": sum(1 for r in uncounted if r[0] > w1),
    }

    # Tail: requests in flight over the run and inside the window.
    intervals = [
        (r[0], r[2], conversation_domain(_split_conversation(r[3])[0]) == "coding")
        for r in rows
    ]
    run_end = max(r[2] for r in rows)
    window_bins = _inflight_bins(intervals, w0, w1)
    in_flight = [b["in_flight"] for b in window_bins]
    plateau = median(in_flight) if in_flight else 0.0
    below = [b for b in window_bins if b["in_flight"] < 0.9 * plateau]
    out["tail"] = {
        "in_flight_median": plateau,
        "window_minutes_below_90pct_of_median": len(below),
        "window_minutes": len(window_bins),
        "run_after_window_s": (run_end - w1) / 1e9,
        "counted_share_of_all_requests": len(counted) / len(rows),
    }

    # Drift: per third of the window by issue time.
    def lat(rs: list[list[Any]]) -> dict[str, float | None]:
        ttft = [(r[1] - r[0]) / 1e6 for r in rs if r[1] is not None]
        tpot = [
            (r[2] - r[1]) / 1e6 / (r[6] - 1)
            for r in rs
            if r[1] is not None and r[6] and r[6] > 1
        ]
        return {
            "n": len(rs),
            "ttft_p50_ms": _pct(ttft, 0.5),
            "ttft_p90_ms": _pct(ttft, 0.9),
            "tpot_p50_ms": _pct(tpot, 0.5),
            "tpot_p90_ms": _pct(tpot, 0.9),
        }

    ordered = sorted(counted, key=lambda r: r[0])
    third = len(ordered) // 3
    thirds = [
        lat(ordered[:third]),
        lat(ordered[third : 2 * third]),
        lat(ordered[2 * third :]),
    ]
    out["latency"] = lat(counted)
    out["drift"] = {
        "thirds": thirds,
        **{
            f"{metric}_last_vs_first_pct": _rel(thirds[2][metric], thirds[0][metric])
            for metric in ("ttft_p50_ms", "ttft_p90_ms", "tpot_p50_ms", "tpot_p90_ms")
        },
    }

    shares = [b["coding_share"] for b in window_bins if b["coding_share"] is not None]
    mean_share = sum(shares) / len(shares) if shares else None
    out["mix"] = {
        "coding_share_in_flight_mean": mean_share,
        "coding_share_worst_minute_dev_pt": 100
        * max(abs(s - mean_share) for s in shares)
        if shares
        else None,
        "per_minute": window_bins,
    }

    osl_counted = [r[6] for r in counted if r[6] is not None]
    osl_all = [r[6] for r in rows if r[6] is not None]
    out["osl"] = {
        "counted_mean": sum(osl_counted) / len(osl_counted) if osl_counted else None,
        "all_requests_mean": sum(osl_all) / len(osl_all) if osl_all else None,
        "counted_with_tokens": len(osl_counted),
    }
    tokens = sum(osl_counted)
    out["throughput"] = {
        "window_tokens_per_s": tokens / window_s if window_s > 0 else None
    }

    if lane_plan:
        measured_phase_s = lane_plan.get("measured_phase_s") or 0.0
        out["throughput"]["measured_phase_tokens_per_s"] = (
            tokens / measured_phase_s if measured_phase_s else None
        )
        # Full load: measurement start until the pool empties, while every slot
        # still has measured work. The rest of the window is the tail, where
        # slots finish their last measured conversation alongside unmeasured
        # padding. Same monotonic clock as events.jsonl.
        start_ns = lane_plan.get("measure_ns")
        pool_empty_ns = lane_plan.get("pool_empty_ns")
        if (
            start_ns is not None
            and pool_empty_ns is not None
            and pool_empty_ns > start_ns
        ):
            full_load_tokens = sum(
                r[6] for r in counted if r[6] and start_ns <= r[2] <= pool_empty_ns
            )
            full_load_s = (pool_empty_ns - start_ns) / 1e9
            out["throughput"]["full_load_tokens_per_s"] = full_load_tokens / full_load_s
            out["throughput"]["full_load_duration_s"] = full_load_s
        out["lanes"] = {
            key: lane_plan.get(key)
            for key in (
                "copies",
                "service_model",
                "estimated_measured_s",
                "measured_phase_s",
                "tail_after_pool_empty_s",
                "measured_start_spread_s",
                "measured_finish_spread_s",
                "slots_measured_from_next_turn",
                "measured_turns_expected",
                "measured_turns_completed",
                "measured_turns_failed",
            )
        }
        # Coverage: counted turns per (source, turn) and per (source, copy, turn).
        per_key: dict[tuple[str, int, int], int] = defaultdict(int)
        for r in counted:
            source, kind, copy = _split_conversation(r[3])
            if kind == "repeat" and r[4] is not None:
                per_key[(source, copy, r[4])] += 1
        per_turn: dict[tuple[str, int], int] = defaultdict(int)
        for (source, _copy, turn), n in per_key.items():
            per_turn[(source, turn)] += n
        counts = list(per_turn.values())
        out["coverage"] = {
            "dataset_turns_seen": len(per_turn),
            "min_times_counted": min(counts) if counts else 0,
            "max_times_counted": max(counts) if counts else 0,
            "duplicate_instance_turns": sum(1 for n in per_key.values() if n > 1),
            "expected_times": lane_plan.get("copies"),
        }
        # Fairness: counted vs uncounted completing in the same minute.
        by_bin: dict[int, tuple[list[list[Any]], list[list[Any]]]] = defaultdict(
            lambda: ([], [])
        )
        for r in rows:
            if r[1] is None:
                continue
            by_bin[(r[2] - w0) // _BIN_NS][0 if r[5] else 1].append(r)
        ratios: dict[str, list[tuple[float, int]]] = {"ttft": [], "tpot": []}
        for counted_bin, uncounted_bin in by_bin.values():
            if (
                len(counted_bin) < _FAIRNESS_MIN_PER_BIN
                or len(uncounted_bin) < _FAIRNESS_MIN_PER_BIN
            ):
                continue
            a, b = lat(counted_bin), lat(uncounted_bin)
            weight = min(len(counted_bin), len(uncounted_bin))
            if a["ttft_p50_ms"] and b["ttft_p50_ms"]:
                ratios["ttft"].append((b["ttft_p50_ms"] / a["ttft_p50_ms"], weight))
            if a["tpot_p50_ms"] and b["tpot_p50_ms"]:
                ratios["tpot"].append((b["tpot_p50_ms"] / a["tpot_p50_ms"], weight))
        out["fairness"] = {
            f"{metric}_uncounted_over_counted": (
                sum(x * w for x, w in vals) / sum(w for _x, w in vals) if vals else None
            )
            for metric, vals in ratios.items()
        }
        out["fairness"]["minutes_compared"] = len(ratios["tpot"])
    return out


def write_steady_diagnostics(
    report_dir: Path, tokenizer_name: str | None
) -> dict[str, Any] | None:
    """Best-effort: compute and write ``steady_diagnostics.json``."""
    events_path = report_dir / "events.jsonl"
    if not events_path.is_file():
        return None
    plan_path = report_dir / "lane_plan.json"
    lane_plan = json.loads(plan_path.read_text()) if plan_path.is_file() else None
    try:
        result = analyse(collect_requests(events_path, tokenizer_name), lane_plan)
    except Exception:
        logger.exception("Steady diagnostics failed")
        return None
    (report_dir / "steady_diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    logger.info("Wrote %s", report_dir / "steady_diagnostics.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = write_steady_diagnostics(args.report_dir, args.tokenizer)
    if result is None:
        raise SystemExit("no diagnostics written")


if __name__ == "__main__":
    main()
