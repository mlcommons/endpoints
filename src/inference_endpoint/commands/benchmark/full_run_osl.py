# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-run OSL — output token lengths over every turn, including the tail.

The performance window closes when the first user finishes its final assigned
trajectory, and turns issued after that point never reach the metrics
aggregator at all (they are dropped at row creation). The windowed OSL is
therefore a concurrency-dependent subsample: at higher concurrency the window
closes earlier, so a different — and systematically shorter — subset of turns
is measured, even when the model produces identical output for every turn.

This module recomputes OSL over every completed turn from ``events.jsonl``,
which retains the full ``TextModelOutput`` (reasoning and tool calls included)
regardless of tracking state. Counting goes through the same
``extract_tokenization_input`` rule the perf-side ``OslTrigger`` uses, so the
two statistics are comparable rather than merely adjacent.

See mlcommons/endpoints#500.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import msgspec

from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    BatchTokenizer,
)
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    TextInput,
    extract_tokenization_input,
)
from inference_endpoint.core.record import EventRecord, EventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.metrics.report import series_metric_dict

logger = logging.getLogger(__name__)

# Matched against each line's leading bytes so the 6+ GB of prompt and response
# payload in a large run is never decoded just to classify the record.
_COMPLETE_MARKER = b'"event_type":"sample.complete"'
_PREFIX_BYTES = 256
# AgenticInferenceInlineScorer's dataset_name; the key its turns live under in
# sample_idx_map.json.
_PERFORMANCE_PHASE = "performance"

TokenCounter = Callable[[MessageInput | TextInput], int]


def _in_population(
    record: EventRecord,
    performance_uuids: Collection[str] | None,
) -> bool:
    """Whether one COMPLETE record belongs to the full-run OSL population.

    This is the policy boundary for "all turns", and the one place to change
    if that definition moves.

    ``events.jsonl`` is written for the whole run, so it also holds the
    accuracy phases' completions (SWE-bench). Bounding the population by the
    performance phase's uuids from ``sample_idx_map.json`` keeps an accuracy
    turn from inflating the performance statistic. ``None`` means "no bound",
    used only when the caller has already narrowed the file.

    Deliberately NOT filtered here:

    - Tail turns issued after STOP_PERFORMANCE_TRACKING. Including them is the
      entire point of this statistic.
    - Salted turns (``agentic_inference.enable_salt``). Salt perturbs the
      prompt to defeat prefix caching; it does not create extra turns, so
      there is nothing to exclude.
    - Empty completions. Those are excluded by ``extract_tokenization_input``
      returning ``None`` — the same rule the perf side applies — and counted
      under ``n_empty`` rather than dropped silently.
    """
    if performance_uuids is None:
        return True
    return record.sample_uuid in performance_uuids


def compute_full_run_osl(
    events_path: Path,
    count_tokens: TokenCounter,
    *,
    performance_uuids: Collection[str] | None = None,
) -> dict[str, Any] | None:
    """Roll up OSL over every completed turn in ``events_path``.

    Returns ``None`` only when the file held no relevant COMPLETE lines at all
    and no expected turn was missing. A file with records but nothing countable
    (all empty, timed out, errored, or undecodable) — or one missing an expected
    performance turn — returns a block with an empty ``output_sequence_lengths``
    and non-zero counters so the run stays visible rather than looking uncomputed.

    ``partial`` is True when any turn was errored, undecodable, or missing
    (reconciled against ``performance_uuids``): the mean is then over a subset of
    the population and must not feed the OSL accuracy gate. Empty turns do not
    make it partial — they are the shared rule's legitimate zero-output exclusion.
    """
    decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
    lengths: list[int] = []
    n_empty = 0
    n_errors = 0
    n_undecodable = 0
    first_error: str | None = None
    seen_uuids: set[str] = set()

    with events_path.open("rb") as events_file:
        for line in events_file:
            if _COMPLETE_MARKER not in line[:_PREFIX_BYTES]:
                continue
            # Decode before the population filter so a corrupt line — which we
            # cannot attribute to the performance population — is skipped rather
            # than charged to n_errors.
            try:
                record = decoder.decode(line)
            except msgspec.DecodeError:
                n_undecodable += 1
                continue
            if not _in_population(record, performance_uuids):
                continue
            seen_uuids.add(record.sample_uuid)
            if record.data is None:
                n_empty += 1  # e.g. a timed-out turn logs COMPLETE with data=None
                continue
            if not isinstance(record.data, TextModelOutput):
                # Unexpected payload (ErrorData/PromptData) on a COMPLETE event.
                n_errors += 1
                if first_error is None:
                    first_error = f"unexpected {type(record.data).__name__} payload"
                continue
            # Output preparation stays inside the per-turn try: a malformed
            # tool-call payload can raise in extract_tokenization_input, and one
            # bad turn must not void the whole statistic.
            try:
                tokenization_input = extract_tokenization_input(record.data)
                if tokenization_input is None:
                    n_empty += 1
                    continue
                lengths.append(count_tokens(tokenization_input))
            except Exception as e:  # noqa: BLE001 - one bad turn must not void the stat
                n_errors += 1
                if first_error is None:
                    first_error = str(e)

    # A performance turn whose COMPLETE record never reached the log is invisible
    # to the counters above; reconcile observed UUIDs against the expected
    # performance population so a lost record is flagged, not silently averaged
    # away (mlcommons/endpoints#504 review).
    n_missing = (
        len(set(performance_uuids) - seen_uuids) if performance_uuids is not None else 0
    )

    # Warn once with the aggregate rather than once per turn, so a systematic
    # failure (bad tokenizer, corrupt log) does not flood the finalize log.
    if n_undecodable:
        logger.warning(
            "Full-run OSL: skipped %d undecodable COMPLETE line(s)", n_undecodable
        )
    if n_errors:
        logger.warning(
            "Full-run OSL: %d turn(s) could not be counted (first: %s)",
            n_errors,
            first_error,
        )
    if n_missing:
        logger.warning(
            "Full-run OSL: %d performance turn(s) had no COMPLETE record in the log",
            n_missing,
        )

    if (
        not lengths
        and not n_empty
        and not n_errors
        and not n_undecodable
        and not n_missing
    ):
        return None
    # ``partial`` is the machine-readable gate signal: errored / undecodable /
    # missing turns mean the mean is over a subset and must not feed the OSL
    # accuracy gate. Empty turns are a legitimate shared-rule exclusion.
    partial = bool(n_errors or n_undecodable or n_missing)
    return {
        "output_sequence_lengths": series_metric_dict(lengths),
        "n_turns_counted": len(lengths),
        "n_empty": n_empty,
        "n_errors": n_errors,
        "n_undecodable": n_undecodable,
        "n_missing": n_missing,
        "partial": partial,
    }


def load_performance_uuids(report_dir: Path) -> set[str] | None:
    """Sample uuids belonging to the performance phase.

    ``sample_idx_map.json`` is keyed by dataset name; the agentic inline
    scorer runs with ``dataset_name="performance"``. Every issued perf turn is
    recorded there unconditionally at issue time, so it is a complete
    enumeration of the population — tail turns included.

    Returns ``None`` when the map is missing, unreadable, or carries no
    performance phase. ``full_run_osl_for_report`` treats that as "skip the
    block" (fail closed) so accuracy-phase turns can never be counted as
    performance OSL; only a caller that has already narrowed the file to the
    performance population may pass ``None`` to ``compute_full_run_osl`` as
    "no further bound".
    """
    path = report_dir / "sample_idx_map.json"
    if not path.is_file():
        return None
    try:
        idx_map = msgspec.json.decode(path.read_bytes())
    except (msgspec.DecodeError, OSError) as e:
        logger.warning("Full-run OSL: unreadable sample_idx_map.json: %s", e)
        return None
    per_phase = idx_map.get(_PERFORMANCE_PHASE) if isinstance(idx_map, dict) else None
    if not isinstance(per_phase, dict) or not per_phase:
        return None
    return set(per_phase)


def full_run_osl_for_report(
    report_dir: Path,
    tokenizer_name: str | None,
) -> dict[str, Any] | None:
    """Compute the full-run OSL block for a finished run's report directory.

    Best-effort by design: a tokenizer that cannot be loaded drops the block
    rather than failing finalize. Counting runs in-process (``n_workers=0``),
    which works with or without a fast (Rust) backend — the same fallback the
    live ``OslTrigger`` uses — so the block is not gated on backend
    availability. Returns ``None`` when there is no performance population to
    bound to (accuracy-only / perf-less runs).
    """
    events_path = report_dir / "events.jsonl"
    if tokenizer_name is None or not events_path.is_file():
        return None
    performance_uuids = load_performance_uuids(report_dir)
    if performance_uuids is None:
        # Fail closed: without the performance-phase uuid set we cannot bound the
        # population to perf turns, and counting unbounded would fold accuracy-phase
        # completions into the accuracy OSL. Absent only for accuracy-only / perf-less
        # runs or an unreadable map.
        logger.info("Full-run OSL skipped: no performance-phase samples to bound to")
        return None
    try:
        with BatchTokenizer(tokenizer_name, live_workers=1, n_workers=0) as tokenizer:
            return compute_full_run_osl(
                events_path,
                tokenizer.count_sync,
                performance_uuids=performance_uuids,
            )
    except Exception as e:  # noqa: BLE001 - optional block; never fail finalize
        logger.warning("Full-run OSL skipped: %s", e)
        return None
