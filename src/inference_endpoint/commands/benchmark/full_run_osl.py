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
which retains each turn's COMPLETE record regardless of tracking state. One
pass counts each turn on both token bases, as the windowed OSL does:

- Client: the full ``TextModelOutput`` (reasoning and tool calls included)
  through the same ``extract_tokenization_input`` rule the perf-side
  ``OslTrigger`` uses, so the two statistics are comparable rather than merely
  adjacent. Needs a tokenizer.
- Server: each turn's reported ``completion_tokens``, read by
  ``read_server_usage``; a turn without usable usage is counted under
  ``n_missing_usage``.

The run's primary basis keeps the base key names and alone sets the block's
``partial``; the other basis sits beside it under suffixed names (see
``basis_key``).

See mlcommons/endpoints#500.
"""

from __future__ import annotations

import logging
from array import array
from collections.abc import Callable, Collection
from contextlib import nullcontext
from dataclasses import dataclass, field
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
from inference_endpoint.core.record import EventRecord, EventType, is_complete_line
from inference_endpoint.core.types import (
    PhaseType,
    TextModelOutput,
    TokenCountSource,
    basis_key,
)
from inference_endpoint.metrics.report import series_metric_dict
from inference_endpoint.metrics.token_count_basis import (
    ServerUsage,
    read_server_usage,
)

logger = logging.getLogger(__name__)

TokenCounter = Callable[[MessageInput | TextInput], int]


@dataclass(slots=True)
class _Tally:
    """Turn lengths and exclusion counters for one token basis.

    ``n_errors`` (turns the tokenizer could not count, and unexpected
    payloads) and ``n_missing_usage`` (turns without usable server usage) are
    the turns this basis should have counted but could not.
    """

    lengths: array[int] = field(default_factory=lambda: array("q"))
    n_empty: int = 0
    n_errors: int = 0
    n_missing_usage: int = 0
    first_error: str | None = None

    def error(self, reason: str) -> None:
        self.n_errors += 1
        if self.first_error is None:
            self.first_error = reason

    def tokenize(self, data: TextModelOutput, count_tokens: TokenCounter) -> None:
        # Output preparation stays inside the try: a malformed tool-call payload
        # can raise in extract_tokenization_input, and one bad turn must not void
        # the whole statistic.
        try:
            tokenization_input = extract_tokenization_input(data)
            if tokenization_input is None:
                self.n_empty += 1
                return
            self.lengths.append(count_tokens(tokenization_input))
        except Exception as e:  # noqa: BLE001 - one bad turn must not void the stat
            self.error(str(e))

    def count_usage(self, record: EventRecord, data: TextModelOutput) -> None:
        usage, completion_tokens = read_server_usage(
            record.prompt_tokens, record.completion_tokens, data
        )
        if completion_tokens is not None:
            self.lengths.append(completion_tokens)
        elif usage is ServerUsage.TRUE_ZERO:
            self.n_empty += 1
        else:
            self.n_missing_usage += 1

    def missed_any(self) -> bool:
        """Whether this basis could not count some turn it saw."""
        return bool(self.n_errors or self.n_missing_usage)


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
    - Empty completions. Those are excluded — by ``extract_tokenization_input``
      returning ``None`` on the client basis (the rule the perf side applies),
      by a reported ``completion_tokens`` of 0 for an empty output on the
      server basis — and counted under ``n_empty`` rather than dropped
      silently.
    """
    if performance_uuids is None:
        return True
    return record.sample_uuid in performance_uuids


def _basis_keys(
    tally: _Tally, basis: TokenCountSource, *, primary: TokenCountSource
) -> dict[str, Any]:
    """One basis's keys, named by ``basis_key``.

    The primary series is ``{}`` when nothing was counted, the other series
    ``None``. ``n_errors`` exists only on the client basis.
    """
    series: dict[str, Any] | None = series_metric_dict(tally.lengths)
    if basis is not primary:
        series = series or None
    keys = {
        basis_key("output_sequence_lengths", basis, primary=primary): series,
        basis_key("n_turns_counted", basis, primary=primary): len(tally.lengths),
        basis_key("n_empty", basis, primary=primary): tally.n_empty,
    }
    if basis is TokenCountSource.CLIENT:
        keys[basis_key("n_errors", basis, primary=primary)] = tally.n_errors
    return keys


def compute_full_run_osl(
    events_path: Path,
    count_tokens: TokenCounter | None,
    *,
    token_count_source: TokenCountSource,
    performance_uuids: Collection[str] | None = None,
) -> dict[str, Any] | None:
    """Roll up OSL over every completed turn in ``events_path``.

    Counts each turn on both bases in one pass: the client basis with
    ``count_tokens``, the server basis from the turn's usage. The primary
    basis, ``token_count_source``, takes the base key names; the client basis
    needs ``count_tokens`` when it is primary. Without ``count_tokens`` a
    client basis that is the other one counts nothing.

    Returns ``None`` only when the file held no relevant COMPLETE lines at all
    and no expected turn was missing. A file with records but nothing countable
    (all empty, timed out, errored, undecodable, or without usage) — or one
    missing an expected performance turn — returns a block with an empty
    ``output_sequence_lengths`` and non-zero counters so the run stays visible
    rather than looking uncomputed.

    ``partial`` is True when the primary basis fell short: a turn errored,
    was undecodable or missing (reconciled against ``performance_uuids``), or,
    on the server basis, lacked usable usage. The mean is then over a subset
    of the population and must not feed the OSL accuracy gate. Empty turns do
    not make it partial — they are a legitimate zero-output exclusion. The
    other basis's shortfall marks only its own series.
    """
    client_primary = token_count_source is TokenCountSource.CLIENT
    if count_tokens is None and client_primary:
        raise ValueError("count_tokens is required for client-basis full-run OSL")
    decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
    tallies = {basis: _Tally() for basis in TokenCountSource}
    client, server = tallies[TokenCountSource.CLIENT], tallies[TokenCountSource.SERVER]
    # A client basis without a tokenizer counts nothing at all.
    counting = (client, server) if count_tokens is not None else (server,)
    n_undecodable = 0
    seen_uuids: set[str] = set()

    with events_path.open("rb") as events_file:
        for line in events_file:
            if not is_complete_line(line):
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
            data = record.data
            if data is None:
                # e.g. a timed-out turn logs COMPLETE with data=None
                for tally in counting:
                    tally.n_empty += 1
                continue
            if not isinstance(data, TextModelOutput):
                # Unexpected payload (ErrorData/PromptData) on a COMPLETE event.
                reason = f"unexpected {type(data).__name__} payload"
                for tally in counting:
                    tally.error(reason)
                continue
            server.count_usage(record, data)
            if count_tokens is not None:
                client.tokenize(data, count_tokens)

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
    for basis, tally in tallies.items():
        if tally.n_errors:
            logger.warning(
                "Full-run OSL: %d turn(s) could not be counted on the %s basis "
                "(first: %s)",
                tally.n_errors,
                basis.value,
                tally.first_error,
            )
    if n_missing:
        logger.warning(
            "Full-run OSL: %d performance turn(s) had no COMPLETE record in the log",
            n_missing,
        )
    if server.n_missing_usage:
        # A shortfall decides the block only on the primary basis.
        logger.log(
            logging.INFO if client_primary else logging.WARNING,
            "Full-run OSL: %d turn(s) carried no usable server token usage",
            server.n_missing_usage,
        )

    if not seen_uuids and not n_undecodable and not n_missing:
        return None

    # Every basis falls short of the population when a line was undecodable or
    # an expected turn is missing.
    population_short = bool(n_undecodable or n_missing)
    primary, other = token_count_source, token_count_source.other
    block = {
        **_basis_keys(tallies[primary], primary, primary=primary),
        "n_undecodable": n_undecodable,
        "n_missing": n_missing,
        # The machine-readable gate signal: the primary mean is over a subset
        # and must not feed the OSL accuracy gate.
        "partial": population_short or tallies[primary].missed_any(),
        "n_missing_usage": server.n_missing_usage,
        **_basis_keys(tallies[other], other, primary=primary),
    }
    # The other basis's shortfall marks only its own series.
    other_series = block[basis_key("output_sequence_lengths", other, primary=primary)]
    if other_series is not None and (population_short or tallies[other].missed_any()):
        other_series["partial"] = True
    return block


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
    per_phase = (
        idx_map.get(PhaseType.PERFORMANCE.value) if isinstance(idx_map, dict) else None
    )
    if not isinstance(per_phase, dict) or not per_phase:
        return None
    return set(per_phase)


def full_run_osl_for_report(
    report_dir: Path,
    tokenizer_name: str | None,
    *,
    token_count_source: TokenCountSource,
) -> dict[str, Any] | None:
    """Compute the full-run OSL block for a finished run's report directory.

    Best-effort by design: it never fails finalize. A client-primary run
    without a tokenizer, or whose tokenizer cannot be loaded, drops the block;
    a server-primary run needs no tokenizer, and without one counts nothing on
    the client basis. Counting runs in-process (``n_workers=0``), which works
    with or without a fast (Rust) backend — the same fallback the live
    ``OslTrigger`` uses — so the block is not gated on backend availability.
    Returns ``None`` when there is no performance population to bound to
    (accuracy-only / perf-less runs).
    """
    client_primary = token_count_source is TokenCountSource.CLIENT
    events_path = report_dir / "events.jsonl"
    if (tokenizer_name is None and client_primary) or not events_path.is_file():
        return None
    performance_uuids = load_performance_uuids(report_dir)
    if performance_uuids is None:
        # Fail closed: without the performance-phase uuid set we cannot bound the
        # population to perf turns, and counting unbounded would fold accuracy-phase
        # completions into the accuracy OSL. Absent only for accuracy-only / perf-less
        # runs or an unreadable map.
        logger.info("Full-run OSL skipped: no performance-phase samples to bound to")
        return None
    tokenizer_cm: BatchTokenizer | nullcontext[None] = nullcontext()
    if tokenizer_name is not None:
        try:
            tokenizer_cm = BatchTokenizer(tokenizer_name, live_workers=1, n_workers=0)
        except Exception as e:  # noqa: BLE001 - optional block; never fail finalize
            if client_primary:
                logger.warning("Full-run OSL skipped: %s", e)
                return None
            logger.warning("Full-run OSL: client counts skipped: %s", e)
    try:
        with tokenizer_cm as tokenizer:
            return compute_full_run_osl(
                events_path,
                tokenizer.count_sync if tokenizer is not None else None,
                token_count_source=token_count_source,
                performance_uuids=performance_uuids,
            )
    except Exception as e:  # noqa: BLE001 - optional block; never fail finalize
        logger.warning("Full-run OSL skipped: %s", e)
        return None
