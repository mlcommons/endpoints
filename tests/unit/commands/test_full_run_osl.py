# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-run OSL: the all-turns statistic that complements the windowed one.

The performance window closes when the first user finishes its final
trajectory, so every turn issued afterwards is invisible to the perf-side
``OslTrigger``. These tests pin the statistic that counts them
(mlcommons/endpoints#500).
"""

from __future__ import annotations

import logging
from pathlib import Path

import msgspec
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    TextInput,
)
from inference_endpoint.commands.benchmark.full_run_osl import (
    compute_full_run_osl,
    full_run_osl_for_report,
    load_performance_uuids,
)
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput

pytestmark = pytest.mark.unit

_ENCODER = msgspec.json.Encoder(enc_hook=EventType.encode_hook)


def _word_count(tok_input: MessageInput | TextInput) -> int:
    """Deterministic stand-in for a real tokenizer."""
    if isinstance(tok_input, TextInput):
        return len(tok_input.text.split())
    parts = [tok_input.content or "", tok_input.reasoning or ""]
    if tok_input.tool_calls:
        parts.append(" ".join(str(tc) for tc in tok_input.tool_calls))
    return len(" ".join(parts).split())


def _write_events(path: Path, records: list[EventRecord]) -> Path:
    path.write_bytes(b"\n".join(_ENCODER.encode(r) for r in records) + b"\n")
    return path


def _complete(turn: int, text: str, uuid: str | None = None) -> EventRecord:
    return EventRecord(
        event_type=SampleEventType.COMPLETE,
        sample_uuid=uuid or f"u{turn}",
        conversation_id="c0",
        turn=turn,
        data=TextModelOutput(output=text),
    )


def _session(ev: SessionEventType) -> EventRecord:
    return EventRecord(event_type=ev)


def test_counts_turns_completed_after_the_performance_window(tmp_path: Path) -> None:
    """Turns after STOP_PERFORMANCE_TRACKING are excluded from the window but
    must appear in the full-run statistic.

    Window sees 2 turns ("one" = 1 token, "two words" = 2). The tail adds two
    more 3-token turns, which the perf-side OslTrigger never sees at all.
    """
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _session(SessionEventType.START_PERFORMANCE_TRACKING),
            _complete(1, "one"),
            _complete(3, "two words"),
            _session(SessionEventType.STOP_PERFORMANCE_TRACKING),
            _complete(5, "a b c"),
            _complete(7, "d e f"),
            _session(SessionEventType.ENDED),
        ],
    )

    result = compute_full_run_osl(events, _word_count)

    assert result is not None
    assert result["n_turns_counted"] == 4
    # 1 + 2 + 3 + 3 = 9 tokens over 4 turns
    assert result["output_sequence_lengths"]["total"] == 9
    assert result["output_sequence_lengths"]["avg"] == 2.25


def test_full_run_osl_is_invariant_to_window_position(tmp_path: Path) -> None:
    """The #500 regression guard.

    Identical per-turn outputs, but the performance window closes at two
    different points — as it does when concurrency changes. The windowed mean
    moves; the full-run mean must not. If this fails, the full-run statistic
    has picked up the window gating and is no longer concurrency-independent.
    """
    turns = [_complete(2 * i + 1, " ".join("w" * (i + 1))) for i in range(6)]

    def build(name: str, stop_after: int) -> Path:
        return _write_events(
            tmp_path / name,
            [
                _session(SessionEventType.STARTED),
                _session(SessionEventType.START_PERFORMANCE_TRACKING),
                *turns[:stop_after],
                _session(SessionEventType.STOP_PERFORMANCE_TRACKING),
                *turns[stop_after:],
                _session(SessionEventType.ENDED),
            ],
        )

    early = compute_full_run_osl(build("early.jsonl", 2), _word_count)
    late = compute_full_run_osl(build("late.jsonl", 5), _word_count)

    assert early is not None and late is not None
    assert early["n_turns_counted"] == late["n_turns_counted"] == 6
    assert early["output_sequence_lengths"] == late["output_sequence_lengths"]


def test_excludes_completions_from_other_phases(tmp_path: Path) -> None:
    """events.jsonl holds every phase's COMPLETE events.

    A SWE-bench accuracy turn must not inflate the performance-phase full-run
    OSL, so the population is bounded by the performance phase's uuids from
    sample_idx_map.json.
    """
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _session(SessionEventType.START_PERFORMANCE_TRACKING),
            _complete(1, "one"),
            _session(SessionEventType.STOP_PERFORMANCE_TRACKING),
            _complete(3, "a b c"),
            _complete(1, "x y z z z z z z", uuid="swebench-1"),
            _session(SessionEventType.ENDED),
        ],
    )

    result = compute_full_run_osl(events, _word_count, performance_uuids={"u1", "u3"})

    assert result is not None
    assert result["n_turns_counted"] == 2
    assert result["output_sequence_lengths"]["total"] == 4


def test_counts_empty_completions_separately(tmp_path: Path) -> None:
    """A failed request logs COMPLETE with output == "".

    It must not be counted as a zero-token sample, but it must still be
    visible — a run full of blanks should not look like a clean short run.
    """
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _complete(1, "one"),
            _complete(3, ""),
            _session(SessionEventType.ENDED),
        ],
    )

    result = compute_full_run_osl(events, _word_count)

    assert result is not None
    assert result["n_turns_counted"] == 1
    assert result["n_empty"] == 1


def test_load_performance_uuids_reads_the_performance_phase(tmp_path: Path) -> None:
    """sample_idx_map.json is keyed by dataset name; the perf phase is
    "performance" (AgenticInferenceInlineScorer's dataset_name)."""
    (tmp_path / "sample_idx_map.json").write_text(
        '{"performance": {"u1": 0, "u3": 1}, "swe_bench": {"s1": 0}}'
    )

    assert load_performance_uuids(tmp_path) == {"u1", "u3"}


def test_load_performance_uuids_returns_none_when_absent(tmp_path: Path) -> None:
    """No map, or no performance phase, returns None; full_run_osl_for_report
    treats that as "skip the block" (fail closed)."""
    assert load_performance_uuids(tmp_path) is None

    (tmp_path / "sample_idx_map.json").write_text('{"swe_bench": {"s1": 0}}')
    assert load_performance_uuids(tmp_path) is None


def test_full_run_mean_invariant_while_windowed_mean_moves(tmp_path: Path) -> None:
    """#500 (f): under the performance uuid bound, the full-run mean is identical
    across window positions even though the windowed (pre-STOP) mean differs. A
    dataset repeat (distinct uuid) is counted; an out-of-population accuracy turn
    is excluded and cannot move the mean."""
    turns = [_complete(2 * i + 1, " ".join(["w"] * (i + 1))) for i in range(6)]
    turns.append(_complete(13, "w w w", uuid="u13"))  # dataset repeat, distinct id
    # An accuracy-phase completion NOT in the performance map — must be excluded.
    swe = _complete(1, "x y z z z", uuid="swe-1")

    perf_map = {r.sample_uuid: i for i, r in enumerate(turns)}
    (tmp_path / "sample_idx_map.json").write_bytes(
        msgspec.json.encode({"performance": perf_map, "swe_bench": {"swe-1": 0}})
    )
    uuids = load_performance_uuids(tmp_path)
    assert uuids is not None and "swe-1" not in uuids

    def build(name: str, stop_after: int) -> Path:
        return _write_events(
            tmp_path / name,
            [
                _session(SessionEventType.STARTED),
                _session(SessionEventType.START_PERFORMANCE_TRACKING),
                *turns[:stop_after],
                _session(SessionEventType.STOP_PERFORMANCE_TRACKING),
                *turns[stop_after:],
                swe,  # out-of-population: must not affect the perf full-run mean
                _session(SessionEventType.ENDED),
            ],
        )

    def windowed_mean(stop_after: int) -> float:
        counted = [_word_count(TextInput(str(r.data))) for r in turns[:stop_after]]
        return sum(counted) / len(counted)

    early = compute_full_run_osl(
        build("e.jsonl", 2), _word_count, performance_uuids=uuids
    )
    late = compute_full_run_osl(
        build("l.jsonl", 5), _word_count, performance_uuids=uuids
    )

    assert windowed_mean(2) != windowed_mean(5)  # the windowed mean moves
    assert early is not None and late is not None
    assert (
        early["n_turns_counted"] == late["n_turns_counted"] == 7
    )  # repeat in, swe out
    assert early["output_sequence_lengths"] == late["output_sequence_lengths"]


def test_data_none_completion_counts_as_empty(tmp_path: Path) -> None:
    """A timed-out turn logs COMPLETE with data=None — counted as empty, not error."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _complete(1, "one"),
            EventRecord(
                event_type=SampleEventType.COMPLETE, sample_uuid="u3", turn=3, data=None
            ),
            _session(SessionEventType.ENDED),
        ],
    )

    result = compute_full_run_osl(events, _word_count)

    assert result is not None
    assert result["n_turns_counted"] == 1
    assert result["n_empty"] == 1
    assert result["n_errors"] == 0


def test_count_failure_is_isolated_to_n_errors(tmp_path: Path) -> None:
    """One uncountable turn increments n_errors without voiding the stat."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _complete(1, "one"),
            _complete(3, "boom"),
            _session(SessionEventType.ENDED),
        ],
    )

    def flaky(tok_input: MessageInput | TextInput) -> int:
        if getattr(tok_input, "text", "") == "boom":
            raise ValueError("tokenizer blew up")
        return _word_count(tok_input)

    result = compute_full_run_osl(events, flaky)

    assert result is not None
    assert result["n_turns_counted"] == 1
    assert result["n_errors"] == 1
    assert result["output_sequence_lengths"]["total"] == 1  # only "one" counted


def test_all_empty_population_returns_a_visible_block(tmp_path: Path) -> None:
    """A population with records but nothing countable returns a block (not None)
    so the run is visible rather than looking uncomputed."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _complete(1, ""),
            EventRecord(
                event_type=SampleEventType.COMPLETE, sample_uuid="u3", turn=3, data=None
            ),
            _session(SessionEventType.ENDED),
        ],
    )

    result = compute_full_run_osl(events, _word_count)

    assert result is not None
    assert result["n_turns_counted"] == 0
    assert result["n_empty"] == 2
    assert result["output_sequence_lengths"] == {}


def test_undecodable_complete_line_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt COMPLETE line goes to n_undecodable (not n_errors/n_empty) and
    warns once."""
    path = tmp_path / "events.jsonl"
    good = _ENCODER.encode(_complete(1, "one"))
    # Trips the prefix marker but is not a decodable EventRecord.
    corrupt = b'{"event_type":"sample.complete","data":<<<not json>>>}'
    path.write_bytes(good + b"\n" + corrupt + b"\n")

    with caplog.at_level(logging.WARNING):
        result = compute_full_run_osl(path, _word_count)

    assert result is not None
    assert result["n_turns_counted"] == 1
    assert result["n_errors"] == 0
    assert result["n_empty"] == 0
    assert result["n_undecodable"] == 1
    messages = [r.getMessage() for r in caplog.records]
    assert sum("undecodable COMPLETE line" in m for m in messages) == 1


def test_all_undecodable_returns_visible_block(tmp_path: Path) -> None:
    """A stream of only corrupt COMPLETE lines returns a block (not None) so the
    accuracy field is not mistaken for uncomputed."""
    corrupt = b'{"event_type":"sample.complete","data":<<<not json>>>}'
    path = tmp_path / "events.jsonl"
    path.write_bytes(corrupt + b"\n" + corrupt + b"\n")

    result = compute_full_run_osl(path, _word_count)

    assert result is not None
    assert result["n_turns_counted"] == 0
    assert result["n_undecodable"] == 2
    assert result["output_sequence_lengths"] == {}


def test_for_report_fails_closed_without_performance_map(tmp_path: Path) -> None:
    """full_run_osl_for_report skips (returns None) when no performance uuid set
    can be loaded — accuracy-phase turns must never count as performance OSL."""
    (tmp_path / "events.jsonl").write_bytes(b"")  # exists so the events gate passes
    # No sample_idx_map.json → load_performance_uuids None → fail closed, before
    # any tokenizer is constructed.
    assert full_run_osl_for_report(tmp_path, "any-tokenizer") is None


def test_for_report_skips_without_tokenizer_or_events(tmp_path: Path) -> None:
    """No tokenizer, or a missing events.jsonl, short-circuits to None."""
    assert full_run_osl_for_report(tmp_path, None) is None  # no tokenizer
    assert full_run_osl_for_report(tmp_path, "any-tokenizer") is None  # no events.jsonl


def test_missing_turn_is_detected_via_uuid_reconciliation(tmp_path: Path) -> None:
    """A perf uuid with no COMPLETE record is reconciled as missing and marks the
    block partial — the silent case where every other counter is zero (#504)."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _complete(1, "a b c"),
            _session(SessionEventType.ENDED),
        ],
    )

    result = compute_full_run_osl(events, _word_count, performance_uuids={"u1", "u3"})

    assert result is not None
    assert result["n_turns_counted"] == 1
    assert result["n_missing"] == 1  # u3 was expected but never logged a COMPLETE
    assert result["n_errors"] == 0 and result["n_undecodable"] == 0
    assert result["partial"] is True


def test_failed_turn_marks_block_partial(tmp_path: Path) -> None:
    """An uncountable turn keeps the surviving mean but flags the block partial."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _complete(1, "one"),
            _complete(3, "boom"),
            _session(SessionEventType.ENDED),
        ],
    )

    def flaky(tok_input: MessageInput | TextInput) -> int:
        if getattr(tok_input, "text", "") == "boom":
            raise ValueError("tokenizer blew up")
        return _word_count(tok_input)

    result = compute_full_run_osl(events, flaky, performance_uuids={"u1", "u3"})

    assert result is not None
    assert result["n_turns_counted"] == 1
    assert result["n_errors"] == 1
    assert result["partial"] is True


def test_clean_run_is_not_partial(tmp_path: Path) -> None:
    """Every expected turn counted → partial is False and nothing is missing."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _complete(1, "a"),
            _complete(3, "b c"),
            _session(SessionEventType.ENDED),
        ],
    )

    result = compute_full_run_osl(events, _word_count, performance_uuids={"u1", "u3"})

    assert result is not None
    assert result["n_turns_counted"] == 2
    assert result["n_missing"] == 0
    assert result["partial"] is False


def test_empty_turns_do_not_make_block_partial(tmp_path: Path) -> None:
    """An empty completion is a legitimate shared-rule exclusion, not a defect,
    so it must not flip the block to partial."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _session(SessionEventType.STARTED),
            _complete(1, "a b"),
            _complete(3, ""),  # empty output → n_empty, not partial
            _session(SessionEventType.ENDED),
        ],
    )

    result = compute_full_run_osl(events, _word_count, performance_uuids={"u1", "u3"})

    assert result is not None
    assert result["n_empty"] == 1
    assert result["n_missing"] == 0
    assert result["partial"] is False
