# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-run OSL: the all-turns statistic that complements the windowed one.

The performance window closes when the first user finishes its final
trajectory, so every turn issued afterwards is invisible to the perf-side
``OslTrigger``. These tests pin the statistic that counts them
(mlcommons/endpoints#500).
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import inference_endpoint.commands.benchmark.full_run_osl as full_run_osl_mod
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
from inference_endpoint.core.types import TextModelOutput, TokenCountSource, basis_key
from inference_endpoint.metrics.token_count_basis import ServerUsage

from tests.test_helpers import LOGGED_SERVER_USAGE_CASES

pytestmark = pytest.mark.unit

_ENCODER = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
CLIENT, SERVER = TokenCountSource.CLIENT, TokenCountSource.SERVER

# The statistic on the client basis, whose behavior these tests share with a
# tokenizer-only run.
_client_full_run_osl = functools.partial(
    compute_full_run_osl, token_count_source=CLIENT
)

# The block keys a client-primary run shares with a tokenizer-only run.
_CLIENT_PRIMARY_KEYS = (
    "output_sequence_lengths",
    "n_turns_counted",
    "n_empty",
    "n_errors",
    "n_undecodable",
    "n_missing",
    "partial",
)


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


def _complete(
    turn: int,
    text: str,
    uuid: str | None = None,
    *,
    prompt_tokens: int | msgspec.UnsetType = msgspec.UNSET,
    completion_tokens: int | msgspec.UnsetType = msgspec.UNSET,
) -> EventRecord:
    return EventRecord(
        event_type=SampleEventType.COMPLETE,
        sample_uuid=uuid or f"u{turn}",
        conversation_id="c0",
        turn=turn,
        data=TextModelOutput(output=text),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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

    result = _client_full_run_osl(events, _word_count)

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

    early = _client_full_run_osl(build("early.jsonl", 2), _word_count)
    late = _client_full_run_osl(build("late.jsonl", 5), _word_count)

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

    result = _client_full_run_osl(events, _word_count, performance_uuids={"u1", "u3"})

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

    result = _client_full_run_osl(events, _word_count)

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

    early = _client_full_run_osl(
        build("e.jsonl", 2), _word_count, performance_uuids=uuids
    )
    late = _client_full_run_osl(
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

    result = _client_full_run_osl(events, _word_count)

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

    result = _client_full_run_osl(events, flaky)

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

    result = _client_full_run_osl(events, _word_count)

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
        result = _client_full_run_osl(path, _word_count)

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

    result = _client_full_run_osl(path, _word_count)

    assert result is not None
    assert result["n_turns_counted"] == 0
    assert result["n_undecodable"] == 2
    assert result["output_sequence_lengths"] == {}


@pytest.mark.parametrize("primary", list(TokenCountSource))
def test_for_report_fails_closed_without_performance_map(
    tmp_path: Path, primary: TokenCountSource
) -> None:
    """full_run_osl_for_report skips (returns None) when no performance uuid set
    can be loaded — accuracy-phase turns must never count as performance OSL."""
    (tmp_path / "events.jsonl").write_bytes(b"")  # exists so the events gate passes
    # No sample_idx_map.json → load_performance_uuids None → fail closed, before
    # any tokenizer is constructed.
    assert (
        full_run_osl_for_report(tmp_path, "any-tokenizer", token_count_source=primary)
        is None
    )


@pytest.mark.parametrize("primary", list(TokenCountSource))
def test_for_report_skips_without_events(
    tmp_path: Path, primary: TokenCountSource
) -> None:
    """A missing events.jsonl short-circuits to None on either basis."""
    assert (
        full_run_osl_for_report(tmp_path, "any-tokenizer", token_count_source=primary)
        is None
    )


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

    result = _client_full_run_osl(events, _word_count, performance_uuids={"u1", "u3"})

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

    result = _client_full_run_osl(events, flaky, performance_uuids={"u1", "u3"})

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

    result = _client_full_run_osl(events, _word_count, performance_uuids={"u1", "u3"})

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

    result = _client_full_run_osl(events, _word_count, performance_uuids={"u1", "u3"})

    assert result is not None
    assert result["n_empty"] == 1
    assert result["n_missing"] == 0
    assert result["partial"] is False


def _perf_run(tmp_path: Path, records: list[EventRecord]) -> None:
    _write_events(tmp_path / "events.jsonl", records)
    perf = {r.sample_uuid: i for i, r in enumerate(records)}
    (tmp_path / "sample_idx_map.json").write_bytes(
        msgspec.json.encode({"performance": perf})
    )


class _WordTokenizer:
    """BatchTokenizer stand-in that counts whitespace-delimited words."""

    def __init__(self, *a: object, **k: object) -> None: ...

    def __enter__(self) -> _WordTokenizer:
        return self

    def __exit__(self, *exc: object) -> None: ...

    def count_sync(self, item: MessageInput | TextInput) -> int:
        return _word_count(item)


class _BrokenTokenizer:
    """BatchTokenizer stand-in whose tokenizer cannot be loaded."""

    def __init__(self, *a: object, **k: object) -> None:
        raise OSError("no such tokenizer")


def _with_usage(turn: int, text: str, completion_tokens: int) -> EventRecord:
    """A turn reporting ``completion_tokens`` beside a prompt count of 1."""
    return _complete(turn, text, prompt_tokens=1, completion_tokens=completion_tokens)


# One turn of each kind. Client counts 1/2/3 tokens and one empty turn; the
# server counts 2/3 tokens, one true zero, and one turn without usage.
_MIXED_TURNS = [
    _with_usage(1, "one", 2),
    _with_usage(3, "two words", 3),
    _complete(5, "a b c"),  # no usage
    _with_usage(7, "", 0),  # true zero on both bases
]
_MIXED_UUIDS = {"u1", "u3", "u5", "u7"}
# Per basis: (total tokens, turns counted, empty turns) over _MIXED_TURNS.
_MIXED_COUNTS = {CLIENT: (6, 3, 1), SERVER: (5, 2, 1)}


def _strip_usage(record: EventRecord) -> EventRecord:
    return msgspec.structs.replace(
        record, prompt_tokens=msgspec.UNSET, completion_tokens=msgspec.UNSET
    )


def test_full_run_unsuffixed_block_identical_with_and_without_usage_keys(
    tmp_path: Path,
) -> None:
    """On the client basis, server usage only adds keys: the shared keys hold
    the same values whether or not the log carries usage."""
    turns = [
        *_MIXED_TURNS,
        _with_usage(9, "x y", 0),  # 0 beside output: missing usage
        EventRecord(event_type=SampleEventType.COMPLETE, sample_uuid="u11"),
    ]
    uuids = {r.sample_uuid for r in turns} | {"u13"}  # u13 never completed
    with_usage = _write_events(tmp_path / "with.jsonl", turns)
    without_usage = _write_events(
        tmp_path / "without.jsonl", [_strip_usage(r) for r in turns]
    )

    with_block = _client_full_run_osl(with_usage, _word_count, performance_uuids=uuids)
    without_block = _client_full_run_osl(
        without_usage, _word_count, performance_uuids=uuids
    )

    assert with_block is not None and without_block is not None
    assert {k: with_block[k] for k in _CLIENT_PRIMARY_KEYS} == {
        k: without_block[k] for k in _CLIENT_PRIMARY_KEYS
    }
    assert with_block["n_turns_counted"] == 4 and with_block["partial"] is True


def test_full_run_missing_usage_never_sets_top_level_partial_for_client_primary(
    tmp_path: Path,
) -> None:
    """Turns without usable usage leave the client-primary block whole: they
    are counted, and only the server series is marked partial."""
    events = _write_events(tmp_path / "events.jsonl", _MIXED_TURNS)
    without_usage = _write_events(
        tmp_path / "bare.jsonl", [_strip_usage(r) for r in _MIXED_TURNS]
    )

    some = _client_full_run_osl(events, _word_count, performance_uuids=_MIXED_UUIDS)
    none = _client_full_run_osl(
        without_usage, _word_count, performance_uuids=_MIXED_UUIDS
    )

    assert some is not None and none is not None
    assert (some["partial"], some["n_missing_usage"]) == (False, 1)
    assert some["output_sequence_lengths_server"]["partial"] is True
    # Nothing counted on the server basis: no series at all. The empty turn
    # without usage is missing usage too, as in the live count.
    assert (none["partial"], none["n_missing_usage"]) == (False, 4)
    assert none["output_sequence_lengths_server"] is None
    assert none["n_turns_counted_server"] == 0


@pytest.mark.parametrize("primary", list(TokenCountSource))
def test_full_run_other_basis_keys_follow_rule(
    tmp_path: Path, primary: TokenCountSource
) -> None:
    """The primary basis keeps the base names, the other basis appends its
    value; ``n_errors`` exists only on the client basis and the remaining
    counters are basis-independent."""
    events = _write_events(tmp_path / "events.jsonl", _MIXED_TURNS)

    block = compute_full_run_osl(
        events,
        _word_count,
        performance_uuids=_MIXED_UUIDS,
        token_count_source=primary,
    )

    def key(base: str, basis: TokenCountSource) -> str:
        return basis_key(base, basis, primary=primary)

    assert block is not None
    assert set(block) == {
        *(
            key(base, basis)
            for basis in TokenCountSource
            for base in ("output_sequence_lengths", "n_turns_counted", "n_empty")
        ),
        key("n_errors", CLIENT),
        "n_undecodable",
        "n_missing",
        "n_missing_usage",
        "partial",
    }
    for basis, (total, counted, empty) in _MIXED_COUNTS.items():
        assert block[key("output_sequence_lengths", basis)]["total"] == total
        assert block[key("n_turns_counted", basis)] == counted
        assert block[key("n_empty", basis)] == empty
    assert (block[key("n_errors", CLIENT)], block["n_missing_usage"]) == (0, 1)
    # The server shortfall sets the top-level flag only when the server basis
    # is primary; otherwise it marks just the server series.
    assert block["partial"] is (primary is SERVER)
    server_series = block[key("output_sequence_lengths", SERVER)]
    assert ("partial" in server_series) is (primary is CLIENT)
    assert "partial" not in block[key("output_sequence_lengths", CLIENT)]


def test_full_run_server_primary_missing_usage_sets_partial(tmp_path: Path) -> None:
    """On the server basis a turn without usable usage, including a 0 beside
    output, makes the block partial; a 0 beside empty output is empty."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            _with_usage(1, "one two", 0),
            _with_usage(3, "", 0),
            _with_usage(5, "x", 4),
        ],
    )

    block = compute_full_run_osl(events, None, token_count_source=SERVER)

    assert block is not None
    assert block["output_sequence_lengths"]["total"] == 4
    assert block["n_turns_counted"] == 1
    assert (block["n_empty"], block["n_missing_usage"]) == (1, 1)
    assert block["partial"] is True
    # No tokenizer: the client keys are present and count nothing.
    assert block["output_sequence_lengths_client"] is None
    assert (block["n_turns_counted_client"], block["n_errors_client"]) == (0, 0)


def test_full_run_client_primary_without_tokenizer_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client basis needs a tokenizer: without one, or with one that
    cannot load, there is no block. The server basis needs none."""
    _perf_run(tmp_path, [_with_usage(1, "one", 4)])

    assert full_run_osl_for_report(tmp_path, None, token_count_source=CLIENT) is None
    server = full_run_osl_for_report(tmp_path, None, token_count_source=SERVER)
    assert server is not None and server["output_sequence_lengths"]["total"] == 4

    monkeypatch.setattr(full_run_osl_mod, "BatchTokenizer", _BrokenTokenizer)
    assert full_run_osl_for_report(tmp_path, "tok", token_count_source=CLIENT) is None
    with pytest.raises(ValueError, match="count_tokens"):
        compute_full_run_osl(tmp_path / "events.jsonl", None, token_count_source=CLIENT)


_LOGGED_USAGE_CASES = [
    pytest.param(*case[1:], id=case[0]) for case in LOGGED_SERVER_USAGE_CASES
]


@pytest.mark.parametrize("primary", list(TokenCountSource))
@pytest.mark.parametrize(
    ("prompt", "completion", "output", "expected"), _LOGGED_USAGE_CASES
)
def test_full_run_missing_usage_matches_rule(
    tmp_path: Path,
    primary: TokenCountSource,
    prompt: Any,
    completion: Any,
    output: TextModelOutput,
    expected: ServerUsage,
) -> None:
    """The full-run statistic counts each turn's usage by the shared rule."""
    events = _write_events(
        tmp_path / "events.jsonl",
        [
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                sample_uuid="u1",
                data=output,
                prompt_tokens=prompt,
                completion_tokens=completion,
            )
        ],
    )

    block = compute_full_run_osl(
        events, _word_count, performance_uuids={"u1"}, token_count_source=primary
    )

    def server(base: str) -> str:
        return basis_key(base, SERVER, primary=primary)

    assert block is not None
    assert block["n_missing_usage"] == int(expected is ServerUsage.MISSING)
    assert block[server("n_turns_counted")] == int(expected is ServerUsage.USABLE)
    assert block[server("n_empty")] == int(expected is ServerUsage.TRUE_ZERO)


@pytest.mark.parametrize(
    ("tok_cls", "client_total"), [(_WordTokenizer, 2), (_BrokenTokenizer, None)]
)
def test_for_report_server_primary_counts_client_beside_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tok_cls: type,
    client_total: int | None,
) -> None:
    """With a tokenizer the server-primary block carries the client count; a
    tokenizer that cannot load leaves the client basis uncounted, with a
    warning."""
    _perf_run(tmp_path, [_with_usage(1, "one two", 3)])
    monkeypatch.setattr(full_run_osl_mod, "BatchTokenizer", tok_cls)

    with caplog.at_level(logging.WARNING):
        block = full_run_osl_for_report(tmp_path, "tok", token_count_source=SERVER)

    assert block is not None and block["output_sequence_lengths"]["total"] == 3
    assert (block["output_sequence_lengths_client"] or {}).get("total") == client_total
    assert ("no such tokenizer" in caplog.text) == (client_total is None)
