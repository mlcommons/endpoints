# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/es_from_events.py — post-hoc ES from an events.jsonl log."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "es_from_events", Path("scripts/es_from_events.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_script()


def _ev(event_type, ts, uuid="", data=None):
    return json.dumps(
        {
            "event_type": event_type,
            "timestamp_ns": ts,
            "sample_uuid": uuid,
            "conversation_id": "",
            "turn": None,
            "data": data,
        }
    )


def _tmo(output=None, reasoning=None, tool_calls=None):
    return ["TextModelOutput", output, reasoning, tool_calls]


def _ws_counts(texts):  # deterministic stand-in tokenizer: whitespace tokens
    return [len(t.split()) for t in texts]


class TestExtractTpotText:
    def test_streamed_output_drops_first_chunk(self):
        assert mod.extract_tpot_text(_tmo(["a", "b c", "d"])) == "b cd"

    def test_single_chunk_output_skips(self):
        assert not mod.extract_tpot_text(_tmo(["only"]))

    def test_nonstreaming_str_output_skips(self):
        assert not mod.extract_tpot_text(_tmo("full text"))

    def test_reasoning_first_chunk_absorbs(self):
        # first chunk lives in reasoning -> reasoning[1:] plus ALL output chunks
        assert mod.extract_tpot_text(_tmo(["o1", "o2"], ["r1", "r2"])) == "r2o1o2"

    def test_tool_call_samples_return_none(self):
        # chat-template tokenization is not replicated -> skip, don't guess
        assert mod.extract_tpot_text(_tmo(["a", "b"], None, [["tc"]])) is None

    def test_non_tmo_data_returns_none(self):
        assert mod.extract_tpot_text(["PromptData", None, [1, 2]]) is None
        assert mod.extract_tpot_text(None) is None


class TestComputeSeries:
    def _fixture(self, tmp_path):
        # "w" issued BEFORE tracking starts -> excluded entirely. "a", "b"
        # tracked; "b" completes after stop (still counted: its row exists).
        # "c" has no recv_first -> latency only, no ttft/tpot.
        lines = [
            _ev("session.started", 0),
            _ev("sample.issued", 5, "w", ["PromptData", None, [1]]),
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.issued", 200, "b"),
            _ev("sample.issued", 300, "c"),
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.recv_first", 260, "b"),
            _ev("sample.recv_first", 50, "w"),
            _ev("sample.complete", 400, "a", _tmo(["x", "one two", "three"])),
            _ev("sample.complete", 500, "w", _tmo(["x", "y"])),
            _ev("session.stop_performance_tracking", 550),
            _ev("sample.complete", 600, "b", _tmo(["only-chunk"])),
            _ev("sample.complete", 700, "c", _tmo("nonstream full")),
            _ev("session.ended", 800),
        ]
        p = tmp_path / "events.jsonl"
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_gating_and_deltas(self, tmp_path):
        s = mod.compute_series(self._fixture(tmp_path), count_tokens=_ws_counts)
        # ttft: a=150-100, b=260-200; w excluded (untracked), c has none
        assert sorted(s["ttft_ns"]) == [50, 60]
        # latency: a=400-100, b=600-200, c=700-300; w excluded
        assert sorted(s["sample_latency_ns"]) == [300, 400, 400]
        # tpot: only "a" — rest = "one two"+"three" -> 2 ws tokens; (400-150)/2
        assert s["tpot_ns"] == [(400 - 150) / 2]

    def test_no_tokenizer_skips_tpot(self, tmp_path):
        s = mod.compute_series(self._fixture(tmp_path), count_tokens=None)
        assert s["tpot_ns"] == []
        assert len(s["sample_latency_ns"]) == 3


class TestEsBlocks:
    def test_blocks_use_product_math_and_shape(self):
        # product reference values: n=10000, p99 -> discarded 77, floor 662
        arr = [float(i) for i in range(10000)]
        blocks = mod.es_blocks(arr, [0.5, 0.99], 0.99)
        by_p = {b["percentile"]: b for b in blocks}
        assert by_p[0.99]["discarded"] == 77
        assert by_p[0.99]["min_queries"] == 662
        assert by_p[0.99]["estimate"] == arr[10000 - 77]
        assert by_p[0.5]["min_queries"] == 11
        for b in blocks:
            assert "tolerance" not in b
            assert b["estimate"] >= b["empirical"]


class TestMalformedInput:
    def _events_with_junk(self, tmp_path):
        lines = [
            _ev("session.start_performance_tracking", 10),
            "this is not json {",
            "123",  # valid JSON, not an object
            '"just a string"',
            _ev("sample.issued", 100, "a"),
            json.dumps({"event_type": "sample.issued", "timestamp_ns": 150}),  # no uuid
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.complete", 400, "a", _tmo(["x", "one two"])),
            _ev("session.stop_performance_tracking", 500),
        ]
        p = tmp_path / "events.jsonl"
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_junk_lines_counted_not_crashed(self, tmp_path, capsys):
        s = mod.compute_series(self._events_with_junk(tmp_path), count_tokens=_ws_counts)
        assert s["ttft_ns"] == [50]
        assert s["sample_latency_ns"] == [300]
        err = capsys.readouterr().err
        assert "malformed" in err and "4" in err  # 3 junk lines + 1 missing-uuid event

    def test_tool_call_samples_warned(self, tmp_path, capsys):
        lines = [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.complete", 400, "a", _tmo(["x", "y"], None, [["tc"]])),
            _ev("session.stop_performance_tracking", 500),
        ]
        p = tmp_path / "events.jsonl"
        p.write_text("\n".join(lines) + "\n")
        s = mod.compute_series(p, count_tokens=_ws_counts)
        assert s["tpot_ns"] == []  # skipped, not guessed
        assert "tool-call" in capsys.readouterr().err

    def test_duplicate_issued_mirrors_aggregator(self, tmp_path):
        # aggregator keeps the row on duplicate ISSUED (recv_first preserved) and
        # refreshes only the issue timestamp — the script must match, or --compare
        # reports spurious mismatches on runs with retries.
        lines = [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.issued", 200, "a"),  # retry between recv_first and complete
            _ev("sample.complete", 400, "a", _tmo(["x", "one two"])),
            _ev("session.stop_performance_tracking", 500),
        ]
        p = tmp_path / "events.jsonl"
        p.write_text("\n".join(lines) + "\n")
        s = mod.compute_series(p, count_tokens=_ws_counts)
        assert s["ttft_ns"] == [50]  # from the first issue
        assert s["sample_latency_ns"] == [200]  # complete - refreshed issue ts
        assert s["tpot_ns"] == [(400 - 150) / 2]  # recv_first survived the retry


class TestCrossCheckAndMain:
    def _summary(self, blocks):
        return {
            "n_samples_completed": 2,
            "ttft": {"early_stopping": blocks},
            "tpot": {},
            "latency": {},
        }

    def test_cross_check_exact_match_and_mismatch(self, capsys):
        values = [1.0, 2.0]
        blocks = mod.es_blocks(values, [0.99], 0.99)
        mod._cross_check(self._summary(blocks), "ttft_ns", values, blocks)
        assert "EXACT MATCH" in capsys.readouterr().out
        tampered = [dict(blocks[0], estimate=123.0)]
        mod._cross_check(self._summary(tampered), "ttft_ns", values, blocks)
        assert "MISMATCH" in capsys.readouterr().out

    def test_main_writes_json_and_validates_args(self, tmp_path, capsys):
        lines = [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.complete", 400, "a", _tmo(["x", "one two"])),
            _ev("session.stop_performance_tracking", 500),
        ]
        events = tmp_path / "events.jsonl"
        events.write_text("\n".join(lines) + "\n")
        out = tmp_path / "blocks.json"
        result = mod.main([str(events), "--json", str(out)])
        assert out.exists()
        assert "ttft" in result and "latency" in result
        capsys.readouterr()  # drain
        with pytest.raises(SystemExit):
            mod.main([str(events), "--percentiles", "1.0"])
        with pytest.raises(SystemExit):
            mod.main([str(events), "--confidence", "1.0"])
