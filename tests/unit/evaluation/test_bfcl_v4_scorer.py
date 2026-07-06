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

"""Unit tests for the BFCL v4 single-turn scorer's output extraction."""

import json
from pathlib import Path

import msgspec
import pytest
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.evaluation.bfcl_v4_scorer import BFCLv4Scorer
from inference_endpoint.evaluation.extractor import FunctionCallExtractor


def _write_events(report_dir: Path, records: list[EventRecord]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
    with (report_dir / "events.jsonl").open("wb") as f:
        for record in records:
            f.write(encoder.encode(record) + b"\n")


def _get_outputs_for(report_dir: Path):
    # get_outputs() only needs report_dir; bypass __init__ so the test does not
    # require the optional bfcl-eval dependency.
    scorer = object.__new__(BFCLv4Scorer)
    scorer.report_dir = report_dir
    return scorer.get_outputs()


class _StubDataset:
    """Minimal Dataset stand-in exposing only what BFCLv4Scorer.score() reads."""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def num_samples(self) -> int:
        return len(self.dataframe)


def _make_scorer(
    report_dir: Path,
    *,
    sample_index_map: dict[str, int],
    dataframe=None,
    extractor=FunctionCallExtractor,
    ground_truth_column: str = "ground_truth",
):
    """Build a BFCLv4Scorer without __init__ (avoids the bfcl-eval import gate)."""
    scorer = object.__new__(BFCLv4Scorer)
    scorer.report_dir = report_dir
    scorer.sample_index_map = sample_index_map
    scorer.dataset = _StubDataset(dataframe) if dataframe is not None else None  # type: ignore[assignment]
    scorer.extractor = extractor
    scorer.ground_truth_column = ground_truth_column
    scorer._breakdown = None
    return scorer


def _tool_call_output(name: str = "get_weather", arguments: str = '{"city": "NYC"}'):
    return TextModelOutput(
        output="",
        tool_calls=[
            {
                "id": "c0",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    )


class TestBFCLv4ScorerGetOutputs:
    """Regression tests for streaming vs non-streaming tool-call serialization."""

    def test_streaming_tool_call_deltas_are_merged(self, tmp_path):
        """Streaming responses store tool_calls as fragmented per-delta chunks.

        Without merging, the serialized output is a list-of-lists that the
        function-call extractor cannot parse, so every function-calling sample
        scores 0. get_outputs() must reassemble the deltas so the extractor
        recovers a complete call.
        """
        report_dir = tmp_path / "report"
        # Mirrors the on-wire streaming shape: a tuple of delta chunks, each a
        # tuple of partial tool-call dicts with fragmented `arguments`.
        streaming_tool_calls = [
            [
                {
                    "index": 0,
                    "id": "abc",
                    "type": "function",
                    "function": {"name": "math_factorial", "arguments": "{"},
                }
            ],
            [{"index": 0, "function": {"arguments": '"number":'}}],
            [{"index": 0, "function": {"arguments": "5"}}],
            [{"index": 0, "function": {"arguments": "}"}}],
        ]
        _write_events(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="s1",
                    data=TextModelOutput(output=[], tool_calls=streaming_tool_calls),
                ),
            ],
        )

        df = _get_outputs_for(report_dir)
        output_text = df.loc[df["sample_uuid"] == "s1", "output"].iloc[0]

        # Serialized output must be a flat list of complete tool-call objects.
        parsed = json.loads(output_text)
        assert isinstance(parsed, list)
        assert all(isinstance(item, dict) for item in parsed)

        extracted = FunctionCallExtractor.extract(output_text, default="[]")
        assert json.loads(extracted) == [
            {"name": "math_factorial", "arguments": {"number": 5}}
        ]

    def test_non_streaming_tool_calls_pass_through(self, tmp_path):
        """Already-complete (non-streaming) tool calls must still serialize cleanly."""
        report_dir = tmp_path / "report"
        complete_tool_calls = [
            {
                "id": "x",
                "type": "function",
                "function": {
                    "name": "math_factorial",
                    "arguments": '{"number": 5}',
                },
            }
        ]
        _write_events(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="s1",
                    data=TextModelOutput(output="", tool_calls=complete_tool_calls),
                ),
            ],
        )

        df = _get_outputs_for(report_dir)
        output_text = df.loc[df["sample_uuid"] == "s1", "output"].iloc[0]

        extracted = FunctionCallExtractor.extract(output_text, default="[]")
        assert json.loads(extracted) == [
            {"name": "math_factorial", "arguments": {"number": 5}}
        ]

    def test_plain_text_response_falls_back_to_string(self, tmp_path):
        """Responses with no tool calls fall back to the full output string."""
        report_dir = tmp_path / "report"
        _write_events(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="s1",
                    data=TextModelOutput(output="I cannot help with that."),
                ),
            ],
        )

        df = _get_outputs_for(report_dir)
        output_text = df.loc[df["sample_uuid"] == "s1", "output"].iloc[0]
        assert "I cannot help with that." in output_text


class TestBFCLv4ScorerEmptyGuards:
    """score() must never KeyError on an empty/unmatched events log."""

    def test_no_complete_events_returns_zero_breakdown(self, tmp_path):
        """An events log with no COMPLETE records yields a column-less frame.

        Without the guard, ``df["sample_uuid"]`` would KeyError. Instead the
        scorer must emit a well-formed zero breakdown.
        """
        report_dir = tmp_path / "report"
        _write_events(report_dir, [])  # empty events.jsonl

        scorer = _make_scorer(report_dir, sample_index_map={"u1": 0})
        overall, n_repeats = scorer.score()

        assert overall == 0.0
        assert n_repeats == 1
        assert scorer.score_breakdown() == {
            "overall_accuracy": 0.0,
            "normalized_single_turn_score": 0.0,
            "category_scores": {},
            "subset_scores": {},
            "unscored_subsets": {},
            "total_samples": 0,
        }

    def test_events_present_but_no_matching_uuid_returns_zero_breakdown(self, tmp_path):
        """COMPLETE events exist but none map to a known sample_uuid.

        The post-filter frame is empty-but-columned; the downstream
        sample_index lookup would KeyError without the guard.
        """
        report_dir = tmp_path / "report"
        _write_events(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="stranger",
                    data=TextModelOutput(output="hello"),
                )
            ],
        )

        scorer = _make_scorer(report_dir, sample_index_map={"u1": 0})
        overall, n_repeats = scorer.score()

        assert overall == 0.0
        assert scorer.score_breakdown()["total_samples"] == 0


class TestBFCLv4ScorerHallucination:
    """_score_hallucination: refusals score 1.0, tool calls score 0.0."""

    def test_plain_text_refusal_scores_one(self, tmp_path):
        scorer = _make_scorer(tmp_path, sample_index_map={})
        assert scorer._score_hallucination("I cannot help with that request.") == 1.0

    def test_structured_tool_call_scores_zero(self, tmp_path):
        scorer = _make_scorer(tmp_path, sample_index_map={})
        raw = json.dumps(
            [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ]
        )
        assert scorer._score_hallucination(raw) == 0.0


class TestFunctionCallExtractorPublicApi:
    """has_native_tool_calls is the stable entry point for hallucination scoring."""

    def test_true_for_serialized_tool_calls(self):
        raw = json.dumps(
            [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ]
        )
        assert FunctionCallExtractor.has_native_tool_calls(raw) is True

    def test_false_for_plain_text(self):
        assert (
            FunctionCallExtractor.has_native_tool_calls("just prose, no calls") is False
        )

    def test_extract_tolerates_non_dict_function_value(self):
        # A malformed tool_calls array whose "function" is not an object must
        # fall through to None instead of raising AttributeError (str/list have
        # no .get), which would otherwise abort scoring for the whole run.
        assert FunctionCallExtractor.extract('[{"function": "foo"}]') is None
        assert FunctionCallExtractor.extract('[{"function": ["x"]}]') is None
        assert FunctionCallExtractor.has_native_tool_calls('[{"function": "foo"}]') is (
            False
        )


class TestBFCLv4ScorerScoreAst:
    """_score_ast branch behavior (empty-expected shortcut + ast_checker path)."""

    def test_empty_expected_no_calls_scores_one(self, tmp_path):
        scorer = _make_scorer(tmp_path, sample_index_map={})
        # Empty expected + no model calls == correctly-abstained == 1.0.
        assert scorer._score_ast("[]", "[]") == 1.0

    def test_empty_expected_with_calls_scores_zero(self, tmp_path):
        scorer = _make_scorer(tmp_path, sample_index_map={})
        model = json.dumps([{"name": "f", "arguments": {}}])
        assert scorer._score_ast(model, "[]") == 0.0

    def test_invalid_ground_truth_scores_zero(self, tmp_path):
        scorer = _make_scorer(tmp_path, sample_index_map={})
        assert scorer._score_ast("[]", "not-json") == 0.0

    def test_ast_checker_result_is_honored(self, tmp_path, monkeypatch):
        import inference_endpoint.evaluation.bfcl_v4_scorer as mod

        monkeypatch.setattr(mod, "Language", {"PYTHON": "python"})
        monkeypatch.setattr(
            mod, "ast_checker", lambda **kw: {"valid": bool(kw["model_output"])}
        )
        scorer = _make_scorer(tmp_path, sample_index_map={})
        model = json.dumps([{"name": "f", "arguments": {}}])
        gt = json.dumps([{"f": {}}])
        assert scorer._score_ast(model, gt, subset="live_simple") == 1.0
        assert scorer._score_ast("[]", gt, subset="live_simple") == 0.0


class TestBFCLv4ScorerFullPath:
    """End-to-end score() incl. sample-weighted category aggregation + breakdown."""

    def test_live_sample_weighted_aggregation(self, tmp_path, monkeypatch):
        import inference_endpoint.evaluation.bfcl_v4_scorer as mod
        import pandas as pd

        monkeypatch.setattr(mod, "Language", {"PYTHON": "python"})
        # "Correct" iff the model produced any tool call (bfcl_output non-empty).
        monkeypatch.setattr(
            mod, "ast_checker", lambda **kw: {"valid": bool(kw["model_output"])}
        )

        report_dir = tmp_path / "report"
        gt = json.dumps([{"f": {}}])
        dataframe = pd.DataFrame(
            [
                {"subset": "live_simple", "ground_truth": gt},  # idx 0: correct
                {"subset": "live_simple", "ground_truth": gt},  # idx 1: wrong
                {"subset": "live_multiple", "ground_truth": gt},  # idx 2: correct
            ]
        )
        _write_events(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u0",
                    data=_tool_call_output(),
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u1",
                    data=TextModelOutput(output="Sorry, I can't."),
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u2",
                    data=_tool_call_output(),
                ),
            ],
        )

        scorer = _make_scorer(
            report_dir,
            sample_index_map={"u0": 0, "u1": 1, "u2": 2},
            dataframe=dataframe,
        )
        overall, n_repeats = scorer.score()
        breakdown = scorer.score_breakdown()

        # overall = mean(1, 0, 1) = 2/3
        assert overall == pytest.approx(2 / 3)
        assert n_repeats == 1
        assert breakdown["subset_scores"]["live_simple"] == 50.0
        assert breakdown["subset_scores"]["live_multiple"] == 100.0
        # live is sample_weighted: (0.5*2 + 1.0*1) / 3 == 2/3
        assert breakdown["category_scores"]["live"] == pytest.approx(66.67, abs=0.01)
        # single category present -> normalized equals that category mean
        assert breakdown["normalized_single_turn_score"] == pytest.approx(
            66.67, abs=0.01
        )
        assert breakdown["overall_accuracy"] == pytest.approx(66.67, abs=0.01)
        assert breakdown["total_samples"] == 3


pytestmark = pytest.mark.unit
