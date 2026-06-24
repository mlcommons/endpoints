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


pytestmark = pytest.mark.unit
