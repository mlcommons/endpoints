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

"""Unit tests for evaluation scoring module."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import pandas as pd
import pytest
from inference_endpoint.async_utils.services.event_logger.file_writer import JSONLWriter
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import PromptData, TextModelOutput
from inference_endpoint.dataset_manager.agentic_inference_dataset import (
    AgenticInferenceDataset,
)
from inference_endpoint.dataset_manager.predefined.shopify_product_catalogue import (
    ProductMetadata,
)
from inference_endpoint.evaluation import scoring as scoring_mod
from inference_endpoint.evaluation.scoring import (
    _PRED_CATEGORY_PAD,
    AgenticInferenceInlineScorer,
    PassAt1Scorer,
    Scorer,
    ShopifyCategoryF1Scorer,
    VBenchScorer,
    _calculate_hierarchical_f1,
    _create_pred_pad_category,
    _IssuedKey,
    _match_hierarchical_paths,
    _parse_response_to_category,
    guess_and_decode,
)
from pydantic import ValidationError


class TestMatchHierarchicalPaths:
    """Tests for _match_hierarchical_paths."""

    def test_exact_match(self):
        pred = "Clothing > Shirts > Polo"
        true = "Clothing > Shirts > Polo"
        inter, p_len, t_len = _match_hierarchical_paths(pred, true)
        assert inter == 3
        assert p_len == 3
        assert t_len == 3

    def test_partial_match_prefix(self):
        pred = "Clothing > Shirts"
        true = "Clothing > Shirts > Polo"
        inter, p_len, t_len = _match_hierarchical_paths(pred, true)
        assert inter == 2
        assert p_len == 2
        assert t_len == 3

    def test_mismatch_at_first_level(self):
        pred = "Electronics > Phones"
        true = "Clothing > Shirts > Polo"
        inter, p_len, t_len = _match_hierarchical_paths(pred, true)
        assert inter == 0
        assert p_len == 2
        assert t_len == 3


class TestCalculateHierarchicalF1:
    """Tests for _calculate_hierarchical_f1."""

    def test_perfect_score(self):
        data = [
            ("A > B > C", "A > B > C"),
            ("X > Y", "X > Y"),
        ]
        assert _calculate_hierarchical_f1(data) == 1.0

    def test_zero_score(self):
        data = [
            ("A > B", "X > Y"),
            ("P > Q", "M > N"),
        ]
        assert _calculate_hierarchical_f1(data) == 0.0

    def test_partial_score(self):
        # 2/3 match on first, 1/2 on second
        # Sample 1: inter=2, pred=3, true=3
        # Sample 2: inter=1, pred=2, true=2
        # total_inter=3, total_pred=5, total_true=5
        # hp=3/5, hr=3/5, f1=3/5
        data = [
            ("A > B > X", "A > B > C"),  # inter=2
            ("A > X", "A > B"),  # inter=1
        ]
        f1 = _calculate_hierarchical_f1(data)
        assert abs(f1 - 0.6) < 1e-6

    def test_empty_data(self):
        assert _calculate_hierarchical_f1([]) == 0.0


class TestCreatePredPadCategory:
    """Tests for _create_pred_pad_category."""

    def test_matching_depth(self):
        gt = "A > B > C"
        result = _create_pred_pad_category(gt, " > ")
        assert (
            result
            == f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"
        )

    def test_single_level(self):
        gt = "Electronics"
        result = _create_pred_pad_category(gt, " > ")
        assert result == _PRED_CATEGORY_PAD


class TestParseResponseToCategory:
    """Tests for _parse_response_to_category."""

    def test_valid_json(self):
        out = json.dumps(
            {"category": "Clothing > Shirts", "brand": "Nike", "is_secondhand": False}
        )
        assert (
            _parse_response_to_category(out, "Clothing > Shirts") == "Clothing > Shirts"
        )

    def test_json_with_extra_whitespace(self):
        out = '  { "category" : " A > B ", "brand": "x", "is_secondhand": false }  '
        assert _parse_response_to_category(out, "A > B") == "A > B"

    def test_raw_json_only(self):
        """Reference implementation expects raw JSON; no markdown stripping."""
        out = '{"category": "X > Y", "brand": "x", "is_secondhand": false}'
        assert _parse_response_to_category(out, "X > Y") == "X > Y"

    def test_invalid_json_uses_pred_pad(self):
        gt = "A > B > C"
        result = _parse_response_to_category("not json", gt)
        expected = f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"
        assert result == expected

    def test_missing_category_uses_pred_pad(self):
        out = '{"brand": "Nike", "is_secondhand": false}'
        result = _parse_response_to_category(out, "Electronics > Phones")
        expected = f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"
        assert result == expected

    def test_invalid_schema_uses_pred_pad(self):
        out = '{"category": "x", "brand": "y"}'  # missing required is_secondhand
        result = _parse_response_to_category(out, "A > B")
        assert result == f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"

    def test_markdown_wrapped_uses_pred_pad(self):
        """Reference passes raw string; markdown-wrapped JSON fails validation."""
        out = (
            '```json\n{"category": "X > Y", "brand": "x", "is_secondhand": false}\n```'
        )
        result = _parse_response_to_category(out, "A > B")
        assert result == f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"


class TestProductMetadata:
    """Tests for ProductMetadata model."""

    def test_valid_parse(self):
        data = '{"category": "A > B", "brand": "Nike", "is_secondhand": false}'
        parsed = ProductMetadata.model_validate_json(data)
        assert parsed.category == "A > B"
        assert parsed.brand == "Nike"
        assert parsed.is_secondhand is False

    def test_missing_field_raises(self):
        data = '{"category": "A", "brand": "x"}'
        with pytest.raises(ValidationError):
            ProductMetadata.model_validate_json(data)


class TestShopifyCategoryF1ScorerRegistration:
    """Test that ShopifyCategoryF1Scorer is registered."""

    def test_scorer_registered(self):
        assert "shopify_category_f1" in Scorer.available_scorers()
        scorer_cls = Scorer.get("shopify_category_f1")
        assert scorer_cls is ShopifyCategoryF1Scorer


class TestShopifyCategoryF1Scorer:
    """Integration tests for ShopifyCategoryF1Scorer."""

    @pytest.fixture
    def mock_dataset(self):
        df = pd.DataFrame(
            {
                "ground_truth_category": [
                    "Clothing > Shirts > Polo",
                    "Electronics > Phones",
                ],
            }
        )
        dataset = MagicMock()
        dataset.dataframe = df
        dataset.num_samples.return_value = 2
        return dataset

    @pytest.fixture
    def report_dir(self, tmp_path):
        return tmp_path

    def test_score_requires_sample_index_map_and_events(self, mock_dataset, report_dir):
        """Scorer raises when sample_idx_map.json missing (checked in __init__)."""
        with pytest.raises(FileNotFoundError, match="Sample index map"):
            ShopifyCategoryF1Scorer(
                dataset_name="test",
                dataset=mock_dataset,
                report_dir=report_dir,
            )


_LINE_ENCODER = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
# Long enough that a layout putting event_type after it starts it past byte 256.
_LONG_ID = "conv-" + "c" * 300


def _compact(record: EventRecord) -> bytes:
    """The line JSONLWriter writes: event_type first, no spaces."""
    return _LINE_ENCODER.encode(record) + b"\n"


def _spaced(record: EventRecord) -> bytes:
    """The same record as valid JSON with json.dumps' default spacing."""
    return json.dumps(json.loads(_LINE_ENCODER.encode(record))).encode() + b"\n"


def _event_type_last(record: EventRecord) -> bytes:
    """The same record with its fields reversed, so event_type comes last."""
    fields = json.loads(_LINE_ENCODER.encode(record))
    reordered = dict(reversed(fields.items()))
    return json.dumps(reordered, separators=(",", ":")).encode() + b"\n"


def _event_type_past_the_lead(record: EventRecord) -> bytes:
    """event_type after the (long) conversation_id, beyond the first 256 bytes."""
    fields = json.loads(_LINE_ENCODER.encode(record))
    reordered = {"conversation_id": fields.pop("conversation_id"), **fields}
    return json.dumps(reordered, separators=(",", ":")).encode() + b"\n"


# Valid layouts of one record: only the first is what the writer emits today.
_LAYOUTS = {
    "compact": _compact,
    "spaced": _spaced,
    "event_type_last": _event_type_last,
    "event_type_past_the_lead": _event_type_past_the_lead,
}


@pytest.mark.unit
class TestGuessAndDecode:
    @staticmethod
    def _line(record: EventRecord) -> bytes:
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        return encoder.encode(record) + b"\n"

    def test_issued_line_decodes_only_key_fields(self):
        line = self._line(
            EventRecord(
                event_type=SampleEventType.ISSUED,
                sample_uuid="u1",
                conversation_id="conv1__repeat_2",
                turn=3,
                data=PromptData(messages=({"role": "user", "content": "hello"},)),
            )
        )

        assert guess_and_decode(line) == _IssuedKey(
            sample_uuid="u1", conversation_id="conv1__repeat_2", turn=3
        )

    def test_complete_line_decodes_full_record(self):
        record = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="u1",
            conversation_id="conv1__repeat_2",
            turn=3,
            data=TextModelOutput(output="done"),
        )

        assert guess_and_decode(self._line(record)) == record

    def test_issued_line_with_data_past_1024_bytes_decodes_in_full(self):
        conversation_id = "c" * 2000
        line = self._line(
            EventRecord(
                event_type=SampleEventType.ISSUED,
                sample_uuid="u1",
                conversation_id=conversation_id,
                turn=3,
                data=PromptData(messages=({"role": "user", "content": "hello"},)),
            )
        )
        assert line.find(b',"data":') > 1024

        assert guess_and_decode(line) == _IssuedKey(
            sample_uuid="u1", conversation_id=conversation_id, turn=3
        )

    def test_other_event_returns_none(self):
        line = self._line(
            EventRecord(
                event_type=SampleEventType.RECV_FIRST,
                sample_uuid="u1",
                conversation_id="conv1",
                turn=3,
            )
        )

        assert guess_and_decode(line) is None

    @pytest.mark.parametrize("layout", list(_LAYOUTS))
    def test_complete_line_in_any_layout(self, layout):
        record = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="u1",
            conversation_id=_LONG_ID,
            turn=3,
            data=TextModelOutput(output="done"),
        )

        assert guess_and_decode(_LAYOUTS[layout](record)) == record

    @pytest.mark.parametrize("layout", list(_LAYOUTS))
    def test_issued_line_in_any_layout(self, layout):
        record = EventRecord(
            event_type=SampleEventType.ISSUED,
            sample_uuid="u1",
            conversation_id=_LONG_ID,
            turn=3,
            data=PromptData(messages=({"role": "user", "content": "hello"},)),
        )

        assert guess_and_decode(_LAYOUTS[layout](record)) == _IssuedKey(
            sample_uuid="u1", conversation_id=_LONG_ID, turn=3
        )

    @pytest.mark.parametrize("layout", list(_LAYOUTS))
    def test_other_event_in_any_layout_returns_none(self, layout):
        record = EventRecord(
            event_type=SampleEventType.RECV_FIRST,
            sample_uuid="u1",
            conversation_id=_LONG_ID,
            turn=3,
        )

        assert guess_and_decode(_LAYOUTS[layout](record)) is None

    def test_complete_whose_payload_mentions_the_issued_marker_stays_complete(self):
        """The event type comes from the line's lead, never from payload bytes."""
        record = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="u1",
            conversation_id="c",
            turn=1,
            data=TextModelOutput(tool_calls=[{"event_type": "sample.issued"}]),
        )
        line = self._line(record)
        # Searching the first bytes for the marker would read this as ISSUED.
        assert b'"event_type":"sample.issued"' in line[:256]

        assert guess_and_decode(line) == record

    def test_issued_cut_inside_a_nested_object_falls_back_to_the_whole_line(self):
        """A "data" key nested before the top-level one is not a field boundary."""
        line = (
            b'{"event_type":"sample.issued","timestamp_ns":1,"sample_uuid":"u1",'
            b'"conversation_id":"c1","turn":1,"meta":{"x":1,"data":2},"data":null}\n'
        )
        cut = line[: line.find(b',"data":')] + b"}"
        with pytest.raises(msgspec.DecodeError):
            msgspec.json.decode(cut)  # the optimization's cut is not valid JSON

        assert guess_and_decode(line) == _IssuedKey(
            sample_uuid="u1", conversation_id="c1", turn=1
        )

    @pytest.mark.parametrize("line", [b"\n", b"   \n", b""])
    def test_blank_line_returns_none(self, line):
        assert guess_and_decode(line) is None

    @pytest.mark.parametrize(
        "line",
        [
            b"not json\n",
            b'{"event_type":"sample.complete","time\n',  # truncated, compact lead
            b'{"event_type": "sample.complete", "time\n',  # truncated, spaced
        ],
    )
    def test_invalid_line_raises(self, line):
        with pytest.raises(msgspec.DecodeError):
            guess_and_decode(line)


def _write_event_lines(report_dir: Path, lines: list[bytes]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "sample_idx_map.json").write_bytes(
        msgspec.json.encode({"performance": {}})
    )
    (report_dir / "events.jsonl").write_bytes(b"".join(lines))


@pytest.mark.unit
class TestEventLogContract:
    """What the event logger writes is what the scorer reads.

    The scorer classifies log lines from their leading bytes. These tests feed it
    the real ``JSONLWriter`` output, and other valid layouts of the same records,
    so a change to the writer or a differently formatted log cannot silently drop
    events or change a score.
    """

    @staticmethod
    def _bash_tool_call(call_id: str, command: str) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"cmd": command})},
        }

    @classmethod
    def _dataset(cls) -> AgenticInferenceDataset:
        return AgenticInferenceDataset(
            pd.DataFrame(
                [
                    {
                        "conversation_id": "code1",
                        "turn": 1,
                        "role": "user",
                        "content": "run the tests",
                    },
                    {
                        "conversation_id": "code1",
                        "turn": 2,
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            cls._bash_tool_call("expected-code", "python -m pytest")
                        ],
                    },
                    {
                        "conversation_id": "sim_1",
                        "turn": 1,
                        "role": "user",
                        "content": "choose the next workflow action",
                    },
                    {
                        "conversation_id": "sim_1",
                        "turn": 2,
                        "role": "assistant",
                        "content": "expected workflow action",
                        "intent_codes": ["I042"],
                    },
                ]
            )
        )

    @classmethod
    def _records(cls) -> list[EventRecord]:
        """Both turns scored correctly, with an unrelated event in between."""
        return [
            EventRecord(event_type=SessionEventType.STARTED),
            EventRecord(
                event_type=SampleEventType.ISSUED,
                sample_uuid="code-response",
                conversation_id="code1",
                turn=1,
                data=PromptData(messages=({"role": "user", "content": "run it"},)),
            ),
            EventRecord(
                event_type=SampleEventType.RECV_FIRST,
                sample_uuid="code-response",
                conversation_id="code1",
                turn=1,
            ),
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                sample_uuid="code-response",
                conversation_id="code1",
                turn=1,
                data=TextModelOutput(
                    tool_calls=[cls._bash_tool_call("model-code", "python test.py")]
                ),
            ),
            EventRecord(
                event_type=SampleEventType.ISSUED,
                sample_uuid="workflow-response",
                conversation_id="sim_1",
                turn=1,
            ),
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                sample_uuid="workflow-response",
                conversation_id="sim_1",
                turn=1,
                data=TextModelOutput(output="intent: I042"),
            ),
            EventRecord(event_type=SessionEventType.ENDED),
        ]

    def _assert_scored_in_full(self, report_dir: Path) -> None:
        score, repeats = AgenticInferenceInlineScorer(
            "performance", self._dataset(), report_dir
        ).score()
        assert (score, repeats) == (1.0, 1)
        scores = json.loads((report_dir / "scores.json").read_text())
        assert scores["turns"]["scored"] == 2

    def test_every_event_type_the_writer_emits_classifies_like_a_full_decode(
        self, tmp_path
    ):
        records = [
            *(EventRecord(event_type=t, sample_uuid="u") for t in SessionEventType),
            *(EventRecord(event_type=t, sample_uuid="u") for t in SampleEventType),
            *(EventRecord(event_type=t, sample_uuid="u") for t in ErrorEventType),
            *self._records(),
        ]
        writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        try:
            for record in records:
                writer.write(record)
        finally:
            writer.close()
        lines = (tmp_path / "events.jsonl").read_bytes().splitlines(keepends=True)
        assert len(lines) == len(records)
        decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)

        for line in lines:
            full = decoder.decode(line)
            got = guess_and_decode(line)
            if full.event_type == SampleEventType.ISSUED:
                assert got == _IssuedKey(
                    full.sample_uuid, full.conversation_id, full.turn
                )
            elif full.event_type == SampleEventType.COMPLETE:
                assert got == full
            else:
                assert got is None

    def test_real_writer_output_is_scored(self, tmp_path):
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode({"performance": {}})
        )
        writer = JSONLWriter(report_dir / "events", flush_interval=1)
        try:
            for record in self._records():
                writer.write(record)
        finally:
            writer.close()

        self._assert_scored_in_full(report_dir)

    @pytest.mark.parametrize("layout", list(_LAYOUTS))
    def test_any_layout_scores_the_same(self, tmp_path, layout):
        report_dir = tmp_path / "report"
        _write_event_lines(
            report_dir, [_LAYOUTS[layout](record) for record in self._records()]
        )

        self._assert_scored_in_full(report_dir)

    def test_iter_complete_reads_every_layout_and_skips_blank_and_bad_lines(
        self, tmp_path, caplog
    ):
        def complete(uuid: str) -> EventRecord:
            return EventRecord(
                event_type=SampleEventType.COMPLETE,
                sample_uuid=uuid,
                conversation_id="c1",
                turn=1,
                data=TextModelOutput(output=uuid),
            )

        report_dir = tmp_path / "report"
        _write_event_lines(
            report_dir,
            [
                _compact(complete("u1")),
                _spaced(complete("u2")),
                _event_type_last(complete("u3")),
                b"\n",
                b"not json\n",
                _compact(
                    EventRecord(
                        event_type=SampleEventType.ISSUED,
                        sample_uuid="u4",
                        conversation_id="c1",
                        turn=1,
                    )
                ),
            ],
        )
        scorer = AgenticInferenceInlineScorer("performance", MagicMock(), report_dir)

        with caplog.at_level(logging.WARNING, logger=scoring_mod.logger.name):
            uuids = [uuid for uuid, _ in scorer._iter_complete()]

        assert uuids == ["u1", "u2", "u3"]
        assert [r.getMessage() for r in caplog.records if "line 5" in r.getMessage()]


@pytest.mark.unit
class TestAgenticInferenceInlineScorer:
    """AgenticInferenceInlineScorer unit tests."""

    @staticmethod
    def _bash_tool_call(call_id: str, command: str) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"cmd": command}),
            },
        }

    @staticmethod
    def _write_report(report_dir: Path, records: list[EventRecord]) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode({"performance": {}})
        )
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        with (report_dir / "events.jsonl").open("wb") as f:
            for record in records:
                f.write(encoder.encode(record) + b"\n")

    def test_scores_coding_and_workflow_turns(self, tmp_path):
        dataset = AgenticInferenceDataset(
            pd.DataFrame(
                [
                    {
                        "conversation_id": "code1",
                        "turn": 1,
                        "role": "user",
                        "content": "run the tests",
                    },
                    {
                        "conversation_id": "code1",
                        "turn": 2,
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            self._bash_tool_call("expected-code", "python -m pytest")
                        ],
                    },
                    {
                        "conversation_id": "sim_1",
                        "turn": 1,
                        "role": "user",
                        "content": "choose the next workflow action",
                    },
                    {
                        "conversation_id": "sim_1",
                        "turn": 2,
                        "role": "assistant",
                        "content": "expected workflow action",
                        "intent_codes": ["I042"],
                    },
                ]
            )
        )
        report_dir = tmp_path / "report"
        self._write_report(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="code-response",
                    conversation_id="code1",
                    turn=1,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="code-response",
                    conversation_id="code1",
                    turn=1,
                    data=TextModelOutput(
                        tool_calls=[
                            self._bash_tool_call("model-code", "python test.py")
                        ]
                    ),
                ),
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="workflow-response",
                    conversation_id="sim_1",
                    turn=1,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="workflow-response",
                    conversation_id="sim_1",
                    turn=1,
                    data=TextModelOutput(output="intent: I042"),
                ),
            ],
        )

        score, repeats = AgenticInferenceInlineScorer(
            "performance", dataset, report_dir
        ).score()

        assert score == 1.0
        assert repeats == 1
        scores = json.loads((report_dir / "scores.json").read_text())
        assert "valid" not in scores
        assert scores["turns"]["scored"] == 2
        assert scores["domains"] == {
            "coding": {"score": 1.0, "scored": 1},
            "workflow": {"score": 1.0, "scored": 1},
        }

    def test_turns_without_ground_truth_are_excluded(self, tmp_path):
        dataset = AgenticInferenceDataset(
            pd.DataFrame(
                [
                    {
                        "conversation_id": "code1",
                        "turn": 1,
                        "role": "user",
                        "content": "run the tests",
                    },
                    {
                        "conversation_id": "code1",
                        "turn": 2,
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            self._bash_tool_call("expected-code", "python -m pytest")
                        ],
                    },
                    {
                        "conversation_id": "code2",
                        "turn": 1,
                        "role": "user",
                        "content": "summarize the repository",
                    },
                    {
                        "conversation_id": "code2",
                        "turn": 2,
                        "role": "assistant",
                        "content": "This turn has no bash action ground truth.",
                    },
                ]
            )
        )
        report_dir = tmp_path / "report"
        self._write_report(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="code-response",
                    conversation_id="code1",
                    turn=1,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="code-response",
                    conversation_id="code1",
                    turn=1,
                    data=TextModelOutput(
                        tool_calls=[
                            self._bash_tool_call("model-code", "python test.py")
                        ]
                    ),
                ),
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="unscored-code-response",
                    conversation_id="code2",
                    turn=1,
                ),
            ],
        )

        score, repeats = AgenticInferenceInlineScorer(
            "performance", dataset, report_dir
        ).score()

        assert score == 1.0
        assert repeats == 1
        scores = json.loads((report_dir / "scores.json").read_text())
        assert "valid" not in scores
        assert scores["turns"] == {
            "issued": 2,
            "expected": 1,
            "observed": 1,
            "missing": 0,
            "scored": 1,
        }
        assert scores["excluded_turns"] == [
            {
                "conversation_id": "code2",
                "turn": 2,
                "domain": "coding",
                "exclude_reason": "no ground truth",
            }
        ]

    def test_scores_issued_turns_without_rounding_to_full_repeats(self, tmp_path):
        dataset = AgenticInferenceDataset(
            pd.DataFrame(
                [
                    {
                        "conversation_id": "code1",
                        "turn": 1,
                        "role": "user",
                        "content": "run the tests",
                    },
                    {
                        "conversation_id": "code1",
                        "turn": 2,
                        "role": "assistant",
                        "tool_calls": [
                            self._bash_tool_call("expected-code", "python -m pytest")
                        ],
                    },
                    {
                        "conversation_id": "sim_1",
                        "turn": 1,
                        "role": "user",
                        "content": "choose the next workflow action",
                    },
                    {
                        "conversation_id": "sim_1",
                        "turn": 2,
                        "role": "assistant",
                        "content": "expected workflow action",
                        "intent_codes": ["I042"],
                    },
                ]
            )
        )
        report_dir = tmp_path / "report"
        self._write_report(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="code-r1",
                    conversation_id="code1",
                    turn=1,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="code-r1",
                    conversation_id="code1",
                    turn=1,
                    data=TextModelOutput(
                        tool_calls=[
                            self._bash_tool_call("model-code-r1", "python test.py")
                        ]
                    ),
                ),
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="workflow-r1",
                    conversation_id="sim_1",
                    turn=1,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="workflow-r1",
                    conversation_id="sim_1",
                    turn=1,
                    data=None,
                ),
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="code-r2",
                    conversation_id="code1__repeat_2",
                    turn=1,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="code-r2",
                    conversation_id="code1__repeat_2",
                    turn=1,
                    data=TextModelOutput(
                        tool_calls=[
                            self._bash_tool_call("model-code-r2", "python test.py")
                        ]
                    ),
                ),
            ],
        )

        score, repeats = AgenticInferenceInlineScorer(
            "performance", dataset, report_dir
        ).score()

        assert score == 0.6667
        assert repeats == 2
        scores = json.loads((report_dir / "scores.json").read_text())
        assert scores["turns"] == {
            "issued": 3,
            "expected": 3,
            "observed": 2,
            "missing": 1,
            "scored": 3,
        }

    def test_issued_records_with_prompts_decode_only_their_keys(self, tmp_path):
        """ISSUED lines carry the whole prompt after "data"; the scorer reads only
        the fields before it, whatever the prompt holds, and still joins each
        repeat's completion (or its failure) to the issued turn."""
        dataset = AgenticInferenceDataset(
            pd.DataFrame(
                [
                    {
                        "conversation_id": "code1",
                        "turn": 1,
                        "role": "user",
                        "content": "run the tests",
                    },
                    {
                        "conversation_id": "code1",
                        "turn": 2,
                        "role": "assistant",
                        "tool_calls": [
                            self._bash_tool_call("expected", "python -m pytest")
                        ],
                    },
                ]
            )
        )
        # A prompt that itself mentions the event-type and "data" markers.
        prompt = PromptData(
            messages=(
                {
                    "role": "user",
                    "content": 'x,"data":"event_type":"sample.complete" ' * 500,
                },
            )
        )
        report_dir = tmp_path / "report"
        self._write_report(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="r1",
                    conversation_id="code1",
                    turn=1,
                    data=prompt,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="r1",
                    conversation_id="code1",
                    turn=1,
                    data=TextModelOutput(
                        tool_calls=[self._bash_tool_call("model", "python test.py")]
                    ),
                ),
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="r2",
                    conversation_id="code1__repeat_2",
                    turn=1,
                    data=prompt,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="r2",
                    conversation_id="code1__repeat_2",
                    turn=1,
                ),
            ],
        )

        score, repeats = AgenticInferenceInlineScorer(
            "performance", dataset, report_dir
        ).score()

        assert (score, repeats) == (0.5, 2)
        scores = json.loads((report_dir / "scores.json").read_text())
        assert scores["turns"]["issued"] == 2
        assert scores["turns"]["observed"] == 1

    @pytest.mark.parametrize("late_field", ["sample_uuid", "conversation_id", "turn"])
    def test_issued_line_in_another_field_order_is_decoded_in_full(
        self, tmp_path, late_field
    ):
        """An ISSUED line with a key field after "data" falls back to decoding
        the whole line."""
        dataset = AgenticInferenceDataset(
            pd.DataFrame(
                [
                    {
                        "conversation_id": "sim_1",
                        "turn": 1,
                        "role": "user",
                        "content": "choose the next workflow action",
                    },
                    {
                        "conversation_id": "sim_1",
                        "turn": 2,
                        "role": "assistant",
                        "content": "expected workflow action",
                        "intent_codes": ["I042"],
                    },
                ]
            )
        )
        report_dir = tmp_path / "report"
        self._write_report(
            report_dir,
            [
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="w1",
                    conversation_id="sim_1",
                    turn=1,
                    data=TextModelOutput(output="intent: I042"),
                ),
            ],
        )
        fields = {"sample_uuid": '"w1"', "conversation_id": '"sim_1"', "turn": "1"}
        early = "".join(
            f'"{name}":{value},' for name, value in fields.items() if name != late_field
        )
        line = (
            f'{{"event_type":"sample.issued",{early}"data":null,'
            f'"{late_field}":{fields[late_field]}}}\n'
        )
        with (report_dir / "events.jsonl").open("a") as f:
            f.write(line)

        score, repeats = AgenticInferenceInlineScorer(
            "performance", dataset, report_dir
        ).score()

        assert (score, repeats) == (1.0, 1)


@pytest.mark.unit
class TestVBenchScorerRegistration:
    def test_scorer_registered(self):
        assert "vbench" in Scorer.available_scorers()
        assert Scorer.get("vbench") is VBenchScorer


@pytest.mark.unit
class TestVBenchScorer:
    """VBenchScorer unit tests with VBench monkey-patched."""

    DIMS = (
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "appearance_style",
        "scene",
    )
    # Per-dim aggregate scores. Mean = 0.55.
    DIM_SCORES = {
        "subject_consistency": 0.9,
        "background_consistency": 0.8,
        "motion_smoothness": 0.7,
        "dynamic_degree": 0.4,
        "appearance_style": 0.3,
        "scene": 0.2,
    }

    @pytest.fixture
    def dataset(self):
        df = pd.DataFrame({"prompt": ["a cat", "a dog", "a tree"]})
        ds = MagicMock()
        ds.dataframe = df
        ds.num_samples.return_value = 3
        return ds

    @pytest.fixture
    def staged(self, tmp_path):
        """Build report_dir with sample_idx_map + events.jsonl pointing at fake mp4s.

        Three samples; each video file is a real (empty) file so symlinking
        works on a real filesystem.
        """
        report_dir = tmp_path / "report"
        report_dir.mkdir()

        video_paths = []
        for i in range(3):
            p = tmp_path / f"video_{i}.mp4"
            p.write_bytes(b"")
            video_paths.append(str(p))

        uuids = [f"uuid-{i}" for i in range(3)]
        sample_idx_map = {"vid_acc": dict(zip(uuids, range(3), strict=True))}
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode(sample_idx_map)
        )

        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        events_path = report_dir / "events.jsonl"
        with events_path.open("wb") as f:
            for uid, vp in zip(uuids, video_paths, strict=True):
                rec = EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid=uid,
                    data=TextModelOutput(output=vp),
                )
                f.write(encoder.encode(rec) + b"\n")
        return report_dir, video_paths

    @pytest.fixture
    def vbench_project(self, tmp_path):
        """Stub accuracy subproject with a vbench_runner.py the scorer can find."""
        project = tmp_path / "accuracy"
        project.mkdir()
        (project / "vbench_runner.py").write_text("# stub\n")
        return project

    @pytest.fixture
    def patch_subprocess(self, monkeypatch):
        """Capture subprocess.run; side-effect writes a fake VBench results JSON.

        The fake parses --out-dir / --name / --dims out of the command list and
        writes results.json shaped like VBench's real output:
        `{dim: [aggregate_score, [per_video_results, ...]]}`.
        """
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            name = cmd[cmd.index("--name") + 1]
            dims = cmd[cmd.index("--dims") + 1].split(",")
            results = {dim: [TestVBenchScorer.DIM_SCORES[dim], []] for dim in dims}
            (out_dir / f"{name}_eval_results.json").write_bytes(
                msgspec.json.encode(results)
            )
            return MagicMock(returncode=0, stdout="ok\n")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        return captured

    def test_score_averages_six_dims(
        self, dataset, staged, vbench_project, patch_subprocess
    ):
        report_dir, video_paths = staged
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        mean_score, n_repeats = scorer.score()

        assert mean_score == pytest.approx(0.55)
        assert n_repeats == 1

        # Subprocess was invoked via `uv run --project <subproject> python ...`.
        cmd = patch_subprocess["cmd"]
        assert cmd[0] == "uv"
        assert cmd[1:3] == ["run", "--project"]
        assert Path(cmd[3]) == vbench_project
        assert Path(cmd[5]) == vbench_project / "vbench_runner.py"
        # All six dims passed through, in declared order.
        dims_arg = cmd[cmd.index("--dims") + 1]
        assert dims_arg.split(",") == list(self.DIMS)

        # Videos were staged as `{prompt}-0.mp4`.
        staged_dir = report_dir / "vbench_videos"
        names = sorted(p.name for p in staged_dir.iterdir())
        assert names == ["a cat-0.mp4", "a dog-0.mp4", "a tree-0.mp4"]
        # And each symlink resolves to the original mp4.
        for staged_file, original in zip(
            sorted(staged_dir.iterdir()), sorted(video_paths), strict=True
        ):
            assert staged_file.resolve() == Path(original).resolve()

    def test_missing_subproject_raises_at_init(self, dataset, staged, tmp_path):
        report_dir, _ = staged
        nonexistent = tmp_path / "no_such_project"
        with pytest.raises(FileNotFoundError, match="vbench_runner.py"):
            VBenchScorer(
                dataset_name="vid_acc",
                dataset=dataset,
                report_dir=report_dir,
                ground_truth_column="prompt",
                vbench_project_path=nonexistent,
            )

    def test_score_filters_empty_outputs(
        self, dataset, tmp_path, vbench_project, patch_subprocess, caplog
    ):
        """Failed queries (record.data=None → output="") must not reach _stage_videos.

        Without filtering, Path("").resolve() returns cwd and the staged
        symlink would point at the repo root.
        """
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        video_paths = []
        for i in range(3):
            p = tmp_path / f"video_{i}.mp4"
            p.write_bytes(b"")
            video_paths.append(str(p))

        uuids = [f"uuid-{i}" for i in range(3)]
        sample_idx_map = {"vid_acc": dict(zip(uuids, range(3), strict=True))}
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode(sample_idx_map)
        )

        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        # Sample 1 has no TextModelOutput data — simulates a failed query
        # where worker set response_output=None.
        with (report_dir / "events.jsonl").open("wb") as f:
            for i, (uid, vp) in enumerate(zip(uuids, video_paths, strict=True)):
                data = None if i == 1 else TextModelOutput(output=vp)
                rec = EventRecord(
                    event_type=SampleEventType.COMPLETE, sample_uuid=uid, data=data
                )
                f.write(encoder.encode(rec) + b"\n")

        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        with caplog.at_level("WARNING"):
            mean_score, _ = scorer.score()

        assert mean_score == pytest.approx(0.55)
        assert "dropped 1" in caplog.text
        # Only the two non-empty samples were staged.
        names = sorted(p.name for p in (report_dir / "vbench_videos").iterdir())
        assert names == ["a cat-0.mp4", "a tree-0.mp4"]

    def test_stage_videos_idempotent_on_rerun(
        self, dataset, staged, vbench_project, patch_subprocess
    ):
        """Calling score() twice on the same report_dir must not raise FileExistsError."""
        report_dir, _ = staged
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        scorer.score()
        # Second call must not crash with FileExistsError on the symlinks.
        scorer.score()

    def test_stage_clears_stale_files_from_prior_run(
        self, dataset, staged, vbench_project, patch_subprocess
    ):
        """Re-scoring must wipe stale `{prompt}-{N}.mp4` from a prior run.

        Otherwise VBench walks the directory and scores zombie videos from a
        higher-repeat earlier run.
        """
        report_dir, _ = staged
        staged_dir = report_dir / "vbench_videos"
        staged_dir.mkdir(parents=True, exist_ok=True)
        # Pretend a prior 3-repeat run left these around.
        zombie = staged_dir / "a cat-2.mp4"
        zombie.write_bytes(b"")
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        scorer.score()
        names = sorted(p.name for p in staged_dir.iterdir())
        assert names == ["a cat-0.mp4", "a dog-0.mp4", "a tree-0.mp4"]
        assert not zombie.exists()

    def test_subprocess_failure_includes_stderr_tail(
        self, dataset, staged, vbench_project, monkeypatch, tmp_path
    ):
        """Non-zero subprocess exit must raise with captured output and log path."""
        report_dir, _ = staged

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=2, stdout="boom: CUDA OOM\nline2\n")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        with pytest.raises(RuntimeError, match=r"(?s)exited with code 2.*CUDA OOM"):
            scorer.score()
        assert (report_dir / "vbench_subprocess.log").read_text() == (
            "boom: CUDA OOM\nline2\n"
        )

    def test_subprocess_timeout_raises(
        self, dataset, staged, vbench_project, monkeypatch
    ):
        report_dir, _ = staged

        def fake_run(cmd, **kwargs):
            import subprocess as _sp

            raise _sp.TimeoutExpired(cmd=cmd, timeout=1)

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
            subprocess_timeout_s=1,
        )
        with pytest.raises(RuntimeError, match="timed out"):
            scorer.score()

    def test_missing_dim_raises_named_error(
        self, dataset, staged, vbench_project, monkeypatch
    ):
        """A missing dim in the results JSON must fail loudly with the dim name."""
        report_dir, _ = staged

        def fake_run(cmd, **kwargs):
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            name = cmd[cmd.index("--name") + 1]
            # Drop `scene` to mimic the RUNBOOK §5 known failure mode.
            results = {
                dim: [TestVBenchScorer.DIM_SCORES[dim], []]
                for dim in TestVBenchScorer.DIMS
                if dim != "scene"
            }
            (out_dir / f"{name}_eval_results.json").write_bytes(
                msgspec.json.encode(results)
            )
            return MagicMock(returncode=0, stdout="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        with pytest.raises(ValueError, match="missing dimensions.*scene"):
            scorer.score()

    def test_malformed_results_entry_raises(
        self, dataset, staged, vbench_project, monkeypatch
    ):
        report_dir, _ = staged

        def fake_run(cmd, **kwargs):
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            name = cmd[cmd.index("--name") + 1]
            # Replace `scene` with an empty list — IndexError on entry[0].
            results: dict = {
                dim: [TestVBenchScorer.DIM_SCORES[dim], []]
                for dim in TestVBenchScorer.DIMS
            }
            results["scene"] = []
            (out_dir / f"{name}_eval_results.json").write_bytes(
                msgspec.json.encode(results)
            )
            return MagicMock(returncode=0, stdout="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        with pytest.raises(ValueError, match="dimension 'scene' is malformed"):
            scorer.score()

    def test_all_failed_returns_none_with_correct_repeats(
        self, dataset, tmp_path, vbench_project, monkeypatch
    ):
        """All-empty outputs → (None, n_repeats), n_repeats from issued count."""
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        uuids = [f"uuid-{i}" for i in range(3)]
        sample_idx_map = {"vid_acc": dict(zip(uuids, range(3), strict=True))}
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode(sample_idx_map)
        )
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        with (report_dir / "events.jsonl").open("wb") as f:
            for uid in uuids:
                rec = EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid=uid,
                    data=None,
                )
                f.write(encoder.encode(rec) + b"\n")

        # subprocess.run must NOT be invoked when there's nothing to score.
        def fail_if_called(cmd, **kwargs):
            raise AssertionError("subprocess.run should not run on empty input")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fail_if_called)
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        score, n_repeats = scorer.score()
        assert score is None
        # n_repeats reflects issued count (3 samples / 3 dataset rows = 1),
        # not surviving rows.
        assert n_repeats == 1

    def test_unsafe_prompt_with_dotdot_rejected(
        self, staged, vbench_project, patch_subprocess
    ):
        """A prompt containing `..` must raise rather than escape staged_dir."""
        report_dir, _ = staged
        # Dataset with a hostile prompt for one of the three rows.
        df = pd.DataFrame({"prompt": ["a cat", "../../etc/passwd", "a tree"]})
        ds = MagicMock()
        ds.dataframe = df
        ds.num_samples.return_value = 3
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=ds,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        with pytest.raises(ValueError, match=r"\.\."):
            scorer.score()

    def test_unsafe_prompt_with_slash_sanitized(
        self, staged, vbench_project, patch_subprocess
    ):
        """A prompt with `/` must be sanitized to `_` and kept inside staged_dir."""
        report_dir, _ = staged
        df = pd.DataFrame({"prompt": ["cat/dog", "a dog", "a tree"]})
        ds = MagicMock()
        ds.dataframe = df
        ds.num_samples.return_value = 3
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=ds,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        scorer.score()
        names = sorted(p.name for p in (report_dir / "vbench_videos").iterdir())
        assert "cat_dog-0.mp4" in names
        # No nested subdir was created.
        for child in (report_dir / "vbench_videos").iterdir():
            assert child.is_symlink()

    def test_missing_src_video_raises_before_subprocess(
        self, dataset, tmp_path, vbench_project, monkeypatch
    ):
        """A non-existent src path must raise at staging, not in VBench."""
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        uuids = [f"uuid-{i}" for i in range(3)]
        sample_idx_map = {"vid_acc": dict(zip(uuids, range(3), strict=True))}
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode(sample_idx_map)
        )
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        with (report_dir / "events.jsonl").open("wb") as f:
            for uid in uuids:
                rec = EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid=uid,
                    data=TextModelOutput(output="/nonexistent/video.mp4"),
                )
                f.write(encoder.encode(rec) + b"\n")

        def fail_if_called(cmd, **kwargs):
            raise AssertionError("subprocess should not run when src is missing")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fail_if_called)
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        with pytest.raises(FileNotFoundError):
            scorer.score()

    def test_env_var_resolves_project_path(
        self, dataset, staged, tmp_path, monkeypatch, patch_subprocess
    ):
        """VBENCH_PROJECT_PATH env var is consulted when no explicit path is given."""
        report_dir, _ = staged
        env_project = tmp_path / "env_project"
        env_project.mkdir()
        (env_project / "vbench_runner.py").write_text("# stub\n")
        monkeypatch.setenv("VBENCH_PROJECT_PATH", str(env_project))
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
        )
        assert scorer.vbench_project_path == env_project


@pytest.mark.unit
class TestGetRawOutputs:
    @staticmethod
    def _write_events(report_dir: Path, records: list[EventRecord]) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        with (report_dir / "events.jsonl").open("wb") as f:
            for record in records:
                f.write(encoder.encode(record) + b"\n")

    def _scorer(self, report_dir: Path) -> Scorer:
        # get_raw_outputs is the base-class read; use an already-registered
        # concrete scorer so the test does not add a class to Scorer.PREDEFINED.
        # Bypass __init__ — get_raw_outputs only reads self.report_dir.
        scorer = object.__new__(PassAt1Scorer)
        scorer.report_dir = report_dir
        return scorer

    def test_raw_outputs_hold_every_complete_row(self, tmp_path):
        self._write_events(
            tmp_path,
            [
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u1",
                    data=TextModelOutput(output="a b"),
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u2",
                    data=TextModelOutput(output="c"),
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u3",
                    data=TextModelOutput(output="d e f"),
                ),
            ],
        )
        frame = self._scorer(tmp_path).get_raw_outputs()
        assert dict(zip(frame["sample_uuid"], frame["output"], strict=True)) == {
            "u1": "a b",
            "u2": "c",
            "u3": "d e f",
        }

    def test_log_without_completes_returns_columned_frame(self, tmp_path):
        # A log with no COMPLETE event must still yield a frame WITH the
        # expected columns (not a column-less pd.DataFrame([])), so callers can
        # index "sample_uuid"/"output" without a KeyError.
        self._write_events(
            tmp_path,
            [
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    sample_uuid="u1",
                ),
            ],
        )
        frame = self._scorer(tmp_path).get_raw_outputs()
        assert frame.empty
        assert list(frame.columns) == ["sample_uuid", "output"]

    def test_iter_complete_outputs_reads_text_and_usage(self, tmp_path):
        out = TextModelOutput(output="a b")
        self._write_events(
            tmp_path,
            [
                EventRecord(event_type=SampleEventType.ISSUED, sample_uuid="u1"),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u1",
                    data=out,
                    prompt_tokens=2,
                    completion_tokens=3,
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE, sample_uuid="u2", data=out
                ),
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u3",
                    prompt_tokens=2,
                    completion_tokens=0,
                ),
            ],
        )
        scorer = self._scorer(tmp_path)

        assert list(scorer.iter_complete_outputs(None)) == [
            ("u1", "a b", 3),
            ("u2", "a b", None),
            ("u3", "", None),
        ]
        bounded = scorer.iter_complete_outputs({"u1", "u3"})
        assert [uuid for uuid, *_ in bounded] == ["u1", "u3"]
        # BFCL unpacks _iter_complete as (sample_uuid, data); the shape is kept.
        assert [u for u, _ in scorer._iter_complete()] == ["u1", "u2", "u3"]

    def test_events_reader_decodes_only_complete_lines(self, tmp_path):
        """Only COMPLETE lines are decoded, so a corrupt line of another event
        type cannot fail scoring."""
        complete = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="u1",
            data=TextModelOutput(output="a"),
        )
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        (tmp_path / "events.jsonl").write_bytes(
            b'{"event_type":"sample.issued",<<<not json>>>}\n'
            + encoder.encode(complete)
            + b"\n"
        )

        frame = self._scorer(tmp_path).get_raw_outputs()

        assert frame.to_dict("records") == [{"sample_uuid": "u1", "output": "a"}]

    def test_non_compact_log_is_decoded(self, tmp_path):
        """A line outside the compact layout is decoded whole, not dropped."""
        complete = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="u1",
            data=TextModelOutput(output="a"),
        )
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        spaced = json.dumps(json.loads(encoder.encode(complete))).encode()
        (tmp_path / "events.jsonl").write_bytes(b"\n" + spaced + b"\n")
        scorer = self._scorer(tmp_path)

        assert scorer.get_raw_outputs().to_dict("records") == [
            {"sample_uuid": "u1", "output": "a"}
        ]
        assert list(scorer.iter_complete_outputs(None)) == [("u1", "a", None)]

    def test_compact_log_after_blank_lines_is_read(self, tmp_path):
        complete = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="u1",
            data=TextModelOutput(output="a"),
        )
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        (tmp_path / "events.jsonl").write_bytes(
            b"\n\n" + encoder.encode(complete) + b"\n"
        )

        frame = self._scorer(tmp_path).get_raw_outputs()

        assert frame.to_dict("records") == [{"sample_uuid": "u1", "output": "a"}]


@pytest.mark.unit
class TestScoreDenominator:
    """Base ``Scorer.score()`` divides by the *issued* count, so samples with no
    COMPLETE event count as failures instead of inflating the surviving subset."""

    @staticmethod
    def _write_events(report_dir: Path, uuid_to_output: dict[str, str]) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        with (report_dir / "events.jsonl").open("wb") as f:
            for uid, out in uuid_to_output.items():
                f.write(
                    encoder.encode(
                        EventRecord(
                            event_type=SampleEventType.COMPLETE,
                            sample_uuid=uid,
                            data=TextModelOutput(output=out),
                        )
                    )
                    + b"\n"
                )

    def _scorer(
        self,
        report_dir: Path,
        ground_truths: list[str],
        sample_index_map: dict[str, int],
    ) -> Scorer:
        # Bypass __init__ (which would read sample_idx_map.json off disk and
        # register the class); set only what score() reads. PassAt1Scorer's
        # score_single_sample is exact match: 1.0 if output == ground truth.
        scorer = object.__new__(PassAt1Scorer)
        scorer.report_dir = report_dir
        scorer.sample_index_map = sample_index_map
        scorer.extractor = None
        scorer.ground_truth_column = "answer"
        dataset = MagicMock()
        dataset.dataframe = pd.DataFrame({"answer": ground_truths})
        dataset.num_samples.return_value = len(ground_truths)
        scorer.dataset = dataset
        return scorer

    def test_missing_samples_count_as_failures(self, tmp_path):
        # 3 issued, only 2 completed (both correct); u2 never produced a COMPLETE
        # event. Accuracy is 2/3, not mean([1.0, 1.0]) == 1.0 over the survivors.
        self._write_events(tmp_path, {"u0": "a", "u1": "b"})
        scorer = self._scorer(
            tmp_path,
            ground_truths=["a", "b", "c"],
            sample_index_map={"u0": 0, "u1": 1, "u2": 2},
        )
        mean, n_repeats = scorer.score()
        assert mean == pytest.approx(2 / 3)
        assert n_repeats == 1  # issued (3) // num_samples (3), not survivors (2)

    def test_all_missing_yields_zero_not_nan(self, tmp_path):
        # No COMPLETE events: 0 correct over 3 issued == 0.0, never NaN.
        self._write_events(tmp_path, {})
        scorer = self._scorer(
            tmp_path,
            ground_truths=["a", "b", "c"],
            sample_index_map={"u0": 0, "u1": 1, "u2": 2},
        )
        mean, n_repeats = scorer.score()
        assert mean == 0.0
        assert n_repeats == 1

    def test_no_samples_issued_returns_none(self, tmp_path):
        # Empty issued set: nothing to score, return None (not a fabricated 0.0),
        # and the issued==0 guard returns before num_samples()==0 divides by zero.
        self._write_events(tmp_path, {})
        scorer = self._scorer(tmp_path, ground_truths=[], sample_index_map={})
        mean, n_repeats = scorer.score()
        assert mean is None
        assert n_repeats == 0
