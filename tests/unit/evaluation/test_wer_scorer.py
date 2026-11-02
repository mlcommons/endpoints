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

"""Unit tests for WERScorer."""

from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import pandas as pd
import pytest
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.evaluation import scoring
from inference_endpoint.evaluation.extractor import PythonCodeExtractor
from inference_endpoint.evaluation.scoring import Scorer, WERScorer
from inference_endpoint.evaluation.wer import project_to_labels
from inference_endpoint.exceptions import InputValidationError
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

DATASET_NAME = "whisper_librispeech"

REFERENCES = [
    "mister quilter is the apostle of the middle classes",
    "and we are glad to welcome his gospel",
    "nor is mister quilter's manner less interesting than his matter",
]


def _ref_word_count(reference: str) -> int:
    """Reference word count as the scorer counts it (post-normalization)."""
    normalizer = EnglishTextNormalizer({})
    return len(normalizer(project_to_labels(reference)).split())


@pytest.fixture
def hermetic_normalizer(monkeypatch: pytest.MonkeyPatch):
    """Use the real normalizer with an empty spelling map - no HF download."""
    monkeypatch.setattr(
        WERScorer, "_normalizer", lambda self: EnglishTextNormalizer({})
    )


@pytest.fixture
def dataset():
    ds = MagicMock()
    ds.dataframe = pd.DataFrame({"transcript": REFERENCES})
    ds.num_samples.return_value = len(REFERENCES)
    return ds


def _stage(report_dir: Path, outputs: dict[int, str]) -> Path:
    """Write sample_idx_map for all references and events for `outputs` only.

    Keys are dataset sample indices; a reference without an entry stands in for
    a sample that never produced a COMPLETE event.
    """
    report_dir.mkdir(exist_ok=True)
    uuids = {i: f"uuid-{i}" for i in range(len(REFERENCES))}
    (report_dir / "sample_idx_map.json").write_bytes(
        msgspec.json.encode({DATASET_NAME: {u: i for i, u in uuids.items()}})
    )
    encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
    with (report_dir / "events.jsonl").open("wb") as f:
        for index, text in outputs.items():
            rec = EventRecord(
                event_type=SampleEventType.COMPLETE,
                sample_uuid=uuids[index],
                data=TextModelOutput(output=text),
            )
            f.write(encoder.encode(rec) + b"\n")
    return report_dir


def _scorer(report_dir: Path, dataset, **kwargs) -> WERScorer:
    return WERScorer(
        DATASET_NAME,
        dataset,
        report_dir,
        ground_truth_column="transcript",
        **kwargs,
    )


@pytest.mark.unit
class TestRegistration:
    def test_registered(self):
        assert "wer" in Scorer.available_scorers()
        assert Scorer.get("wer") is WERScorer

    def test_requires_no_extractor(self):
        assert WERScorer.REQUIRES_EXTRACTOR is False

    def test_runs_on_endpoint_outputs(self):
        assert WERScorer.SKIP_ENDPOINT_PHASE is False


@pytest.mark.unit
@pytest.mark.usefixtures("hermetic_normalizer")
class TestScore:
    def test_perfect_transcription_scores_100(self, tmp_path: Path, dataset):
        report_dir = _stage(tmp_path / "r", dict(enumerate(REFERENCES)))
        score, n_repeats = _scorer(report_dir, dataset).score()
        assert score == pytest.approx(100.0)
        assert n_repeats == 1

    def test_casing_and_punctuation_do_not_count_as_errors(
        self, tmp_path: Path, dataset
    ):
        outputs = {i: f"{r.capitalize()}." for i, r in enumerate(REFERENCES)}
        report_dir = _stage(tmp_path / "r", outputs)
        assert _scorer(report_dir, dataset).score()[0] == pytest.approx(100.0)

    def test_one_substitution_over_the_corpus(self, tmp_path: Path, dataset):
        outputs = dict(enumerate(REFERENCES))
        outputs[0] = REFERENCES[0].replace("apostle", "epistle")
        report_dir = _stage(tmp_path / "r", outputs)

        scorer = _scorer(report_dir, dataset)
        score, _ = scorer.score()
        breakdown = scorer.score_breakdown()
        assert breakdown is not None

        assert breakdown["edit_distance"] == 1
        # Corpus-level: 1 error over every reference word, not a per-sample mean.
        assert breakdown["wer_percent"] == pytest.approx(
            100.0 / breakdown["reference_words"]
        )
        assert score == pytest.approx(100.0 - breakdown["wer_percent"])

    def test_empty_transcription_counts_every_word_as_an_error(
        self, tmp_path: Path, dataset
    ):
        report_dir = _stage(tmp_path / "r", {i: "" for i in range(len(REFERENCES))})
        scorer = _scorer(report_dir, dataset)
        score, _ = scorer.score()
        assert score == pytest.approx(0.0)
        breakdown = scorer.score_breakdown()
        assert breakdown is not None
        assert breakdown["empty_responses"] == len(REFERENCES)

    def test_concatenated_words_are_credited(self, tmp_path: Path, dataset):
        ds = MagicMock()
        ds.dataframe = pd.DataFrame({"transcript": ["i can not go"]})
        ds.num_samples.return_value = 1
        report_dir = _stage(tmp_path / "r", {})
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode({DATASET_NAME: {"uuid-0": 0}})
        )
        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        (report_dir / "events.jsonl").write_bytes(
            encoder.encode(
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="uuid-0",
                    data=TextModelOutput(output="I cannot go."),
                )
            )
            + b"\n"
        )
        assert _scorer(report_dir, ds).score()[0] == pytest.approx(100.0)


@pytest.mark.unit
@pytest.mark.usefixtures("hermetic_normalizer")
class TestPartialRuns:
    def test_missing_sample_is_scored_as_an_empty_transcript(
        self, tmp_path: Path, dataset
    ):
        # Only 2 of 3 issued samples completed, both perfectly. The third
        # contributes its reference words as deletions rather than vanishing
        # from the denominator, so the score must fall below 100.
        report_dir = _stage(tmp_path / "r", {0: REFERENCES[0], 1: REFERENCES[1]})
        scorer = _scorer(report_dir, dataset)
        score, _ = scorer.score()
        breakdown = scorer.score_breakdown()
        assert breakdown is not None
        assert breakdown["missing_samples"] == 1
        # Counted through the same normalizer the scorer uses: it rewrites
        # contractions, so a raw str.split() is not the reference word count.
        assert breakdown["edit_distance"] == _ref_word_count(REFERENCES[2])
        assert breakdown["reference_words"] == sum(
            _ref_word_count(r) for r in REFERENCES
        )
        assert score == pytest.approx(100.0 - breakdown["wer_percent"])
        assert score < 100.0

    def test_missing_samples_mark_the_score_incomplete(self, tmp_path: Path, dataset):
        report_dir = _stage(tmp_path / "r", {0: REFERENCES[0], 1: REFERENCES[1]})
        scorer = _scorer(report_dir, dataset)
        scorer.score()
        assert scorer.complete is False

    def test_a_full_run_is_complete(self, tmp_path: Path, dataset):
        report_dir = _stage(tmp_path / "r", dict(enumerate(REFERENCES)))
        scorer = _scorer(report_dir, dataset)
        scorer.score()
        assert scorer.complete is True

    def test_an_empty_response_does_not_mark_the_run_incomplete(
        self, tmp_path: Path, dataset
    ):
        # A blank transcript is a response: the sample completed and scored 0.
        report_dir = _stage(tmp_path / "r", {i: "" for i in range(len(REFERENCES))})
        scorer = _scorer(report_dir, dataset)
        scorer.score()
        assert scorer.complete is True
        breakdown = scorer.score_breakdown()
        assert breakdown is not None
        assert breakdown["missing_samples"] == 0

    def test_no_completions_scores_zero_and_is_incomplete(
        self, tmp_path: Path, dataset
    ):
        # Every issued sample is scored as an empty transcript, so the whole
        # corpus is deletions: WER 100% -> score 0, not an absent number.
        report_dir = _stage(tmp_path / "r", {})
        scorer = _scorer(report_dir, dataset)
        score, _ = scorer.score()
        assert score == pytest.approx(0.0)
        assert scorer.complete is False
        breakdown = scorer.score_breakdown()
        assert breakdown is not None
        assert breakdown["missing_samples"] == len(REFERENCES)
        assert breakdown["wer_percent"] == pytest.approx(100.0)

    def test_nothing_issued_yields_no_score(self, tmp_path: Path, dataset):
        report_dir = tmp_path / "r"
        report_dir.mkdir()
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode({DATASET_NAME: {}})
        )
        (report_dir / "events.jsonl").write_bytes(b"")
        scorer = _scorer(report_dir, dataset)
        assert scorer.score() == (None, 0)
        assert scorer.complete is False


@pytest.mark.unit
@pytest.mark.usefixtures("hermetic_normalizer")
class TestContract:
    def test_score_single_sample_is_not_usable(self, tmp_path: Path, dataset):
        report_dir = _stage(tmp_path / "r", {0: REFERENCES[0]})
        with pytest.raises(RuntimeError, match="corpus-level"):
            _scorer(report_dir, dataset).score_single_sample("a", "b")

    def test_breakdown_is_none_before_scoring(self, tmp_path: Path, dataset):
        report_dir = _stage(tmp_path / "r", {0: REFERENCES[0]})
        assert _scorer(report_dir, dataset).score_breakdown() is None

    def test_default_normalizer_source(self, tmp_path: Path, dataset):
        report_dir = _stage(tmp_path / "r", {})
        scorer = _scorer(report_dir, dataset)
        assert scorer.normalizer_source == "openai/whisper-large-v3"

    def test_normalizer_source_override(self, tmp_path: Path, dataset):
        report_dir = _stage(tmp_path / "r", {})
        scorer = _scorer(report_dir, dataset, normalizer_source="/local/whisper")
        assert scorer.normalizer_source == "/local/whisper"


@pytest.mark.unit
class TestNormalizerResolution:
    def test_tokenizer_without_spelling_map_is_rejected(
        self, tmp_path: Path, dataset, monkeypatch: pytest.MonkeyPatch
    ):
        tokenizer = MagicMock(spec=[])  # no english_spelling_normalizer attribute
        monkeypatch.setattr(
            scoring.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer
        )
        report_dir = _stage(tmp_path / "r", {0: REFERENCES[0]})
        with pytest.raises(ValueError, match="english_spelling_normalizer"):
            _scorer(report_dir, dataset)._normalizer()


@pytest.mark.unit
class TestPreflight:
    """The normalizer is resolved before the run, not after the perf phase."""

    def test_preflight_resolves_the_default_source(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        seen: list[str] = []

        def _from_pretrained(source, *a, **k):
            seen.append(source)
            return MagicMock(english_spelling_normalizer={"a": "b"})

        monkeypatch.setattr(scoring.AutoTokenizer, "from_pretrained", _from_pretrained)
        WERScorer.preflight({})
        assert seen == [WERScorer.DEFAULT_NORMALIZER_SOURCE]

    def test_preflight_honours_the_extras_override(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        seen: list[str] = []

        def _from_pretrained(source, *a, **k):
            seen.append(source)
            return MagicMock(english_spelling_normalizer={"a": "b"})

        monkeypatch.setattr(scoring.AutoTokenizer, "from_pretrained", _from_pretrained)
        WERScorer.preflight({"normalizer_source": "/local/whisper"})
        assert seen == ["/local/whisper"]

    def test_preflight_raises_on_an_unusable_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # The whole point: this surfaces before any load is issued.
        monkeypatch.setattr(
            scoring.AutoTokenizer,
            "from_pretrained",
            lambda *a, **k: MagicMock(spec=[]),
        )
        with pytest.raises(ValueError, match="english_spelling_normalizer"):
            WERScorer.preflight({})


@pytest.mark.unit
@pytest.mark.usefixtures("hermetic_normalizer")
class TestExtractorIsRejected:
    def test_configured_extractor_is_an_error(self, tmp_path: Path, dataset):
        # WER scores the transcript verbatim; an extractor would be ignored.
        report_dir = _stage(tmp_path / "r", {0: REFERENCES[0]})
        with pytest.raises(InputValidationError, match="extractor"):
            _scorer(report_dir, dataset, extractor=PythonCodeExtractor)

    def test_no_extractor_constructs(self, tmp_path: Path, dataset):
        report_dir = _stage(tmp_path / "r", {0: REFERENCES[0]})
        assert _scorer(report_dir, dataset).extractor is None
