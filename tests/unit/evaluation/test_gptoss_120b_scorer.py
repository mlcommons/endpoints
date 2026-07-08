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

"""Unit tests for the composite gpt-oss-120b accuracy scorer.

The three subsets route in-process (aime25/gpqa via extractor + exact match) or
out-of-band (livecodebench via the lcb-service WebSocket, monkeypatched). Tests
isolate the per-subset routing, the unique-count weighting, the top-level 0-100
scale, and completeness propagation.
"""

from unittest.mock import MagicMock

import msgspec
import pandas as pd
import pytest
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.evaluation import scoring as scoring_mod
from inference_endpoint.evaluation.scoring import GptOss120bAccuracyScorer, Scorer

DATASET_NAME = "gptoss_120b_accuracy"

# Five unique problems across three subsets: aime25 (2, both correct -> 1.0),
# gpqa (2, both wrong -> 0.0), livecodebench (1, graded by the mocked container).
SUBSETS = ["aime25", "aime25", "gpqa", "gpqa", "livecodebench"]
GROUND_TRUTH = ["8", "42", "choice2", "choice1", "lcbq0"]
OUTPUTS = [
    r"reasoning... \boxed{8}",  # aime25 correct
    r"\boxed{42}",  # aime25 correct
    "Answer: C",  # gpqa -> choice3, gt choice2 -> wrong
    "Answer: D",  # gpqa -> choice4, gt choice1 -> wrong
    "```python\ndef solve():\n    pass\n```",  # livecodebench (graded by mock)
]
QUESTIONS = [f"q{i}" for i in range(5)]


def _stage(report_dir, uuids, indices, outputs):
    """Write sample_idx_map.json + events.jsonl for the given issued samples."""
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_idx_map = {DATASET_NAME: dict(zip(uuids, indices, strict=True))}
    (report_dir / "sample_idx_map.json").write_bytes(
        msgspec.json.encode(sample_idx_map)
    )
    encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
    with (report_dir / "events.jsonl").open("wb") as f:
        for uid, out in zip(uuids, outputs, strict=True):
            rec = EventRecord(
                event_type=SampleEventType.COMPLETE,
                sample_uuid=uid,
                data=TextModelOutput(output=out),
            )
            f.write(encoder.encode(rec) + b"\n")


@pytest.fixture
def dataset():
    df = pd.DataFrame(
        {
            "prompt": [f"prompt-{i}" for i in range(5)],
            "subset": SUBSETS,
            "ground_truth": GROUND_TRUTH,
            "question": QUESTIONS,
        }
    )
    ds = MagicMock()
    ds.dataframe = df
    ds.num_samples.return_value = 5
    return ds


@pytest.fixture
def staged(tmp_path):
    """One COMPLETE event per unique sample (n_repeats == 1)."""
    report_dir = tmp_path / "report"
    uuids = [f"uuid-{i}" for i in range(5)]
    _stage(report_dir, uuids, list(range(5)), OUTPUTS)
    return report_dir


@pytest.fixture
def lcb_pass(monkeypatch):
    """lcb-service that passes the single livecodebench problem (1/1 -> 1.0)."""
    monkeypatch.setattr(
        scoring_mod,
        "_lcb_ws_evaluate",
        lambda url, codes, timeout: {"total_samples": 1, "results": {"lcbq0": [True]}},
    )


@pytest.mark.unit
class TestRegistration:
    def test_registered(self):
        assert "gptoss_120b_accuracy" in Scorer.available_scorers()
        assert Scorer.get("gptoss_120b_accuracy") is GptOss120bAccuracyScorer

    def test_requires_no_extractor(self):
        assert GptOss120bAccuracyScorer.REQUIRES_EXTRACTOR is False


@pytest.mark.unit
class TestScore:
    def test_routing_weighting_and_scale(self, dataset, staged, lcb_pass):
        scorer = GptOss120bAccuracyScorer(DATASET_NAME, dataset, staged)
        score, n_repeats = scorer.score()

        # Unique-count weighted: (1.0*2 + 0.0*2 + 1.0*1) / 5 = 0.6 -> 60.0 (0-100).
        assert score == pytest.approx(60.0)
        assert n_repeats == 1
        assert scorer.complete is True

    def test_breakdown_per_subset(self, dataset, staged, lcb_pass):
        scorer = GptOss120bAccuracyScorer(DATASET_NAME, dataset, staged)
        assert scorer.score_breakdown() is None  # not populated until score()
        scorer.score()

        bd = scorer.score_breakdown()
        assert bd is not None
        assert bd["overall_accuracy"] == pytest.approx(60.0)
        # Breakdown values are on the shared 0-100 scale.
        assert bd["subset_scores"] == {
            "aime25": 100.0,
            "gpqa": 0.0,
            "livecodebench": 100.0,
        }
        # total_samples is the summed unique problem count (2 + 2 + 1).
        assert bd["total_samples"] == 5
        assert bd["complete"] is True
        assert bd["per_subset_status"] == {
            "aime25": "in-process",
            "gpqa": "in-process",
            "livecodebench": "lcb-service",
        }

    def test_lcb_service_down_excludes_and_marks_incomplete(
        self, dataset, staged, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod, "_lcb_ws_evaluate", lambda url, codes, timeout: None
        )
        scorer = GptOss120bAccuracyScorer(DATASET_NAME, dataset, staged)
        score, _ = scorer.score()

        # livecodebench dropped from weight + subset_scores; aime/gpqa still count.
        # (1.0*2 + 0.0*2) / 4 = 0.5 -> 50.0.
        assert score == pytest.approx(50.0)
        assert scorer.complete is False
        bd = scorer.score_breakdown()
        assert "livecodebench" not in bd["subset_scores"]
        assert bd["per_subset_status"]["livecodebench"] == "unscored"
        assert bd["total_samples"] == 4
        assert bd["complete"] is False

    def test_lcb_port_none_skips_ws_call(self, dataset, staged, monkeypatch):
        def _boom(url, codes, timeout):
            raise AssertionError(
                "_lcb_ws_evaluate must not be called when port is None"
            )

        monkeypatch.setattr(scoring_mod, "_lcb_ws_evaluate", _boom)
        scorer = GptOss120bAccuracyScorer(
            DATASET_NAME, dataset, staged, lcb_websocket_port=None
        )
        score, _ = scorer.score()

        assert score == pytest.approx(50.0)  # only aime + gpqa
        assert scorer.complete is False
        assert scorer.score_breakdown()["per_subset_status"]["livecodebench"] == (
            "unscored"
        )

    def test_n_repeats_counts_duplicate_issues(self, tmp_path, dataset, lcb_pass):
        # Two issued copies of every sample -> n_repeats == 2.
        report_dir = tmp_path / "report2"
        uuids = [f"uuid-{i}" for i in range(10)]
        indices = list(range(5)) + list(range(5))
        _stage(report_dir, uuids, indices, OUTPUTS + OUTPUTS)

        scorer = GptOss120bAccuracyScorer(DATASET_NAME, dataset, report_dir)
        _, n_repeats = scorer.score()
        assert n_repeats == 2

    def test_empty_outputs_returns_none(self, dataset, tmp_path):
        report_dir = tmp_path / "empty"
        _stage(report_dir, [], [], [])
        scorer = GptOss120bAccuracyScorer(DATASET_NAME, dataset, report_dir)
        score, _ = scorer.score()
        assert score is None
        assert scorer.complete is False
        assert scorer.score_breakdown() is None

    def test_score_single_sample_raises(self, dataset, staged):
        scorer = GptOss120bAccuracyScorer(DATASET_NAME, dataset, staged)
        with pytest.raises(RuntimeError, match="per-subset routing"):
            scorer.score_single_sample("x", "y")
