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

"""Tests for the combined gpt-oss accuracy orchestrator.

Members are graded by stand-in scorers so the test isolates the roll-up:
sample-weighting by unique problem count, the [0,1] -> [0,100] breakdown scale,
and completeness propagation.
"""

from __future__ import annotations

import pytest
from inference_endpoint.evaluation.scoring import GptOssAccuracyScorer, GptOssMember


class _FakeDataset:
    def __init__(self, n: int, score: float | None, complete: bool = True):
        self._n = n
        self.score_value = score
        self.complete_value = complete

    def num_samples(self) -> int:
        return self._n


class _FakeScorer:
    """Duck-typed member scorer returning its dataset's preset score."""

    def __init__(
        self,
        name,
        dataset,
        report_dir,
        extractor=None,
        ground_truth_column=None,
        **extras,
    ):
        self._dataset = dataset
        self.complete = dataset.complete_value

    def score(self):
        return self._dataset.score_value, 1


def _member(subset: str, n: int, score: float | None, complete: bool = True):
    return GptOssMember(
        full_name=f"{subset}::gptoss",
        subset=subset,
        scorer_cls=_FakeScorer,  # type: ignore[arg-type]  # duck-typed stand-in
        extractor=None,
        dataset=_FakeDataset(n, score, complete),  # type: ignore[arg-type]
        ground_truth_column=None,
        extras={},
    )


@pytest.mark.unit
class TestGptOssAccuracyScorer:
    def test_weighted_by_unique_counts(self, tmp_path):
        members = [
            _member("aime25", 30, 0.8),
            _member("gpqa", 198, 0.9),
            _member("livecodebench", 1055, 0.6),
        ]
        scorer = GptOssAccuracyScorer(members, tmp_path)
        overall01, n_repeats = scorer.score()

        expected = (0.8 * 30 + 0.9 * 198 + 0.6 * 1055) / (30 + 198 + 1055)
        assert overall01 == pytest.approx(expected)
        assert n_repeats == 1
        assert scorer.complete is True

        bd = scorer.score_breakdown()
        assert bd["overall_accuracy"] == pytest.approx(expected * 100, abs=0.01)
        assert bd["subset_scores"] == {
            "aime25": 80.0,
            "gpqa": 90.0,
            "livecodebench": 60.0,
        }
        assert bd["total_samples"] == 1283
        assert bd["complete"] is True
        # Per-member outcomes exposed for the caller's per-subset audit entries.
        assert scorer.member_scores == {
            "aime25::gptoss": (0.8, True),
            "gpqa::gptoss": (0.9, True),
            "livecodebench::gptoss": (0.6, True),
        }

    def test_failed_subset_marks_incomplete_and_excludes(self, tmp_path):
        members = [
            _member("aime25", 30, 0.8),
            _member("livecodebench", 1055, None),
        ]
        scorer = GptOssAccuracyScorer(members, tmp_path)
        overall01, _ = scorer.score()

        # livecodebench failed -> excluded from both weight and subset_scores.
        assert overall01 == pytest.approx(0.8)
        assert scorer.complete is False
        bd = scorer.score_breakdown()
        assert bd["subset_scores"] == {"aime25": 80.0}
        assert bd["total_samples"] == 30
        assert bd["complete"] is False

    def test_partial_member_scorer_marks_incomplete(self, tmp_path):
        # A member that scored but flagged itself partial shrinks completeness.
        members = [
            _member("aime25", 30, 0.8, complete=False),
            _member("gpqa", 198, 0.9),
        ]
        scorer = GptOssAccuracyScorer(members, tmp_path)
        scorer.score()
        assert scorer.complete is False

    def test_all_failed_returns_none(self, tmp_path):
        scorer = GptOssAccuracyScorer([_member("aime25", 30, None)], tmp_path)
        score, _ = scorer.score()
        assert score is None
        assert scorer.complete is False
        assert scorer.score_breakdown() is None

    def test_rejects_out_of_range_member(self, tmp_path):
        # A 0-100 scale score (e.g. a mis-grouped DeepSeek member) is rejected
        # rather than silently scaled to ~10000%.
        scorer = GptOssAccuracyScorer([_member("x", 10, 81.36)], tmp_path)
        with pytest.raises(ValueError, match=r"scalar scores in \[0, 1\]"):
            scorer.score()

    def test_rejects_non_numeric_member(self, tmp_path):
        # A dict-returning member (e.g. RougeScorer) is rejected, not float(dict).
        scorer = GptOssAccuracyScorer(
            [_member("x", 10, {"rouge": 0.5})],  # type: ignore[arg-type]
            tmp_path,
        )
        with pytest.raises(ValueError):
            scorer.score()
