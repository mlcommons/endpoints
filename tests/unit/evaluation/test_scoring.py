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
from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import pandas as pd
import pytest
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.dataset_manager.predefined.shopify_product_catalogue import (
    ProductMetadata,
)
from inference_endpoint.evaluation import scoring as scoring_mod
from inference_endpoint.evaluation.scoring import (
    _PRED_CATEGORY_PAD,
    Scorer,
    ShopifyCategoryF1Scorer,
    VBenchScorer,
    _calculate_hierarchical_f1,
    _create_pred_pad_category,
    _match_hierarchical_paths,
    _parse_response_to_category,
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
        captured = {}

        def fake_run(cmd, check):
            captured["cmd"] = cmd
            captured["check"] = check
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            name = cmd[cmd.index("--name") + 1]
            dims = cmd[cmd.index("--dims") + 1].split(",")
            results = {dim: [TestVBenchScorer.DIM_SCORES[dim], []] for dim in dims}
            (out_dir / f"{name}_eval_results.json").write_bytes(
                msgspec.json.encode(results)
            )
            return MagicMock(returncode=0)

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
