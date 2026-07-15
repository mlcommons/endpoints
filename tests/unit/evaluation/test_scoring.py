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

    def test_stage_videos_renames_non_mp4_sources_to_mp4(
        self, dataset, staged, vbench_project, tmp_path
    ):
        """Non-mp4 sources (e.g. .avi) stage under .mp4 names for VBench."""
        report_dir, _ = staged
        avi = tmp_path / "video_x.avi"
        avi.write_bytes(b"")
        scorer = VBenchScorer(
            dataset_name="vid_acc",
            dataset=dataset,
            report_dir=report_dir,
            ground_truth_column="prompt",
            vbench_project_path=vbench_project,
        )
        staged_dir = report_dir / "vbench_videos"
        scorer._stage_videos(staged_dir, [str(avi)], ["a cat"])
        (staged_file,) = staged_dir.iterdir()
        assert staged_file.name == "a cat-0.mp4"
        assert staged_file.resolve() == avi.resolve()

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

    def test_wanted_uuids_bounds_the_frame(self, tmp_path):
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
        scorer = self._scorer(tmp_path)
        # Unbounded read returns every COMPLETE row.
        assert set(scorer.get_raw_outputs()["sample_uuid"]) == {"u1", "u2", "u3"}
        # wanted_uuids filters the frame to the requested population only.
        bounded = scorer.get_raw_outputs({"u1", "u3"})
        assert set(bounded["sample_uuid"]) == {"u1", "u3"}
        assert set(bounded["output"]) == {"a b", "d e f"}

    def test_empty_filter_returns_columned_frame(self, tmp_path):
        # Filtering out every uuid must still yield a frame WITH the expected
        # columns (not a column-less pd.DataFrame([])), so callers can index
        # "sample_uuid"/"output" without a KeyError.
        self._write_events(
            tmp_path,
            [
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid="u1",
                    data=TextModelOutput(output="a b"),
                ),
            ],
        )
        frame = self._scorer(tmp_path).get_raw_outputs({"absent"})
        assert frame.empty
        assert list(frame.columns) == ["sample_uuid", "output"]


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
