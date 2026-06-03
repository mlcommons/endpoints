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

"""Unit tests for the MLPerf DeepSeek-R1 accuracy scorer (subprocess mocked)."""

from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import pandas as pd
import pytest
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.evaluation import scoring as scoring_mod
from inference_endpoint.evaluation.scoring import DeepSeekR1Scorer, Scorer


@pytest.mark.unit
class TestDeepSeekR1ScorerRegistration:
    def test_scorer_registered(self):
        assert "deepseek_r1" in Scorer.available_scorers()
        assert Scorer.get("deepseek_r1") is DeepSeekR1Scorer


@pytest.mark.unit
class TestDeepSeekR1Scorer:
    """DeepSeekR1Scorer unit tests with the eval subprocess monkey-patched."""

    # Three samples across three subsets.
    OUTPUTS = [
        r"reasoning... \boxed{8}",  # math500, correct
        "ANSWER: B",  # gpqa, correct
        r"\boxed{0}",  # aime, wrong
    ]
    GROUND_TRUTH = ["8", "B", "42"]
    SUBSETS = ["math500", "gpqa", "aime1983"]
    QUESTIONS = ["q0", "q1", "q2"]

    @pytest.fixture
    def dataset(self):
        df = pd.DataFrame(
            {
                "ground_truth": self.GROUND_TRUTH,
                "dataset": self.SUBSETS,
                "question": self.QUESTIONS,
            }
        )
        ds = MagicMock()
        ds.dataframe = df
        ds.num_samples.return_value = 3
        return ds

    @pytest.fixture
    def staged(self, tmp_path):
        """report_dir with sample_idx_map + events.jsonl for three COMPLETE samples."""
        report_dir = tmp_path / "report"
        report_dir.mkdir()

        uuids = [f"uuid-{i}" for i in range(3)]
        sample_idx_map = {"dsr1_acc": dict(zip(uuids, range(3), strict=True))}
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode(sample_idx_map)
        )

        encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        with (report_dir / "events.jsonl").open("wb") as f:
            for uid, out in zip(uuids, self.OUTPUTS, strict=True):
                rec = EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid=uid,
                    data=TextModelOutput(output=out),
                )
                f.write(encoder.encode(rec) + b"\n")
        return report_dir

    @pytest.fixture
    def project(self, tmp_path):
        """Stub accuracy subproject with a deepseek_eval_runner.py the scorer finds."""
        project = tmp_path / "accuracy"
        project.mkdir()
        (project / "deepseek_eval_runner.py").write_text("# stub\n")
        return project

    @pytest.fixture
    def patch_subprocess(self, monkeypatch):
        """Capture subprocess.run; read the input parquet, write an aggregate JSON.

        Mirrors the real runner: reads model_output/ground_truth/dataset, writes
        {exact_match, tokens_per_sample, num_samples, per_dataset} so the scorer
        parses a real file rather than a hand-faked one.
        """
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            in_parquet = Path(cmd[cmd.index("--input") + 1])
            out_json = Path(cmd[cmd.index("--output") + 1])
            df = pd.read_parquet(in_parquet)
            captured["input_df"] = df
            results = {
                "exact_match": 66.6667,
                "tokens_per_sample": 123.0,
                "num_samples": int(len(df)),
                "evaluated_samples": int(len(df)),
                "complete": True,
                "per_dataset": {},
            }
            out_json.write_bytes(msgspec.json.encode(results))
            return MagicMock(returncode=0, stdout="ok\n")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        return captured

    def test_score_returns_exact_match(
        self, dataset, staged, project, patch_subprocess
    ):
        scorer = DeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
        )
        score, n_repeats = scorer.score()

        assert score == pytest.approx(66.6667)
        assert n_repeats == 1

        # Invoked via `uv run --project <subproject> python deepseek_eval_runner.py`.
        cmd = patch_subprocess["cmd"]
        assert cmd[0] == "uv"
        assert cmd[1:3] == ["run", "--project"]
        assert Path(cmd[3]) == project
        assert Path(cmd[5]) == project / "deepseek_eval_runner.py"

    def test_eval_dataframe_columns_and_mapping(
        self, dataset, staged, project, patch_subprocess
    ):
        """The parquet handed to the subprocess has the evaluator's columns,
        with model_output (from events) joined to the correct dataset row."""
        scorer = DeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
        )
        scorer.score()

        df = patch_subprocess["input_df"]
        assert set(df.columns) == {
            "model_output",
            "ground_truth",
            "dataset",
            "question",
        }
        # Row order follows sample_index 0,1,2 -> outputs aligned to subsets.
        assert list(df["model_output"]) == self.OUTPUTS
        assert list(df["ground_truth"]) == self.GROUND_TRUTH
        assert list(df["dataset"]) == self.SUBSETS

    def test_missing_runner_raises(self, dataset, staged, tmp_path):
        empty_project = tmp_path / "empty"
        empty_project.mkdir()
        with pytest.raises(FileNotFoundError, match="deepseek_eval_runner.py"):
            DeepSeekR1Scorer(
                dataset_name="dsr1_acc",
                dataset=dataset,
                report_dir=staged,
                deepseek_eval_project_path=empty_project,
            )

    def test_none_score_when_no_exact_match(
        self, dataset, staged, project, monkeypatch
    ):
        """Subprocess yields no exact_match -> scorer returns (None, n_repeats)."""

        def fake_run(cmd, **kwargs):
            out_json = Path(cmd[cmd.index("--output") + 1])
            out_json.write_bytes(msgspec.json.encode({"exact_match": None}))
            return MagicMock(returncode=0, stdout="ok\n")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        scorer = DeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
        )
        score, n_repeats = scorer.score()
        assert score is None
        assert n_repeats == 1
