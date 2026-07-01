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
from inference_endpoint.evaluation.scoring import LegacyMLPerfDeepSeekR1Scorer, Scorer


@pytest.mark.unit
class TestLegacyMLPerfDeepSeekR1ScorerRegistration:
    def test_scorer_registered(self):
        assert "legacy_mlperf_deepseek_r1" in Scorer.available_scorers()
        assert Scorer.get("legacy_mlperf_deepseek_r1") is LegacyMLPerfDeepSeekR1Scorer


@pytest.mark.unit
class TestLegacyMLPerfDeepSeekR1Scorer:
    """LegacyMLPerfDeepSeekR1Scorer unit tests with the eval subprocess monkey-patched."""

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
        scorer = LegacyMLPerfDeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
        )
        score, n_repeats = scorer.score()

        assert score == pytest.approx(66.6667)
        assert n_repeats == 1
        assert scorer.complete is True

        # Invoked via `uv run --project <subproject> python deepseek_eval_runner.py`.
        cmd = patch_subprocess["cmd"]
        assert cmd[0] == "uv"
        assert cmd[1:3] == ["run", "--project"]
        assert Path(cmd[3]) == project
        assert Path(cmd[5]) == project / "deepseek_eval_runner.py"

    def test_subprocess_pins_subproject_venv(
        self, dataset, staged, project, patch_subprocess
    ):
        """The eval subprocess pins UV_PROJECT_ENVIRONMENT to the subproject's own
        .venv, so an inherited value (e.g. /opt/venv in the dev image) can't
        redirect uv to the parent environment."""
        scorer = LegacyMLPerfDeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
        )
        scorer.score()
        env = patch_subprocess["kwargs"]["env"]
        assert env["UV_PROJECT_ENVIRONMENT"] == str(project / ".venv")

    def test_eval_dataframe_columns_and_mapping(
        self, dataset, staged, project, patch_subprocess
    ):
        """The parquet handed to the subprocess has the evaluator's columns,
        with model_output (from events) joined to the correct dataset row."""
        scorer = LegacyMLPerfDeepSeekR1Scorer(
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
            LegacyMLPerfDeepSeekR1Scorer(
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
        scorer = LegacyMLPerfDeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
        )
        score, n_repeats = scorer.score()
        assert score is None
        assert n_repeats == 1
        assert scorer.complete is False


@pytest.mark.unit
class TestLegacyMLPerfDeepSeekR1ScorerContainer:
    """Container path: text subsets graded by the subprocess, livecodebench graded
    via the lcb-service WebSocket, merged into one 5-subset number."""

    OUTPUTS = [
        r"reasoning \boxed{8}",  # math500
        "```python\nprint(1)\n```",  # livecodebench
    ]
    GROUND_TRUTH = ["8", "lcb-q0"]
    SUBSETS = ["math500", "livecodebench"]

    @pytest.fixture
    def dataset(self):
        df = pd.DataFrame(
            {
                "ground_truth": self.GROUND_TRUTH,
                "dataset": self.SUBSETS,
                "question": ["q0", "q1"],
            }
        )
        ds = MagicMock()
        ds.dataframe = df
        ds.num_samples.return_value = 2
        return ds

    @pytest.fixture
    def staged(self, tmp_path):
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        uuids = [f"uuid-{i}" for i in range(2)]
        (report_dir / "sample_idx_map.json").write_bytes(
            msgspec.json.encode({"dsr1_acc": dict(zip(uuids, range(2), strict=True))})
        )
        enc = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
        with (report_dir / "events.jsonl").open("wb") as f:
            for uid, out in zip(uuids, self.OUTPUTS, strict=True):
                rec = EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    sample_uuid=uid,
                    data=TextModelOutput(output=out),
                )
                f.write(enc.encode(rec) + b"\n")
        return report_dir

    @pytest.fixture
    def project(self, tmp_path):
        project = tmp_path / "accuracy"
        project.mkdir()
        (project / "deepseek_eval_runner.py").write_text("# stub\n")
        return project

    @pytest.fixture
    def patch_subprocess(self, monkeypatch):
        """Emulate the runner: grade text subsets, mark --external-subsets ones
        external (exact_match=None). Records call count + external args."""
        calls: dict[str, object] = {"n": 0, "external": []}

        def fake_run(cmd, **kwargs):
            calls["n"] = int(calls["n"]) + 1  # type: ignore[arg-type]
            ext = (
                cmd[cmd.index("--external-subsets") + 1].split(",")
                if "--external-subsets" in cmd
                else []
            )
            calls["external"].append(ext)  # type: ignore[attr-defined]
            df = pd.read_parquet(Path(cmd[cmd.index("--input") + 1]))
            out_json = Path(cmd[cmd.index("--output") + 1])
            per: dict[str, dict] = {}
            for sub, g in df.groupby("dataset"):
                if str(sub) in ext:
                    per[str(sub)] = {
                        "exact_match": None,
                        "tokens_per_sample": 200.0,
                        "num_samples": int(len(g)),
                        "status": "external",
                    }
                else:
                    per[str(sub)] = {
                        "exact_match": 100.0,
                        "tokens_per_sample": 50.0,
                        "num_samples": int(len(g)),
                        "status": "ok",
                    }
            out_json.write_bytes(
                msgspec.json.encode(
                    {
                        "exact_match": 100.0,
                        "tokens_per_sample": 100.0,
                        "num_samples": int(len(df)),
                        "evaluated_samples": int(len(df)),
                        "complete": True,
                        "per_dataset": per,
                    }
                )
            )
            return MagicMock(returncode=0, stdout="ok\n")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        return calls

    def _scorer(self, dataset, staged, project):
        return LegacyMLPerfDeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
            lcb_websocket_port=13835,
        )

    def _results(self, staged):
        return msgspec.json.decode(
            (staged / "deepseek_eval" / "dsr1_acc_results.json").read_bytes()
        )

    def test_container_merges_lcb_in_run(
        self, dataset, staged, project, patch_subprocess, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod,
            "_lcb_ws_evaluate",
            lambda url, codes, timeout: {
                "total_samples": 1,
                "results": {"lcb-q0": [True]},
            },
        )
        scorer = self._scorer(dataset, staged, project)
        score, n_repeats = scorer.score()

        assert score == pytest.approx(100.0)  # (1 text + 1 LCB) / 2
        assert n_repeats == 1
        assert scorer.complete is True
        # Subprocess ran exactly once, with livecodebench marked external.
        assert patch_subprocess["n"] == 1
        assert patch_subprocess["external"][0] == ["livecodebench"]
        res = self._results(staged)
        assert res["complete"] is True
        assert res["evaluated_samples"] == 2
        lcb = res["per_dataset"]["livecodebench"]
        assert lcb["status"] == "lcb-service"
        assert lcb["exact_match"] == pytest.approx(100.0)
        assert lcb["tokens_per_sample"] == 200.0  # preserved from the runner

    def test_container_unreachable_leaves_lcb_unscored(
        self, dataset, staged, project, patch_subprocess, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod, "_lcb_ws_evaluate", lambda url, codes, timeout: None
        )
        scorer = self._scorer(dataset, staged, project)
        scorer.score()

        # No in-process LCB re-grade: subprocess still ran exactly once.
        assert patch_subprocess["n"] == 1
        # The partial result is visible to callers via the scorer attribute,
        # not only the sidecar JSON (execute.py stores this in accuracy_scores).
        assert scorer.complete is False
        res = self._results(staged)
        assert res["complete"] is False
        assert res["per_dataset"]["livecodebench"]["status"] == "unscored"

    def test_lcb_count_divergence_marks_incomplete(
        self, dataset, staged, project, patch_subprocess, monkeypatch
    ):
        """Container reports more LCB samples than were issued -> total_n
        (text 1 + lcb 2) != expected_n (2): a number is still produced but the
        run is flagged incomplete (the headline-validity guard)."""
        monkeypatch.setattr(
            scoring_mod,
            "_lcb_ws_evaluate",
            lambda url, codes, timeout: {
                "total_samples": 2,
                "results": {"lcb-q0": [True, True]},
            },
        )
        scorer = self._scorer(dataset, staged, project)
        score, _ = scorer.score()
        assert score is not None
        assert scorer.complete is False
        assert self._results(staged)["complete"] is False

    def test_subprocess_nonzero_exit_raises(
        self, dataset, staged, project, monkeypatch
    ):
        """A non-zero eval-subprocess exit surfaces as RuntimeError, not a
        silent partial score."""
        monkeypatch.setattr(
            scoring_mod.subprocess,
            "run",
            lambda cmd, **kw: MagicMock(returncode=1, stdout="boom\ntraceback\n"),
        )
        scorer = self._scorer(dataset, staged, project)
        with pytest.raises(RuntimeError, match="exited with code 1"):
            scorer.score()

    def test_subprocess_timeout_raises(self, dataset, staged, project, monkeypatch):
        """A timed-out eval subprocess surfaces as RuntimeError."""

        def _timeout(cmd, **kw):
            raise scoring_mod.subprocess.TimeoutExpired(cmd, 1, output="partial\n")

        monkeypatch.setattr(scoring_mod.subprocess, "run", _timeout)
        scorer = self._scorer(dataset, staged, project)
        with pytest.raises(RuntimeError, match="timed out after"):
            scorer.score()

    def test_no_container_with_lcb_rows_refuses(
        self, dataset, staged, project, patch_subprocess, monkeypatch
    ):
        """With livecodebench rows but no container, refuse to run untrusted code
        in-process unless ALLOW_LCB_LOCAL is set."""
        monkeypatch.delenv("ALLOW_LCB_LOCAL", raising=False)
        scorer = LegacyMLPerfDeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
            lcb_websocket_port=None,
        )
        with pytest.raises(RuntimeError, match="ALLOW_LCB_LOCAL"):
            scorer.score()

    def test_allow_lcb_local_opt_in_grades_in_process(
        self, dataset, staged, project, patch_subprocess, monkeypatch
    ):
        """ALLOW_LCB_LOCAL=1 opts into in-process LCB grading (no container)."""
        monkeypatch.setenv("ALLOW_LCB_LOCAL", "1")
        scorer = LegacyMLPerfDeepSeekR1Scorer(
            dataset_name="dsr1_acc",
            dataset=dataset,
            report_dir=staged,
            deepseek_eval_project_path=project,
            lcb_websocket_port=None,
        )
        score, _ = scorer.score()
        assert score is not None
        # Graded all subsets in-process: no subset marked --external-subsets.
        assert patch_subprocess["external"][0] == []

    def test_failed_text_subset_marks_incomplete(
        self, dataset, staged, project, monkeypatch
    ):
        """A text subset that fails to grade must NOT be reported complete."""

        def fake_run(cmd, **kwargs):
            df = pd.read_parquet(Path(cmd[cmd.index("--input") + 1]))
            out_json = Path(cmd[cmd.index("--output") + 1])
            ext = (
                cmd[cmd.index("--external-subsets") + 1].split(",")
                if "--external-subsets" in cmd
                else []
            )
            per: dict[str, dict] = {}
            for sub, g in df.groupby("dataset"):
                status = "external" if str(sub) in ext else "failed: boom"
                per[str(sub)] = {
                    "exact_match": None,
                    "tokens_per_sample": 50.0,
                    "num_samples": int(len(g)),
                    "status": status,
                }
            out_json.write_bytes(
                msgspec.json.encode(
                    {
                        "exact_match": None,
                        "tokens_per_sample": 100.0,
                        "num_samples": int(len(df)),
                        "evaluated_samples": 0,
                        "complete": False,
                        "per_dataset": per,
                    }
                )
            )
            return MagicMock(returncode=0, stdout="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(
            scoring_mod,
            "_lcb_ws_evaluate",
            lambda url, codes, timeout: {
                "total_samples": 1,
                "results": {"lcb-q0": [True]},
            },
        )
        scorer = self._scorer(dataset, staged, project)
        scorer.score()
        # math500 failed -> must be flagged incomplete (was forced True pre-fix).
        assert self._results(staged)["complete"] is False
        assert scorer.complete is False
