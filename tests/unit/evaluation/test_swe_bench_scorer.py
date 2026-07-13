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

"""Unit tests for SWEBenchScorer."""

import concurrent.futures
import json
from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import pandas as pd
import pytest
import yaml
from inference_endpoint.evaluation import scoring as scoring_mod
from inference_endpoint.evaluation.scoring import (
    Scorer,
    SWEBenchScorer,
)
from inference_endpoint.exceptions import SetupError

pytestmark = pytest.mark.unit

_DATASET_NAME = "swe_bench_acc"
_MODEL_NAME = "TestOrg/test-model-7b"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_benchmark_config(report_dir: Path, model_params: dict | None = None) -> None:
    mp: dict = {"name": _MODEL_NAME}
    defaults = {
        "model_params": mp,
        "endpoint_config": {"endpoints": ["http://localhost:30000"]},
    }
    if model_params is not None:
        mp.update(model_params)
    (report_dir / "config.yaml").write_text(yaml.dump(defaults))


def _write_sample_idx_map(report_dir: Path, n: int = 3) -> None:
    idx_map = {_DATASET_NAME: {f"uuid-{i}": i for i in range(n)}}
    (report_dir / "sample_idx_map.json").write_bytes(msgspec.json.encode(idx_map))


def _make_dataset(n: int = 3) -> MagicMock:
    df = pd.DataFrame(
        {
            "instance_id": [f"repo__repo-{i}" for i in range(n)],
            "prompt": ["placeholder"] * n,
        }
    )
    ds = MagicMock()
    ds.dataframe = df
    ds.num_samples.return_value = n
    return ds


def _write_toolcall_patch_files(swe_bench_project: Path) -> None:
    for filename in SWEBenchScorer.TOOLCALL_PATCH_REPLACEMENTS:
        (swe_bench_project / filename).write_text(f"# patched {filename}\n")


def _make_scorer(
    report_dir: Path,
    swe_bench_project: Path,
    template_yaml: Path,
    *,
    dataset=None,
    **kwargs,
) -> SWEBenchScorer:
    return SWEBenchScorer(
        dataset_name=_DATASET_NAME,
        dataset=dataset or _make_dataset(),
        report_dir=report_dir,
        swe_bench_project_path=swe_bench_project,
        swebench_config_template=template_yaml,
        **kwargs,
    )


def _patch_config_and_read(
    scorer: SWEBenchScorer,
    output_dir: Path,
    benchmark_cfg: dict,
) -> dict:
    output_dir.mkdir()
    patched_path = scorer._patch_config(output_dir, benchmark_cfg)
    return yaml.safe_load(patched_path.read_text())


@pytest.fixture
def swe_bench_project(tmp_path: Path) -> Path:
    """Fake accuracy subproject directory with a minimal pyproject.toml."""
    d = tmp_path / "accuracy"
    d.mkdir(parents=True)
    (d / "pyproject.toml").write_text("[project]\nname = 'swe-bench-accuracy'\n")
    return d


@pytest.fixture
def template_yaml(tmp_path: Path) -> Path:
    """Minimal swebench template YAML."""
    tmpl = {
        "model": {
            "model_name": "",
            "model_kwargs": {
                "custom_llm_provider": "openai",
                "api_base": "",
            },
        }
    }
    p = tmp_path / "swebench_template.yaml"
    p.write_text(yaml.dump(tmpl))
    return p


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    d = tmp_path / "report"
    d.mkdir()
    _write_benchmark_config(d)
    _write_sample_idx_map(d)
    return d


def _make_fake_run(cmd, **kwargs):
    """Return a fake subprocess.run result with returncode=0."""
    return MagicMock(returncode=0, stdout="")


def _make_staged_run(on_eval_cmd):
    """Return a fake subprocess.run that handles mini-extra successfully, then delegates."""

    def fake_run(cmd, **kwargs):
        if "mini-extra" in " ".join(cmd):
            output_dir = Path(cmd[cmd.index("--output") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "preds.json").write_text(json.dumps({}))
            return MagicMock(returncode=0, stdout="")
        return on_eval_cmd(cmd, **kwargs)

    return fake_run


class _CapturedSubprocessRuns(list):
    def __init__(self):
        super().__init__()
        self.kwargs = []


class _FakeTqdm:
    instances: list["_FakeTqdm"] = []

    def __init__(self, *, total: int, desc: str, unit: str):
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def update(self, n: int) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.instances = []


class _FakeFuture:
    def __init__(self, fn, image: str):
        self._fn = fn
        self.image = image
        self.started = False
        self.cancelled = False
        self.done = False
        self.result_value = None
        self.error: BaseException | None = None

    def run(self) -> None:
        if self.done or self.cancelled:
            return
        self.started = True
        try:
            self.result_value = self._fn(self.image)
        except BaseException as exc:  # pragma: no cover - exercised via result()
            self.error = exc
        self.done = True

    def result(self):
        if not self.done:
            self.run()
        if self.cancelled:
            raise concurrent.futures.CancelledError()
        if self.error is not None:
            raise self.error
        return self.result_value

    def cancel(self) -> bool:
        if self.started:
            return False
        self.cancelled = True
        return True


class _FakeThreadPoolExecutor:
    instances: list["_FakeThreadPoolExecutor"] = []
    completion_order: list[str] | None = None

    def __init__(self, *, max_workers: int):
        self.max_workers = max_workers
        self.submitted: list[_FakeFuture] = []
        self.shutdown_wait: bool | None = None
        self.shutdown_cancel_futures: bool | None = None
        type(self).instances.append(self)

    def submit(self, fn, image: str) -> _FakeFuture:
        future = _FakeFuture(fn, image)
        self.submitted.append(future)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_wait = wait
        self.shutdown_cancel_futures = cancel_futures
        if cancel_futures:
            for future in self.submitted:
                future.cancel()

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.completion_order = None


def _fake_as_completed(futures):
    executor = _FakeThreadPoolExecutor.instances[-1]
    future_by_image = {future.image: future for future in executor.submitted}
    order = _FakeThreadPoolExecutor.completion_order or [
        future.image for future in executor.submitted
    ]
    pending = executor.submitted[executor.max_workers :]
    for future in executor.submitted[: executor.max_workers]:
        future.started = True
    for image in order:
        future = future_by_image[image]
        if future.cancelled or not future.started:
            continue
        future.run()
        yield future
        if pending:
            pending.pop(0).started = True


@pytest.fixture
def patch_subprocess(monkeypatch, report_dir: Path, swe_bench_project: Path):
    """Patch subprocess.run to write fake preds.json and result JSON."""
    captured = _CapturedSubprocessRuns()

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        captured.kwargs.append(kwargs)
        cmd_str = " ".join(cmd)
        if "mini-extra" in cmd_str:
            output_dir = Path(cmd[cmd.index("--output") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "preds.json").write_text(json.dumps({}))
        elif "run_evaluation" in cmd_str:
            cwd = Path(kwargs["cwd"])
            run_id = cmd[cmd.index("--run_id") + 1]
            safe_model = _MODEL_NAME.replace("/", "__")
            (cwd / f"{safe_model}.{run_id}.json").write_text(
                json.dumps(
                    {
                        "resolved_instances": 1,
                        "submitted_instances": 3,
                        "total_instances": 500,
                    }
                )
            )
        return MagicMock(returncode=0, stdout="")

    monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
    return captured


class TestSWEBenchScorerRegistration:
    def test_registered(self):
        assert "swe_bench_scorer" in Scorer.PREDEFINED
        assert Scorer.get("swe_bench_scorer") is SWEBenchScorer

    def test_skip_endpoint_phase(self):
        assert SWEBenchScorer.SKIP_ENDPOINT_PHASE is True

    def test_external_sample_count(self):
        assert SWEBenchScorer.external_sample_count({"num_instances": 50}) == 50
        assert (
            SWEBenchScorer.external_sample_count({})
            == SWEBenchScorer.DEFAULT_NUM_INSTANCES
        )
        assert SWEBenchScorer.external_sample_count({"num_instances": "bad"}) is None
        assert SWEBenchScorer.external_sample_count({"num_instances": 0}) is None
        assert SWEBenchScorer.external_sample_count({"num_instances": -1}) is None


class TestSWEBenchScorer:
    def test_score_happy_path(
        self, report_dir, swe_bench_project, template_yaml, patch_subprocess
    ):
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        score, n_repeats = scorer.score()

        assert score == pytest.approx(1 / 3)
        assert n_repeats == 1
        assert (report_dir / "swe_bench_results.json").exists()
        assert {
            kwargs["env"]["UV_PROJECT_ENVIRONMENT"]
            for kwargs in patch_subprocess.kwargs
        } == {str(swe_bench_project / ".venv")}
        assert {
            cmd[cmd.index("--project") + 1]
            for cmd in patch_subprocess
            if cmd[:3] == ["uv", "run", "--project"]
        } == {str(swe_bench_project.resolve())}
        eval_cmd = patch_subprocess[1]
        instance_idx = eval_cmd.index("--instance_ids") + 1
        assert eval_cmd[instance_idx:] == [
            "repo__repo-0",
            "repo__repo-1",
            "repo__repo-2",
        ]
        assert not (report_dir / "swe_bench_output" / "swebench_patched.yaml").exists()

    def test_score_does_not_install_toolcall_patch_by_default(
        self,
        report_dir,
        swe_bench_project,
        template_yaml,
        patch_subprocess,
        monkeypatch,
    ):
        def fail_overlay(cls, swe_bench_project_path, overlay_root):
            pytest.fail("toolcall patch overlay should not be created by default")

        monkeypatch.setattr(
            SWEBenchScorer,
            "_create_toolcall_patch_overlay",
            classmethod(fail_overlay),
        )
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)

        score, n_repeats = scorer.score()

        assert score == pytest.approx(1 / 3)
        assert n_repeats == 1

    def test_score_rejects_toolcall_patch_for_non_qwen(
        self, report_dir, swe_bench_project, template_yaml, monkeypatch
    ):
        _write_toolcall_patch_files(swe_bench_project)

        def fail_overlay(cls, swe_bench_project_path, overlay_root):
            pytest.fail("toolcall patch should be rejected before overlay creation")

        monkeypatch.setattr(
            SWEBenchScorer,
            "_create_toolcall_patch_overlay",
            classmethod(fail_overlay),
        )
        scorer = _make_scorer(
            report_dir,
            swe_bench_project,
            template_yaml,
            enable_swebench_toolcall_patch=True,
        )

        with pytest.raises(ValueError, match="only supported for Qwen"):
            scorer.score()

    def test_score_uses_toolcall_patch_overlay_for_qwen_agent_run_only(
        self,
        report_dir,
        swe_bench_project,
        template_yaml,
        patch_subprocess,
        monkeypatch,
        tmp_path,
    ):
        _write_toolcall_patch_files(swe_bench_project)
        _write_benchmark_config(
            report_dir,
            model_params={"name": "Qwen/Qwen3.6-35B-A3B"},
        )
        overlay_root = tmp_path / "overlay"
        overlay_root.mkdir()
        calls: list[Path] = []

        def fake_overlay(cls, swe_bench_project_path, root):
            calls.append(root)
            return overlay_root

        monkeypatch.setattr(
            SWEBenchScorer,
            "_create_toolcall_patch_overlay",
            classmethod(fake_overlay),
        )
        scorer = _make_scorer(
            report_dir,
            swe_bench_project,
            template_yaml,
            enable_swebench_toolcall_patch=True,
        )

        score, n_repeats = scorer.score()

        assert score == pytest.approx(1 / 3)
        assert n_repeats == 1
        assert len(calls) == 1
        agent_env = patch_subprocess.kwargs[0]["env"]
        eval_env = patch_subprocess.kwargs[1]["env"]
        assert agent_env["PYTHONPATH"].split(":")[0] == str(overlay_root)
        assert (
            "PYTHONPATH" not in eval_env
            or str(overlay_root) not in eval_env["PYTHONPATH"]
        )

    def test_create_toolcall_patch_overlay_leaves_site_packages_unchanged(
        self, swe_bench_project, tmp_path, monkeypatch
    ):
        _write_toolcall_patch_files(swe_bench_project)
        site_packages = tmp_path / "site-packages"
        actions_dest = (
            site_packages / "minisweagent" / "models" / "utils" / "actions_toolcall.py"
        )
        litellm_dest = site_packages / "minisweagent" / "models" / "litellm_model.py"
        actions_dest.parent.mkdir(parents=True)
        litellm_dest.parent.mkdir(parents=True, exist_ok=True)
        actions_dest.write_text("# original actions\n")
        litellm_dest.write_text("# original litellm\n")

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout=str(actions_dest) + "\n", stderr="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)

        overlay_root = tmp_path / "overlay"
        SWEBenchScorer._create_toolcall_patch_overlay(swe_bench_project, overlay_root)

        assert actions_dest.read_text() == "# original actions\n"
        assert litellm_dest.read_text() == "# original litellm\n"
        assert (
            overlay_root / "minisweagent" / "models" / "utils" / "actions_toolcall.py"
        ).read_text() == "# patched actions_toolcall.py\n"
        assert (
            overlay_root / "minisweagent" / "models" / "litellm_model.py"
        ).read_text() == "# patched litellm_model.py\n"

    def test_create_toolcall_patch_overlay_missing_package_raises(
        self, swe_bench_project, tmp_path, monkeypatch
    ):
        _write_toolcall_patch_files(swe_bench_project)
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        actions_toolcall_file = (
            site_packages / "minisweagent" / "models" / "utils" / "actions_toolcall.py"
        )

        def fake_run(cmd, **kwargs):
            return MagicMock(
                returncode=0, stdout=f"{actions_toolcall_file}\n", stderr=""
            )

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)

        with pytest.raises(SetupError, match="package directory not found"):
            SWEBenchScorer._create_toolcall_patch_overlay(
                swe_bench_project, tmp_path / "overlay"
            )

    def test_create_toolcall_patch_overlay_propagates_copy_failure_without_mutation(
        self, swe_bench_project, tmp_path, monkeypatch
    ):
        _write_toolcall_patch_files(swe_bench_project)
        site_packages = tmp_path / "site-packages"
        actions_dest = (
            site_packages / "minisweagent" / "models" / "utils" / "actions_toolcall.py"
        )
        litellm_dest = site_packages / "minisweagent" / "models" / "litellm_model.py"
        actions_dest.parent.mkdir(parents=True)
        actions_dest.write_text("# original actions\n")
        litellm_dest.write_text("# original litellm\n")

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout=str(actions_dest) + "\n", stderr="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(
            scoring_mod.shutil,
            "copy2",
            MagicMock(side_effect=OSError("disk full")),
        )

        with pytest.raises(OSError, match="disk full"):
            SWEBenchScorer._create_toolcall_patch_overlay(
                swe_bench_project, tmp_path / "overlay"
            )

        assert actions_dest.read_text() == "# original actions\n"
        assert litellm_dest.read_text() == "# original litellm\n"

    def test_score_missing_config_yaml_raises(
        self, tmp_path, swe_bench_project, template_yaml
    ):
        report_dir = tmp_path / "empty_report"
        report_dir.mkdir()
        _write_sample_idx_map(report_dir)
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        with pytest.raises(FileNotFoundError, match="config.yaml not found"):
            scorer.score()

    def test_score_requires_loaded_dataset(
        self, report_dir, swe_bench_project, template_yaml
    ):
        dataset = _make_dataset()
        dataset.dataframe = None
        scorer = _make_scorer(
            report_dir, swe_bench_project, template_yaml, dataset=dataset
        )
        with pytest.raises(RuntimeError, match="dataset must be loaded"):
            scorer.score()

    def test_dataset_loader_kwargs_resolves_dataset_options(self):
        assert SWEBenchScorer.dataset_loader_kwargs(
            {"subset": "lite", "split": "train"}
        ) == {"subset": "lite", "split": "train"}

    def test_score_single_sample_raises(
        self, report_dir, swe_bench_project, template_yaml
    ):
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        with pytest.raises(RuntimeError, match="call score\\(\\) instead"):
            scorer.score_single_sample("value", "ground_truth")

    def test_missing_subproject_raises_at_init(
        self, report_dir, tmp_path, template_yaml
    ):
        empty_dir = tmp_path / "empty_project"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="SWE-bench subproject not found"):
            SWEBenchScorer(
                dataset_name=_DATASET_NAME,
                dataset=_make_dataset(),
                report_dir=report_dir,
                swe_bench_project_path=empty_dir,
                swebench_config_template=template_yaml,
            )

    def test_missing_template_raises_at_init(
        self, report_dir, swe_bench_project, tmp_path
    ):
        nonexistent = tmp_path / "no_such_template.yaml"
        with pytest.raises(FileNotFoundError, match="swebench template"):
            SWEBenchScorer(
                dataset_name=_DATASET_NAME,
                dataset=_make_dataset(),
                report_dir=report_dir,
                swe_bench_project_path=swe_bench_project,
                swebench_config_template=nonexistent,
            )

    def test_missing_preds_returns_none(
        self, report_dir, swe_bench_project, template_yaml, monkeypatch
    ):
        monkeypatch.setattr(scoring_mod.subprocess, "run", _make_fake_run)
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        score, n_repeats = scorer.score()
        assert score is None
        assert n_repeats == 1
        assert scorer.complete is False

    def test_config_patching_all_fields(self, report_dir, swe_bench_project, tmp_path):
        tmpl = {
            "model": {
                "model_name": "",
                "model_kwargs": {
                    "api_base": "",
                    "temperature": None,
                    "top_k": None,
                },
            }
        }
        template_path = tmp_path / "tmpl.yaml"
        template_path.write_text(yaml.dump(tmpl))

        _write_benchmark_config(
            report_dir,
            model_params={
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 15,
                "chat_template_kwargs": {"preserve_thinking": True},
            },
        )

        scorer = _make_scorer(report_dir, swe_bench_project, template_path)
        output_dir = tmp_path / "out"
        with (report_dir / "config.yaml").open() as f:
            benchmark_cfg = yaml.safe_load(f)
        patched = _patch_config_and_read(scorer, output_dir, benchmark_cfg)

        assert patched["model"]["model_name"] == _MODEL_NAME
        assert (
            patched["model"]["model_kwargs"]["api_base"] == "http://localhost:30000/v1"
        )
        assert patched["model"]["model_kwargs"]["temperature"] == pytest.approx(0.8)
        assert patched["model"]["model_kwargs"]["top_p"] == pytest.approx(0.9)
        assert patched["model"]["model_kwargs"]["top_k"] == 15
        assert patched["model"]["model_kwargs"]["chat_template_kwargs"] == {
            "preserve_thinking": True
        }

    def test_config_patching_omits_none_fields(
        self, report_dir, swe_bench_project, tmp_path
    ):
        tmpl = {
            "model": {
                "model_name": "",
                "model_kwargs": {"api_base": "", "top_k": 20},
            }
        }
        template_path = tmp_path / "tmpl.yaml"
        template_path.write_text(yaml.dump(tmpl))

        # model_params has no top_k — should be removed from patched config
        _write_benchmark_config(report_dir)

        scorer = _make_scorer(report_dir, swe_bench_project, template_path)
        output_dir = tmp_path / "out"
        with (report_dir / "config.yaml").open() as f:
            benchmark_cfg = yaml.safe_load(f)
        patched = _patch_config_and_read(scorer, output_dir, benchmark_cfg)

        assert "top_k" not in patched["model"]["model_kwargs"]

    def test_config_patching_max_new_tokens(
        self, report_dir, swe_bench_project, tmp_path
    ):
        tmpl = {
            "model": {
                "model_name": "",
                "model_kwargs": {"api_base": ""},
            }
        }
        template_path = tmp_path / "tmpl.yaml"
        template_path.write_text(yaml.dump(tmpl))

        _write_benchmark_config(report_dir, model_params={"max_new_tokens": 4096})

        scorer = _make_scorer(report_dir, swe_bench_project, template_path)
        output_dir = tmp_path / "out"
        with (report_dir / "config.yaml").open() as f:
            benchmark_cfg = yaml.safe_load(f)
        patched = _patch_config_and_read(scorer, output_dir, benchmark_cfg)

        assert patched["model"]["model_kwargs"]["max_tokens"] == 4096

    def test_config_patching_omits_max_tokens_when_not_set(
        self, report_dir, swe_bench_project, tmp_path
    ):
        tmpl = {
            "model": {
                "model_name": "",
                "model_kwargs": {"api_base": "", "max_tokens": 999},
            }
        }
        template_path = tmp_path / "tmpl.yaml"
        template_path.write_text(yaml.dump(tmpl))

        # model_params has no max_new_tokens — max_tokens should be removed
        _write_benchmark_config(report_dir)

        scorer = _make_scorer(report_dir, swe_bench_project, template_path)
        output_dir = tmp_path / "out"
        with (report_dir / "config.yaml").open() as f:
            benchmark_cfg = yaml.safe_load(f)
        patched = _patch_config_and_read(scorer, output_dir, benchmark_cfg)

        assert "max_tokens" not in patched["model"]["model_kwargs"]

    def test_config_patching_no_endpoints_clears_api_base(
        self, report_dir, swe_bench_project, template_yaml, tmp_path
    ):
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        patched = _patch_config_and_read(
            scorer,
            tmp_path / "out",
            {"model_params": {"name": _MODEL_NAME}, "endpoint_config": {}},
        )

        assert patched["model"]["model_kwargs"]["api_base"] == ""

    def test_config_patching_strips_trailing_v1_from_endpoint(
        self, report_dir, swe_bench_project, template_yaml, tmp_path
    ):
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        patched = _patch_config_and_read(
            scorer,
            tmp_path / "out",
            {
                "model_params": {"name": _MODEL_NAME},
                "endpoint_config": {"endpoints": ["http://localhost:30000/v1"]},
            },
        )

        assert (
            patched["model"]["model_kwargs"]["api_base"] == "http://localhost:30000/v1"
        )

    @pytest.mark.parametrize("api_key", ["secret-key", None])
    def test_config_patching_updates_api_key(
        self, report_dir, swe_bench_project, tmp_path, api_key
    ):
        tmpl = {
            "model": {
                "model_name": "",
                "model_kwargs": {"api_base": "", "api_key": "test"},
            }
        }
        template_path = tmp_path / "tmpl.yaml"
        template_path.write_text(yaml.dump(tmpl))

        scorer = _make_scorer(report_dir, swe_bench_project, template_path)
        endpoint_cfg = {"endpoints": ["http://localhost:30000"]}
        if api_key is not None:
            endpoint_cfg["api_key"] = api_key
        patched = _patch_config_and_read(
            scorer,
            tmp_path / "out",
            {
                "model_params": {"name": _MODEL_NAME},
                "endpoint_config": endpoint_cfg,
            },
        )

        model_kwargs = patched["model"]["model_kwargs"]
        if api_key is None:
            assert "api_key" not in model_kwargs
        else:
            assert model_kwargs["api_key"] == api_key

    @pytest.mark.parametrize(
        "num_instances, expected_slice",
        [
            (5, "0:5"),
            (100, "0:100"),
        ],
    )
    def test_num_instances_produces_correct_slice(
        self,
        num_instances,
        expected_slice,
        report_dir,
        swe_bench_project,
        template_yaml,
        patch_subprocess,
    ):
        scorer = _make_scorer(
            report_dir,
            swe_bench_project,
            template_yaml,
            dataset=_make_dataset(n=num_instances),
            num_instances=num_instances,
        )
        scorer.score()
        agent_cmd = patch_subprocess[0]
        assert agent_cmd[agent_cmd.index("--slice") + 1] == expected_slice

    @pytest.mark.parametrize(
        "subset, expected_hf_name",
        [
            ("lite", "princeton-nlp/SWE-bench_Lite"),
            ("verified", "princeton-nlp/SWE-bench_Verified"),
        ],
    )
    def test_subset_maps_to_correct_hf_dataset_name(
        self,
        subset,
        expected_hf_name,
        report_dir,
        swe_bench_project,
        template_yaml,
        patch_subprocess,
    ):
        scorer = _make_scorer(
            report_dir, swe_bench_project, template_yaml, subset=subset
        )
        scorer.score()
        eval_cmd = patch_subprocess[1]
        assert eval_cmd[eval_cmd.index("--dataset_name") + 1] == expected_hf_name

    def test_unknown_subset_raises_at_init(
        self, report_dir, swe_bench_project, template_yaml
    ):
        with pytest.raises(ValueError, match="Unknown SWE-bench subset"):
            SWEBenchScorer(
                dataset_name=_DATASET_NAME,
                dataset=_make_dataset(),
                report_dir=report_dir,
                swe_bench_project_path=swe_bench_project,
                swebench_config_template=template_yaml,
                subset="full",
            )

    def test_missing_model_name_raises_clear_error(
        self, report_dir, swe_bench_project, tmp_path
    ):
        tmpl = {
            "model": {
                "model_name": "",
                "model_kwargs": {"api_base": ""},
            }
        }
        template_path = tmp_path / "tmpl.yaml"
        template_path.write_text(yaml.dump(tmpl))

        scorer = _make_scorer(report_dir, swe_bench_project, template_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with pytest.raises(ValueError, match="model_params.name is required"):
            scorer._patch_config(output_dir, {"model_params": {}})

    def test_score_missing_model_name_raises(
        self, report_dir, swe_bench_project, template_yaml
    ):
        (report_dir / "config.yaml").write_text(
            yaml.dump({"model_params": {}, "endpoint_config": {"endpoints": []}})
        )
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        with pytest.raises(ValueError, match="model_params.name is required"):
            scorer.score()

    @pytest.mark.parametrize("config_text", ["", "[]", "not-a-mapping"])
    def test_score_non_mapping_config_raises(
        self, report_dir, swe_bench_project, template_yaml, config_text
    ):
        (report_dir / "config.yaml").write_text(config_text)
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)

        with pytest.raises(ValueError, match="benchmark config.*YAML mapping"):
            scorer.score()

    def test_template_missing_model_kwargs_raises(
        self, report_dir, swe_bench_project, tmp_path
    ):
        bad_template = tmp_path / "bad_template.yaml"
        bad_template.write_text(yaml.dump({"model": {"model_name": ""}}))
        with pytest.raises(ValueError, match="model.model_kwargs"):
            SWEBenchScorer(
                dataset_name=_DATASET_NAME,
                dataset=_make_dataset(),
                report_dir=report_dir,
                swe_bench_project_path=swe_bench_project,
                swebench_config_template=bad_template,
            )

    @pytest.mark.parametrize("template_text", ["", "[]", "not-a-mapping"])
    def test_template_non_mapping_raises(
        self, report_dir, swe_bench_project, tmp_path, template_text
    ):
        bad_template = tmp_path / "bad_template.yaml"
        bad_template.write_text(template_text)

        with pytest.raises(ValueError, match="YAML mapping"):
            SWEBenchScorer(
                dataset_name=_DATASET_NAME,
                dataset=_make_dataset(),
                report_dir=report_dir,
                swe_bench_project_path=swe_bench_project,
                swebench_config_template=bad_template,
            )

    @pytest.mark.parametrize(
        ("max_eval_workers", "expected_match"),
        [(0, ">= 1"), (-1, ">= 1"), ("bad", "must be an integer")],
    )
    def test_invalid_max_eval_workers_raises(
        self,
        report_dir,
        swe_bench_project,
        template_yaml,
        max_eval_workers,
        expected_match,
    ):
        with pytest.raises(ValueError, match=expected_match):
            _make_scorer(
                report_dir,
                swe_bench_project,
                template_yaml,
                max_eval_workers=max_eval_workers,
            )

    @pytest.mark.parametrize("num_instances", [0, -1])
    def test_invalid_num_instances_raises(
        self, report_dir, swe_bench_project, template_yaml, num_instances
    ):
        with pytest.raises(SetupError, match=r"num_instances must be >= 1"):
            _make_scorer(
                report_dir,
                swe_bench_project,
                template_yaml,
                num_instances=num_instances,
            )

    def test_run_subprocess_replaces_decode_errors(
        self, report_dir, swe_bench_project, template_yaml, tmp_path, monkeypatch
    ):
        captured_kwargs: dict = {}

        def fake_run(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock(returncode=0, stdout="ok")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)

        scorer._run_subprocess(
            ["python", "-c", "print('ok')"], tmp_path / "run.log", tmp_path
        )

        assert captured_kwargs["encoding"] == "utf-8"
        assert captured_kwargs["errors"] == "replace"

    def test_subprocess_failure_raises(
        self, report_dir, swe_bench_project, template_yaml, monkeypatch
    ):
        def _fail_eval(cmd, **kwargs):
            return MagicMock(returncode=2, stdout="docker error: permission denied")

        monkeypatch.setattr(
            scoring_mod.subprocess,
            "run",
            _make_staged_run(_fail_eval),
        )
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        with pytest.raises(RuntimeError, match="exited with code 2"):
            scorer.score()

    def test_subprocess_timeout_raises(
        self, report_dir, swe_bench_project, template_yaml, monkeypatch
    ):
        def _timeout_eval(cmd, **kwargs):
            raise scoring_mod.subprocess.TimeoutExpired(cmd=cmd, timeout=300)

        monkeypatch.setattr(
            scoring_mod.subprocess, "run", _make_staged_run(_timeout_eval)
        )
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        with pytest.raises(RuntimeError, match="timed out after"):
            scorer.score()

    def test_result_glob_fallback(
        self, report_dir, swe_bench_project, template_yaml, monkeypatch
    ):
        def _write_alt_prefix(cmd, **kwargs):
            if "run_evaluation" in " ".join(cmd):
                cwd = Path(kwargs["cwd"])
                run_id = cmd[cmd.index("--run_id") + 1]
                # Write under a different prefix so exact name won't match; glob will find it
                (cwd / f"alt_prefix.{run_id}.json").write_text(
                    json.dumps(
                        {
                            "resolved_instances": 1,
                            "submitted_instances": 5,
                            "total_instances": 500,
                        }
                    )
                )
            return MagicMock(returncode=0, stdout="")

        monkeypatch.setattr(
            scoring_mod.subprocess, "run", _make_staged_run(_write_alt_prefix)
        )
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        score, n_repeats = scorer.score()
        assert score == pytest.approx(1 / 3)
        assert n_repeats == 1

    def test_zero_submitted_instances_returns_none(
        self, report_dir, swe_bench_project, template_yaml, monkeypatch
    ):
        def _write_zero_results(cmd, **kwargs):
            if "run_evaluation" in " ".join(cmd):
                cwd = Path(kwargs["cwd"])
                run_id = cmd[cmd.index("--run_id") + 1]
                safe_model = _MODEL_NAME.replace("/", "__")
                (cwd / f"{safe_model}.{run_id}.json").write_text(
                    json.dumps(
                        {
                            "resolved_instances": 0,
                            "submitted_instances": 0,
                            "total_instances": 500,
                        }
                    )
                )
            return MagicMock(returncode=0, stdout="")

        monkeypatch.setattr(
            scoring_mod.subprocess, "run", _make_staged_run(_write_zero_results)
        )
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        score, n_repeats = scorer.score()
        assert score is None
        assert n_repeats == 1
        assert scorer.complete is False

    def test_missing_result_file_returns_none_and_marks_incomplete(
        self, report_dir, swe_bench_project, template_yaml, monkeypatch
    ):
        def _no_result(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", _make_staged_run(_no_result))
        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        score, n_repeats = scorer.score()
        assert score is None
        assert n_repeats == 1
        assert scorer.complete is False

    def test_zero_evaluated_instances_returns_none_and_marks_incomplete(
        self, report_dir, swe_bench_project, template_yaml, patch_subprocess
    ):
        scorer = _make_scorer(
            report_dir,
            swe_bench_project,
            template_yaml,
            dataset=_make_dataset(n=0),
        )
        score, n_repeats = scorer.score()
        assert score is None
        assert n_repeats == 1
        assert scorer.complete is False

    def test_num_instances_exceeding_dataset_warns_and_clamps(
        self, report_dir, swe_bench_project, template_yaml, patch_subprocess, caplog
    ):
        scorer = _make_scorer(
            report_dir,
            swe_bench_project,
            template_yaml,
            dataset=_make_dataset(n=3),
            num_instances=10,
        )
        with caplog.at_level("WARNING"):
            scorer.score()

        assert any("exceeds dataset size" in message for message in caplog.messages)
        agent_cmd = patch_subprocess[0]
        assert agent_cmd[agent_cmd.index("--slice") + 1] == "0:3"

    def test_score_removes_pre_existing_output_dir(
        self, report_dir, swe_bench_project, template_yaml, patch_subprocess
    ):
        output_dir = report_dir / "swe_bench_output"
        output_dir.mkdir()
        stale_file = output_dir / "stale.txt"
        stale_file.write_text("stale")

        scorer = _make_scorer(report_dir, swe_bench_project, template_yaml)
        scorer.score()

        assert not stale_file.exists()
        assert output_dir.exists()


class TestSWEBenchScorerPreflight:
    def _extras(self, swe_bench_project: Path, **overrides) -> dict:
        return {"swe_bench_project_path": str(swe_bench_project), **overrides}

    def _patch_fake_executor(self, monkeypatch) -> None:
        _FakeThreadPoolExecutor.reset()
        monkeypatch.setattr(
            scoring_mod.concurrent.futures,
            "ThreadPoolExecutor",
            _FakeThreadPoolExecutor,
        )
        monkeypatch.setattr(
            scoring_mod.concurrent.futures, "as_completed", _fake_as_completed
        )

    def test_toolcall_patch_overlay_probes_minisweagent_with_uv_project_env(
        self, swe_bench_project, tmp_path, monkeypatch
    ):
        actions_path = (
            tmp_path
            / "site-packages"
            / "minisweagent"
            / "models"
            / "utils"
            / "actions_toolcall.py"
        )
        litellm_path = (
            tmp_path / "site-packages" / "minisweagent" / "models" / "litellm_model.py"
        )
        actions_path.parent.mkdir(parents=True)
        litellm_path.parent.mkdir(parents=True, exist_ok=True)
        actions_path.write_text("old actions\n")
        litellm_path.write_text("old litellm\n")
        (swe_bench_project / "actions_toolcall.py").write_text("new actions\n")
        (swe_bench_project / "litellm_model.py").write_text("new litellm\n")
        captured: list[list[str]] = []
        captured_kwargs: list[dict] = []

        def fake_run(cmd, **kw):
            captured.append(list(cmd))
            captured_kwargs.append(kw)
            return MagicMock(returncode=0, stdout=f"{actions_path}\n", stderr="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)

        overlay_root = tmp_path / "overlay"
        SWEBenchScorer._create_toolcall_patch_overlay(swe_bench_project, overlay_root)

        probe_cmd = captured[0]
        assert probe_cmd[:5] == [
            "uv",
            "run",
            "--project",
            str(swe_bench_project.resolve()),
            "python",
        ]
        assert captured_kwargs[0]["env"]["UV_PROJECT_ENVIRONMENT"] == str(
            swe_bench_project.resolve() / ".venv"
        )
        assert actions_path.read_text() == "old actions\n"
        assert litellm_path.read_text() == "old litellm\n"
        assert (
            overlay_root / "minisweagent" / "models" / "utils" / "actions_toolcall.py"
        ).read_text() == "new actions\n"
        assert (
            overlay_root / "minisweagent" / "models" / "litellm_model.py"
        ).read_text() == "new litellm\n"

    @pytest.mark.parametrize(
        "result, expected_match",
        [
            (
                MagicMock(returncode=1, stdout="", stderr="ModuleNotFoundError"),
                "Could not locate minisweagent install",
            ),
            (MagicMock(returncode=0, stdout="\n", stderr=""), "empty output"),
            (
                MagicMock(
                    returncode=0,
                    stdout="/path/that/does/not/exist/actions_toolcall.py\n",
                    stderr="",
                ),
                "does not exist",
            ),
        ],
    )
    def test_resolve_site_packages_invalid_probe_result_raises(
        self, swe_bench_project, monkeypatch, result, expected_match
    ):
        monkeypatch.setattr(scoring_mod.subprocess, "run", lambda cmd, **kw: result)

        with pytest.raises(SetupError, match=expected_match):
            SWEBenchScorer._resolve_minisweagent_site_packages(swe_bench_project)

    def test_preflight_parallelizes_cached_and_missing_images(
        self, swe_bench_project, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        _FakeTqdm.reset()
        self._patch_fake_executor(monkeypatch)
        monkeypatch.setattr(scoring_mod, "tqdm", _FakeTqdm)
        captured: list[list[str]] = []
        captured_kwargs: list[dict] = []
        _FakeThreadPoolExecutor.completion_order = [
            "docker.io/swebench/missing-a:latest",
            "docker.io/swebench/cached:latest",
            "docker.io/swebench/missing-b:latest",
        ]

        def fake_run(cmd, **kw):
            captured.append(list(cmd))
            captured_kwargs.append(kw)
            cmd_str = " ".join(cmd)
            if "get_swebench_docker_image_name" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "👋 This is mini-swe-agent version 2.3.0.\n"
                        "minisweagent: INFO: Instance slice: 500 -> 2 instances\n"
                        + json.dumps(
                            [
                                "docker.io/swebench/cached:latest",
                                "docker.io/swebench/missing-a:latest",
                                "docker.io/swebench/missing-b:latest",
                            ]
                        )
                    ),
                    stderr="",
                )
            if cmd[:3] == ["docker", "image", "inspect"]:
                image = cmd[3]
                if image == "docker.io/swebench/cached:latest":
                    return MagicMock(returncode=0, stdout="", stderr=b"")
                return MagicMock(returncode=1, stdout="", stderr=b"missing")
            if cmd[:2] == ["docker", "pull"]:
                return MagicMock(returncode=0, stdout="", stderr=b"")
            return MagicMock(returncode=0, stdout="", stderr=b"")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        SWEBenchScorer.preflight(
            self._extras(
                swe_bench_project,
                subset="lite",
                split="test",
                num_instances=3,
                workers=5,
            ),
            loaded_sample_count=2,
        )

        derive_cmd = next(
            cmd for cmd in captured if "get_swebench_docker_image_name" in " ".join(cmd)
        )
        derive_idx = captured.index(derive_cmd)
        compile(derive_cmd[6], "<swebench-derive-images>", "exec")
        assert derive_cmd[-3:] == ["lite", "test", "2"]
        assert captured_kwargs[derive_idx]["env"]["UV_PROJECT_ENVIRONMENT"] == str(
            swe_bench_project / ".venv"
        )
        uv_run_idxs = [
            i for i, cmd in enumerate(captured) if cmd[:3] == ["uv", "run", "--project"]
        ]
        assert len(uv_run_idxs) == 3
        assert {
            captured_kwargs[i]["env"]["UV_PROJECT_ENVIRONMENT"] for i in uv_run_idxs
        } == {str(swe_bench_project / ".venv")}
        assert ["docker", "pull", "docker.io/swebench/cached:latest"] not in captured
        assert ["docker", "pull", "docker.io/swebench/missing-a:latest"] in captured
        assert ["docker", "pull", "docker.io/swebench/missing-b:latest"] in captured
        assert len(_FakeTqdm.instances) == 1
        assert len(_FakeThreadPoolExecutor.instances) == 1
        assert _FakeThreadPoolExecutor.instances[0].max_workers == 3
        assert _FakeThreadPoolExecutor.instances[0].shutdown_wait is True
        assert _FakeThreadPoolExecutor.instances[0].shutdown_cancel_futures is False

    def test_preflight_skips_cached_images(self, swe_bench_project, monkeypatch):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        _FakeTqdm.reset()
        self._patch_fake_executor(monkeypatch)
        monkeypatch.setattr(scoring_mod, "tqdm", _FakeTqdm)
        captured: list[list[str]] = []

        def fake_run(cmd, **kw):
            captured.append(list(cmd))
            cmd_str = " ".join(cmd)
            if "get_swebench_docker_image_name" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        [
                            "docker.io/swebench/cached-a:latest",
                            "docker.io/swebench/cached-b:latest",
                        ]
                    ),
                    stderr="",
                )
            if cmd[:3] == ["docker", "image", "inspect"]:
                return MagicMock(returncode=0, stdout="", stderr=b"")
            return MagicMock(returncode=0, stdout="", stderr=b"")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        SWEBenchScorer.preflight(self._extras(swe_bench_project, workers=4))

        assert not any(cmd[:2] == ["docker", "pull"] for cmd in captured)
        assert len(_FakeTqdm.instances) == 1
        assert _FakeThreadPoolExecutor.instances[0].max_workers == 2
        assert _FakeThreadPoolExecutor.instances[0].shutdown_wait is True

    def test_preflight_fails_uv_missing(self, swe_bench_project, monkeypatch):
        monkeypatch.setattr(scoring_mod.shutil, "which", lambda name: None)
        with pytest.raises(SetupError, match="uv is not on PATH"):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    def test_preflight_fails_mini_extra_missing(self, swe_bench_project, monkeypatch):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )

        def fake_run(cmd, **kw):
            if "mini-extra" in cmd:
                return MagicMock(returncode=1, stderr=b"not found")
            return MagicMock(returncode=0)

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(
            SetupError, match=r"mini-extra is not available.*stderr: not found"
        ):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    @pytest.mark.parametrize(
        "timeout_match, command_match",
        [
            ("Timed out probing mini-extra", lambda cmd: "mini-extra" in cmd),
            (
                "Timed out probing swebench",
                lambda cmd: "import swebench" in " ".join(cmd),
            ),
        ],
    )
    def test_preflight_probe_timeout_raises_setup_error(
        self, swe_bench_project, monkeypatch, timeout_match, command_match
    ):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )

        def fake_run(cmd, **kw):
            if command_match(cmd):
                raise scoring_mod.subprocess.TimeoutExpired(cmd=cmd, timeout=30)
            return MagicMock(returncode=0)

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(SetupError, match=timeout_match):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    def test_preflight_fails_swebench_missing(self, swe_bench_project, monkeypatch):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )

        def fake_run(cmd, **kw):
            if "import swebench" in " ".join(cmd):
                return MagicMock(returncode=1, stderr=b"ModuleNotFoundError")
            return MagicMock(returncode=0)

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(
            SetupError,
            match=r"swebench is not available.*stderr: ModuleNotFoundError",
        ):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    def test_preflight_fails_docker_not_running(self, swe_bench_project, monkeypatch):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )

        def fake_run(cmd, **kw):
            if "docker" in cmd:
                return MagicMock(
                    returncode=1, stderr=b"Cannot connect to Docker daemon"
                )
            return MagicMock(returncode=0)

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(SetupError, match="Docker daemon is not running"):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    def test_preflight_fails_docker_missing_from_path(
        self, swe_bench_project, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod.shutil,
            "which",
            lambda name: None if name == "docker" else f"/usr/bin/{name}",
        )
        monkeypatch.setattr(
            scoring_mod.subprocess, "run", lambda cmd, **kw: MagicMock(returncode=0)
        )
        with pytest.raises(SetupError, match="docker is not on PATH"):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    def test_preflight_fails_docker_command_raises(
        self, swe_bench_project, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )

        def fake_run(cmd, **kw):
            if cmd[:1] == ["docker"]:
                raise OSError("docker binary not executable")
            return MagicMock(returncode=0)

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(SetupError, match="Failed to execute docker command"):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    def test_preflight_wraps_resolve_options_failure(
        self, report_dir, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        missing_project = tmp_path / "missing_project"
        with pytest.raises(SetupError, match="SWE-bench subproject not found"):
            SWEBenchScorer.preflight(self._extras(missing_project))

    def test_preflight_fails_when_pull_fails(self, swe_bench_project, monkeypatch):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        _FakeTqdm.reset()
        self._patch_fake_executor(monkeypatch)
        monkeypatch.setattr(scoring_mod, "tqdm", _FakeTqdm)
        _FakeThreadPoolExecutor.completion_order = [
            "docker.io/swebench/test:latest",
        ]

        def fake_run(cmd, **kw):
            cmd_str = " ".join(cmd)
            if "get_swebench_docker_image_name" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        [
                            "docker.io/swebench/cached:latest",
                            "docker.io/swebench/test:latest",
                            "docker.io/swebench/pending:latest",
                        ]
                    ),
                    stderr="",
                )
            if cmd[:3] == ["docker", "image", "inspect"]:
                if cmd[3] == "docker.io/swebench/cached:latest":
                    return MagicMock(returncode=0, stdout="", stderr=b"")
                if cmd[3] == "docker.io/swebench/pending:latest":
                    return MagicMock(returncode=1, stdout="", stderr=b"missing")
                return MagicMock(returncode=1, stdout="", stderr=b"missing")
            if cmd[:2] == ["docker", "pull"]:
                return MagicMock(
                    returncode=1,
                    stdout="",
                    stderr=b"rate limit exceeded",
                )
            return MagicMock(returncode=0, stdout="", stderr=b"")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(
            SetupError,
            match=r"docker\.io/swebench/test:latest.*rate limit exceeded",
        ):
            SWEBenchScorer.preflight(self._extras(swe_bench_project, workers=2))
        assert _FakeThreadPoolExecutor.instances[0].max_workers == 2
        assert _FakeThreadPoolExecutor.instances[0].shutdown_wait is True
        assert _FakeThreadPoolExecutor.instances[0].shutdown_cancel_futures is True

    def test_derive_required_images_nonzero_returncode_raises(
        self, swe_bench_project, monkeypatch
    ):
        def fake_run(cmd, **kw):
            return MagicMock(returncode=1, stdout="", stderr="permission denied")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(
            SetupError, match=r"Failed to derive required.*permission denied"
        ):
            SWEBenchScorer._derive_required_images(
                swe_bench_project_path=swe_bench_project,
                subset="lite",
                split="test",
                num_instances=3,
            )

    def test_derive_required_images_unparseable_output_raises(
        self, swe_bench_project, monkeypatch
    ):
        def fake_run(cmd, **kw):
            return MagicMock(returncode=0, stdout="not json", stderr="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(SetupError, match="Failed to parse the required"):
            SWEBenchScorer._derive_required_images(
                swe_bench_project_path=swe_bench_project,
                subset="lite",
                split="test",
                num_instances=3,
            )

    def test_derive_required_images_invalid_list_raises(
        self, swe_bench_project, monkeypatch
    ):
        def fake_run(cmd, **kw):
            return MagicMock(returncode=0, stdout=json.dumps([1, 2]), stderr="")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(SetupError, match="invalid SWE-bench Docker image list"):
            SWEBenchScorer._derive_required_images(
                swe_bench_project_path=swe_bench_project,
                subset="lite",
                split="test",
                num_instances=3,
            )

    def test_prepull_image_inspect_timeout_raises_setup_error(self, monkeypatch):
        def fake_run(cmd, **kw):
            if cmd[:3] == ["docker", "image", "inspect"]:
                raise scoring_mod.subprocess.TimeoutExpired(cmd=cmd, timeout=30)
            return MagicMock(returncode=0, stdout="", stderr=b"")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(SetupError, match="Timed out inspecting"):
            SWEBenchScorer._prepull_image("docker.io/swebench/test:latest")

    def test_preflight_derive_images_timeout_raises_setup_error(
        self, swe_bench_project, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )

        def fake_run(cmd, **kw):
            if "get_swebench_docker_image_name" in " ".join(cmd):
                raise scoring_mod.subprocess.TimeoutExpired(cmd=cmd, timeout=300)
            return MagicMock(returncode=0, stdout="", stderr=b"")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(SetupError, match="Timed out deriving required"):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    def test_preflight_prepull_timeout_raises_setup_error(
        self, swe_bench_project, monkeypatch
    ):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        _FakeTqdm.reset()
        self._patch_fake_executor(monkeypatch)
        monkeypatch.setattr(scoring_mod, "tqdm", _FakeTqdm)
        _FakeThreadPoolExecutor.completion_order = [
            "docker.io/swebench/test:latest",
        ]

        def fake_run(cmd, **kw):
            cmd_str = " ".join(cmd)
            if "get_swebench_docker_image_name" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(["docker.io/swebench/test:latest"]),
                    stderr="",
                )
            if cmd[:3] == ["docker", "image", "inspect"]:
                return MagicMock(returncode=1, stdout="", stderr=b"missing")
            if cmd[:2] == ["docker", "pull"]:
                raise scoring_mod.subprocess.TimeoutExpired(cmd=cmd, timeout=300)
            return MagicMock(returncode=0, stdout="", stderr=b"")

        monkeypatch.setattr(scoring_mod.subprocess, "run", fake_run)
        with pytest.raises(
            SetupError, match=r"Timed out pulling.*docker\.io/swebench/test:latest"
        ):
            SWEBenchScorer.preflight(self._extras(swe_bench_project))

    def test_preflight_fails_invalid_workers(self, swe_bench_project, monkeypatch):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        with pytest.raises(
            SetupError, match=r"accuracy_config\.extras\.workers must be >= 1"
        ):
            SWEBenchScorer.preflight(self._extras(swe_bench_project, workers=0))

    def test_preflight_enabled_toolcall_patch_requires_replacements(
        self, swe_bench_project
    ):
        with pytest.raises(SetupError, match="replacement files are missing"):
            SWEBenchScorer.preflight(
                self._extras(
                    swe_bench_project,
                    enable_swebench_toolcall_patch=True,
                )
            )


class TestSWEBenchExampleConfigs:
    @staticmethod
    def _swe_bench_extras(path: Path) -> dict:
        config = yaml.safe_load(path.read_text())
        swe_bench_dataset = next(
            dataset for dataset in config["datasets"] if dataset["name"] == "swe_bench"
        )
        return swe_bench_dataset["accuracy_config"]["extras"]

    @pytest.mark.parametrize(
        "config_path",
        [
            _REPO_ROOT / "examples/10_Agentic_Inference/swe_bench_accuracy.yaml",
            _REPO_ROOT / "examples/10_Agentic_Inference/qwen_agentic_benchmark.yaml",
        ],
    )
    def test_qwen_configs_enable_toolcall_patch(self, config_path):
        extras = self._swe_bench_extras(config_path)

        assert extras["enable_swebench_toolcall_patch"] is True
        assert (
            extras["swebench_config_template"]
            == "examples/10_Agentic_Inference/accuracy/swebench_qwen_tools_template.yaml"
        )

    def test_kimi_config_leaves_toolcall_patch_disabled(self):
        extras = self._swe_bench_extras(
            _REPO_ROOT / "examples/10_Agentic_Inference/kimi_agentic_benchmark.yaml"
        )

        assert "enable_swebench_toolcall_patch" not in extras
        assert "swebench_config_template" not in extras
