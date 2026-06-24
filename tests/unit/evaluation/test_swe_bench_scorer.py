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


class _FakeTqdm:
    instances: list["_FakeTqdm"] = []

    def __init__(self, *, total: int, desc: str, unit: str):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.updates: list[int] = []
        self.closed = False
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True

    def update(self, n: int) -> None:
        self.updates.append(n)

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
        self.shutdown_calls: list[dict[str, bool]] = []
        type(self).instances.append(self)

    def submit(self, fn, image: str) -> _FakeFuture:
        future = _FakeFuture(fn, image)
        self.submitted.append(future)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})
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
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
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
                        "resolved_instances": 3,
                        "submitted_instances": 10,
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
        assert SWEBenchScorer.external_sample_count({"num_instances": 100}) == 100
        assert SWEBenchScorer.external_sample_count({}) is None
        assert SWEBenchScorer.external_sample_count({"num_instances": "bad"}) is None


class TestSWEBenchScorer:
    def test_score_happy_path(
        self, report_dir, swe_bench_project, template_yaml, patch_subprocess
    ):
        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_yaml,
        )
        score, n_repeats = scorer.score()

        assert score == pytest.approx(0.3)
        assert n_repeats == 1
        assert (report_dir / "swe_bench_results.json").exists()

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
        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_yaml,
        )
        score, n_repeats = scorer.score()
        assert score is None
        assert n_repeats == 1

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

        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_path,
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        with (report_dir / "config.yaml").open() as f:
            benchmark_cfg = yaml.safe_load(f)
        patched_path = scorer._patch_config(output_dir, benchmark_cfg)
        patched = yaml.safe_load(patched_path.read_text())

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

        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_path,
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        with (report_dir / "config.yaml").open() as f:
            benchmark_cfg = yaml.safe_load(f)
        patched_path = scorer._patch_config(output_dir, benchmark_cfg)
        patched = yaml.safe_load(patched_path.read_text())

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

        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_path,
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        with (report_dir / "config.yaml").open() as f:
            benchmark_cfg = yaml.safe_load(f)
        patched_path = scorer._patch_config(output_dir, benchmark_cfg)
        patched = yaml.safe_load(patched_path.read_text())

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

        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_path,
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        with (report_dir / "config.yaml").open() as f:
            benchmark_cfg = yaml.safe_load(f)
        patched_path = scorer._patch_config(output_dir, benchmark_cfg)
        patched = yaml.safe_load(patched_path.read_text())

        assert "max_tokens" not in patched["model"]["model_kwargs"]

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
        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(n=num_instances),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_yaml,
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
        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_yaml,
            subset=subset,
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

    def test_missing_model_name_raises_clear_error(self, swe_bench_project, tmp_path):
        tmpl = {
            "model": {
                "model_name": "",
                "model_kwargs": {"api_base": ""},
            }
        }
        template_path = tmp_path / "tmpl.yaml"
        template_path.write_text(yaml.dump(tmpl))

        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=tmp_path,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_path,
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with pytest.raises(ValueError, match="model_params.name is required"):
            scorer._patch_config(output_dir, {"model_params": {}})

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
        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_yaml,
        )
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
        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_yaml,
        )
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
        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_yaml,
        )
        score, n_repeats = scorer.score()
        assert score == pytest.approx(1 / 5)
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
        scorer = SWEBenchScorer(
            dataset_name=_DATASET_NAME,
            dataset=_make_dataset(),
            report_dir=report_dir,
            swe_bench_project_path=swe_bench_project,
            swebench_config_template=template_yaml,
        )
        score, n_repeats = scorer.score()
        assert score is None
        assert n_repeats == 1


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
        _FakeThreadPoolExecutor.completion_order = [
            "docker.io/swebench/missing-a:latest",
            "docker.io/swebench/cached:latest",
            "docker.io/swebench/missing-b:latest",
        ]

        def fake_run(cmd, **kw):
            captured.append(list(cmd))
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
            )
        )

        derive_cmd = next(
            cmd for cmd in captured if "get_swebench_docker_image_name" in " ".join(cmd)
        )
        compile(derive_cmd[6], "<swebench-derive-images>", "exec")
        assert derive_cmd[-3:] == ["lite", "test", "3"]
        assert ["docker", "pull", "docker.io/swebench/cached:latest"] not in captured
        assert ["docker", "pull", "docker.io/swebench/missing-a:latest"] in captured
        assert ["docker", "pull", "docker.io/swebench/missing-b:latest"] in captured
        assert len(_FakeTqdm.instances) == 1
        assert _FakeTqdm.instances[0].total == 3
        assert _FakeTqdm.instances[0].desc == "SWE-bench images"
        assert _FakeTqdm.instances[0].updates == [1, 1, 1]
        assert _FakeTqdm.instances[0].closed is True
        assert len(_FakeThreadPoolExecutor.instances) == 1
        assert _FakeThreadPoolExecutor.instances[0].max_workers == 3
        assert _FakeThreadPoolExecutor.instances[0].shutdown_calls == [
            {"wait": True, "cancel_futures": False}
        ]

    def test_preflight_all_cached_still_completes_progress_bar(
        self, swe_bench_project, monkeypatch
    ):
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
        assert _FakeTqdm.instances[0].total == 2
        assert _FakeTqdm.instances[0].updates == [1, 1]
        assert _FakeTqdm.instances[0].closed is True
        assert _FakeThreadPoolExecutor.instances[0].max_workers == 2

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
                if cmd[2] == "docker.io/swebench/pending:latest":
                    pytest.fail(
                        "pending pull should have been cancelled before starting"
                    )
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
        assert len(_FakeTqdm.instances) == 1
        assert _FakeTqdm.instances[0].total == 3
        assert _FakeTqdm.instances[0].updates == []
        assert _FakeTqdm.instances[0].closed is True
        executor = _FakeThreadPoolExecutor.instances[0]
        assert executor.max_workers == 2
        assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]
        future_by_image = {future.image: future for future in executor.submitted}
        assert future_by_image["docker.io/swebench/pending:latest"].cancelled is True

    def test_preflight_fails_invalid_workers(self, swe_bench_project, monkeypatch):
        monkeypatch.setattr(
            scoring_mod.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        with pytest.raises(
            SetupError, match=r"accuracy_config\.extras\.workers must be >= 1"
        ):
            SWEBenchScorer.preflight(self._extras(swe_bench_project, workers=0))
