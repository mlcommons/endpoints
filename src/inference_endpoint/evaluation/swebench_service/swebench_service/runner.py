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

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

import msgspec.json
import yaml

from .artifacts import atomic_write_bytes, redact_secrets, redact_text
from .schemas import RunRequest, TemplateName

logger = logging.getLogger(__name__)


class RunnerError(RuntimeError):
    pass


class RunCancelled(RunnerError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            process = self._process
        if process is not None:
            _terminate_process(process)

    def attach(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._process = process
            cancelled = self._event.is_set()
        if cancelled:
            _terminate_process(process)

    def detach(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            if self._process is process:
                self._process = None


TEMPLATE_FILES: dict[TemplateName, str] = {
    "default": "swebench_template.yaml",
    "qwen_tools": "swebench_qwen_tools_template.yaml",
}

_LOG_TAIL_MAX_BYTES = 64 * 1024
_LOG_TAIL_MAX_LINES = 50
_RUN_LABEL = "com.mlcommons.endpoints.swebench-run"
_PROCESS_TERMINATE_TIMEOUT_S = 10
_PYXIS_PRINT_LOCK = threading.Lock()
_PYXIS_EVAL_SCRIPT = r"""set -eu

patch_path=$1
eval_path=$2
output_path=$3
timeout_s=$4

cd /testbed
if git apply --verbose "$patch_path" || \
    git apply --verbose --reject "$patch_path" || \
    patch --batch --fuzz=5 -p1 -i "$patch_path"; then
    echo ">>>>> Applied Patch"
else
    echo ">>>>> Patch Apply Failed"
    exit 1
fi

set +e
timeout "$timeout_s" /bin/bash "$eval_path" >"$output_path" 2>&1
status=$?
set -e
cat "$output_path"
if [[ $status -eq 124 ]]; then
    echo "Timeout error: $timeout_s seconds exceeded." >>"$output_path"
    exit 124
fi
exit 0
"""


def _normalize_endpoint_base(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    hostname = parsed.hostname or ""
    if hostname == "localhost":
        hostname = "127.0.0.1"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunparse(
        parsed._replace(netloc=netloc, path=path, params="", query="", fragment="")
    )


def _exact_instance_filter(instance_ids: list[str]) -> str:
    return (
        "^(?:" + "|".join(re.escape(instance_id) for instance_id in instance_ids) + ")$"
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate the local process group; containers are cleaned separately."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("SWE-bench subprocess did not exit after SIGKILL")


def _run_subprocess(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
    cancel_token: CancellationToken | None = None,
) -> None:
    if cancel_token is not None and cancel_token.is_cancelled():
        raise RunCancelled(f"subprocess cancelled before start: {cmd}")
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(cwd),
                env=env,
                start_new_session=os.name != "nt",
            )
            if cancel_token is not None:
                cancel_token.attach(process)
            deadline = time.monotonic() + timeout_s
            while True:
                if cancel_token is not None and cancel_token.is_cancelled():
                    _terminate_process(process)
                    raise RunCancelled(f"subprocess cancelled: {cmd}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_process(process)
                    raise RunnerError(f"subprocess timed out after {timeout_s}s: {cmd}")
                try:
                    process.communicate(timeout=min(0.5, remaining))
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise RunCancelled(f"subprocess cancelled: {cmd}")
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        if process is not None and cancel_token is not None:
            cancel_token.detach(process)

    if process.returncode != 0:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - _LOG_TAIL_MAX_BYTES))
            tail_bytes = log_file.read()
        tail = "\n".join(
            tail_bytes.decode("utf-8", errors="replace").splitlines()[
                -_LOG_TAIL_MAX_LINES:
            ]
        )
        raise RunnerError(
            f"subprocess exited with code {process.returncode}: {cmd}\n{tail}"
        )


class RunnerProtocol(Protocol):
    """Structural interface used by the service to execute a SWE-bench run."""

    def run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]: ...


class SweBenchRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        subprocess_timeout_s: int,
    ):
        self.project_root = project_root.resolve()
        self.subprocess_timeout_s = subprocess_timeout_s

    def run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        try:
            return self._run(request, run_dir, cancel_token)
        finally:
            try:
                cleanup_kwargs: dict[str, Any] = {}
                eval_run_id_path = run_dir / "swe_bench_eval_run_id.txt"
                if eval_run_id_path.exists():
                    eval_run_id = eval_run_id_path.read_text().strip()
                    if eval_run_id:
                        cleanup_kwargs = {
                            "eval_run_id": eval_run_id,
                            "instance_ids": request.evaluated_instance_ids,
                        }
                self._cleanup_containers(run_dir.name, **cleanup_kwargs)
            except Exception:
                logger.warning(
                    "Could not clean up SWE-bench containers for run %s",
                    run_dir.name,
                    exc_info=True,
                )

    def _run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        secret_values = (
            {request.endpoint_api_key} if request.endpoint_api_key else set()
        )
        (run_dir / "request.json").write_bytes(
            msgspec.json.encode(
                redact_secrets(request.model_dump(), secret_values=secret_values)
            )
        )

        output_dir = run_dir / "swe_bench_output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        with tempfile.TemporaryDirectory(prefix="swebench_config_") as config_tmp:
            patched_config = self._patch_config(
                Path(config_tmp),
                request,
                run_id=run_dir.name,
            )
            self._run_agent(
                request,
                patched_config,
                output_dir,
                run_dir,
                secret_values,
                cancel_token,
            )

        preds_path = output_dir / "preds.json"
        if not preds_path.exists():
            raise RunnerError("mini-extra did not produce preds.json")
        self._validate_prediction_ids(request, preds_path)
        shutil.copy2(preds_path, run_dir / "preds.json")

        result_path = self._run_eval(
            request, preds_path, output_dir, run_dir, secret_values, cancel_token
        )
        shutil.copy2(result_path, run_dir / "swe_bench_results.json")
        return msgspec.json.decode(result_path.read_bytes(), type=dict)

    def _load_template(self, request: RunRequest) -> dict[str, Any]:
        template_path = self._template_dir / TEMPLATE_FILES[request.template]
        with template_path.open() as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise RunnerError("swebench template must be a YAML mapping")
        model_cfg = loaded.get("model")
        if not isinstance(model_cfg, dict):
            raise RunnerError("swebench template must define model")
        if not isinstance(model_cfg.get("model_kwargs"), dict):
            raise RunnerError("swebench template must define model.model_kwargs")
        return loaded

    @property
    def _template_dir(self) -> Path:
        return Path(__file__).resolve().parent / "templates"

    def _patch_config(
        self, config_dir: Path, request: RunRequest, *, run_id: str
    ) -> Path:
        cfg = self._load_template(request)
        model_cfg = cfg["model"]
        model_kwargs = model_cfg["model_kwargs"]

        model_cfg["model_name"] = request.model_name
        if request.template == "qwen_tools":
            model_cfg["model_class"] = (
                "swebench_service.qwen_tools_model.QwenToolsModel"
            )
        else:
            model_cfg.pop("model_class", None)
        if request.endpoint_urls:
            base = _normalize_endpoint_base(str(request.endpoint_urls[0]))
            model_kwargs["api_base"] = base + "/v1"
        else:
            base = ""
            model_kwargs["api_base"] = ""

        model_kwargs.pop("api_key", None)

        for field in (
            "temperature",
            "seed",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            val = request.generation_params.get(field)
            if val is not None:
                model_kwargs[field] = val
            else:
                model_kwargs.pop(field, None)

        if (
            max_new_tokens := request.generation_params.get("max_new_tokens")
        ) is not None:
            model_kwargs["max_tokens"] = max_new_tokens
        else:
            model_kwargs.pop("max_tokens", None)

        if (
            chat_tmpl := request.generation_params.get("chat_template_kwargs")
        ) is not None:
            model_kwargs["chat_template_kwargs"] = chat_tmpl
        else:
            model_kwargs.pop("chat_template_kwargs", None)

        environment_cfg = cfg.get("environment")
        if not isinstance(environment_cfg, dict):
            raise RunnerError("swebench template must define environment")
        self._configure_environment(environment_cfg, run_id)

        config_dir.mkdir(parents=True, exist_ok=True)
        patched_path = config_dir / "swebench_patched.yaml"
        with patched_path.open("w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        return patched_path

    def _configure_environment(
        self, environment_cfg: dict[str, Any], run_id: str
    ) -> None:
        run_args: list[str] = []
        if docker_runtime := os.environ.get("SWEBENCH_DOCKER_RUNTIME", "").strip():
            run_args.extend(["--runtime", docker_runtime])
        run_args.extend(
            [
                "--rm",
                "--label",
                f"{_RUN_LABEL}={run_id}",
            ]
        )
        environment_cfg["run_args"] = run_args

    def _run_agent(
        self,
        request: RunRequest,
        patched_config: Path,
        output_dir: Path,
        run_dir: Path,
        secret_values: set[str],
        cancel_token: CancellationToken | None = None,
    ) -> None:
        instance_filter = _exact_instance_filter(request.evaluated_instance_ids)
        cmd = [
            "mini-extra",
            "swebench",
            "--model",
            request.model_name,
            "--config",
            str(patched_config),
            "--subset",
            request.subset,
            "--split",
            request.split,
            "--filter",
            instance_filter,
            "--workers",
            str(request.workers),
            "--output",
            str(output_dir),
        ]
        self._run_logged_subprocess(
            cmd,
            run_dir / "swe_bench_agent.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=self._base_env(request),
            secret_values=secret_values,
            cancel_token=cancel_token,
        )

    @staticmethod
    def _run_logged_subprocess(
        cmd: list[str],
        public_log_path: Path,
        *,
        cwd: Path,
        timeout_s: int,
        env: dict[str, str],
        secret_values: set[str],
        cancel_token: CancellationToken | None,
    ) -> None:
        raw_log_path = public_log_path.with_name(f".{public_log_path.name}.raw")
        try:
            _run_subprocess(
                cmd,
                raw_log_path,
                cwd=cwd,
                timeout_s=timeout_s,
                env=env,
                cancel_token=cancel_token,
            )
        finally:
            try:
                if raw_log_path.exists():
                    atomic_write_bytes(
                        public_log_path,
                        redact_text(
                            raw_log_path.read_text(errors="replace"), secret_values
                        ).encode(),
                    )
            finally:
                raw_log_path.unlink(missing_ok=True)

    def _base_env(self, request: RunRequest) -> dict[str, str]:
        env = dict(os.environ)
        no_proxy = {"127.0.0.1", "localhost"}
        for endpoint in request.endpoint_urls:
            host = urlparse(str(endpoint)).hostname
            if host:
                no_proxy.add(host)
        existing = env.get("NO_PROXY") or env.get("no_proxy")
        if existing:
            no_proxy.update(
                part.strip() for part in existing.split(",") if part.strip()
            )
        no_proxy_value = ",".join(sorted(no_proxy))
        env["NO_PROXY"] = no_proxy_value
        env["no_proxy"] = no_proxy_value
        endpoint_host = (
            urlparse(str(request.endpoint_urls[0])).hostname
            if request.endpoint_urls
            else None
        )
        if request.endpoint_api_key:
            env["OPENAI_API_KEY"] = request.endpoint_api_key
        elif endpoint_host in {"localhost", "127.0.0.1", "::1"}:
            env["OPENAI_API_KEY"] = "EMPTY"
        else:
            env.pop("OPENAI_API_KEY", None)
        return env

    def _cleanup_containers(
        self,
        run_id: str,
        *,
        eval_run_id: str | None = None,
        instance_ids: list[str] | None = None,
    ) -> None:
        docker = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
        label_filter = f"label={_RUN_LABEL}={run_id}"
        try:
            listed = subprocess.run(
                [docker, "ps", "-aq", "--filter", label_filter],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            container_ids = listed.stdout.split()
            if eval_run_id is not None:
                expected_names = {
                    f"sweb.eval.{instance_id.lower()}.{eval_run_id}"
                    for instance_id in instance_ids or []
                }
                listed_eval = subprocess.run(
                    [
                        docker,
                        "ps",
                        "-a",
                        "--filter",
                        f"name={eval_run_id}",
                        "--format",
                        "{{.ID}}\t{{.Names}}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                for line in listed_eval.stdout.splitlines():
                    container_id, separator, container_name = line.partition("\t")
                    if (
                        separator
                        and container_id
                        and container_name in expected_names
                        and container_id not in container_ids
                    ):
                        container_ids.append(container_id)
            if container_ids:
                subprocess.run(
                    [docker, "rm", "-f", *container_ids],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RunnerError(
                f"failed to clean up Docker containers for SWE-bench run {run_id}"
            ) from exc

    def _validate_prediction_ids(self, request: RunRequest, preds_path: Path) -> None:
        try:
            preds = msgspec.json.decode(preds_path.read_bytes(), type=dict)
        except msgspec.DecodeError as exc:
            raise RunnerError("mini-extra produced invalid preds.json") from exc
        expected = set(request.evaluated_instance_ids)
        actual = {str(instance_id) for instance_id in preds}
        unexpected = sorted(actual - expected)
        if unexpected:
            raise RunnerError(
                "mini-extra produced predictions for unexpected SWE-bench "
                f"instances: {', '.join(unexpected[:10])}"
            )
        missing = sorted(expected - actual)
        if missing:
            logger.warning(
                "mini-extra omitted predictions for %d expected SWE-bench "
                "instances: %s",
                len(missing),
                ", ".join(missing[:10]),
            )

    def _run_eval(
        self,
        request: RunRequest,
        preds_path: Path,
        output_dir: Path,
        run_dir: Path,
        secret_values: set[str],
        cancel_token: CancellationToken | None = None,
    ) -> Path:
        run_id = f"endpoints_{uuid.uuid4().hex[:8]}"
        (run_dir / "swe_bench_eval_run_id.txt").write_text(run_id)
        dataset_name = {
            "verified": "princeton-nlp/SWE-bench_Verified",
            "lite": "princeton-nlp/SWE-bench_Lite",
        }.get(request.subset)
        if dataset_name is None:
            raise RunnerError(f"unknown SWE-bench subset: {request.subset}")
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            request.split,
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            str(request.max_eval_workers),
            "--run_id",
            run_id,
            "--instance_ids",
            *request.evaluated_instance_ids,
        ]
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        self._run_logged_subprocess(
            cmd,
            run_dir / "swe_bench_eval.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=env,
            secret_values=secret_values,
            cancel_token=cancel_token,
        )
        safe_model = request.model_name.replace("/", "__")
        result_path = output_dir / f"{safe_model}.{run_id}.json"
        if result_path.exists():
            return result_path
        candidates = sorted(output_dir.rglob(f"*{run_id}*.json"))
        if not candidates:
            raise RunnerError(f"SWE-bench result file not found for run_id={run_id}")
        if len(candidates) > 1:
            raise RunnerError(
                f"multiple SWE-bench result files found for run_id={run_id}"
            )
        return candidates[0]


class PyxisSweBenchRunner(SweBenchRunner):
    def __init__(
        self,
        *,
        project_root: Path,
        subprocess_timeout_s: int,
        image_registry: str,
    ):
        super().__init__(
            project_root=project_root,
            subprocess_timeout_s=subprocess_timeout_s,
        )
        self.image_registry = image_registry

    def _configure_environment(
        self, environment_cfg: dict[str, Any], run_id: str
    ) -> None:
        for key in ("run_args", "pull_timeout", "container_timeout"):
            environment_cfg.pop(key, None)
        environment_cfg["environment_class"] = (
            "swebench_service.pyxis_environment.PyxisEnvironment"
        )
        environment_cfg["run_id"] = run_id

    def _run_agent(
        self,
        request: RunRequest,
        patched_config: Path,
        output_dir: Path,
        run_dir: Path,
        secret_values: set[str],
        cancel_token: CancellationToken | None = None,
    ) -> None:
        command = [
            sys.executable,
            "-m",
            "swebench_service.runner",
            "pyxis-agent",
            "--model",
            request.model_name,
            "--config",
            str(patched_config),
            "--subset",
            request.subset,
            "--split",
            request.split,
            "--filter",
            _exact_instance_filter(request.evaluated_instance_ids),
            "--workers",
            str(request.workers),
            "--output",
            str(output_dir),
            "--image-registry",
            self.image_registry,
        ]
        self._run_logged_subprocess(
            command,
            run_dir / "swe_bench_agent.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=self._base_env(request),
            secret_values=secret_values,
            cancel_token=cancel_token,
        )

    def _run_eval(
        self,
        request: RunRequest,
        preds_path: Path,
        output_dir: Path,
        run_dir: Path,
        secret_values: set[str],
        cancel_token: CancellationToken | None = None,
    ) -> Path:
        run_id = f"endpoints_{uuid.uuid4().hex[:8]}"
        (run_dir / "swe_bench_eval_run_id.txt").write_text(run_id)
        dataset_name = {
            "verified": "princeton-nlp/SWE-bench_Verified",
            "lite": "princeton-nlp/SWE-bench_Lite",
        }.get(request.subset)
        if dataset_name is None:
            raise RunnerError(f"unknown SWE-bench subset: {request.subset}")
        command = [
            sys.executable,
            "-m",
            "swebench_service.runner",
            "pyxis-eval",
            "--dataset-name",
            dataset_name,
            "--split",
            request.split,
            "--predictions-path",
            str(preds_path),
            "--max-workers",
            str(request.max_eval_workers),
            "--run-id",
            run_id,
            "--image-registry",
            self.image_registry,
            "--output-dir",
            str(output_dir),
            "--instance-ids",
            *request.evaluated_instance_ids,
        ]
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        self._run_logged_subprocess(
            command,
            run_dir / "swe_bench_eval.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=env,
            secret_values=secret_values,
            cancel_token=cancel_token,
        )
        result_path = (
            output_dir / f"{request.model_name.replace('/', '__')}.{run_id}.json"
        )
        if not result_path.exists():
            raise RunnerError(f"SWE-bench result file not found for run_id={run_id}")
        return result_path

    def _cleanup_containers(
        self,
        run_id: str,
        *,
        eval_run_id: str | None = None,
        instance_ids: list[str] | None = None,
    ) -> None:
        from .pyxis_environment import build_host_srun_command, safe_srun_env

        del eval_run_id, instance_ids
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)[:24]
        prefix = f"pyxis_mswe_{safe_run_id}_"
        try:
            listed = subprocess.run(
                build_host_srun_command(["enroot", "list", "-f"]),
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
                env=safe_srun_env(),
            )
            for line in listed.stdout.splitlines():
                fields = line.split(maxsplit=1)
                name = fields[0] if fields else ""
                if name.startswith(prefix):
                    subprocess.run(
                        build_host_srun_command(["enroot", "remove", "-f", name]),
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=safe_srun_env(),
                    )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RunnerError(
                f"failed to clean up Pyxis containers for SWE-bench run {run_id}"
            ) from exc


def _enable_live_trajectory_saves(swebench: Any, output_dir: Path) -> None:
    base_agent = swebench.ProgressTrackingAgent

    class LiveTrajectoryAgent(base_agent):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, instance_id: str = "", **kwargs: Any):
            kwargs["output_path"] = (
                output_dir / instance_id / f"{instance_id}.live.json"
            )
            super().__init__(*args, instance_id=instance_id, **kwargs)

    swebench.ProgressTrackingAgent = LiveTrajectoryAgent


def _pyxis_agent_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--filter", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-registry", required=True)
    args = parser.parse_args(argv)

    from minisweagent.environments import get_environment
    from minisweagent.run.benchmarks import swebench

    from .pyxis_environment import resolve_image

    def get_pyxis_environment(config: dict, instance: dict):
        environment_config = copy.deepcopy(config.get("environment", {}))
        environment_config["image"] = resolve_image(
            args.image_registry, instance["instance_id"]
        )
        return get_environment(environment_config)

    swebench.get_sb_environment = get_pyxis_environment
    _enable_live_trajectory_saves(swebench, args.output)
    swebench.main(
        subset=args.subset,
        split=args.split,
        slice_spec="",
        filter_spec=args.filter,
        shuffle=False,
        output=str(args.output),
        workers=args.workers,
        model=args.model,
        model_class=None,
        redo_existing=False,
        config_spec=[str(args.config)],
        environment_class=("swebench_service.pyxis_environment.PyxisEnvironment"),
    )


def _write_pyxis_run_report(
    *,
    output_dir: Path,
    run_id: str,
    instance_ids: list[str],
    predictions: dict[str, dict[str, Any]],
) -> Path:
    from swebench.harness.reporting import make_run_report

    output_dir = output_dir.resolve()
    with contextlib.chdir(output_dir):
        result_path = make_run_report(
            predictions,
            [{"instance_id": instance_id} for instance_id in instance_ids],
            run_id,
            client=None,
        )
    return output_dir / result_path


def _evaluate_pyxis_instance(
    *,
    test_spec: Any,
    prediction: dict[str, Any],
    image: str | Path,
    output_dir: Path,
    run_id: str,
    timeout_s: int,
) -> None:
    from .pyxis_environment import build_srun_command, safe_srun_env

    instance_id = test_spec.instance_id
    safe_model = prediction["model_name_or_path"].replace("/", "__")
    log_dir = output_dir / "logs" / "run_evaluation" / run_id / safe_model / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    patch_path = log_dir / "patch.diff"
    eval_path = log_dir / "eval.sh"
    output_path = log_dir / "test_output.txt"
    report_path = log_dir / "report.json"
    patch_path.write_text(prediction["model_patch"])
    eval_path.write_text(test_spec.eval_script)
    output_path.write_text("")
    report_path.unlink(missing_ok=True)

    command = build_srun_command(
        image=image,
        name=None,
        mounts=[
            (patch_path, "/tmp/swebench_patch.diff"),
            (eval_path, "/tmp/swebench_eval.sh"),
            (output_path, "/tmp/swebench_test_output.txt"),
        ],
        workdir="/testbed",
        argv=[
            "bash",
            "-c",
            _PYXIS_EVAL_SCRIPT,
            "pyxis-eval",
            "/tmp/swebench_patch.diff",
            "/tmp/swebench_eval.sh",
            "/tmp/swebench_test_output.txt",
            str(timeout_s),
        ],
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s + 60,
        env=safe_srun_env(),
    )
    with _PYXIS_PRINT_LOCK:
        print(f"[{instance_id}]\n{result.stdout}{result.stderr}", flush=True)
    if result.returncode != 0:
        return

    from swebench.harness.grading import get_eval_report

    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=output_path,
        include_tests_status=True,
    )
    atomic_write_bytes(report_path, (json.dumps(report, indent=4) + "\n").encode())


def _pyxis_eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image-registry", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--instance-ids", nargs="+", required=True)
    args = parser.parse_args(argv)

    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.utils import (
        get_predictions_from_file,
        load_swebench_dataset,
    )

    from .pyxis_environment import resolve_image

    predictions = {
        prediction["instance_id"]: prediction
        for prediction in get_predictions_from_file(
            str(args.predictions_path), args.dataset_name, args.split
        )
        if prediction["instance_id"] in args.instance_ids
    }
    rows = load_swebench_dataset(args.dataset_name, args.split, args.instance_ids)
    images = {
        instance_id: resolve_image(args.image_registry, instance_id)
        for instance_id in args.instance_ids
    }
    payloads = []
    for row in rows:
        instance_id = row["instance_id"]
        prediction = predictions.get(instance_id)
        if prediction is None or prediction.get("model_patch") in {"", None}:
            continue
        payloads.append(
            {
                "test_spec": make_test_spec(row, arch="arm64"),
                "prediction": prediction,
                "image": images[instance_id],
                "output_dir": args.output_dir,
                "run_id": args.run_id,
                "timeout_s": args.timeout,
            }
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [
            executor.submit(_evaluate_pyxis_instance, **payload) for payload in payloads
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with _PYXIS_PRINT_LOCK:
                    print(f"Pyxis evaluation failed: {exc}", flush=True)

    _write_pyxis_run_report(
        output_dir=args.output_dir,
        run_id=args.run_id,
        instance_ids=args.instance_ids,
        predictions=predictions,
    )


def create_runner(
    runtime: str,
    *,
    project_root: Path,
    subprocess_timeout_s: int,
    image_registry: str | None,
) -> RunnerProtocol:
    if runtime == "docker":
        return SweBenchRunner(
            project_root=project_root,
            subprocess_timeout_s=subprocess_timeout_s,
        )
    if runtime == "pyxis":
        if image_registry is None:
            raise ValueError("Pyxis runtime requires an image registry")
        return PyxisSweBenchRunner(
            project_root=project_root,
            subprocess_timeout_s=subprocess_timeout_s,
            image_registry=image_registry,
        )
    raise ValueError(f"unknown SWE-bench runtime: {runtime}")


def _internal_main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("expected internal command: pyxis-agent or pyxis-eval")
    command, *command_args = argv
    if command == "pyxis-agent":
        _pyxis_agent_main(command_args)
    elif command == "pyxis-eval":
        _pyxis_eval_main(command_args)
    else:
        raise SystemExit(f"unknown internal command: {command}")


if __name__ == "__main__":
    _internal_main()
