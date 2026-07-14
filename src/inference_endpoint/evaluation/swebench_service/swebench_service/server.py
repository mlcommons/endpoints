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

import asyncio
import inspect
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import msgspec.json
import yaml
from aiohttp import web
from pydantic import ValidationError

from . import API_VERSION, CAPABILITIES
from .artifacts import redact_secrets, redact_text, resolve_artifact
from .config import ServiceConfig
from .runner import CancellationToken, RunCancelled, SwebenchRunner
from .schemas import ArtifactInfo, RunRequest, RunStatus


class RunManager:
    def __init__(self, *, config: ServiceConfig, runner: Any):
        self.config = config
        self.runner = runner
        self.runs: dict[str, RunStatus] = {}
        self._requests: dict[str, RunRequest] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_tokens: dict[str, CancellationToken] = {}
        self._secret_values: dict[str, set[str]] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_runs)
        self._submit_lock = asyncio.Lock()

    def active_count(self) -> int:
        return sum(
            1 for run in self.runs.values() if run.status in {"queued", "running"}
        )

    async def submit(self, request: RunRequest) -> RunStatus:
        async with self._submit_lock:
            if self.active_count() >= self.config.max_concurrent_runs:
                raise web.HTTPTooManyRequests(text="too many active SWE-bench runs")
            run_id = uuid.uuid4().hex
            now = time.time()
            status = RunStatus(
                run_id=run_id,
                status="queued",
                created_at=now,
                updated_at=now,
                phase="queued",
                agent_total=len(request.evaluated_instance_ids),
                agent_completed=0,
                eval_total=0,
                eval_completed=0,
                message="queued",
            )
            self.runs[run_id] = status
            self._requests[run_id] = request
            self._cancel_tokens[run_id] = CancellationToken()
            self._secret_values[run_id] = self._secrets_for_request(request)
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "request.json").write_bytes(
            msgspec.json.encode(
                redact_secrets(
                    request.model_dump(mode="json"),
                    secret_values=self._secret_values[run_id],
                )
            )
        )
        self._write_status(status)
        self._tasks[run_id] = asyncio.create_task(
            self._execute(run_id, request, run_dir)
        )
        return status

    async def _execute(self, run_id: str, request: RunRequest, run_dir: Path) -> None:
        token = self._cancel_tokens[run_id]
        try:
            async with self._semaphore:
                if token.is_cancelled():
                    self._update(run_id, status="cancelled", phase="cancelled")
                    return
                self._update(run_id, status="running", phase="agent", message="agent")
                try:
                    result = await asyncio.to_thread(
                        self._run_runner, request, run_dir, token
                    )
                except RunCancelled:
                    self._update(run_id, status="cancelled", phase="cancelled")
                    return
                except Exception as exc:
                    if token.is_cancelled():
                        self._update(run_id, status="cancelled", phase="cancelled")
                        return
                    self._update(
                        run_id,
                        status="failed",
                        phase="failed",
                        error=redact_text(
                            str(exc), self._secret_values.get(run_id, set())
                        ),
                    )
                    return
                if token.is_cancelled():
                    self._update(run_id, status="cancelled", phase="cancelled")
                    return
                self._refresh_progress(run_id)
                final_progress = self._terminal_progress(run_id, "succeeded")
                artifacts = [
                    ArtifactInfo(name=name, url=f"/v1/runs/{run_id}/artifacts/{name}")
                    for name in (
                        "preds.json",
                        "swe_bench_agent.log",
                        "swe_bench_eval.log",
                        "swe_bench_results.json",
                        "status.json",
                    )
                    if (run_dir / name).exists()
                ]
                self._update(
                    run_id,
                    status="succeeded",
                    result=result,
                    artifacts=artifacts,
                    **final_progress,
                )
        finally:
            self._tasks.pop(run_id, None)
            self._prune_completed_runs()

    def _run_runner(
        self, request: RunRequest, run_dir: Path, token: CancellationToken
    ) -> dict[str, Any]:
        try:
            signature = inspect.signature(self.runner.run)
        except (TypeError, ValueError):
            return self.runner.run(request, run_dir)
        if "cancel_token" not in signature.parameters:
            return self.runner.run(request, run_dir)
        return self.runner.run(request, run_dir, cancel_token=token)

    def cancel(self, run_id: str) -> RunStatus:
        status = self.get(run_id)
        if status.status not in {"queued", "running"}:
            return status
        token = self._cancel_tokens.get(run_id)
        if token is not None:
            token.cancel()
        self._update(
            run_id, status="cancelled", **self._terminal_progress(run_id, "cancelled")
        )
        return self.runs[run_id]

    async def cancel_all_active(self) -> None:
        active = [
            run.run_id
            for run in self.runs.values()
            if run.status in {"queued", "running"}
        ]
        for run_id in active:
            self.cancel(run_id)
        tasks = [self._tasks[run_id] for run_id in active if run_id in self._tasks]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _secrets_for_request(self, request: RunRequest) -> set[str]:
        secrets: set[str] = set()
        if request.endpoint_api_key:
            secrets.add(request.endpoint_api_key)
        if self.config.auth_token:
            secrets.add(self.config.auth_token)
        return secrets

    def _update(self, run_id: str, **updates: Any) -> None:
        current = self.runs[run_id]
        data = current.model_dump()
        data.update(updates)
        data["updated_at"] = time.time()
        updated = RunStatus.model_validate(data)
        self.runs[run_id] = updated
        self._write_status(updated)

    def _write_status(self, status: RunStatus) -> None:
        run_dir = self.run_dir(status.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        secret_values = self._secret_values.get(status.run_id, set())
        (run_dir / "status.json").write_bytes(
            msgspec.json.encode(
                redact_secrets(
                    status.model_dump(mode="json"), secret_values=secret_values
                )
            )
        )

    def get(self, run_id: str) -> RunStatus:
        try:
            self._refresh_progress(run_id)
            return self.runs[run_id]
        except KeyError as exc:
            raise web.HTTPNotFound(text="unknown run_id") from exc

    def _refresh_progress(self, run_id: str) -> None:
        status = self.runs[run_id]
        if status.status in {"failed", "cancelled"}:
            progress = self._terminal_progress(run_id, status.status)
        elif status.status == "succeeded":
            progress = self._terminal_progress(run_id, "succeeded")
        else:
            progress = self._derived_progress(run_id)

        current = status.model_dump()
        changed = any(current.get(key) != value for key, value in progress.items())
        if not changed:
            return
        current.update(progress)
        current["updated_at"] = time.time()
        updated = RunStatus.model_validate(current)
        self.runs[run_id] = updated
        self._write_status(updated)

    def _derived_progress(self, run_id: str) -> dict[str, Any]:
        request = self._requests.get(run_id)
        run_dir = self.run_dir(run_id)
        output_dir = run_dir / "swe_bench_output"
        agent_total = len(request.evaluated_instance_ids) if request else 0
        agent_completed = min(self._count_agent_completed(output_dir), agent_total)

        eval_run_id_path = run_dir / "swe_bench_eval_run_id.txt"
        eval_total = self._count_predictions(output_dir / "preds.json") or agent_total
        eval_completed = 0
        if eval_run_id_path.exists():
            phase = "eval"
            eval_run_id = eval_run_id_path.read_text().strip()
            eval_completed = min(
                self._count_eval_completed(output_dir, eval_run_id),
                eval_total,
            )
            message = "eval"
        else:
            phase = "queued" if self.runs[run_id].status == "queued" else "agent"
            eval_total = 0
            message = phase

        return {
            "phase": phase,
            "agent_total": agent_total,
            "agent_completed": agent_completed,
            "eval_total": eval_total,
            "eval_completed": eval_completed,
            "message": message,
        }

    def _terminal_progress(self, run_id: str, phase: str) -> dict[str, Any]:
        progress = self._derived_progress(run_id)
        if phase == "succeeded":
            agent_total = progress["agent_total"] or 0
            eval_total = progress["eval_total"] or self._count_predictions(
                self.run_dir(run_id) / "swe_bench_output" / "preds.json"
            )
            if not eval_total:
                eval_total = agent_total
            progress["agent_completed"] = agent_total
            progress["eval_total"] = eval_total
            progress["eval_completed"] = eval_total
        progress["phase"] = phase
        progress["message"] = phase
        return progress

    @staticmethod
    def _count_agent_completed(output_dir: Path) -> int:
        exit_statuses = sorted(
            output_dir.glob("exit_statuses_*.yaml"),
            key=lambda path: path.stat().st_mtime,
        )
        if exit_statuses:
            try:
                loaded = yaml.safe_load(exit_statuses[-1].read_text()) or {}
            except (OSError, yaml.YAMLError):
                loaded = {}
            instances_by_status = loaded.get("instances_by_exit_status")
            if isinstance(instances_by_status, dict):
                completed: set[str] = set()
                for instance_ids in instances_by_status.values():
                    if isinstance(instance_ids, list):
                        completed.update(
                            str(instance_id) for instance_id in instance_ids
                        )
                return len(completed)

        traj_ids = {
            path.name.removesuffix(".traj.json")
            for path in output_dir.glob("*/*.traj.json")
        }
        if traj_ids:
            return len(traj_ids)
        return RunManager._count_predictions(output_dir / "preds.json")

    @staticmethod
    def _count_predictions(preds_path: Path) -> int:
        if not preds_path.exists():
            return 0
        try:
            loaded = msgspec.json.decode(preds_path.read_bytes(), type=dict)
        except (OSError, msgspec.DecodeError):
            return 0
        return len(loaded)

    @staticmethod
    def _count_eval_completed(output_dir: Path, eval_run_id: str) -> int:
        if not eval_run_id:
            return 0
        run_root = output_dir / "logs" / "run_evaluation" / eval_run_id
        if not run_root.exists():
            return 0
        return sum(1 for _ in run_root.rglob("report.json"))

    def run_dir(self, run_id: str) -> Path:
        return self.config.artifact_root / run_id

    def _prune_completed_runs(self) -> None:
        limit = self.config.max_stored_runs
        if limit <= 0:
            return
        completed = [
            run for run in self.runs.values() if run.status not in {"queued", "running"}
        ]
        overflow = len(completed) - limit
        if overflow <= 0:
            return
        for run in sorted(completed, key=lambda item: item.updated_at)[:overflow]:
            self.runs.pop(run.run_id, None)
            self._requests.pop(run.run_id, None)
            self._cancel_tokens.pop(run.run_id, None)
            self._secret_values.pop(run.run_id, None)
            shutil.rmtree(self.run_dir(run.run_id), ignore_errors=True)


MANAGER_KEY = web.AppKey("manager", RunManager)


def create_app(config: ServiceConfig, runner: Any | None = None) -> web.Application:
    config = ServiceConfig(
        host=config.host,
        port=config.port,
        artifact_root=config.artifact_root.expanduser().resolve(),
        max_concurrent_runs=config.max_concurrent_runs,
        subprocess_timeout_s=config.subprocess_timeout_s,
        auth_token=config.auth_token,
        max_stored_runs=config.max_stored_runs,
    )
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    runner = runner or SwebenchRunner(
        project_root=Path(__file__).resolve().parents[1],
        subprocess_timeout_s=config.subprocess_timeout_s,
    )
    manager = RunManager(config=config, runner=runner)

    @web.middleware
    async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        if config.auth_token and request.headers.get("Authorization") != (
            f"Bearer {config.auth_token}"
        ):
            raise web.HTTPUnauthorized(text="unauthorized")
        return await handler(request)

    app = web.Application(middlewares=[auth_middleware])
    app[MANAGER_KEY] = manager

    async def shutdown_active_runs(app: web.Application) -> None:
        await app[MANAGER_KEY].cancel_all_active()

    app.on_shutdown.append(shutdown_active_runs)

    async def health(request: web.Request) -> web.Response:
        return web.json_response(
            {"api_version": API_VERSION, "capabilities": CAPABILITIES, "status": "ok"}
        )

    async def post_run(request: web.Request) -> web.Response:
        try:
            data = await request.json()
            run_request = RunRequest.model_validate(data)
        except (ValidationError, ValueError) as exc:
            raise web.HTTPBadRequest(text="invalid run request") from exc
        status = await manager.submit(run_request)
        return web.json_response(
            redact_secrets(
                status.model_dump(mode="json"),
                secret_values=manager._secret_values.get(status.run_id, set()),
            ),
            status=202,
        )

    async def get_run(request: web.Request) -> web.Response:
        status = manager.get(request.match_info["run_id"])
        return web.json_response(
            redact_secrets(
                status.model_dump(mode="json"),
                secret_values=manager._secret_values.get(status.run_id, set()),
            )
        )

    async def cancel_run(request: web.Request) -> web.Response:
        status = manager.cancel(request.match_info["run_id"])
        return web.json_response(
            redact_secrets(
                status.model_dump(mode="json"),
                secret_values=manager._secret_values.get(status.run_id, set()),
            )
        )

    async def get_artifact(request: web.Request) -> web.FileResponse:
        run_id = request.match_info["run_id"]
        name = request.match_info["name"]
        manager.get(run_id)
        try:
            path = resolve_artifact(manager.run_dir(run_id), name)
        except FileNotFoundError as exc:
            raise web.HTTPNotFound(text="artifact not found") from exc
        return web.FileResponse(path)

    app.router.add_get("/health", health)
    app.router.add_post("/v1/runs", post_run)
    app.router.add_get("/v1/runs/{run_id}", get_run)
    app.router.add_post("/v1/runs/{run_id}/cancel", cancel_run)
    app.router.add_get("/v1/runs/{run_id}/artifacts/{name}", get_artifact)
    return app
