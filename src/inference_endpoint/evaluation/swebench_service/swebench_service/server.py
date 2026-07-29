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
import hmac
import logging
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec.json
import yaml
from aiohttp import web
from pydantic import ValidationError

from . import API_VERSION, CAPABILITIES
from .artifacts import (
    atomic_write_bytes,
    redact_secrets,
    redact_text,
    resolve_artifact,
)
from .config import ServiceConfig
from .runner import CancellationToken, RunCancelled, RunnerProtocol, SweBenchRunner
from .schemas import ArtifactInfo, RunRequest, RunStatus

_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
_PROGRESS_REFRESH_INTERVAL_S = 1.0
_ARTIFACT_RETENTION_GRACE_S = 5 * 60
_PRUNE_RETRY_INTERVAL_S = 1.0
_LOG_ARTIFACT_NAMES = frozenset({"swe_bench_agent.log", "swe_bench_eval.log"})
logger = logging.getLogger(__name__)


class _LeasedFileResponse(web.FileResponse):
    def __init__(self, path: Path, release: Callable[[], None]):
        super().__init__(path)
        self._release = release

    async def prepare(self, request: web.Request) -> Any:
        try:
            return await super().prepare(request)
        finally:
            release, self._release = self._release, lambda: None
            release()


class RunManager:
    def __init__(self, *, config: ServiceConfig, runner: RunnerProtocol):
        self.config = config
        self.runner = runner
        self.runs: dict[str, RunStatus] = {}
        self._requests: dict[str, RunRequest] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_tokens: dict[str, CancellationToken] = {}
        self._secret_values: dict[str, set[str]] = {}
        self._progress_refresh_locks: dict[str, asyncio.Lock] = {}
        self._status_persist_locks: dict[str, asyncio.Lock] = {}
        self._last_progress_refresh: dict[str, float] = {}
        self._artifact_readers: dict[str, int] = {}
        self._prune_task: asyncio.Task[None] | None = None
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
        try:
            await asyncio.to_thread(
                self._persist_submission,
                run_id,
                request,
                status,
            )
        except OSError as exc:
            self._discard_run(run_id)
            raise web.HTTPInternalServerError(
                text="could not persist SWE-bench run metadata"
            ) from exc
        self._tasks[run_id] = asyncio.create_task(
            self._execute(run_id, request, run_dir)
        )
        return status

    def _persist_submission(
        self, run_id: str, request: RunRequest, status: RunStatus
    ) -> None:
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            run_dir / "request.json",
            msgspec.json.encode(
                redact_secrets(
                    request.model_dump(mode="json"),
                    secret_values=self._secret_values[run_id],
                )
            ),
        )
        self._write_status(status)

    async def _execute(self, run_id: str, request: RunRequest, run_dir: Path) -> None:
        token = self._cancel_tokens[run_id]
        try:
            async with self._semaphore:
                if token.is_cancelled():
                    await self._transition(
                        run_id, status="cancelled", phase="cancelled"
                    )
                    return
                await self._transition(
                    run_id, status="running", phase="agent", message="agent"
                )
                try:
                    result = await asyncio.to_thread(
                        self.runner.run,
                        request,
                        run_dir,
                        cancel_token=token,
                    )
                except RunCancelled:
                    await self._transition(
                        run_id, status="cancelled", phase="cancelled"
                    )
                    return
                except Exception as exc:
                    if token.is_cancelled():
                        await self._transition(
                            run_id, status="cancelled", phase="cancelled"
                        )
                        return
                    await self._transition(
                        run_id,
                        status="failed",
                        phase="failed",
                        error=redact_text(
                            str(exc), self._secret_values.get(run_id, set())
                        ),
                    )
                    return
                if token.is_cancelled():
                    await self._transition(
                        run_id, status="cancelled", phase="cancelled"
                    )
                    return
                try:
                    await self._refresh_progress(run_id, force=True)
                    final_progress = await self._terminal_progress_async(
                        run_id, "succeeded"
                    )
                    artifacts = [
                        ArtifactInfo(
                            name=name, url=f"/v1/runs/{run_id}/artifacts/{name}"
                        )
                        for name in (
                            "preds.json",
                            "swe_bench_agent.log",
                            "swe_bench_eval.log",
                            "swe_bench_results.json",
                            "status.json",
                        )
                        if (run_dir / name).exists()
                    ]
                    await self._transition(
                        run_id,
                        status="succeeded",
                        result=result,
                        artifacts=artifacts,
                        **final_progress,
                    )
                except Exception as exc:
                    await self._transition(
                        run_id,
                        status="failed",
                        phase="failed",
                        error=redact_text(
                            f"could not finalize SWE-bench run: {exc}",
                            self._secret_values.get(run_id, set()),
                        ),
                    )
        finally:
            self._tasks.pop(run_id, None)
            self._schedule_prune()

    async def cancel(self, run_id: str) -> RunStatus:
        status = await self.get(run_id)
        if status.status not in {"queued", "running"}:
            return status
        token = self._cancel_tokens.get(run_id)
        if token is not None:
            await asyncio.to_thread(token.cancel)
        try:
            progress = await self._terminal_progress_async(run_id, "cancelled")
        except OSError:
            progress = {"phase": "cancelled", "message": "cancelled"}
        await self._transition(run_id, status="cancelled", **progress)
        return self.runs[run_id]

    async def cancel_all_active(self) -> None:
        active = [
            run.run_id
            for run in self.runs.values()
            if run.status in {"queued", "running"}
        ]
        if active:
            await asyncio.gather(*(self.cancel(run_id) for run_id in active))
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        await self.cancel_all_active()
        prune_task = self._prune_task
        if prune_task is not None and not prune_task.done():
            prune_task.cancel()
        if prune_task is not None:
            await asyncio.gather(prune_task, return_exceptions=True)
        self._prune_task = None

    def _secrets_for_request(self, request: RunRequest) -> set[str]:
        secrets: set[str] = set()
        if request.endpoint_api_key:
            secrets.add(request.endpoint_api_key)
        if self.config.auth_token:
            secrets.add(self.config.auth_token)
        return secrets

    def _update_in_memory(self, run_id: str, **updates: Any) -> RunStatus:
        current = self.runs[run_id]
        data = current.model_dump()
        data.update(updates)
        data["updated_at"] = time.time()
        if data["status"] in _TERMINAL_STATES and current.finished_at is None:
            data["finished_at"] = data["updated_at"]
        updated = RunStatus.model_validate(data)
        self.runs[run_id] = updated
        return updated

    async def _transition(self, run_id: str, **updates: Any) -> RunStatus:
        updated = self._update_in_memory(run_id, **updates)
        lock = self._status_persist_locks.setdefault(run_id, asyncio.Lock())
        try:
            async with lock:
                await asyncio.to_thread(self._write_status, updated)
        except OSError:
            # The in-memory transition releases capacity even if the disk status
            # record cannot be updated (for example, a full artifact volume).
            logger.exception("Could not persist status for SWE-bench run %s", run_id)
        return updated

    def _write_status(self, status: RunStatus) -> None:
        run_dir = self.run_dir(status.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        secret_values = self._secret_values.get(status.run_id, set())
        atomic_write_bytes(
            run_dir / "status.json",
            msgspec.json.encode(
                redact_secrets(
                    status.model_dump(mode="json"), secret_values=secret_values
                )
            ),
        )

    def _discard_run(self, run_id: str) -> None:
        self.runs.pop(run_id, None)
        self._requests.pop(run_id, None)
        self._cancel_tokens.pop(run_id, None)
        self._secret_values.pop(run_id, None)
        self._progress_refresh_locks.pop(run_id, None)
        self._status_persist_locks.pop(run_id, None)
        self._last_progress_refresh.pop(run_id, None)
        self._artifact_readers.pop(run_id, None)

    async def get(self, run_id: str) -> RunStatus:
        try:
            await self._refresh_progress(run_id)
            return self.runs[run_id]
        except KeyError as exc:
            raise web.HTTPNotFound(text="unknown run_id") from exc

    async def _refresh_progress(self, run_id: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self._last_progress_refresh.get(run_id, 0)
            < _PROGRESS_REFRESH_INTERVAL_S
        ):
            return
        lock = self._progress_refresh_locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            if (
                not force
                and now - self._last_progress_refresh.get(run_id, 0)
                < _PROGRESS_REFRESH_INTERVAL_S
            ):
                return
            status = self.runs[run_id]
            request = self._requests.get(run_id)
            try:
                progress = await asyncio.to_thread(
                    self._read_progress, run_id, status, request
                )
            except OSError:
                logger.warning(
                    "Could not refresh progress for SWE-bench run %s",
                    run_id,
                    exc_info=True,
                )
                return
            self._last_progress_refresh[run_id] = now
            current_status = self.runs.get(run_id)
            if current_status is None or current_status.status != status.status:
                return
            self._apply_progress(run_id, progress)

    def _apply_progress(self, run_id: str, progress: dict[str, Any]) -> None:
        status = self.runs[run_id]
        current = status.model_dump()
        changed = any(current.get(key) != value for key, value in progress.items())
        if not changed:
            return
        self._update_in_memory(run_id, **progress)

    def _read_progress(
        self,
        run_id: str,
        status: RunStatus,
        request: RunRequest | None,
    ) -> dict[str, Any]:
        if status.status in {"failed", "cancelled"}:
            return self._terminal_progress(run_id, status.status, status, request)
        elif status.status == "succeeded":
            return self._terminal_progress(run_id, "succeeded", status, request)
        return self._derived_progress(run_id, status, request)

    async def _terminal_progress_async(self, run_id: str, phase: str) -> dict[str, Any]:
        status = self.runs[run_id]
        request = self._requests.get(run_id)
        return await asyncio.to_thread(
            self._terminal_progress, run_id, phase, status, request
        )

    def _derived_progress(
        self,
        run_id: str,
        status: RunStatus,
        request: RunRequest | None,
    ) -> dict[str, Any]:
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
            phase = "queued" if status.status == "queued" else "agent"
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

    def _terminal_progress(
        self,
        run_id: str,
        phase: str,
        status: RunStatus,
        request: RunRequest | None,
    ) -> dict[str, Any]:
        progress = self._derived_progress(run_id, status, request)
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

    def acquire_artifact_reader(self, run_id: str) -> None:
        if run_id not in self.runs:
            raise web.HTTPNotFound(text="unknown run_id")
        self._artifact_readers[run_id] = self._artifact_readers.get(run_id, 0) + 1

    def release_artifact_reader(self, run_id: str) -> None:
        readers = self._artifact_readers.get(run_id, 0)
        if readers <= 1:
            self._artifact_readers.pop(run_id, None)
        else:
            self._artifact_readers[run_id] = readers - 1

    def _schedule_prune(self) -> None:
        if self._prune_task is None or self._prune_task.done():
            self._prune_task = asyncio.create_task(self._prune_completed_runs())

    async def _prune_completed_runs(self) -> None:
        limit = self.config.max_stored_runs
        if limit <= 0:
            return
        while True:
            completed = [
                run
                for run in self.runs.values()
                if run.status in _TERMINAL_STATES and run.finished_at is not None
            ]
            overflow = len(completed) - limit
            if overflow <= 0:
                return
            now = time.time()
            eligible: list[RunStatus] = []
            pinned_expired = False
            next_expiry: float | None = None
            for run in completed:
                finished_at = run.finished_at
                if finished_at is None:
                    continue
                expires_at = finished_at + _ARTIFACT_RETENTION_GRACE_S
                if now < expires_at:
                    next_expiry = (
                        expires_at
                        if next_expiry is None
                        else min(next_expiry, expires_at)
                    )
                    continue
                if self._artifact_readers.get(run.run_id, 0) > 0:
                    pinned_expired = True
                    continue
                eligible.append(run)
            eligible.sort(key=lambda run: run.finished_at or 0)
            if not eligible:
                delays = []
                if next_expiry is not None:
                    delays.append(max(0, next_expiry - now))
                if pinned_expired:
                    delays.append(_PRUNE_RETRY_INTERVAL_S)
                await asyncio.sleep(min(delays) if delays else _PRUNE_RETRY_INTERVAL_S)
                continue
            for run in eligible[:overflow]:
                run_id = run.run_id
                run_dir = self.run_dir(run_id)
                self._discard_run(run_id)
                await asyncio.to_thread(shutil.rmtree, run_dir, ignore_errors=True)


MANAGER_KEY = web.AppKey("manager", RunManager)


def create_app(
    config: ServiceConfig,
    runner: RunnerProtocol | None = None,
) -> web.Application:
    if config.auth_token and config.allow_unauthenticated:
        raise ValueError(
            "SWE-bench service cannot set both auth_token and " "allow_unauthenticated"
        )
    if not config.auth_token and not config.allow_unauthenticated:
        raise ValueError(
            "SWE-bench service requires --auth-token; use "
            "--allow-unauthenticated only for trusted development"
        )
    config = ServiceConfig(
        host=config.host,
        port=config.port,
        artifact_root=config.artifact_root.expanduser().resolve(),
        max_concurrent_runs=config.max_concurrent_runs,
        subprocess_timeout_s=config.subprocess_timeout_s,
        auth_token=config.auth_token,
        allow_unauthenticated=config.allow_unauthenticated,
        max_stored_runs=config.max_stored_runs,
    )
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    runner = runner or SweBenchRunner(
        project_root=Path(__file__).resolve().parents[1],
        subprocess_timeout_s=config.subprocess_timeout_s,
    )
    manager = RunManager(config=config, runner=runner)

    @web.middleware
    async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        if (
            request.path != "/health"
            and config.auth_token
            and not hmac.compare_digest(
                request.headers.get("Authorization", ""),
                f"Bearer {config.auth_token}",
            )
        ):
            raise web.HTTPUnauthorized(text="unauthorized")
        return await handler(request)

    app = web.Application(middlewares=[auth_middleware])
    app[MANAGER_KEY] = manager

    async def shutdown_manager(app: web.Application) -> None:
        await app[MANAGER_KEY].shutdown()

    app.on_shutdown.append(shutdown_manager)

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
        status = await manager.get(request.match_info["run_id"])
        return web.json_response(
            redact_secrets(
                status.model_dump(mode="json"),
                secret_values=manager._secret_values.get(status.run_id, set()),
            )
        )

    async def cancel_run(request: web.Request) -> web.Response:
        status = await manager.cancel(request.match_info["run_id"])
        return web.json_response(
            redact_secrets(
                status.model_dump(mode="json"),
                secret_values=manager._secret_values.get(status.run_id, set()),
            )
        )

    async def get_artifact(request: web.Request) -> web.StreamResponse:
        run_id = request.match_info["run_id"]
        name = request.match_info["name"]
        await manager.get(run_id)
        manager.acquire_artifact_reader(run_id)
        try:
            path = resolve_artifact(manager.run_dir(run_id), name)
        except FileNotFoundError as exc:
            manager.release_artifact_reader(run_id)
            raise web.HTTPNotFound(text="artifact not found") from exc
        if name in _LOG_ARTIFACT_NAMES:
            try:
                text = await asyncio.to_thread(
                    lambda: redact_text(
                        path.read_text(errors="replace"),
                        manager._secret_values.get(run_id, set()),
                    )
                )
            finally:
                manager.release_artifact_reader(run_id)
            return web.Response(
                text=text,
                content_type="text/plain",
                headers={"Content-Disposition": f'attachment; filename="{name}"'},
            )
        return _LeasedFileResponse(
            path, lambda: manager.release_artifact_reader(run_id)
        )

    app.router.add_get("/health", health)
    app.router.add_post("/v1/runs", post_run)
    app.router.add_get("/v1/runs/{run_id}", get_run)
    app.router.add_post("/v1/runs/{run_id}/cancel", cancel_run)
    app.router.add_get("/v1/runs/{run_id}/artifacts/{name}", get_artifact)
    return app
