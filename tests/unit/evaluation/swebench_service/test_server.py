# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from inference_endpoint.evaluation.swebench_service.swebench_service import (
    server as server_mod,
)
from inference_endpoint.evaluation.swebench_service.swebench_service.config import (
    ServiceConfig,
)
from inference_endpoint.evaluation.swebench_service.swebench_service.schemas import (
    RunRequest,
    RunStatus,
)
from inference_endpoint.evaluation.swebench_service.swebench_service.server import (
    MANAGER_KEY,
    RunManager,
    create_app,
)

pytestmark = pytest.mark.unit


class FakeRunner:
    def __init__(self, *, delay: float = 0.0, fail: bool = False):
        self.delay = delay
        self.fail = fail
        self.requests: list[object] = []

    def run(self, request, run_dir: Path, cancel_token=None):
        self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("runner failed")
        (run_dir / "preds.json").write_text("{}")
        (run_dir / "swe_bench_results.json").write_text(
            '{"resolved_instances":1,"submitted_instances":1}'
        )
        return {"resolved_instances": 1, "submitted_instances": 1}


class AgentProgressRunner:
    def __init__(self):
        self.ready = threading.Event()
        self.release = threading.Event()

    def run(self, request, run_dir: Path, cancel_token=None):
        output_dir = run_dir / "swe_bench_output"
        output_dir.mkdir()
        (output_dir / "exit_statuses_0002.yaml").write_text(
            "\n".join(
                [
                    "instances_by_exit_status:",
                    "  submitted:",
                    "    - repo__repo-1",
                    "  resolved:",
                    "    - repo__repo-2",
                    "",
                ]
            )
        )
        self.ready.set()
        assert self.release.wait(2)
        (run_dir / "preds.json").write_text("{}")
        (run_dir / "swe_bench_results.json").write_text(
            '{"resolved_instances":1,"submitted_instances":2}'
        )
        return {"resolved_instances": 1, "submitted_instances": 2}


class EvalProgressRunner:
    def __init__(self):
        self.ready = threading.Event()
        self.release = threading.Event()

    def run(self, request, run_dir: Path, cancel_token=None):
        output_dir = run_dir / "swe_bench_output"
        report_dir = (
            output_dir / "logs" / "run_evaluation" / "eval-run-1" / "test-model"
        )
        report_dir.mkdir(parents=True)
        (run_dir / "swe_bench_eval_run_id.txt").write_text("eval-run-1")
        (output_dir / "preds.json").write_text(
            '{"repo__repo-1":"patch","repo__repo-2":"patch","repo__repo-3":"patch"}'
        )
        (report_dir / "repo__repo-1" / "report.json").parent.mkdir()
        (report_dir / "repo__repo-1" / "report.json").write_text("{}")
        (report_dir / "repo__repo-2" / "report.json").parent.mkdir()
        (report_dir / "repo__repo-2" / "report.json").write_text("{}")
        self.ready.set()
        assert self.release.wait(2)
        (run_dir / "preds.json").write_text("{}")
        (run_dir / "swe_bench_results.json").write_text(
            '{"resolved_instances":2,"submitted_instances":3}'
        )
        return {"resolved_instances": 2, "submitted_instances": 3}


class CancellationAwareRunner:
    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.cleaned = threading.Event()

    def run(self, request, run_dir: Path, cancel_token=None):
        try:
            self.started.set()
            deadline = time.monotonic() + 5
            while (
                cancel_token is not None
                and not cancel_token.is_cancelled()
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            if cancel_token is not None and cancel_token.is_cancelled():
                self.cancelled.set()
            return {"resolved_instances": 0, "submitted_instances": 0}
        finally:
            self.cleaned.set()


def _payload() -> dict:
    return {
        "model_name": "test-model",
        "endpoint_urls": ["http://endpoint"],
        "endpoint_api_key": "secret",
        "generation_params": {"temperature": 0.0},
        "subset": "lite",
        "split": "test",
        "num_instances": 1,
        "workers": 1,
        "max_eval_workers": 1,
        "evaluated_instance_ids": ["repo__repo-1"],
    }


def _payload_with_instances(n: int) -> dict:
    payload = _payload()
    payload["num_instances"] = n
    payload["evaluated_instance_ids"] = [f"repo__repo-{i + 1}" for i in range(n)]
    return payload


async def _client(tmp_path: Path, runner) -> TestClient:
    app = create_app(
        ServiceConfig(
            artifact_root=tmp_path,
            max_concurrent_runs=1,
            allow_unauthenticated=True,
        ),
        runner=runner,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _auth_client(tmp_path: Path, runner) -> TestClient:
    app = create_app(
        ServiceConfig(artifact_root=tmp_path, max_concurrent_runs=1, auth_token="tok"),
        runner=runner,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_health_response_schema(tmp_path):
    client = await _client(tmp_path, FakeRunner())
    try:
        resp = await client.get("/health")
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 200
    assert body["api_version"] == "v1"
    assert "swebench.run" in body["capabilities"]
    assert "swebench.cancel" in body["capabilities"]
    assert "artifacts.download" in body["capabilities"]
    assert "swebench.progress" in body["capabilities"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        {"model_name": ""},
        {"endpoint_urls": ["ftp://endpoint"]},
        {"endpoint_urls": ["http:///v1"]},
        {"subset": "unsupported"},
        {"num_instances": 2},
    ],
)
async def test_post_run_validates_requests(tmp_path, updates):
    runner = FakeRunner()
    client = await _client(tmp_path, runner)
    payload = _payload()
    payload.update(updates)
    try:
        resp = await client.post("/v1/runs", json=payload)
    finally:
        await client.close()

    assert resp.status == 400
    assert runner.requests == []


@pytest.mark.asyncio
async def test_post_run_rejects_multiple_endpoint_urls(tmp_path):
    runner = FakeRunner()
    client = await _client(tmp_path, runner)
    payload = _payload()
    payload["endpoint_urls"] = ["http://endpoint-a", "http://endpoint-b"]
    try:
        resp = await client.post("/v1/runs", json=payload)
    finally:
        await client.close()

    assert resp.status == 400
    assert runner.requests == []


@pytest.mark.asyncio
async def test_auth_token_protects_run_routes_but_not_health(tmp_path):
    client = await _auth_client(tmp_path, FakeRunner())
    try:
        health = await client.get("/health")
        unauthorized = await client.post("/v1/runs", json=_payload())
        authorized = await client.post(
            "/v1/runs", json=_payload(), headers={"Authorization": "Bearer tok"}
        )
    finally:
        await client.close()

    assert unauthorized.status == 401
    assert health.status == 200
    assert authorized.status == 202


def test_service_requires_auth_or_explicit_development_override(tmp_path):
    with pytest.raises(ValueError, match="requires --auth-token"):
        create_app(ServiceConfig(artifact_root=tmp_path), runner=FakeRunner())


@pytest.mark.asyncio
async def test_runner_transitions_to_succeeded(tmp_path):
    runner = FakeRunner()
    client = await _client(tmp_path, runner)
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
            status = await status_resp.json()
            if status["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
    finally:
        await client.close()

    assert submit.status == 202
    assert status["status"] == "succeeded"
    assert status["result"] == {"resolved_instances": 1, "submitted_instances": 1}
    assert runner.requests[0].evaluated_instance_ids == ["repo__repo-1"]
    assert status["phase"] == "succeeded"
    assert status["agent_completed"] == 1
    assert status["eval_completed"] == 1


@pytest.mark.asyncio
async def test_status_reports_agent_progress_from_exit_statuses(tmp_path):
    runner = AgentProgressRunner()
    client = await _client(tmp_path, runner)
    try:
        submit = await client.post("/v1/runs", json=_payload_with_instances(3))
        submitted = await submit.json()
        assert await asyncio.to_thread(runner.ready.wait, 2)

        status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
        status = await status_resp.json()
    finally:
        runner.release.set()
        await client.close()

    assert status_resp.status == 200
    assert status["phase"] == "agent"
    assert status["agent_total"] == 3
    assert status["agent_completed"] == 2
    assert status["eval_total"] == 0
    assert status["eval_completed"] == 0


@pytest.mark.asyncio
async def test_status_reports_eval_progress_from_harness_reports(tmp_path):
    runner = EvalProgressRunner()
    client = await _client(tmp_path, runner)
    try:
        submit = await client.post("/v1/runs", json=_payload_with_instances(3))
        submitted = await submit.json()
        assert await asyncio.to_thread(runner.ready.wait, 2)

        status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
        status = await status_resp.json()
    finally:
        runner.release.set()
        await client.close()

    assert status_resp.status == 200
    assert status["phase"] == "eval"
    assert status["eval_total"] == 3
    assert status["eval_completed"] == 2


@pytest.mark.asyncio
async def test_status_reports_zero_progress_when_files_are_missing(tmp_path):
    client = await _client(tmp_path, FakeRunner(delay=0.2))
    try:
        submit = await client.post("/v1/runs", json=_payload_with_instances(2))
        submitted = await submit.json()

        status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
        status = await status_resp.json()
    finally:
        await client.close()

    assert status_resp.status == 200
    assert status["phase"] in {"queued", "agent"}
    assert status["agent_total"] == 2
    assert status["agent_completed"] == 0
    assert status["eval_completed"] == 0


@pytest.mark.asyncio
async def test_progress_polling_is_off_thread_and_memory_only(monkeypatch, tmp_path):
    manager = RunManager(
        config=ServiceConfig(artifact_root=tmp_path),
        runner=FakeRunner(),
    )
    run_id = "run-progress"
    now = time.time()
    manager.runs[run_id] = RunStatus(
        run_id=run_id,
        status="running",
        created_at=now,
        updated_at=now,
        phase="agent",
        agent_total=1,
        agent_completed=0,
    )
    manager._requests[run_id] = RunRequest.model_validate(_payload())
    progress_started = threading.Event()
    progress_release = threading.Event()

    def gated_progress(*args):
        progress_started.set()
        assert progress_release.wait(2)
        return {
            "phase": "agent",
            "agent_total": 1,
            "agent_completed": 1,
            "eval_total": 0,
            "eval_completed": 0,
            "message": "agent",
        }

    monkeypatch.setattr(manager, "_read_progress", gated_progress)
    monkeypatch.setattr(
        manager,
        "_write_status",
        lambda status: pytest.fail("progress polling must not persist status.json"),
    )

    poll_task = asyncio.create_task(manager.get(run_id))
    try:
        assert await asyncio.to_thread(progress_started.wait, 2)
        loop_responded = asyncio.Event()
        asyncio.get_running_loop().call_soon(loop_responded.set)
        await asyncio.wait_for(loop_responded.wait(), timeout=2)
        assert not poll_task.done()
    finally:
        progress_release.set()
    status = await poll_task

    assert status.agent_completed == 1


@pytest.mark.asyncio
async def test_runner_transitions_to_failed(tmp_path):
    client = await _client(tmp_path, FakeRunner(fail=True))
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
            status = await status_resp.json()
            if status["status"] == "failed":
                break
            await asyncio.sleep(0.01)
    finally:
        await client.close()

    assert status["status"] == "failed"
    assert "runner failed" in status["error"]


@pytest.mark.asyncio
async def test_cancel_run_marks_cancelled_and_signals_runner(tmp_path):
    runner = CancellationAwareRunner()
    client = await _client(tmp_path, runner)
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        assert await asyncio.to_thread(runner.started.wait, 2)

        cancel_resp = await client.post(f"/v1/runs/{submitted['run_id']}/cancel")
        cancelled = await cancel_resp.json()

        assert await asyncio.to_thread(runner.cancelled.wait, 2)
    finally:
        await client.close()

    assert cancel_resp.status == 200
    assert cancelled["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancellation_keeps_event_loop_responsive(monkeypatch, tmp_path):
    runner = CancellationAwareRunner()
    manager = RunManager(
        config=ServiceConfig(artifact_root=tmp_path, max_concurrent_runs=1),
        runner=runner,
    )
    submitted = await manager.submit(RunRequest.model_validate(_payload()))
    assert await asyncio.to_thread(runner.started.wait, 2)
    token = manager._cancel_tokens[submitted.run_id]
    original_cancel = token.cancel
    cancel_started = threading.Event()
    cancel_release = threading.Event()

    def gated_cancel():
        cancel_started.set()
        assert cancel_release.wait(2)
        original_cancel()

    monkeypatch.setattr(token, "cancel", gated_cancel)
    cancel_task = asyncio.create_task(manager.cancel(submitted.run_id))

    try:
        assert await asyncio.to_thread(cancel_started.wait, 2)
        loop_responded = asyncio.Event()
        asyncio.get_running_loop().call_soon(loop_responded.set)
        await asyncio.wait_for(loop_responded.wait(), timeout=2)
        assert not cancel_task.done()
    finally:
        cancel_release.set()
    cancelled = await cancel_task
    await manager.shutdown()

    assert cancelled.status == "cancelled"
    assert runner.cleaned.is_set()


@pytest.mark.asyncio
async def test_shutdown_cancels_active_runs(tmp_path):
    runner = CancellationAwareRunner()
    app = create_app(
        ServiceConfig(
            artifact_root=tmp_path,
            max_concurrent_runs=1,
            allow_unauthenticated=True,
        ),
        runner=runner,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    await client.post("/v1/runs", json=_payload())
    assert await asyncio.to_thread(runner.started.wait, 2)
    prune_task = asyncio.create_task(asyncio.sleep(60))
    app[MANAGER_KEY]._prune_task = prune_task

    await client.close()

    assert await asyncio.to_thread(runner.cancelled.wait, 2)
    assert await asyncio.to_thread(runner.cleaned.wait, 2)
    assert prune_task.cancelled()
    assert app[MANAGER_KEY]._prune_task is None


@pytest.mark.asyncio
async def test_bounded_concurrency_returns_429(tmp_path):
    client = await _client(tmp_path, FakeRunner(delay=0.2))
    try:
        first = await client.post("/v1/runs", json=_payload())
        second = await client.post("/v1/runs", json=_payload())
    finally:
        await client.close()

    assert first.status == 202
    assert second.status == 429


@pytest.mark.asyncio
async def test_concurrent_submit_reserves_capacity_once(tmp_path):
    manager = RunManager(
        config=ServiceConfig(artifact_root=tmp_path, max_concurrent_runs=1),
        runner=FakeRunner(delay=0.2),
    )
    try:
        manager_request = RunRequest.model_validate(_payload())
        results = await asyncio.gather(
            manager.submit(manager_request),
            manager.submit(manager_request),
            return_exceptions=True,
        )
    finally:
        await manager.cancel_all_active()

    accepted = [result for result in results if isinstance(result, RunStatus)]
    rejected = [
        result for result in results if isinstance(result, web.HTTPTooManyRequests)
    ]
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert manager.active_count() <= 1


@pytest.mark.asyncio
async def test_pruning_waits_for_active_artifact_reader(monkeypatch, tmp_path):
    monkeypatch.setattr(server_mod, "_ARTIFACT_RETENTION_GRACE_S", 60)
    monkeypatch.setattr(server_mod, "_PRUNE_RETRY_INTERVAL_S", 0.01)
    manager = RunManager(
        config=ServiceConfig(artifact_root=tmp_path, max_stored_runs=1),
        runner=FakeRunner(),
    )
    now = time.time()
    for run_id, finished_at in (
        ("run-old", now - 120),
        ("run-new", now),
    ):
        manager.runs[run_id] = RunStatus(
            run_id=run_id,
            status="succeeded",
            created_at=finished_at - 1,
            updated_at=finished_at,
            finished_at=finished_at,
        )
        manager.run_dir(run_id).mkdir()
        (manager.run_dir(run_id) / "preds.json").write_text("{}")

    manager.acquire_artifact_reader("run-old")
    prune_task = asyncio.create_task(manager._prune_completed_runs())
    await asyncio.sleep(0)

    assert "run-old" in manager.runs
    assert manager.run_dir("run-old").exists()

    manager.release_artifact_reader("run-old")
    assert manager._artifact_readers.get("run-old", 0) == 0
    await asyncio.wait_for(prune_task, timeout=0.2)

    assert "run-old" not in manager.runs
    assert not manager.run_dir("run-old").exists()
    assert "run-new" in manager.runs


@pytest.mark.asyncio
async def test_artifact_endpoint_blocks_path_traversal(tmp_path):
    client = await _client(tmp_path, FakeRunner())
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        run_id = submitted["run_id"]
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{run_id}")
            status = await status_resp.json()
            if status["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        ok = await client.get(f"/v1/runs/{run_id}/artifacts/preds.json")
        await ok.read()
        blocked = await client.get(f"/v1/runs/{run_id}/artifacts/..%2Fstatus.json")
        active_readers = client.app[MANAGER_KEY]._artifact_readers
    finally:
        await client.close()

    assert ok.status == 200
    assert blocked.status == 404
    assert active_readers.get(run_id, 0) == 0


@pytest.mark.asyncio
async def test_log_artifact_redacts_short_secret(tmp_path):
    class LogRunner(FakeRunner):
        def run(self, request, run_dir: Path, cancel_token=None):
            (run_dir / "swe_bench_agent.log").write_text(
                f"Authorization: Bearer {request.endpoint_api_key}"
            )
            return super().run(request, run_dir, cancel_token=cancel_token)

    client = await _client(tmp_path, LogRunner())
    payload = _payload()
    payload["endpoint_api_key"] = "abc"
    try:
        submit = await client.post("/v1/runs", json=payload)
        submitted = await submit.json()
        run_id = submitted["run_id"]
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{run_id}")
            status = await status_resp.json()
            if status["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        response = await client.get(f"/v1/runs/{run_id}/artifacts/swe_bench_agent.log")
        text = await response.text()
    finally:
        await client.close()

    assert response.status == 200
    assert "abc" not in text
    assert "<redacted>" in text


@pytest.mark.asyncio
async def test_finalization_failure_marks_run_failed_and_releases_capacity(
    monkeypatch, tmp_path
):
    manager = RunManager(
        config=ServiceConfig(artifact_root=tmp_path, max_concurrent_runs=1),
        runner=FakeRunner(),
    )

    async def fail_terminal_progress(*args, **kwargs):
        raise OSError("status volume full")

    monkeypatch.setattr(manager, "_terminal_progress_async", fail_terminal_progress)
    first = await manager.submit(RunRequest.model_validate(_payload()))
    await manager._tasks[first.run_id]

    assert manager.runs[first.run_id].status == "failed"
    assert manager.active_count() == 0
    second = await manager.submit(RunRequest.model_validate(_payload()))
    await manager.cancel_all_active()
    assert second.status == "queued"


@pytest.mark.asyncio
async def test_status_redacts_api_keys(tmp_path):
    client = await _client(tmp_path, FakeRunner(delay=0.05))
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        status_path = tmp_path / submitted["run_id"] / "status.json"
        request_path = tmp_path / submitted["run_id"] / "request.json"
        for _ in range(100):
            if request_path.exists():
                break
            await asyncio.sleep(0.01)
        request_text = request_path.read_text()
        status_text = status_path.read_text()
    finally:
        await client.close()

    assert "secret" not in request_text
    assert "<redacted>" in request_text
    assert "secret" not in status_text


@pytest.mark.asyncio
async def test_failed_status_redacts_secret_values(tmp_path):
    class SecretFailRunner:
        def run(self, request, run_dir: Path, cancel_token=None):
            raise RuntimeError(f"failed with api_key={request.endpoint_api_key}")

    client = await _client(tmp_path, SecretFailRunner())
    try:
        submit = await client.post("/v1/runs", json=_payload())
        submitted = await submit.json()
        for _ in range(20):
            status_resp = await client.get(f"/v1/runs/{submitted['run_id']}")
            status = await status_resp.json()
            if status["status"] == "failed":
                break
            await asyncio.sleep(0.01)
    finally:
        await client.close()

    assert "secret" not in status["error"]
    assert "<redacted>" in status["error"]
