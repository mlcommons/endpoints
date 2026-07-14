# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_SERVICE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "inference_endpoint"
    / "evaluation"
    / "swebench_service"
)
sys.path.insert(0, str(_SERVICE_ROOT))

from swebench_service.config import ServiceConfig  # noqa: E402
from swebench_service.schemas import RunRequest, RunStatus  # noqa: E402
from swebench_service.server import RunManager, create_app  # noqa: E402

pytestmark = pytest.mark.unit


class FakeRunner:
    def __init__(self, *, delay: float = 0.0, fail: bool = False):
        self.delay = delay
        self.fail = fail
        self.requests: list[object] = []

    def run(self, request, run_dir: Path):
        self.requests.append(request)
        if self.delay:
            import time

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

    def run(self, request, run_dir: Path):
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
        time.sleep(0.2)
        (run_dir / "preds.json").write_text("{}")
        (run_dir / "swe_bench_results.json").write_text(
            '{"resolved_instances":1,"submitted_instances":2}'
        )
        return {"resolved_instances": 1, "submitted_instances": 2}


class EvalProgressRunner:
    def __init__(self):
        self.ready = threading.Event()

    def run(self, request, run_dir: Path):
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
        time.sleep(0.2)
        (run_dir / "preds.json").write_text("{}")
        (run_dir / "swe_bench_results.json").write_text(
            '{"resolved_instances":2,"submitted_instances":3}'
        )
        return {"resolved_instances": 2, "submitted_instances": 3}


class CancellationAwareRunner:
    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def run(self, request, run_dir: Path, cancel_token=None):
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
        ServiceConfig(artifact_root=tmp_path, max_concurrent_runs=1),
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
async def test_post_run_validates_requests(tmp_path):
    client = await _client(tmp_path, FakeRunner())
    try:
        resp = await client.post("/v1/runs", json={"model_name": ""})
    finally:
        await client.close()

    assert resp.status == 400


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
async def test_optional_auth_token_is_enforced(tmp_path):
    client = await _auth_client(tmp_path, FakeRunner())
    try:
        unauthorized = await client.get("/health")
        authorized = await client.get(
            "/health", headers={"Authorization": "Bearer tok"}
        )
    finally:
        await client.close()

    assert unauthorized.status == 401
    assert authorized.status == 200


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
async def test_shutdown_cancels_active_runs(tmp_path):
    runner = CancellationAwareRunner()
    client = await _client(tmp_path, runner)
    await client.post("/v1/runs", json=_payload())
    assert await asyncio.to_thread(runner.started.wait, 2)

    await client.close()

    assert await asyncio.to_thread(runner.cancelled.wait, 2)


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
        blocked = await client.get(f"/v1/runs/{run_id}/artifacts/..%2Fstatus.json")
    finally:
        await client.close()

    assert ok.status == 200
    assert blocked.status == 404


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
        def run(self, request, run_dir: Path):
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
