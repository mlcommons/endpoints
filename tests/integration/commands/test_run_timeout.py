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

"""Whole-run watchdog (settings.timeouts.run_timeout_s) integration tests.

Locking invariant: a fired run watchdog must never produce a COMPLETE
report. The session still publishes ENDED, then the services get one bounded
graceful teardown before remaining children are killed; ``run_benchmark``
exits non-zero via ``ExecutionError``. A fire after the session's terminal
ENDED comes too late for the session's INTERRUPTED marker, so a
``final_snapshot.json`` written within the grace legitimately reads
``state=complete``; ``result_summary.json`` and the raised error are the
run-level truth.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import inference_endpoint.commands.benchmark.execute as execute_mod
import pytest
from inference_endpoint.commands.benchmark.execute import (
    run_benchmark,
    run_benchmark_async,
    setup_benchmark,
)
from inference_endpoint.commands.benchmark.pipeline import MetricsPipeline
from inference_endpoint.commands.benchmark.watchdog import RunWatchdog
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    Dataset,
    DatasetType,
    EndpointConfig,
    LoadPattern,
    LoadPatternType,
    ModelParams,
    RuntimeConfig,
    Settings,
    StreamingMode,
    TestMode,
    TestType,
    Timeouts,
    WarmupConfig,
)
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.exceptions import ExecutionError
from inference_endpoint.metrics.report import Report

from tests.test_helpers import write_large_prompt_dataset

# Local character-level tokenizer: lets the metrics aggregator tokenize
# ISL/OSL without a HuggingFace Hub download (same trick as
# test_benchmark_command.py).
_CHAR_TOKENIZER_DIR = Path(__file__).resolve().parents[2] / "assets/tokenizers/char"

_FAST_CLIENT = HTTPClientConfig(num_workers=1, warmup_connections=0, max_connections=10)


def _read_final_snapshot(report_dir: Path) -> dict[str, Any]:
    snapshot_path = report_dir / "metrics" / "final_snapshot.json"
    assert snapshot_path.exists(), "aggregator must still write a final snapshot"
    return json.loads(snapshot_path.read_text())


def _read_result_summary(report_dir: Path) -> dict[str, Any]:
    return json.loads((report_dir / "performance" / "result_summary.json").read_text())


class _DrainDeadlineWatchdog(RunWatchdog):
    """A RunWatchdog with no wall-clock timer.

    ``expire()`` queues the real fire path, but only while the run has not
    cancelled the watchdog yet.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        deadline: float | None,
        abort_event: asyncio.Event,
    ) -> None:
        super().__init__(loop, None, abort_event)
        self.armed = True

    def cancel(self) -> None:
        self.armed = False
        super().cancel()

    def expire(self) -> None:
        if self.armed:
            asyncio.get_running_loop().call_soon(self._fire)


def _expire_run_deadline_at_metrics_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> list[bool]:
    """Fire the run watchdog when the parent starts waiting on the metrics drain.

    A wall-clock run_timeout_s cannot target that window: service and worker
    startup alone can outlast a small budget, so the fire lands before the
    session exists and takes the launch-abort path instead.

    Returns one entry per drain: whether its abort_event was set when it
    returned.
    """
    watchdogs: list[_DrainDeadlineWatchdog] = []
    drain_aborted: list[bool] = []

    def make_watchdog(*args: Any) -> _DrainDeadlineWatchdog:
        watchdogs.append(_DrainDeadlineWatchdog(*args))
        return watchdogs[-1]

    drain = MetricsPipeline.drain_and_build_report

    async def drain_past_deadline(
        self: MetricsPipeline, **kwargs: Any
    ) -> Report | None:
        (watchdog,) = watchdogs
        # The fire runs at the drain's first suspension: after the session's
        # ENDED, while the parent waits on the aggregator.
        watchdog.expire()
        report = await drain(self, **kwargs)
        drain_aborted.append(kwargs["abort_event"].is_set())
        return report

    monkeypatch.setattr(execute_mod, "RunWatchdog", make_watchdog)
    monkeypatch.setattr(MetricsPipeline, "drain_and_build_report", drain_past_deadline)
    return drain_aborted


def _make_config(
    endpoint_url: str,
    dataset_path: Path,
    report_dir: Path,
    *,
    test_type: TestType = TestType.OFFLINE,
    model_name: str = "echo-server",
    load_pattern: LoadPattern = LoadPattern(),
    runtime: RuntimeConfig | None = None,
    timeouts: Timeouts | None = None,
    streaming: StreamingMode = StreamingMode.OFF,
    metrics_tokenizer_workers: int | None = None,
) -> BenchmarkConfig:
    settings_kwargs: dict[str, Any] = {
        "load_pattern": load_pattern,
        "client": _FAST_CLIENT,
        "warmup": WarmupConfig(enabled=False),
    }
    if runtime is not None:
        settings_kwargs["runtime"] = runtime
    if timeouts is not None:
        settings_kwargs["timeouts"] = timeouts
    if metrics_tokenizer_workers is not None:
        settings_kwargs["metrics_tokenizer_workers"] = metrics_tokenizer_workers
    return BenchmarkConfig(
        type=test_type,
        endpoint_config=EndpointConfig(endpoints=[endpoint_url]),
        model_params=ModelParams(name=model_name, streaming=streaming),
        datasets=[Dataset(path=str(dataset_path), type=DatasetType.PERFORMANCE)],
        report_dir=report_dir,
        settings=Settings(**settings_kwargs),
    )


@pytest.mark.integration
@pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
def test_run_timeout_produces_interrupted_report(
    mock_http_echo_server, ds_dataset_path, tmp_path, streaming
):
    """run_timeout_s firing mid-run aborts with an INTERRUPTED report."""
    config = _make_config(
        mock_http_echo_server.url,
        ds_dataset_path,
        tmp_path,
        test_type=TestType.ONLINE,
        load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=5),
        # 600 samples at 5 QPS is a ~120 s workload, so only the watchdog
        # can end the run. The budget must comfortably exceed service +
        # worker startup (a fire before the session exists aborts the
        # launch instead, without mid-run artifacts — a different path,
        # covered by test_run_timeout_during_service_launch_aborts_promptly).
        runtime=RuntimeConfig(n_samples_to_issue=600),
        timeouts=Timeouts(run_timeout_s=6.0),
        streaming=streaming,
    )

    with pytest.raises(ExecutionError, match="Run timeout"):
        run_benchmark(config, TestMode.PERF)

    snapshot = _read_final_snapshot(tmp_path)
    assert snapshot["state"] == "interrupted"

    # Locking invariant: a fired run watchdog must never yield a COMPLETE report.
    summary = _read_result_summary(tmp_path)
    assert summary["complete"] is False


@pytest.mark.integration
def test_run_timeout_during_metrics_drain_interrupts(
    mock_http_echo_server, tmp_path, monkeypatch
):
    """The watchdog stays armed through the post-ENDED metrics drain.

    The session ends normally and leaves the aggregator a deferred
    tokenization backlog (large prompts echoed back as outputs,
    metrics_tokenizer_workers=0). The run deadline expires as the parent starts
    waiting on that drain. The session already published ENDED without an
    INTERRUPTED marker, so the aggregator finishes draining within the teardown
    grace and final_snapshot.json records the normally-ended run it observed.
    result_summary.json and the "Run timeout" error carry the run outcome: the
    AGENTS.md "Run-outcome precedence" contract that
    test_sigint_in_drain_window_keeps_summary_authoritative pins for ^C.
    """
    drain_aborted = _expire_run_deadline_at_metrics_drain(monkeypatch)
    dataset_path = write_large_prompt_dataset(tmp_path, 30)
    report_dir = tmp_path / "report"
    config = _make_config(
        mock_http_echo_server.url,
        dataset_path,
        report_dir,
        model_name=str(_CHAR_TOKENIZER_DIR),
        # Defer every ISL/OSL tokenization to the end-of-run drain.
        metrics_tokenizer_workers=0,
        # run_timeout_s only labels the error: _DrainDeadlineWatchdog arms no
        # timer. The metrics drain stays unlimited, and the grace leaves the
        # aggregator room to finish it.
        timeouts=Timeouts(run_timeout_s=600.0, interrupted_teardown_grace_s=120.0),
    )

    # Any exception escaping the run after the fire (a pre-session cancel, or
    # a drain/report failure) raises "aborted before a report could be
    # produced" instead.
    with pytest.raises(
        ExecutionError, match=r"Run timeout .*report marked INTERRUPTED"
    ):
        run_benchmark(config, TestMode.PERF)

    # The fire reached the drain as an abort, which bounds it by the grace.
    assert drain_aborted == [True]

    summary = _read_result_summary(report_dir)
    assert summary["state"] == "interrupted"
    assert summary["complete"] is False

    snapshot = _read_final_snapshot(report_dir)
    assert snapshot["state"] == "complete"
    assert snapshot["n_pending_tasks"] == 0


@pytest.mark.integration
def test_metrics_drain_timeout_fails_run(mock_http_echo_server, tmp_path):
    """An expired metrics_drain_timeout_s fails the run instead of exiting 0.

    The aggregator finalizes as COMPLETE with a pending tokenization backlog
    (state "complete", n_pending_tasks > 0). Artifacts must still be written
    with complete: false, and run_benchmark must raise so partial ISL/OSL
    stats can never look like a clean exit.
    """
    dataset_path = write_large_prompt_dataset(tmp_path, 100)
    config = _make_config(
        mock_http_echo_server.url,
        dataset_path,
        tmp_path / "report",
        model_name=str(_CHAR_TOKENIZER_DIR),
        # Defer every ISL/OSL tokenization to the end-of-run drain, then
        # give the drain a budget far below the ~50M-char backlog. No run
        # watchdog: the drain deadline itself must fail the run.
        metrics_tokenizer_workers=0,
        timeouts=Timeouts(metrics_drain_timeout_s=1.0),
    )
    report_dir = tmp_path / "report"

    with pytest.raises(ExecutionError, match="Metrics tokenization did not finish"):
        run_benchmark(config, TestMode.PERF)

    snapshot = _read_final_snapshot(report_dir)
    assert snapshot["state"] == "complete"
    assert snapshot["n_pending_tasks"] > 0

    summary = _read_result_summary(report_dir)
    assert summary["complete"] is False


@pytest.mark.integration
def test_run_timeout_during_service_launch_aborts_promptly(
    mock_http_echo_server, ds_dataset_path, tmp_path
):
    """A deadline expiring before the session exists cancels the launch.

    The watchdog's pre-session fire path cancels the orchestration task so
    pending service-launch/endpoint-connect awaits unwind immediately instead
    of running out their own readiness timeouts, and the abort is attributed
    to the run timeout (ExecutionError), not to a secondary launch error.
    """
    config = _make_config(
        mock_http_echo_server.url,
        ds_dataset_path,
        tmp_path,
        test_type=TestType.ONLINE,
        load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=5),
        runtime=RuntimeConfig(n_samples_to_issue=10),
    )
    ctx = setup_benchmark(config, TestMode.PERF)

    start = time.monotonic()
    # An already-expired deadline fires the watchdog on the first event-loop
    # iteration — deterministically before the metrics services report ready.
    with pytest.raises(ExecutionError, match="Run timeout"):
        run_benchmark_async(ctx, deadline=time.monotonic())
    elapsed = time.monotonic() - start

    # Prompt unwind: nowhere near the 30 s service_ready_timeout_s the
    # pre-fix behavior would have waited out.
    assert elapsed < 15.0, f"launch abort took {elapsed:.1f}s"
