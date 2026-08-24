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
exits non-zero via ``ExecutionError``.
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest
from inference_endpoint.commands.benchmark.execute import (
    run_benchmark,
    run_benchmark_async,
    setup_benchmark,
)
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
def test_run_timeout_during_metrics_drain_interrupts(mock_http_echo_server, tmp_path):
    """The watchdog stays armed through the metrics drain.

    The session itself finishes quickly, but the aggregator is left with a
    deliberately huge tokenization backlog (large prompts echoed back as
    outputs, metrics_tokenizer_workers=0 so nothing tokenizes mid-run, and
    the metrics drain unlimited). The watchdog must fire while the
    aggregator drains, bound the graceful teardown, and surface the run as
    INTERRUPTED with a non-zero exit.
    """
    dataset_path = write_large_prompt_dataset(tmp_path, 100)
    config = _make_config(
        mock_http_echo_server.url,
        dataset_path,
        tmp_path / "report",
        model_name=str(_CHAR_TOKENIZER_DIR),
        # Defer every ISL/OSL tokenization to the end-of-run drain.
        # metrics_drain_timeout_s stays None (unlimited): only the run
        # watchdog can end the drain.
        metrics_tokenizer_workers=0,
        timeouts=Timeouts(run_timeout_s=2.5),
    )
    report_dir = tmp_path / "report"

    with pytest.raises(ExecutionError, match="Run timeout"):
        run_benchmark(config, TestMode.PERF)

    snapshot = _read_final_snapshot(report_dir)
    assert snapshot["state"] == "interrupted"


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
