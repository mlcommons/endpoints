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
report. The watchdog stops the session (ENDED still flows) and then
SIGTERMs the metrics aggregator, whose handler writes an INTERRUPTED
final snapshot; ``run_benchmark`` exits non-zero via ``ExecutionError``.
"""

import json
from pathlib import Path

import pytest
from inference_endpoint.commands.benchmark.execute import run_benchmark
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

# Local character-level tokenizer: lets the metrics aggregator tokenize
# ISL/OSL without a HuggingFace Hub download (same trick as
# test_benchmark_command.py).
_CHAR_TOKENIZER_DIR = Path(__file__).resolve().parents[2] / "assets/tokenizers/char"

_FAST_CLIENT = HTTPClientConfig(num_workers=1, warmup_connections=0, max_connections=10)


def _read_final_snapshot(report_dir: Path) -> dict:
    snapshot_path = report_dir / "metrics" / "final_snapshot.json"
    assert snapshot_path.exists(), "aggregator must still write a final snapshot"
    return json.loads(snapshot_path.read_text())


def _read_result_summary(report_dir: Path) -> dict:
    return json.loads((report_dir / "performance" / "result_summary.json").read_text())


@pytest.mark.integration
def test_run_timeout_produces_interrupted_report(
    mock_http_echo_server, ds_dataset_path, tmp_path
):
    """run_timeout_s firing mid-run aborts with an INTERRUPTED report."""
    config = BenchmarkConfig(
        type=TestType.ONLINE,
        endpoint_config=EndpointConfig(endpoints=[mock_http_echo_server.url]),
        model_params=ModelParams(name="echo-server", streaming=StreamingMode.OFF),
        datasets=[Dataset(path=str(ds_dataset_path), type=DatasetType.PERFORMANCE)],
        report_dir=tmp_path,
        settings=Settings(
            load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=5),
            client=_FAST_CLIENT,
            # 600 samples at 5 QPS is a ~120 s workload, so only the watchdog
            # can end the run.
            runtime=RuntimeConfig(n_samples_to_issue=600),
            timeouts=Timeouts(run_timeout_s=2.0),
            warmup=WarmupConfig(enabled=False),
        ),
    )

    with pytest.raises(ExecutionError, match="Run timeout"):
        run_benchmark(config, TestMode.PERF)

    snapshot = _read_final_snapshot(tmp_path)
    assert snapshot["state"] == "interrupted"

    # Locking invariant: a fired run watchdog must never yield a COMPLETE report.
    summary = _read_result_summary(tmp_path)
    assert summary["complete"] is False


@pytest.mark.integration
def test_generous_run_timeout_completes_normally(
    mock_http_echo_server, ds_dataset_path, tmp_path
):
    """A run_timeout_s far above the workload length never fires: the run
    finishes cleanly and publishes a COMPLETE report."""
    config = BenchmarkConfig(
        type=TestType.OFFLINE,
        endpoint_config=EndpointConfig(endpoints=[mock_http_echo_server.url]),
        model_params=ModelParams(name="echo-server", streaming=StreamingMode.OFF),
        datasets=[Dataset(path=str(ds_dataset_path), type=DatasetType.PERFORMANCE)],
        report_dir=tmp_path,
        settings=Settings(
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
            client=_FAST_CLIENT,
            timeouts=Timeouts(run_timeout_s=300.0),
            warmup=WarmupConfig(enabled=False),
        ),
    )

    run_benchmark(config, TestMode.PERF)  # must not raise

    snapshot = _read_final_snapshot(tmp_path)
    assert snapshot["state"] == "complete"
    summary = _read_result_summary(tmp_path)
    assert summary["complete"] is True


@pytest.mark.integration
def test_run_timeout_during_metrics_drain_interrupts(mock_http_echo_server, tmp_path):
    """The watchdog stays armed through the metrics drain.

    The session itself finishes quickly, but the aggregator is left with a
    deliberately huge tokenization backlog (large prompts echoed back as
    outputs, metrics_tokenizer_workers=0 so nothing tokenizes mid-run, and
    the metrics drain unlimited). The watchdog must fire while the
    aggregator drains, SIGTERM it, and surface the run as INTERRUPTED with
    a non-zero exit.
    """
    # ~25 MB of prompt text; the echo server doubles it into OSL, so the
    # drain has ~50M characters to tokenize — far more than run_timeout_s
    # allows on any hardware.
    dataset_path = tmp_path / "big_prompts.jsonl"
    prompt = "lorem ipsum " * 21_000  # ~250 KB per sample
    with dataset_path.open("w") as f:
        for i in range(100):
            f.write(json.dumps({"prompt": f"{i} {prompt}"}) + "\n")

    report_dir = tmp_path / "report"
    config = BenchmarkConfig(
        type=TestType.OFFLINE,
        endpoint_config=EndpointConfig(endpoints=[mock_http_echo_server.url]),
        model_params=ModelParams(
            name=str(_CHAR_TOKENIZER_DIR), streaming=StreamingMode.OFF
        ),
        datasets=[Dataset(path=str(dataset_path), type=DatasetType.PERFORMANCE)],
        report_dir=report_dir,
        settings=Settings(
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
            client=_FAST_CLIENT,
            # Defer every ISL/OSL tokenization to the end-of-run drain.
            metrics_tokenizer_workers=0,
            # metrics_drain_timeout_s stays None (unlimited): only the
            # run watchdog can end the drain.
            timeouts=Timeouts(run_timeout_s=2.5),
            warmup=WarmupConfig(enabled=False),
        ),
    )

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
    dataset_path = tmp_path / "big_prompts.jsonl"
    prompt = "lorem ipsum " * 21_000  # ~250 KB per sample
    with dataset_path.open("w") as f:
        for i in range(100):
            f.write(json.dumps({"prompt": f"{i} {prompt}"}) + "\n")

    report_dir = tmp_path / "report"
    config = BenchmarkConfig(
        type=TestType.OFFLINE,
        endpoint_config=EndpointConfig(endpoints=[mock_http_echo_server.url]),
        model_params=ModelParams(
            name=str(_CHAR_TOKENIZER_DIR), streaming=StreamingMode.OFF
        ),
        datasets=[Dataset(path=str(dataset_path), type=DatasetType.PERFORMANCE)],
        report_dir=report_dir,
        settings=Settings(
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
            client=_FAST_CLIENT,
            # Defer every ISL/OSL tokenization to the end-of-run drain, then
            # give the drain a budget far below the ~50M-char backlog. No run
            # watchdog: the drain deadline itself must fail the run.
            metrics_tokenizer_workers=0,
            timeouts=Timeouts(metrics_drain_timeout_s=1.0),
            warmup=WarmupConfig(enabled=False),
        ),
    )

    with pytest.raises(ExecutionError, match="Metrics drain timed out"):
        run_benchmark(config, TestMode.PERF)

    snapshot = _read_final_snapshot(report_dir)
    assert snapshot["state"] == "complete"
    assert snapshot["n_pending_tasks"] > 0

    summary = _read_result_summary(report_dir)
    assert summary["complete"] is False
