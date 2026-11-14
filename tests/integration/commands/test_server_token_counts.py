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

"""Server-reported token counts end to end, against the echo server.

The echo server reports ``prompt_tokens = len(user content)`` and
``completion_tokens = len(emitted text)``, which equal the hermetic char
tokenizer's counts for plain text. Covers the missing-usage failure paths,
the ``client`` opt-out, and which requests carry ``stream_options``.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from inference_endpoint.commands.benchmark.execute import run_benchmark
from inference_endpoint.config.schema import (
    AccuracyConfig,
    AgenticInferenceConfig,
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
    WarmupConfig,
)
from inference_endpoint.core.types import APIType, TokenCountSource
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.exceptions import ExecutionError

_CHAR = Path(__file__).resolve().parents[2] / "assets/tokenizers/char"
_FAST_CLIENT = HTTPClientConfig(num_workers=1, warmup_connections=0, max_connections=10)
# 12 characters each: char-tokenizer count == echo usage == 96 in total.
_PROMPTS = [f"alpha beta {i}" for i in range(8)]

_USAGE = "8 of 8 successful tracked samples returned no server token usage"
_FIX = (
    "set --token-count-source client if this server cannot report usage "
    "(YAML: settings.token_count_source: client)"
)
_EMPTY = (
    "8 had empty output — likely server errors or cut streams; "
    "client mode will not fix these"
)

# The echo answers its prompt, so the first three rows score and the last two
# do not.
_ACC = [
    ("alpha", "alpha"),
    ("beta", "beta"),
    ("gamma", "gamma"),
    ("What is the answer?", "42"),
    ("Tell me a joke", "knock knock"),
]


def _rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _summary(report_dir: Path) -> dict[str, Any]:
    return json.loads((report_dir / "performance/result_summary.json").read_text())


def _counters(report_dir: Path) -> dict[str, int]:
    snapshot = json.loads((report_dir / "metrics/final_snapshot.json").read_text())
    return {
        m["name"]: m["value"] for m in snapshot["metrics"] if m["type"] == "counter"
    }


def _config(
    url: str,
    dataset: Path,
    report_dir: Path,
    *,
    source: TokenCountSource = TokenCountSource.SERVER,
    streaming: StreamingMode = StreamingMode.ON,
    api_type: APIType = APIType.OPENAI,
    model: str = str(_CHAR),
    n_samples: int | None = 8,
    agentic: bool = False,
    stream_include_usage: bool = True,
) -> BenchmarkConfig:
    if agentic:
        test_type = TestType.ONLINE
        load_pattern = LoadPattern(
            type=LoadPatternType.AGENTIC_INFERENCE, target_concurrency=2
        )
        dataset_config = Dataset(
            path=str(dataset), agentic_inference=AgenticInferenceConfig()
        )
    else:
        test_type = TestType.OFFLINE
        load_pattern = LoadPattern()
        dataset_config = Dataset(path=str(dataset))
    return BenchmarkConfig(
        type=test_type,
        endpoint_config=EndpointConfig(endpoints=[url], api_type=api_type),
        model_params=ModelParams(name=model, streaming=streaming),
        datasets=[dataset_config],
        report_dir=report_dir,
        settings=Settings(
            load_pattern=load_pattern,
            client=_FAST_CLIENT.with_updates(stream_include_usage=stream_include_usage),
            warmup=WarmupConfig(enabled=False),
            runtime=RuntimeConfig(n_samples_to_issue=n_samples),
            token_count_source=source,
        ),
    )


@pytest.mark.integration
def test_missing_usage_exits_4_and_writes_artifacts(
    recording_echo_server_factory, tmp_path
):
    server = recording_echo_server_factory(report_usage=False)
    report_dir, config_path = tmp_path / "report", tmp_path / "bench.yaml"
    dataset = _rows(tmp_path / "p.jsonl", [{"prompt": p} for p in _PROMPTS])
    _config(server.url, dataset, report_dir).to_yaml_file(config_path)
    cli = shutil.which("inference-endpoint")
    assert cli is not None

    proc = subprocess.run(
        [cli, "benchmark", "from-config", "-c", str(config_path)],
        capture_output=True,
        text=True,
        timeout=240,
    )

    log = proc.stdout + proc.stderr
    assert proc.returncode == 4, log[-3000:]
    assert f"{_USAGE}; {_FIX}" in log
    assert "tokenization did not finish" not in log and "drain timed out" not in log
    s, c = _summary(report_dir), _counters(report_dir)
    assert s["complete"] is False
    assert (
        s["n_samples_missing_usage"],
        s["n_samples_missing_usage_empty_output"],
    ) == (8, 0)
    assert (
        c["tracked_samples_missing_usage"],
        c["tracked_samples_missing_usage_empty_output"],
    ) == (8, 0)
    assert _USAGE in (report_dir / "report.txt").read_text()
    assert (report_dir / "events.jsonl").exists()


@pytest.mark.integration
def test_missing_usage_with_empty_output_names_server_errors(
    recording_echo_server_factory, tmp_path
):
    server = recording_echo_server_factory(report_usage=False)
    dataset = _rows(tmp_path / "p.jsonl", [{"prompt": ""}] * 8)

    with pytest.raises(ExecutionError) as exc:
        run_benchmark(_config(server.url, dataset, tmp_path / "r"), TestMode.PERF)

    assert str(exc.value) == f"{_USAGE}; {_EMPTY}"
    s = _summary(tmp_path / "r")
    assert (
        s["n_samples_missing_usage"],
        s["n_samples_missing_usage_empty_output"],
    ) == (8, 8)


@pytest.mark.integration
@pytest.mark.parametrize("stream_include_usage", [True, False])
def test_client_mode_passes_without_usage(
    recording_echo_server_factory, tmp_path, stream_include_usage
):
    server = recording_echo_server_factory(report_usage=False)
    dataset = _rows(tmp_path / "p.jsonl", [{"prompt": p} for p in _PROMPTS])

    run_benchmark(
        _config(
            server.url,
            dataset,
            tmp_path / "r",
            source=TokenCountSource.CLIENT,
            stream_include_usage=stream_include_usage,
        ),
        TestMode.PERF,
    )

    s = _summary(tmp_path / "r")
    # Missing usage is counted, never fatal, and leaves no *_server block.
    assert s["complete"] is True and s["n_samples_missing_usage"] == 8
    assert s["token_count_source"] == "client"
    assert "token_count_source" not in s["run_config"]
    assert s["output_sequence_lengths"]["total"] == 96
    assert s["output_sequence_lengths_server"] is None and s["tps_server"] is None
    # The run setting alone decides whether streaming requests ask for usage.
    expected = {"include_usage": True} if stream_include_usage else None
    assert len(server.bodies) == 8
    assert all(b["stream"] is True for b in server.bodies)
    assert [b.get("stream_options") for b in server.bodies] == [expected] * 8


@pytest.mark.integration
def test_server_mode_matches_char_tokenizer_on_plain_text(
    recording_echo_server_factory, tmp_path
):
    server = recording_echo_server_factory()
    dataset = _rows(tmp_path / "p.jsonl", [{"prompt": p} for p in _PROMPTS])

    run_benchmark(_config(server.url, dataset, tmp_path / "r"), TestMode.PERF)

    s = _summary(tmp_path / "r")
    assert s["complete"] is True and s["token_count_source"] == "server"
    assert s["n_samples_missing_usage"] == 0
    for key in (
        "input_sequence_lengths",
        "input_sequence_lengths_client",
        "output_sequence_lengths",
        "output_sequence_lengths_client",
    ):
        assert s[key]["total"] == 96
    assert s["osl_paired"] == {
        "n_samples": 8,
        "client_tokens": 96,
        "server_tokens": 96,
        "mean_diff": 0.0,
        "ratio": 1.0,
    }
    assert [b.get("stream_options") for b in server.bodies] == [
        {"include_usage": True}
    ] * 8


@pytest.mark.integration
def test_agentic_rows_request_usage_only_when_streaming(
    recording_echo_server_factory, tmp_path
):
    server = recording_echo_server_factory()
    # Every row sets stream: a stream column with gaps loads as float64 and is
    # sent as "stream": 1.0 / 0.0.
    rows = [
        {
            "conversation_id": "c1",
            "turn": 1,
            "role": "user",
            "content": "alpha",
            "stream": True,
        },
        {
            "conversation_id": "c1",
            "turn": 2,
            "role": "assistant",
            "content": "beta",
            "stream": False,
        },
        {
            "conversation_id": "c1",
            "turn": 3,
            "role": "user",
            "content": "gamma",
            "stream": True,
        },
        {
            "conversation_id": "c2",
            "turn": 1,
            "role": "user",
            "content": "delta",
            "stream": False,
        },
    ]

    run_benchmark(
        _config(
            server.url,
            _rows(tmp_path / "a.jsonl", rows),
            tmp_path / "r",
            model="echo-server",
            n_samples=None,
            agentic=True,
        ),
        TestMode.PERF,
    )

    streamed = [b for b in server.bodies if b["stream"] is True]
    unstreamed = [b for b in server.bodies if b["stream"] is False]
    assert (len(streamed), len(unstreamed)) == (2, 1)
    assert all(b["stream_options"] == {"include_usage": True} for b in streamed)
    assert "stream_options" not in unstreamed[0]
    s = _summary(tmp_path / "r")
    assert s["complete"] is True and s["n_samples_failed"] == 0
    assert s["n_samples_missing_usage"] == 0


@pytest.mark.integration
@pytest.mark.parametrize("streaming", [StreamingMode.ON, StreamingMode.OFF])
def test_completions_run_requests_usage_only_when_streaming(
    recording_echo_server_factory, tmp_path, streaming
):
    # The echo answers stream_options without stream with HTTP 400.
    server = recording_echo_server_factory()

    run_benchmark(
        _config(
            server.url,
            _rows(tmp_path / "t.jsonl", [{"input_tokens": [1, 2, 3]}] * 4),
            tmp_path / "r",
            api_type=APIType.OPENAI_COMPLETIONS,
            model="echo-server",
            streaming=streaming,
            n_samples=4,
        ),
        TestMode.PERF,
    )

    expected = {"include_usage": True} if streaming is StreamingMode.ON else None
    assert [b.get("stream_options") for b in server.bodies] == [expected] * 4
    s = _summary(tmp_path / "r")
    assert s["complete"] is True and s["n_samples_missing_usage"] == 0
    # Echo usage per sample: P = len([1, 2, 3]) = 3, C = len("1 2 3") = 5.
    assert (
        s["input_sequence_lengths"]["total"],
        s["output_sequence_lengths"]["total"],
    ) == (12, 20)
    # Token-ID prompts give a client ISL without a tokenizer.
    assert s["input_sequence_lengths_client"]["total"] == 12


@pytest.mark.integration
def test_accuracy_only_run_without_usage_succeeds(
    recording_echo_server_factory, tmp_path
):
    # An accuracy-only run never opens performance tracking, so missing usage
    # cannot fail it.
    server = recording_echo_server_factory(report_usage=False)
    acc = Dataset(
        name="echo_accuracy",
        type=DatasetType.ACCURACY,
        path=str(
            _rows(
                tmp_path / "acc.jsonl",
                [{"prompt": p, "ground_truth": g} for p, g in _ACC],
            )
        ),
        accuracy_config=AccuracyConfig(
            eval_method="string_match",
            ground_truth="ground_truth",
            extractor="identity_extractor",
        ),
    )
    config = _config(
        server.url,
        _rows(tmp_path / "p.jsonl", [{"prompt": p} for p in _PROMPTS]),
        tmp_path / "r",
        n_samples=None,
    )

    run_benchmark(config.with_updates(datasets=[*config.datasets, acc]), TestMode.ACC)

    s = _summary(tmp_path / "r")
    assert s["complete"] is True and s["n_samples_missing_usage"] == 0
    entry = json.loads((tmp_path / "r/accuracy/accuracy_results.json").read_text())[
        "accuracy_scores"
    ][0]
    assert entry["score"] == pytest.approx(0.6)
    assert entry["response_counts"]["scored"] == 5
    # The char tokenizer supplies the client counts the server did not.
    assert (
        entry["n_osl_samples"],
        entry["n_osl_samples_client"],
        entry["n_missing_usage"],
    ) == (0, 5, 5)
    # Requests still ask for usage.
    assert [b.get("stream_options") for b in server.bodies] == [
        {"include_usage": True}
    ] * 5
