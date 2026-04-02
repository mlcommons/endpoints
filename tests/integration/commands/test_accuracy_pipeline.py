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

"""Integration test: full accuracy scoring pipeline with echo server.

The echo server returns the user message content unchanged. We create a
dataset where some prompts match their ground_truth (correct) and some
don't (incorrect), then verify the scorer produces the expected accuracy.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from inference_endpoint.commands.benchmark.execute import run_benchmark
from inference_endpoint.config.schema import (
    AccuracyConfig,
    BenchmarkConfig,
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
)
from inference_endpoint.config.schema import Dataset as DatasetConfig
from inference_endpoint.endpoint_client.config import HTTPClientConfig


def _create_accuracy_dataset(tmp_path: Path) -> Path:
    """Create a CSV dataset with some correct and some incorrect ground truths.

    The echo server returns the prompt verbatim. So:
    - If ground_truth == prompt → score 1.0 (correct)
    - If ground_truth != prompt → score 0.0 (incorrect)

    Dataset: 5 samples, 3 correct + 2 incorrect = 60% accuracy.
    """
    data = {
        "prompt": [
            "alpha",  # correct: echo returns "alpha", ground_truth is "alpha"
            "beta",  # correct
            "gamma",  # correct
            "What is the answer?",  # INCORRECT: echo returns prompt, ground_truth is "42"
            "Tell me a joke",  # INCORRECT: echo returns prompt, ground_truth is "knock knock"
        ],
        "ground_truth": [
            "alpha",
            "beta",
            "gamma",
            "42",
            "knock knock",
        ],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "accuracy_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _create_perf_dataset(tmp_path: Path) -> Path:
    """Create a minimal perf dataset (CSV with prompt column)."""
    data = {"prompt": ["hello"] * 3}
    df = pd.DataFrame(data)
    csv_path = tmp_path / "perf_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.integration
class TestAccuracyPipeline:
    def test_accuracy_scoring_with_echo_server(
        self, mock_http_echo_server, tmp_path, caplog
    ):
        """Full end-to-end: perf phase + accuracy phase + scoring.

        Expected: 3/5 correct = 60% accuracy (0.6 score).
        """
        perf_path = _create_perf_dataset(tmp_path)
        acc_path = _create_accuracy_dataset(tmp_path)

        report_dir = tmp_path / "report"
        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            endpoint_config=EndpointConfig(endpoints=[mock_http_echo_server.url]),
            model_params=ModelParams(name="echo-server", streaming=StreamingMode.OFF),
            datasets=[
                DatasetConfig(
                    path=str(perf_path),
                    type=DatasetType.PERFORMANCE,
                ),
                DatasetConfig(
                    name="echo_accuracy",
                    path=str(acc_path),
                    type=DatasetType.ACCURACY,
                    accuracy_config=AccuracyConfig(
                        eval_method="string_match",
                        ground_truth="ground_truth",
                        extractor="identity_extractor",
                    ),
                ),
            ],
            settings=Settings(
                runtime=RuntimeConfig(min_duration_ms=0),
                load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
                client=HTTPClientConfig(
                    num_workers=1, warmup_connections=0, max_connections=10
                ),
            ),
            report_dir=str(report_dir),
        )

        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.BOTH)

        # Verify scoring artifacts were written
        assert (report_dir / "sample_idx_map.json").exists()
        assert (report_dir / "events.jsonl").exists()

        # Verify sample_idx_map has both phases
        with (report_dir / "sample_idx_map.json").open("rb") as f:
            import msgspec.json

            idx_map = msgspec.json.decode(f.read())
        assert "performance" in idx_map
        assert "echo_accuracy" in idx_map
        assert len(idx_map["echo_accuracy"]) == 5  # 5 accuracy samples

        # Verify events.jsonl has COMPLETE events (EventRecord format: "sample.complete")
        events_path = report_dir / "events.jsonl"
        with events_path.open() as f:
            events = [msgspec.json.decode(line.strip()) for line in f if line.strip()]
        complete_events = [
            e for e in events if e.get("event_type") == "sample.complete"
        ]
        # Should have both perf (3) and accuracy (5) completions
        assert len(complete_events) >= 5

        # Verify results.json was written with accuracy scores
        results_path = report_dir / "results.json"
        assert results_path.exists()
        with results_path.open() as f:
            results = json.load(f)

        assert "accuracy_scores" in results
        assert "echo_accuracy" in results["accuracy_scores"]
        score_data = results["accuracy_scores"]["echo_accuracy"]
        score = score_data["score"]

        # 3 correct out of 5 = 0.6 accuracy
        assert abs(score - 0.6) < 0.01, f"Expected 0.6, got {score}"

        # Verify logs mention scoring
        assert "Score for echo_accuracy" in caplog.text
