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

"""Integration test: a videogen benchmark and its TEST04 audit end to end."""

import json

import pytest
from inference_endpoint.commands.audit import run_audit
from inference_endpoint.commands.benchmark.execute import run_benchmark
from inference_endpoint.config.schema import (
    AuditConfig,
    AuditTestId,
    BenchmarkConfig,
    Dataset,
    DatasetType,
    EndpointConfig,
    ModelParams,
    Settings,
    StreamingMode,
    TestMode,
    TestType,
    WarmupConfig,
)
from inference_endpoint.core.types import APIType, TokenCountSource
from inference_endpoint.endpoint_client.config import HTTPClientConfig


@pytest.mark.integration
def test_videogen_run_and_audit_resolve_to_client_counts(mock_trtllm_serve, tmp_path):
    """Videogen endpoints report no usage, so every phase counts with the client."""
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(
        "".join(json.dumps({"prompt": f"a cat {i}"}) + "\n" for i in range(5))
    )
    config = BenchmarkConfig(
        type=TestType.OFFLINE,
        audit=AuditConfig(
            test=AuditTestId.OUTPUT_CACHING_TEST,
            samples=5,
            audit_samples=5,
            sample_index=0,
            threshold=0.9,
        ),
        endpoint_config=EndpointConfig(
            endpoints=[mock_trtllm_serve.url], api_type=APIType.VIDEOGEN
        ),
        model_params=ModelParams(name="wan22", streaming=StreamingMode.OFF),
        datasets=[Dataset(path=str(dataset), type=DatasetType.PERFORMANCE)],
        settings=Settings(
            client=HTTPClientConfig(
                num_workers=1, warmup_connections=0, max_connections=10
            ),
            warmup=WarmupConfig(enabled=False),
            token_count_source=TokenCountSource.SERVER,
        ),
        report_dir=str(tmp_path / "r"),
    )
    # The setting asks for the server basis; the resolver overrides it.
    assert config.settings.token_count_source is TokenCountSource.SERVER

    report_dir = run_benchmark(config, TestMode.PERF)
    audit_dir = report_dir / "audit"
    result = run_audit(config, audit_dir)

    # verify() ran on complete phases and wrote its verdict.
    assert result.test_id == AuditTestId.OUTPUT_CACHING_TEST.value
    verdict = json.loads((audit_dir / "audit_output_caching_test.json").read_text())
    assert (verdict["test"], verdict["passed"]) == (result.test_id, result.passed)
    assert (audit_dir / "verify_OUTPUT_CACHING_TEST.txt").is_file()
    for phase in ("", "audit/reference/", "audit/output_caching/"):
        s = json.loads(
            (report_dir / f"{phase}performance/result_summary.json").read_text()
        )
        assert s["complete"] is True
        assert s["token_count_source"] == "client"
        assert "token_count_source" not in s["run_config"]
