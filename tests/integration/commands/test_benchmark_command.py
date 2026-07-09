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

"""Integration tests for benchmark commands against echo server."""

import json
import re
from pathlib import Path

import pytest
import yaml
from inference_endpoint.commands.audit import run_audit
from inference_endpoint.commands.benchmark.execute import run_benchmark
from inference_endpoint.config.schema import (
    AuditConfig,
    AuditTestId,
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
)
from inference_endpoint.endpoint_client.config import HTTPClientConfig

_TEST_SETTINGS = Settings(
    runtime=RuntimeConfig(min_duration_ms=0),
    load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    client=HTTPClientConfig(num_workers=1, warmup_connections=0, max_connections=10),
)


def _config(endpoint_url: str, dataset_path: str, **overrides) -> BenchmarkConfig:
    """Build a minimal BenchmarkConfig for testing."""
    defaults = {
        "type": TestType.OFFLINE,
        "endpoint_config": EndpointConfig(endpoints=[endpoint_url]),
        "model_params": ModelParams(name="echo-server", streaming=StreamingMode.OFF),
        "datasets": [Dataset(path=dataset_path, type=DatasetType.PERFORMANCE)],
        "settings": _TEST_SETTINGS,
    }
    return BenchmarkConfig(**(defaults | overrides))


def _poisson_settings(target_qps: float, duration_s: int = 2) -> Settings:
    return Settings(
        runtime=RuntimeConfig(min_duration_ms=duration_s * 1000),
        load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=target_qps),
        client=HTTPClientConfig(
            num_workers=1, warmup_connections=0, max_connections=10
        ),
    )


class TestBenchmarkCommandIntegration:
    """Integration tests for benchmark commands with echo server."""

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
    def test_offline_benchmark(
        self, mock_http_echo_server, ds_dataset_path, caplog, streaming
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            model_params=ModelParams(name="echo-server", streaming=streaming),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Completed in" in caplog.text
        assert "successful" in caplog.text
        assert "QPS:" in caplog.text
        assert "Starting phase:" in caplog.text

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
    def test_poisson_benchmark(
        self, mock_http_echo_server, ds_dataset_path, caplog, streaming
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            type=TestType.ONLINE,
            model_params=ModelParams(name="echo-server", streaming=streaming),
            settings=_poisson_settings(target_qps=50),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Completed in" in caplog.text
        assert "successful" in caplog.text
        assert "Starting phase:" in caplog.text

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
    def test_concurrency_benchmark(
        self, mock_http_echo_server, ds_dataset_path, caplog, streaming
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            type=TestType.ONLINE,
            model_params=ModelParams(name="echo-server", streaming=streaming),
            settings=Settings(
                runtime=RuntimeConfig(min_duration_ms=2000),
                load_pattern=LoadPattern(
                    type=LoadPatternType.CONCURRENCY, target_concurrency=4
                ),
                client=HTTPClientConfig(
                    num_workers=1, warmup_connections=0, max_connections=10
                ),
            ),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Completed in" in caplog.text
        assert "successful" in caplog.text

    @pytest.mark.integration
    def test_results_json_output(
        self, mock_http_echo_server, ds_dataset_path, tmp_path
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            report_dir=tmp_path,
        )
        run_benchmark(config, TestMode.PERF)

        results_path = tmp_path / "results.json"
        assert results_path.exists()
        results = json.loads(results_path.read_text())
        assert "config" in results
        assert results["results"]["total"] > 0
        assert results["results"]["successful"] >= 0

    @pytest.mark.integration
    def test_result_summary_self_complete(
        self, mock_http_echo_server, ds_dataset_path, tmp_path
    ):
        """result_summary.json carries qps/tps without needing any sidecar."""
        run_benchmark(
            _config(mock_http_echo_server.url, ds_dataset_path, report_dir=tmp_path),
            TestMode.PERF,
        )

        summary = json.loads((tmp_path / "result_summary.json").read_text())
        assert summary["qps"] > 0
        assert "tps" in summary
        # report.txt is the human-readable companion — kept alongside the JSON.
        assert (tmp_path / "report.txt").exists()

    @pytest.mark.integration
    def test_mode_logging(self, mock_http_echo_server, ds_dataset_path, caplog):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            type=TestType.ONLINE,
            settings=_poisson_settings(target_qps=20),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Mode:" in caplog.text
        assert "QPS: 20" in caplog.text
        assert "Responses: False" in caplog.text

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "test_type,settings",
        [
            (
                TestType.OFFLINE,
                Settings(
                    runtime=RuntimeConfig(min_duration_ms=0),
                    load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
                    client=HTTPClientConfig(
                        num_workers=1, warmup_connections=0, max_connections=10
                    ),
                ),
            ),
            (
                TestType.ONLINE,
                Settings(
                    runtime=RuntimeConfig(min_duration_ms=0),
                    load_pattern=LoadPattern(
                        type=LoadPatternType.CONCURRENCY, target_concurrency=1
                    ),
                    client=HTTPClientConfig(
                        num_workers=1, warmup_connections=0, max_connections=10
                    ),
                ),
            ),
        ],
        ids=["offline", "single-stream"],
    )
    def test_audit_output_caching_two_phase_flow(
        self,
        mock_http_echo_server,
        ds_dataset_path,
        tmp_path,
        caplog,
        test_type,
        settings,
    ):
        """Output-caching audit (MLPerf TEST04) runs reference + output_caching
        phases for offline and single-stream.

        Exercises the redesigned audit: config block → run_audit orchestrator →
        AuditResult. Asserts both phase subdirs are created, the result file is
        written, and — against the no-caching echo server — the audit PASSes
        (result.passed and the verify .txt).

        Equal per-phase counts avoid a count bias, and a wide threshold absorbs
        connection-warmup skew: the audit phase runs second with warm pools, so
        on a trivial echo server it can be measurably faster than the reference
        without any caching. The exact threshold boundary is unit-tested in
        tests/unit/compliance/test_output_caching.py; here we only assert the
        no-caching PASS path is plumbed end-to-end.
        """
        config = BenchmarkConfig(
            type=test_type,
            audit=AuditConfig(
                test=AuditTestId.OUTPUT_CACHING_TEST,
                samples=5,
                audit_samples=5,
                sample_index=0,
                threshold=0.9,
            ),
            endpoint_config=EndpointConfig(endpoints=[mock_http_echo_server.url]),
            model_params=ModelParams(name="echo-server", streaming=StreamingMode.OFF),
            datasets=[Dataset(path=ds_dataset_path, type=DatasetType.PERFORMANCE)],
            settings=settings,
            report_dir=str(tmp_path),
        )

        # run_benchmark does the main run and returns report_dir; the caller
        # (cli._run, mirrored here) dispatches the audit under <report_dir>/audit/.
        with caplog.at_level("INFO"):
            report_dir = run_benchmark(config, TestMode.PERF)
            result = run_audit(config, report_dir / "audit")

        # All audit artifacts live under <report_dir>/audit/.
        assert (tmp_path / "audit" / "reference").is_dir()
        assert (tmp_path / "audit" / "output_caching").is_dir()
        # Orchestrator returned a result and wrote both result files.
        assert result is not None
        # No caching on the echo server → the audit must PASS (a regression that
        # always FAILs, or mis-plumbs the threshold, would otherwise slip by).
        assert result.passed is True
        assert result.test_id == AuditTestId.OUTPUT_CACHING_TEST.value
        result_path = tmp_path / "audit" / "audit_result.json"
        assert result_path.exists()
        verify_txt = (tmp_path / "audit" / "verify_OUTPUT_CACHING_TEST.txt").read_text()
        assert "Performance check pass: True" in verify_txt
        result_json = json.loads(result_path.read_text())
        assert result_json["passed"] is True

    @pytest.mark.integration
    def test_cli_run_dispatches_main_run_before_audit(
        self, mock_http_echo_server, ds_dataset_path, tmp_path, monkeypatch
    ):
        """cli._run must dispatch the main benchmark run before the audit
        (upstream MLPerf order)."""
        from inference_endpoint.commands.benchmark import cli

        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            audit=AuditConfig(
                test=AuditTestId.OUTPUT_CACHING_TEST,
                samples=5,
                audit_samples=5,
                sample_index=0,
                threshold=0.9,
            ),
            endpoint_config=EndpointConfig(endpoints=[mock_http_echo_server.url]),
            model_params=ModelParams(name="echo-server", streaming=StreamingMode.OFF),
            datasets=[Dataset(path=ds_dataset_path, type=DatasetType.PERFORMANCE)],
            settings=Settings(
                runtime=RuntimeConfig(min_duration_ms=0),
                load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
                client=HTTPClientConfig(
                    num_workers=1, warmup_connections=0, max_connections=10
                ),
            ),
            report_dir=str(tmp_path),
        )

        call_order: list[str] = []
        real_run_audit = cli.run_audit
        real_run_benchmark = cli.run_benchmark

        def _spy_run_audit(cfg, base_report_dir):
            call_order.append("audit")
            return real_run_audit(cfg, base_report_dir)

        def _spy_run_benchmark(cfg, mode):
            call_order.append("benchmark")
            return real_run_benchmark(cfg, mode)

        monkeypatch.setattr(cli, "run_audit", _spy_run_audit)
        monkeypatch.setattr(cli, "run_benchmark", _spy_run_benchmark)

        cli._run(config, [], TestMode.PERF)

        assert call_order == ["benchmark", "audit"]
        # Both phases still land under the one shared report_dir.
        assert (tmp_path / "audit" / "audit_result.json").exists()
        assert (tmp_path / "config.yaml").exists()


TEMPLATE_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "inference_endpoint"
    / "config"
    / "templates"
)

# Templates generated by regenerate_templates.py (excludes handwritten eval/submission)
_GENERATED_TEMPLATES = sorted(
    p.name
    for p in TEMPLATE_DIR.glob("*_template*.yaml")
    if p.name.startswith(("offline_", "online_", "concurrency_"))
)


# Local character-level tokenizer fixture used in place of the templates'
# default (which references gated `meta-llama/Llama-3.1-*`). The echo-server
# e2e path doesn't care about the model identity, only that a tokenizer
# loads for the metrics aggregator's ISL/OSL/TPOT triggers. Using a local
# fixture removes the HuggingFace Hub dependency from CI: no network call,
# no ~1 MB download, no HF_TOKEN requirement, and the load completes in
# milliseconds rather than seconds — well inside the parent launcher's
# readiness timeout. ``AutoTokenizer.from_pretrained`` supports local
# directories as a first-class input, so this uses the same production
# code path with no test-only hooks.
_TEST_TOKENIZER_DIR = Path(__file__).resolve().parents[2] / "assets/tokenizers/char"
_TEST_MODEL_NAME = str(_TEST_TOKENIZER_DIR)


def _resolve_template(template_path: Path, server_url: str) -> dict:
    """Load a template YAML, strip <PLACEHOLDER> wrappers, and patch for testing.

    Replaces placeholders with working values, swaps the gated default
    model for a non-gated tokenizer (so tests run without ``HF_TOKEN``),
    and caps ``n_samples_to_issue``. Everything else stays as the template
    defines it.
    """
    raw = template_path.read_text()
    # Strip <PLACEHOLDER eg: value> → value (all templates use eg: form)
    raw = re.sub(r"<[^>]*eg:\s*([^>]+)>", r"\1", raw)
    # Replace endpoint URLs with the test server
    raw = re.sub(r"http://localhost:\d+", server_url, raw)
    data = yaml.safe_load(raw)

    # Swap the placeholder-default model name for a non-gated tokenizer
    # (see _TEST_MODEL_NAME above) so these tests can run in CI without
    # HF_TOKEN.
    if "model_params" in data and isinstance(data["model_params"], dict):
        data["model_params"]["name"] = _TEST_MODEL_NAME

    # Cap total samples so test finishes in seconds
    data.setdefault("settings", {})
    data["settings"].setdefault("runtime", {})
    data["settings"]["runtime"]["n_samples_to_issue"] = 10

    # Bump the worker-init timeout for CI. The production default (60 s) is
    # tight on small CI runners where Python's `spawn`-mode multiprocessing
    # pays a full re-import cost per worker on top of ZMQ IPC setup; cold-
    # start of the *first* parametrized template (alphabetical, so
    # `concurrency_template.yaml`) consistently exceeds the budget in CI.
    # The other 5 templates benefit from warm module / IPC caches and don't
    # need the headroom. 120 s is a generous safety margin that does not
    # change the production default, only this integration test.
    data["settings"].setdefault("client", {})
    data["settings"]["client"]["worker_initialization_timeout"] = 120.0

    # Accuracy datasets can't run e2e against echo server (no scorer), so keep only performance datasets.
    data["datasets"] = [
        ds for ds in data.get("datasets", []) if ds.get("type") != "accuracy"
    ]
    return data


class TestTemplateIntegration:
    """Verify generated templates run end-to-end against a local server."""

    @pytest.mark.integration
    @pytest.mark.parametrize("template", _GENERATED_TEMPLATES)
    def test_template_runs(self, mock_http_echo_server, tmp_path, caplog, template):
        data = _resolve_template(TEMPLATE_DIR / template, mock_http_echo_server.url)
        tmp_yaml = tmp_path / template
        tmp_yaml.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        config = BenchmarkConfig.from_yaml_file(tmp_yaml)
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)
        assert "Completed in" in caplog.text
