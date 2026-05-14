"""Unit tests for check validators and structural gate checks."""

from __future__ import annotations

from pathlib import Path

from submission_checker.models import (
    AccuracyResult,
    CheckResult,
    ClientConfig,
    Division,
    LoadPatternConfig,
    ModelContext,
    PercentileStats,
    PublicationStatus,
    RunConfig,
    RunResult,
    RunSummary,
    RuntimeConfig,
    SettingsConfig,
    Severity,
    SystemDescription,
    compute_regions,
)
from submission_checker.structure import ModelDir, SrcDir, SubmissionDir, SystemPareto

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_HW = {
    "submitter": "Test Org",
    "system_name": "test-sys",
    "system_type": "datacenter",
    "system_type_detail": "",
    "number_of_nodes": 1,
    "host_processors_per_node": 2,
    "host_processor_model_name": "AMD EPYC",
    "host_processor_core_count": 64,
    "host_memory_capacity": "512 GB",
    "host_storage_type": "NVMe",
    "host_storage_capacity": "10 TB",
    "host_networking": "InfiniBand",
    "host_networking_topology": "Single switch",
    "accelerators_per_node": 8,
    "accelerator_model_name": "H100",
    "accelerator_memory_capacity": "80 GB",
    "operating_system": "Ubuntu 22.04",
}

_M = 1024
_REGIONS = compute_regions(_M)


def _system_desc(
    division: Division = Division.STANDARDIZED,
    benchmark_model: str = "llama3-70b",
    **kwargs,
) -> SystemDescription:
    return SystemDescription(
        division=division,
        publication_status=PublicationStatus.AVAILABLE,
        benchmark_model=benchmark_model,
        max_supported_concurrency=_M,
        endpoint_url="http://localhost",
        serving_framework="vLLM",
        **_HW,
        **kwargs,
    )


def _config(concurrency: int = 64, stream: bool = True, lp_type: str = "concurrency") -> RunConfig:
    return RunConfig(
        settings=SettingsConfig(
            load_pattern=LoadPatternConfig(type=lp_type, target_concurrency=concurrency),
            runtime=RuntimeConfig(min_duration_ms=1_200_000),
            client=ClientConfig(stream_all_chunks=stream),
        ),
        datasets=[{"name": "mlperf-perf-dataset-v1"}],
    )


def _summary(
    n_completed: int = 1000,
    n_issued: int = 1000,
    n_failed: int = 0,
    duration_ns: float = 1_200_000_000_000.0,
    total_tokens: float = 500_000.0,
) -> RunSummary:
    return RunSummary(
        n_samples_completed=n_completed,
        n_samples_issued=n_issued,
        n_samples_failed=n_failed,
        duration_ns=duration_ns,
        ttft=PercentileStats(total=0.0, percentiles={"50": 150_000_000.0, "95": 300_000_000.0}),
        output_sequence_lengths=PercentileStats(total=total_tokens),
    )


def _passed(results: list[CheckResult]) -> bool:
    return all(r.severity != Severity.ERROR for r in results)


def _model_ctx(
    tmp_path: Path,
    all_run_count: int = 7,
    valid_runs: list[tuple[Path, RunConfig]] | None = None,
    loaded_results: list[tuple[RunConfig, RunSummary]] | None = None,
    system_desc: SystemDescription | None = None,
    model_name: str = "llama3-70b",
    accuracy_result: AccuracyResult | None = None,
) -> ModelContext:
    model_dir = tmp_path / model_name
    model_dir.mkdir(exist_ok=True)
    (model_dir / "runs").mkdir(exist_ok=True)
    (model_dir / "results").mkdir(exist_ok=True)
    (model_dir / "accuracy").mkdir(exist_ok=True)
    return ModelContext(
        system_id="test-sys",
        system_desc=system_desc or _system_desc(benchmark_model=model_name),
        model_dir=model_dir,
        regions=_REGIONS,
        runs_dir=model_dir / "runs",
        accuracy_dir=model_dir / "accuracy",
        all_run_count=all_run_count,
        valid_runs=valid_runs or [],
        loaded_results=loaded_results or [],
        accuracy_result=accuracy_result,
    )


# ---------------------------------------------------------------------------
# structure.SubmissionDir
# ---------------------------------------------------------------------------


class TestSubmissionDir:
    def test_missing_dir(self, tmp_path):
        (tmp_path / "systems").mkdir()
        # pareto/ intentionally absent
        sd = SubmissionDir(root=tmp_path)
        rules = {r.rule for r in sd._check_results if r.severity == Severity.ERROR}
        assert "required-dir" in rules

    def test_both_present(self, tmp_path):
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        sd = SubmissionDir(root=tmp_path)
        assert _passed(sd._check_results)

    def test_computed_paths(self, tmp_path):
        sd = SubmissionDir(root=tmp_path)
        assert sd.systems_dir == tmp_path / "systems"
        assert sd.pareto_dir == tmp_path / "pareto"


# ---------------------------------------------------------------------------
# structure.SystemPareto
# ---------------------------------------------------------------------------


class TestSystemPareto:
    def test_missing_system_pareto(self, tmp_path):
        sp = SystemPareto(pareto_dir=tmp_path, system_id="sys-x")
        assert any(r.severity == Severity.ERROR for r in sp._check_results)

    def test_present(self, tmp_path):
        (tmp_path / "sys-x").mkdir()
        sp = SystemPareto(pareto_dir=tmp_path, system_id="sys-x")
        assert _passed(sp._check_results)

    def test_system_dir_computed(self, tmp_path):
        sp = SystemPareto(pareto_dir=tmp_path, system_id="sys-x")
        assert sp.system_dir == tmp_path / "sys-x"


# ---------------------------------------------------------------------------
# structure.ModelDir
# ---------------------------------------------------------------------------


class TestModelDir:
    def test_missing_subdir(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "runs").mkdir()
        (model_dir / "results").mkdir()
        # accuracy/ absent
        md = ModelDir(root=model_dir, system_id="sys-x", benchmark_model="llama3-70b")
        assert any(r.severity == Severity.ERROR for r in md._check_results)

    def test_all_present(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        for d in ("runs", "results", "accuracy"):
            (model_dir / d).mkdir()
        md = ModelDir(root=model_dir, system_id="sys-x", benchmark_model="llama3-70b")
        assert _passed(md._check_results)

    def test_computed_paths(self, tmp_path):
        md = ModelDir(root=tmp_path, system_id="sys-x", benchmark_model="llama3-70b")
        assert md.runs_dir == tmp_path / "runs"
        assert md.results_dir == tmp_path / "results"
        assert md.accuracy_dir == tmp_path / "accuracy"


# ---------------------------------------------------------------------------
# structure.SrcDir
# ---------------------------------------------------------------------------


class TestSrcDir:
    def test_standardized_missing_src(self, tmp_path):
        sd = SrcDir(root=tmp_path, division=Division.STANDARDIZED)
        assert any(r.severity == Severity.ERROR for r in sd._check_results)

    def test_standardized_src_present(self, tmp_path):
        (tmp_path / "src").mkdir()
        sd = SrcDir(root=tmp_path, division=Division.STANDARDIZED)
        assert _passed(sd._check_results)

    def test_non_standardized_skipped(self, tmp_path):
        sd = SrcDir(root=tmp_path, division=Division.SERVICED)
        assert sd._check_results == []


# ---------------------------------------------------------------------------
# RunConfig validators
# ---------------------------------------------------------------------------


class TestLoadPatternValidator:
    def test_wrong_type(self, tmp_path):
        config = RunConfig.model_validate(
            {"settings": {"load_pattern": {"type": "qps", "target_qps": 10.0}}},
            context={"yaml_path": tmp_path / "run_64.yaml"},
        )
        errors = [
            r
            for r in config._check_results
            if r.severity == Severity.ERROR and r.rule == "load-pattern"
        ]
        assert errors

    def test_missing_target_concurrency(self, tmp_path):
        config = RunConfig.model_validate(
            {"settings": {"load_pattern": {"type": "concurrency", "target_concurrency": None}}},
            context={"yaml_path": tmp_path / "run_64.yaml"},
        )
        errors = [
            r
            for r in config._check_results
            if r.severity == Severity.ERROR and r.rule == "load-pattern"
        ]
        assert errors

    def test_valid(self, tmp_path):
        config = RunConfig.model_validate(
            {"settings": {"load_pattern": {"type": "concurrency", "target_concurrency": 64}}},
            context={"yaml_path": tmp_path / "run_64.yaml"},
        )
        assert all(
            r.severity != Severity.ERROR for r in config._check_results if r.rule == "load-pattern"
        )


class TestStreamingValidator:
    def test_stream_false_errors(self, tmp_path):
        config = RunConfig.model_validate(
            {
                "settings": {
                    "load_pattern": {"type": "concurrency", "target_concurrency": 64},
                    "client": {"stream_all_chunks": False},
                }
            },
            context={"yaml_path": tmp_path / "run_64.yaml"},
        )
        errors = [
            r
            for r in config._check_results
            if r.rule == "streaming-config" and r.severity == Severity.ERROR
        ]
        assert errors

    def test_stream_true_passes(self, tmp_path):
        config = RunConfig.model_validate(
            {"settings": {"load_pattern": {"type": "concurrency", "target_concurrency": 64}}},
            context={"yaml_path": tmp_path / "run_64.yaml"},
        )
        assert all(
            r.severity != Severity.ERROR
            for r in config._check_results
            if r.rule == "streaming-config"
        )


class TestConcurrencyInRangeValidator:
    def test_out_of_range(self, tmp_path):
        config = RunConfig.model_validate(
            {"settings": {"load_pattern": {"type": "concurrency", "target_concurrency": 9999}}},
            context={"yaml_path": tmp_path / "run_9999.yaml", "regions": _REGIONS},
        )
        errors = [
            r
            for r in config._check_results
            if r.rule == "concurrency-in-range" and r.severity == Severity.ERROR
        ]
        assert errors

    def test_in_range(self, tmp_path):
        config = RunConfig.model_validate(
            {"settings": {"load_pattern": {"type": "concurrency", "target_concurrency": 64}}},
            context={"yaml_path": tmp_path / "run_64.yaml", "regions": _REGIONS},
        )
        assert all(
            r.severity != Severity.ERROR
            for r in config._check_results
            if r.rule == "concurrency-in-range"
        )

    def test_no_regions_skips_check(self, tmp_path):
        config = RunConfig.model_validate(
            {"settings": {"load_pattern": {"type": "concurrency", "target_concurrency": 9999}}},
            context={"yaml_path": tmp_path / "run_9999.yaml"},
        )
        # No regions in context — concurrency-in-range should not be present
        rules = {r.rule for r in config._check_results}
        assert "concurrency-in-range" not in rules


# ---------------------------------------------------------------------------
# RunResult validators
# ---------------------------------------------------------------------------


class TestRunDurationValidator:
    def test_short_duration_warns(self, tmp_path):
        config = _config(concurrency=64)
        short = _summary(duration_ns=100_000_000_000.0)  # 100 s — below any minimum
        run_result = RunResult.model_validate(
            {"config": config, "summary": short, "yaml_path": tmp_path / "run_64.yaml"},
            context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "run-duration" and r.severity == Severity.WARNING
            for r in run_result._check_results
        )

    def test_sufficient_duration_passes(self, tmp_path):
        config = _config(concurrency=64)
        long_s = _summary(duration_ns=1_200_000_000_000.0)  # 1200 s
        run_result = RunResult.model_validate(
            {"config": config, "summary": long_s, "yaml_path": tmp_path / "run_64.yaml"},
            context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
        )
        assert all(
            r.severity != Severity.ERROR
            for r in run_result._check_results
            if r.rule == "run-duration"
        )

    def test_out_of_range_concurrency_skipped(self, tmp_path):
        config = _config(concurrency=9999)
        run_result = RunResult.model_validate(
            {"config": config, "summary": _summary(), "yaml_path": tmp_path / "run_9999.yaml"},
            context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
        )
        # run-duration should not appear because concurrency is out of range
        assert not any(r.rule == "run-duration" for r in run_result._check_results)

    def test_no_regions_skips(self, tmp_path):
        config = _config(concurrency=64)
        run_result = RunResult.model_validate(
            {"config": config, "summary": _summary(), "yaml_path": tmp_path / "run_64.yaml"},
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert not any(r.rule == "run-duration" for r in run_result._check_results)


class TestMetricConsistencyValidator:
    def test_zero_duration_errors(self, tmp_path):
        config = _config(concurrency=64)
        run_result = RunResult.model_validate(
            {
                "config": config,
                "summary": _summary(duration_ns=0.0),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-duration" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )

    def test_negative_duration_errors(self, tmp_path):
        config = _config(concurrency=64)
        run_result = RunResult.model_validate(
            {
                "config": config,
                "summary": _summary(duration_ns=-1.0),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-duration" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )

    def test_accounting_mismatch_errors(self, tmp_path):
        config = _config(concurrency=64)
        run_result = RunResult.model_validate(
            {
                "config": config,
                "summary": _summary(n_completed=990, n_issued=1000, n_failed=5),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        # completed(990) + failed(5) = 995 ≠ issued(1000)
        assert any(
            r.rule == "metric-consistency-accounting" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )

    def test_valid_summary_passes(self, tmp_path):
        config = _config(concurrency=64)
        run_result = RunResult.model_validate(
            {"config": config, "summary": _summary(), "yaml_path": tmp_path / "run_64.yaml"},
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert all(r.severity != Severity.ERROR for r in run_result._check_results)

    def test_negative_output_tokens_errors(self, tmp_path):
        """Negative output token total must produce a metric-consistency-output-tokens error."""
        config = _config(concurrency=64)
        run_result = RunResult.model_validate(
            {
                "config": config,
                "summary": _summary(total_tokens=-1.0),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-output-tokens" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )


# ---------------------------------------------------------------------------
# ModelContext validators
# ---------------------------------------------------------------------------


class TestRunCountValidator:
    def test_too_few(self, tmp_path):
        ctx = _model_ctx(tmp_path, all_run_count=3)
        assert any(
            r.rule == "run-count" and r.severity == Severity.ERROR for r in ctx._check_results
        )

    def test_cap_exceeded(self, tmp_path):
        ctx = _model_ctx(tmp_path, all_run_count=33)
        assert any(r.rule == "run-cap" and r.severity == Severity.ERROR for r in ctx._check_results)

    def test_valid_count(self, tmp_path):
        ctx = _model_ctx(tmp_path, all_run_count=10)
        assert all(
            r.severity != Severity.ERROR
            for r in ctx._check_results
            if r.rule in ("run-count", "run-cap")
        )


class TestRegionalCoverageValidator:
    def test_no_runs_all_regions_missing(self, tmp_path):
        ctx = _model_ctx(tmp_path, valid_runs=[])
        coverage_rules = {
            "low-latency-coverage",
            "low-throughput-coverage",
            "med-throughput-coverage",
            "high-throughput-coverage",
        }
        errors = {
            r.rule
            for r in ctx._check_results
            if r.severity == Severity.ERROR and r.rule in coverage_rules
        }
        assert errors == coverage_rules

    def test_concurrency_in_low_latency(self, tmp_path):
        yaml_path = tmp_path / "llama3-70b" / "runs" / "run_16.yaml"
        valid_runs = [(yaml_path, _config(concurrency=16))]
        ctx = _model_ctx(tmp_path, valid_runs=valid_runs)
        assert all(
            r.severity != Severity.ERROR
            for r in ctx._check_results
            if r.rule == "low-latency-coverage"
        )


class TestConfigConsistencyValidator:
    def test_inconsistent_datasets(self, tmp_path):
        c1 = RunConfig(
            settings=SettingsConfig(
                load_pattern=LoadPatternConfig(type="concurrency", target_concurrency=64),
                runtime=RuntimeConfig(),
                client=ClientConfig(),
            ),
            datasets=[{"name": "dataset-a"}],
        )
        c2 = RunConfig(
            settings=SettingsConfig(
                load_pattern=LoadPatternConfig(type="concurrency", target_concurrency=128),
                runtime=RuntimeConfig(),
                client=ClientConfig(),
            ),
            datasets=[{"name": "dataset-b"}],
        )
        s = _summary()
        ctx = _model_ctx(tmp_path, loaded_results=[(c1, s), (c2, s)])
        assert any(
            r.rule == "config-consistency-dataset" and r.severity == Severity.ERROR
            for r in ctx._check_results
        )

    def test_model_name_mismatch(self, tmp_path):
        # model_dir.name will be "wrong-model", system_desc.benchmark_model is "wrong-model"
        # but _model_ctx uses model_name for both, so test a case where they differ
        model_dir = tmp_path / "actual-name"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "runs").mkdir(exist_ok=True)
        (model_dir / "results").mkdir(exist_ok=True)
        (model_dir / "accuracy").mkdir(exist_ok=True)
        s = _summary()
        ctx = ModelContext(
            system_id="test-sys",
            system_desc=_system_desc(benchmark_model="expected-name"),
            model_dir=model_dir,
            regions=_REGIONS,
            runs_dir=model_dir / "runs",
            accuracy_dir=model_dir / "accuracy",
            all_run_count=7,
            valid_runs=[],
            loaded_results=[(_config(), s)],
            accuracy_result=None,
        )
        assert any(
            r.rule == "config-consistency-model" and r.severity == Severity.WARNING
            for r in ctx._check_results
        )

    def test_empty_results_skips(self, tmp_path):
        ctx = _model_ctx(tmp_path, loaded_results=[])
        # config-consistency-dataset should not appear when loaded_results is empty
        assert not any(r.rule == "config-consistency-dataset" for r in ctx._check_results)


class TestAccuracyGateValidator:
    def test_no_accuracy_result_skips(self, tmp_path):
        ctx = _model_ctx(tmp_path, accuracy_result=None)
        assert not any(r.rule == "accuracy-gate" for r in ctx._check_results)

    def test_passed_accuracy_gate(self, tmp_path):
        ar = AccuracyResult(metric="rouge1", score=0.45, quality_target=0.43, passed=True)
        ctx = _model_ctx(tmp_path, accuracy_result=ar)
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.INFO for r in ctx._check_results
        )
        assert not any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR for r in ctx._check_results
        )

    def test_failed_accuracy_gate(self, tmp_path):
        ar = AccuracyResult(metric="rouge1", score=0.30, quality_target=0.43, passed=False)
        ctx = _model_ctx(tmp_path, accuracy_result=ar)
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR for r in ctx._check_results
        )
