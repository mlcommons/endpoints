"""Tests for Pydantic submission models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from submission_checker.models import (
    AccuracyResult,
    CheckResult,
    Division,
    PercentileStats,
    PublicationStatus,
    Report,
    PointConfig,
    PointSummary,
    RuntimeSettings,
    Severity,
    SystemDescription,
)

# ---------------------------------------------------------------------------
# CheckResult / Report (infrastructure models)
# ---------------------------------------------------------------------------


def test_check_result_passed_for_non_error():
    result = CheckResult(rule="r", message="ok", severity=Severity.WARNING)
    assert result.passed


def test_check_result_not_passed_for_error():
    result = CheckResult(rule="r", message="bad", severity=Severity.ERROR)
    assert not result.passed


def test_check_result_is_immutable():
    result = CheckResult(rule="r", message="ok")
    with pytest.raises(ValidationError):
        result.rule = "changed"  # type: ignore[misc]


def test_check_result_serialises_to_dict():
    result = CheckResult(rule="r", message="ok", severity=Severity.INFO)
    data = result.model_dump()
    assert data["rule"] == "r"
    assert data["severity"] == Severity.INFO
    assert data["passed"] is True


def test_check_result_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        CheckResult(rule="r", message="x", severity="not-a-severity")  # type: ignore[arg-type]


def test_report_errors_and_warnings(tmp_path: Path):
    report = Report(submission_path=tmp_path)
    report.results = [
        CheckResult(rule="a", message="err", severity=Severity.ERROR),
        CheckResult(rule="b", message="warn", severity=Severity.WARNING),
        CheckResult(rule="c", message="info", severity=Severity.INFO),
    ]
    assert len(report.errors) == 1
    assert len(report.warnings) == 1
    assert not report.passed


def test_report_passed_with_no_errors(tmp_path: Path):
    report = Report(submission_path=tmp_path)
    report.results = [CheckResult(rule="b", message="warn", severity=Severity.WARNING)]
    assert report.passed


def test_report_model_dump_includes_computed_fields(tmp_path: Path):
    report = Report(
        submission_path=tmp_path,
        results=[CheckResult(rule="r", message="bad", severity=Severity.ERROR)],
    )
    data = report.model_dump()
    assert data["passed"] is False
    assert len(data["errors"]) == 1


# ---------------------------------------------------------------------------
# SystemDescription
# ---------------------------------------------------------------------------

_HW_FIELDS = {
    "submitter": "Test Org",
    "system_name": "test-node",
    "system_type": "datacenter",
    "system_type_detail": "",
    "number_of_nodes": 1,
    "host_processors_per_node": 2,
    "host_processor_model_name": "Intel Xeon Gold 6148",
    "host_processor_core_count": 20,
    "host_memory_capacity": "384 GB",
    "host_storage_type": "NVMe SSD",
    "host_storage_capacity": "10 TB",
    "host_networking": "InfiniBand EDR",
    "host_networking_topology": "Single switch",
    "accelerators_per_node": 8,
    "accelerator_model_name": "NVIDIA A100-SXM4-80GB",
    "accelerator_memory_capacity": "80 GB HBM2e",
    "operating_system": "Ubuntu 20.04",
}


def test_system_description_valid():
    sd = SystemDescription(
        division=Division.STANDARDIZED,
        publication_status=PublicationStatus.AVAILABLE,
        benchmark_model="llama3-70b",
        max_supported_concurrency=1024,
        endpoint_url="https://example.com",
        serving_framework="vLLM 0.4.0",
        **_HW_FIELDS,
    )
    assert sd.division == Division.STANDARDIZED
    assert sd.max_supported_concurrency == 1024


def test_system_description_rejects_m_le_32():
    with pytest.raises(ValidationError):
        SystemDescription(
            division=Division.STANDARDIZED,
            publication_status=PublicationStatus.AVAILABLE,
            benchmark_model="llama3-70b",
            max_supported_concurrency=32,  # must be > 32
            endpoint_url="https://example.com",
            serving_framework="vLLM 0.4.0",
            **_HW_FIELDS,
        )


def test_system_description_rejects_missing_core_and_vcpu():
    hw = {k: v for k, v in _HW_FIELDS.items() if k not in ("host_processor_core_count",)}
    with pytest.raises(ValidationError):
        SystemDescription(
            division=Division.STANDARDIZED,
            publication_status=PublicationStatus.AVAILABLE,
            benchmark_model="llama3-70b",
            max_supported_concurrency=1024,
            endpoint_url="https://example.com",
            serving_framework="vLLM 0.4.0",
            **hw,
        )


def test_system_description_accepts_vcpu_without_core_count():
    hw = {k: v for k, v in _HW_FIELDS.items() if k != "host_processor_core_count"}
    sd = SystemDescription(
        division=Division.STANDARDIZED,
        publication_status=PublicationStatus.AVAILABLE,
        benchmark_model="llama3-70b",
        max_supported_concurrency=1024,
        endpoint_url="https://example.com",
        serving_framework="vLLM 0.4.0",
        host_processor_vcpu_count=40,
        **hw,
    )
    assert sd.host_processor_vcpu_count == 40
    assert sd.host_processor_core_count is None


def test_system_description_allows_extra_fields():
    sd = SystemDescription(
        division=Division.RDI,
        publication_status=PublicationStatus.RDI,
        benchmark_model="test-model",
        max_supported_concurrency=64,
        endpoint_url="http://localhost",
        serving_framework="custom",
        extra_hw_detail="some extra info",  # allowed via extra="allow"
        **_HW_FIELDS,
    )
    assert sd.model_extra["extra_hw_detail"] == "some extra info"


# ---------------------------------------------------------------------------
# PointConfig
# ---------------------------------------------------------------------------


def test_measurement_point_config_valid():
    cfg = PointConfig(
        concurrency=64,
        dataset="test-ds",
        runtime_settings=RuntimeSettings(min_duration_ms=1_200_000),
    )
    assert cfg.concurrency == 64
    assert cfg.dataset == "test-ds"
    assert cfg.runtime_settings.stream_all_chunks is True


def test_measurement_point_config_concurrency_stored():
    cfg = PointConfig(concurrency=128)
    assert cfg.concurrency == 128


def test_measurement_point_config_empty_dataset():
    cfg = PointConfig(concurrency=32)
    assert cfg.dataset == ""


def test_measurement_point_config_load_pattern_stored():
    cfg = PointConfig(concurrency=64)
    assert cfg.runtime_settings.load_pattern == "concurrency"


# ---------------------------------------------------------------------------
# PointSummary
# ---------------------------------------------------------------------------


def _make_summary(
    n_completed: int = 1000,
    n_issued: int = 1000,
    n_failed: int = 0,
    duration_ns: float = 600_000_000_000.0,
    total_tokens: float = 500_000.0,
    ttft_p50_ns: float = 150_000_000.0,
    ttft_p95_ns: float = 300_000_000.0,
) -> PointSummary:
    return PointSummary(
        n_samples_completed=n_completed,
        n_samples_issued=n_issued,
        n_samples_failed=n_failed,
        duration_ns=duration_ns,
        ttft=PercentileStats(
            total=ttft_p50_ns * n_completed, percentiles={"50": ttft_p50_ns, "95": ttft_p95_ns}
        ),
        output_sequence_lengths=PercentileStats(total=total_tokens),
    )


def test_point_result_summary_duration_ms():
    s = _make_summary(duration_ns=1_200_000_000_000.0)
    assert abs(s.duration_ms - 1_200_000.0) < 1.0


def test_point_result_summary_sample_count_alias():
    s = _make_summary(n_completed=512)
    assert s.sample_count == 512


def test_point_result_summary_system_tps():
    # 600 s run, 60000 tokens → 100 tok/s
    s = _make_summary(duration_ns=600_000_000_000.0, total_tokens=60_000.0)
    assert abs(s.system_tps - 100.0) < 0.01


def test_point_result_summary_ttft_ms_conversion():
    s = _make_summary(ttft_p50_ns=150_000_000.0, ttft_p95_ns=300_000_000.0)
    assert abs(s.ttft_p50_ms - 150.0) < 0.001
    assert abs(s.ttft_p95_ms - 300.0) < 0.001


def test_point_result_summary_total_output_tokens():
    s = _make_summary(total_tokens=22_407_098.0)
    assert s.total_output_tokens == 22_407_098


# ---------------------------------------------------------------------------
# AccuracyResult
# ---------------------------------------------------------------------------


def test_accuracy_result_passed():
    ar = AccuracyResult(metric="rouge1", score=0.45, quality_target=0.43, passed=True)
    assert ar.passed


def test_accuracy_result_failed():
    ar = AccuracyResult(metric="rouge1", score=0.38, quality_target=0.43, passed=False)
    assert not ar.passed
