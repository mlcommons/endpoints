"""Tests for SubmissionChecker using pre-built fixtures from test_submissions/."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from submission_checker.checker import SubmissionChecker
from submission_checker.models import CheckResult, Report, Severity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _errors(report: Report, rule: str) -> list[CheckResult]:
    return [r for r in report.results if r.rule == rule and r.severity == Severity.ERROR]


def _warnings(report: Report, rule: str) -> list[CheckResult]:
    return [r for r in report.results if r.rule == rule and r.severity == Severity.WARNING]


def _check(path: Path) -> Report:
    return SubmissionChecker(path).run()


# ---------------------------------------------------------------------------
# valid_standardized — fully compliant synthetic fixture, must pass everything
# ---------------------------------------------------------------------------


class TestValidStandardized:
    def test_passes_overall(self, valid_standardized):
        assert _check(valid_standardized).passed

    def test_all_regions_covered(self, valid_standardized):
        report = _check(valid_standardized)
        for rule in [
            "low-latency-coverage",
            "low-throughput-coverage",
            "med-throughput-coverage",
            "high-throughput-coverage",
        ]:
            assert not _errors(report, rule), f"{rule} should pass"

    def test_metric_consistency(self, valid_standardized):
        report = _check(valid_standardized)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")

    def test_accuracy_gate(self, valid_standardized):
        assert not _errors(_check(valid_standardized), "accuracy-gate")

    def test_point_count(self, valid_standardized):
        assert not _errors(_check(valid_standardized), "run-count")


# ---------------------------------------------------------------------------
# invalid_submission — 3 points, failed accuracy; must trigger specific errors
# ---------------------------------------------------------------------------


class TestInvalidSubmission:
    def test_fails_overall(self, invalid_submission):
        assert not _check(invalid_submission).passed

    def test_point_count_error(self, invalid_submission):
        assert _errors(_check(invalid_submission), "run-count")

    def test_accuracy_gate_error(self, invalid_submission):
        assert _errors(_check(invalid_submission), "accuracy-gate")

    def test_missing_throughput_regions(self, invalid_submission):
        report = _check(invalid_submission)
        # Only 3 points (c16, c38, c88) — no high-throughput coverage
        assert _errors(report, "high-throughput-coverage")


# ---------------------------------------------------------------------------
# sub_a / sub_b — MI355X 8/16-GPU, gpt-oss-120b, M=2048, 7 points
# Concurrencies: 4,16,64,128,512,1024,2048 — jumps 16→64, skipping LT (33–44)
# ---------------------------------------------------------------------------


class TestSubA:
    def test_point_count_passes(self, sub_a):
        assert not _errors(_check(sub_a), "run-count")

    def test_low_latency_covered(self, sub_a):
        assert not _errors(_check(sub_a), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_a):
        assert _errors(_check(sub_a), "low-throughput-coverage")

    def test_metric_consistency(self, sub_a):
        report = _check(sub_a)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")

    def test_accuracy_passes(self, sub_a):
        assert not _errors(_check(sub_a), "accuracy-gate")


class TestSubB:
    def test_low_throughput_missing(self, sub_b):
        assert _errors(_check(sub_b), "low-throughput-coverage")

    def test_metric_consistency(self, sub_b):
        report = _check(sub_b)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# sub_c / sub_d — TPU 4/8-chip, qwen3-coder-480b, M=512/1024, 7/8 points
# ---------------------------------------------------------------------------


class TestSubC:
    def test_point_count_passes(self, sub_c):
        assert not _errors(_check(sub_c), "run-count")

    def test_low_latency_covered(self, sub_c):
        assert not _errors(_check(sub_c), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_c):
        assert _errors(_check(sub_c), "low-throughput-coverage")

    def test_metric_consistency(self, sub_c):
        report = _check(sub_c)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


class TestSubD:
    def test_low_throughput_missing(self, sub_d):
        assert _errors(_check(sub_d), "low-throughput-coverage")

    def test_metric_consistency(self, sub_d):
        report = _check(sub_d)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# sub_e / sub_f — Gaudi, llama3-8b, M=1024, 11 points (1–1024)
# Concurrencies include 32 (LL) and 64 (MT) — LT (33–42) still skipped
# ---------------------------------------------------------------------------


class TestSubE:
    def test_point_count_passes(self, sub_e):
        assert not _errors(_check(sub_e), "run-count")

    def test_low_latency_covered(self, sub_e):
        assert not _errors(_check(sub_e), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_e):
        assert _errors(_check(sub_e), "low-throughput-coverage")

    def test_high_throughput_covered(self, sub_e):
        assert not _errors(_check(sub_e), "high-throughput-coverage")

    def test_metric_consistency(self, sub_e):
        report = _check(sub_e)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


class TestSubF:
    def test_metric_consistency(self, sub_f):
        report = _check(sub_f)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# sub_g / sub_h — 8-GPU vLLM/SGLang, llama3-70b, M=2048, 10 points
# Minimum concurrency is 64 — both LL (1–32) and LT (33–44) missing
# ---------------------------------------------------------------------------


class TestSubG:
    def test_low_latency_missing(self, sub_g):
        assert _errors(_check(sub_g), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_g):
        assert _errors(_check(sub_g), "low-throughput-coverage")

    def test_point_count_passes(self, sub_g):
        assert not _errors(_check(sub_g), "run-count")

    def test_metric_consistency(self, sub_g):
        report = _check(sub_g)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


class TestSubH:
    def test_low_latency_missing(self, sub_h):
        assert _errors(_check(sub_h), "low-latency-coverage")

    def test_metric_consistency(self, sub_h):
        report = _check(sub_h)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# sub_i — H200 8-GPU, deepseek-r1, M=512, 10 points (1–512)
# LT region is 33–40; 32 is LL, 64 is MT — LT skipped
# Short durations → run-duration WARNINGs (not errors) on some points
# ---------------------------------------------------------------------------


class TestSubI:
    def test_low_latency_covered(self, sub_i):
        assert not _errors(_check(sub_i), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_i):
        assert _errors(_check(sub_i), "low-throughput-coverage")

    def test_metric_consistency(self, sub_i):
        report = _check(sub_i)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")

    def test_run_duration_warnings_not_errors(self, sub_i):
        report = _check(sub_i)
        assert not _errors(report, "run-duration"), "run-duration fires as WARNING, not ERROR"


# ---------------------------------------------------------------------------
# sub_j — GB300 72-GPU, deepseek-r1, M=16384, 10 points (32–16384)
# LT region is 33–57; 32 is LL, 64 is MT — LT skipped
# ---------------------------------------------------------------------------


class TestSubJ:
    def test_low_latency_covered(self, sub_j):
        assert not _errors(_check(sub_j), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_j):
        assert _errors(_check(sub_j), "low-throughput-coverage")

    def test_high_throughput_covered(self, sub_j):
        assert not _errors(_check(sub_j), "high-throughput-coverage")

    def test_metric_consistency(self, sub_j):
        report = _check(sub_j)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# Targeted edge-case tests that build synthetic dirs to cover checker.py paths
# ---------------------------------------------------------------------------

_SYSTEM_DESC = {
    "division": "Serviced",
    "publication_status": "Available",
    "benchmark_model": "llama3-70b",
    "max_supported_concurrency": 1024,
    "endpoint_url": "http://localhost",
    "serving_framework": "vLLM",
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

_SUMMARY = {
    "n_samples_issued": 1000,
    "n_samples_completed": 1000,
    "n_samples_failed": 0,
    "duration_ns": 1_200_000_000_000.0,
    "ttft": {"total": 0.0, "percentiles": {"50": 150_000_000.0, "95": 300_000_000.0}},
    "output_sequence_lengths": {"total": 500_000.0, "percentiles": {}},
}

_ACCURACY = {"metric": "rouge1", "score": 0.45, "quality_target": 0.43, "passed": True}

# Concurrencies that cover all four regions for M=1024
# LT: 33–42 → 38; MT: 43–175 → 88; HT: 176–1126 → 256, 512, 768, 1000
_CONCURRENCIES = [16, 38, 88, 256, 512, 768, 1000]


def _make_run_yaml(concurrency: int) -> dict:
    return {
        "concurrency": concurrency,
        "dataset": "llm-perf-dataset-v1",
        "runtime_settings": {
            "load_pattern": "concurrency",
            "min_duration_ms": 1_200_000,
            "stream_all_chunks": True,
        },
    }


def _build_submission(
    root: Path,
    system_id: str = "test-sys",
    system_desc: dict | None = None,
    concurrencies: list[int] | None = None,
    write_runs: bool = True,
    write_results: bool = True,
    write_accuracy: bool = True,
    write_accuracy_json: bool = True,
    accuracy_data: dict | None = None,
    model: str = "llama3-70b",
) -> Path:
    """Build a minimal valid (or deliberately broken) submission directory."""
    desc = system_desc if system_desc is not None else _SYSTEM_DESC.copy()
    desc["benchmark_model"] = model
    concs = concurrencies if concurrencies is not None else _CONCURRENCIES

    systems_dir = root / "systems"
    systems_dir.mkdir(parents=True)
    (systems_dir / f"{system_id}.json").write_text(json.dumps(desc))

    pareto_dir = root / "pareto"
    model_dir = pareto_dir / system_id / model
    points_dir = model_dir / "points"
    results_dir = model_dir / "results"
    accuracy_dir = model_dir / "accuracy"
    for d in (points_dir, results_dir, accuracy_dir):
        d.mkdir(parents=True)

    if write_runs:
        for c in concs:
            (points_dir / f"point_{c}.yaml").write_text(yaml.dump(_make_run_yaml(c)))

    if write_results:
        for c in concs:
            result_dir = results_dir / f"point_{c}"
            result_dir.mkdir(parents=True)
            (result_dir / "mlperf_endpoints_log_summary.json").write_text(json.dumps(_SUMMARY))

    if write_accuracy:
        (accuracy_dir / "accuracy.txt").write_text("ROUGE-1: 0.45")
    if write_accuracy_json:
        data = accuracy_data if accuracy_data is not None else _ACCURACY
        (accuracy_dir / "accuracy_result.json").write_text(json.dumps(data))

    return root


class TestCheckerEdgeCases:
    """Targeted tests to cover checker.py error paths not exercised by fixture tests."""

    def test_nonexistent_path(self, tmp_path):
        """path-exists error when submission_path does not exist."""
        report = _check(tmp_path / "does_not_exist")
        assert _errors(report, "path-exists")

    def test_missing_required_dirs_early_exit(self, tmp_path):
        """SubmissionDir structure errors cause early return from run()."""
        # Only systems/ present — pareto/ missing → structure error → early exit
        (tmp_path / "systems").mkdir()
        report = _check(tmp_path)
        assert _errors(report, "required-dir")
        # Should not have processed any systems
        assert not any(r.rule == "system-description-present" for r in report.results)

    def test_no_system_json_files(self, tmp_path):
        """system-description-present error when systems/ has no *.json files."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        report = _check(tmp_path)
        assert _errors(report, "system-description-present")

    def test_invalid_system_json(self, tmp_path):
        """system-description-valid error when system JSON is malformed."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        (tmp_path / "systems" / "bad-sys.json").write_text("{bad json")
        report = _check(tmp_path)
        assert _errors(report, "system-description-valid")

    def test_missing_pareto_system_dir_early_exit(self, tmp_path):
        """pareto-dir-exists error when pareto/<system_id>/ is absent."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        (tmp_path / "systems" / "test-sys.json").write_text(json.dumps(_SYSTEM_DESC))
        report = _check(tmp_path)
        assert _errors(report, "pareto-dir-exists")

    def test_empty_pareto_system_dir(self, tmp_path):
        """benchmark-model-dir error when pareto/<system_id>/ has no subdirectories."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "systems" / "test-sys.json").write_text(json.dumps(_SYSTEM_DESC))
        pareto_sys = tmp_path / "pareto" / "test-sys"
        pareto_sys.mkdir(parents=True)
        report = _check(tmp_path)
        assert _errors(report, "benchmark-model-dir")

    def test_missing_model_subdirs_early_exit(self, tmp_path):
        """pareto-subdir error when points/ or results/ or accuracy/ is absent."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "systems" / "test-sys.json").write_text(json.dumps(_SYSTEM_DESC))
        model_dir = tmp_path / "pareto" / "test-sys" / "llama3-70b"
        # Only points/ present — results/ and accuracy/ missing
        (model_dir / "points").mkdir(parents=True)
        report = _check(tmp_path)
        assert _errors(report, "pareto-subdir")
        # Should not attempt to list point_*.yaml (early exit after structure errors)
        assert not any(r.rule == "measurement-runs-present" for r in report.results)

    def test_no_run_yamls(self, tmp_path):
        """measurement-runs-present error when runs/ has no run_*.yaml files."""
        root = _build_submission(tmp_path, write_runs=False, write_results=False)
        report = _check(root)
        assert _errors(report, "measurement-runs-present")

    def test_missing_result_log(self, tmp_path):
        """result-file-present error when results/run_<N>/ log is absent."""
        root = _build_submission(tmp_path, write_results=False)
        report = _check(root)
        assert _errors(report, "result-file-present")

    def test_invalid_result_log(self, tmp_path):
        """result-file-valid error when the result log JSON is malformed."""
        root = _build_submission(tmp_path)
        # Overwrite one summary with invalid JSON
        bad_path = root / "pareto" / "test-sys" / "llama3-70b" / "results" / "point_16"
        bad_path.mkdir(parents=True, exist_ok=True)
        (bad_path / "mlperf_endpoints_log_summary.json").write_text("{bad")
        report = _check(root)
        assert _errors(report, "result-file-valid")

    def test_missing_accuracy_txt(self, tmp_path):
        """accuracy-file error when accuracy/accuracy.txt is absent."""
        root = _build_submission(tmp_path, write_accuracy=False)
        report = _check(root)
        assert _errors(report, "accuracy-file")

    def test_missing_accuracy_json(self, tmp_path):
        """accuracy-file error when accuracy/accuracy_result.json is absent."""
        root = _build_submission(tmp_path, write_accuracy_json=False)
        report = _check(root)
        assert _errors(report, "accuracy-file")

    def test_invalid_accuracy_json(self, tmp_path):
        """accuracy-valid error when accuracy_result.json is malformed."""
        root = _build_submission(
            tmp_path,
            accuracy_data={"metric": "rouge1"},  # missing required fields
        )
        report = _check(root)
        assert _errors(report, "accuracy-valid")

    def test_run_filename_concurrency_mismatch(self, tmp_path):
        """run-filename-concurrency warning when filename concurrency ≠ declared concurrency."""
        root = _build_submission(tmp_path)
        # Add a point file whose name says 999 but YAML declares 64
        mismatch_yaml = root / "pareto" / "test-sys" / "llama3-70b" / "points" / "point_999.yaml"
        mismatch_yaml.write_text(yaml.dump(_make_run_yaml(64)))
        # Also add the matching result dir so it doesn't error on result-file-present
        result_dir = root / "pareto" / "test-sys" / "llama3-70b" / "results" / "point_64"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "mlperf_endpoints_log_summary.json").write_text(json.dumps(_SUMMARY))
        report = _check(root)
        assert _warnings(report, "run-filename-concurrency")

    def test_invalid_run_yaml_is_skipped(self, tmp_path):
        """A point_*.yaml that fails validation does not crash the checker."""
        root = _build_submission(tmp_path)
        bad_yaml = root / "pareto" / "test-sys" / "llama3-70b" / "points" / "point_99.yaml"
        bad_yaml.write_text("{bad yaml [")
        report = _check(root)
        # Should produce a run-config-valid error for the bad file
        assert _errors(report, "run-config-valid")

    def test_region_computation_error(self, tmp_path):
        """region-computation error when compute_regions raises ValueError."""
        # compute_regions only raises if M <= 32, but SystemDescription enforces M > 32.
        # Patch compute_regions to simulate an unexpected ValueError.
        (tmp_path / "systems").mkdir()
        (tmp_path / "systems" / "test-sys.json").write_text(json.dumps(_SYSTEM_DESC))
        pareto_sys = tmp_path / "pareto" / "test-sys"
        pareto_sys.mkdir(parents=True)
        (pareto_sys / "llama3-70b").mkdir()
        with patch(
            "submission_checker.checker.compute_regions",
            side_effect=ValueError("M must be > 32"),
        ):
            report = _check(tmp_path)
        assert _errors(report, "region-computation")

    def test_run_filename_non_numeric_suffix_ignored(self, tmp_path):
        """Filename parsing errors (non-numeric suffix) are silently ignored."""
        root = _build_submission(tmp_path)
        # point_abc.yaml — stem is "point_abc", int("abc") raises ValueError
        # This tests the except (IndexError, ValueError): pass branch
        bad_name = root / "pareto" / "test-sys" / "llama3-70b" / "points" / "point_abc.yaml"
        bad_name.write_text(yaml.dump(_make_run_yaml(64)))
        result_dir = root / "pareto" / "test-sys" / "llama3-70b" / "results" / "point_64"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "mlperf_endpoints_log_summary.json").write_text(json.dumps(_SUMMARY))
        report = _check(root)
        # No run-filename-concurrency warning — the ValueError was swallowed
        assert not _warnings(report, "run-filename-concurrency")
