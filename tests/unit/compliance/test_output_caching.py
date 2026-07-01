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

"""Unit tests for output-caching (MLPerf TEST04) audit logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pydantic
import pytest

from inference_endpoint import compliance
from inference_endpoint.commands.audit import run_audit
from inference_endpoint.compliance import (
    AuditRunArtifacts,
    AuditRunStats,
    get_audit_test,
)
from inference_endpoint.compliance.audit_test.output_caching_test import (
    OutputCachingAudit,
    verify_output_caching,
)
from inference_endpoint.compliance.result import AuditResult, write_result
from inference_endpoint.config.schema import (
    AuditTestId,
    LoadPatternType,
    OutputCachingTestConfig,
)
from inference_endpoint.exceptions import ExecutionError, SetupError

# Load patterns the audit must reject, derived from the enum so the test stays
# correct regardless of which patterns exist on the base (only max_throughput
# and concurrency are accepted).
_REJECTED_LOAD_PATTERNS = [
    p
    for p in LoadPatternType
    if p
    not in (
        LoadPatternType.MAX_THROUGHPUT,
        LoadPatternType.CONCURRENCY,
        LoadPatternType.POISSON,
    )
]

# ---------------------------------------------------------------------------
# verify_output_caching — pure function, no I/O
# ---------------------------------------------------------------------------


class TestVerifyOutputCaching:
    @pytest.mark.unit
    def test_pass_when_audit_qps_below_threshold(self):
        ref = AuditRunStats(qps=100.0, n_completed=1000, n_requested=1000)
        audit = AuditRunStats(qps=105.0, n_completed=1000, n_requested=1000)
        result = verify_output_caching(ref, audit, threshold=0.10)
        assert result.passed is True
        assert result.test_id == AuditTestId.OUTPUT_CACHING_TEST.value

    @pytest.mark.unit
    def test_fail_when_audit_qps_clearly_above_threshold(self):
        ref = AuditRunStats(qps=100.0, n_completed=1000, n_requested=1000)
        # audit 120 > limit 110 → FAIL
        audit = AuditRunStats(qps=120.0, n_completed=1000, n_requested=1000)
        result = verify_output_caching(ref, audit, threshold=0.10)
        assert result.passed is False

    @pytest.mark.unit
    def test_fail_when_audit_qps_exactly_at_limit(self):
        # audit_qps sits exactly on the limit ref_qps * (1 + threshold).
        # MLPerf's verify_performance.py uses a strict `<`, so a run exactly on
        # the boundary FAILs; match that reference behavior. Compute the limit
        # the same way the code does to avoid float-rounding ambiguity.
        ref_qps, threshold = 100.0, 0.10
        limit = ref_qps * (1.0 + threshold)
        ref = AuditRunStats(qps=ref_qps, n_completed=1000, n_requested=1000)
        audit = AuditRunStats(qps=limit, n_completed=1000, n_requested=1000)
        result = verify_output_caching(ref, audit, threshold=threshold)
        assert result.passed is False

    @pytest.mark.unit
    def test_fail_when_reference_phase_incomplete(self):
        ref = AuditRunStats(qps=100.0, n_completed=800, n_requested=1000)
        audit = AuditRunStats(qps=50.0, n_completed=1000, n_requested=1000)
        result = verify_output_caching(ref, audit, threshold=0.10)
        assert result.passed is False
        assert "Phase incomplete" in result.details["reason"]

    @pytest.mark.unit
    def test_fail_when_audit_phase_incomplete(self):
        ref = AuditRunStats(qps=100.0, n_completed=1000, n_requested=1000)
        audit = AuditRunStats(qps=50.0, n_completed=800, n_requested=1000)
        result = verify_output_caching(ref, audit, threshold=0.10)
        assert result.passed is False
        assert "Phase incomplete" in result.details["reason"]

    @pytest.mark.unit
    def test_details_contain_qps_values(self):
        ref = AuditRunStats(qps=100.0, n_completed=1000, n_requested=1000)
        audit = AuditRunStats(qps=50.0, n_completed=1000, n_requested=1000)
        result = verify_output_caching(ref, audit)
        assert result.details["ref_qps"] == 100.0
        assert result.details["audit_qps"] == 50.0
        assert result.details["threshold"] == 0.10

    @pytest.mark.unit
    def test_custom_threshold(self):
        ref = AuditRunStats(qps=100.0, n_completed=1000, n_requested=1000)
        audit = AuditRunStats(qps=105.0, n_completed=1000, n_requested=1000)
        # With threshold=0.02 the limit is 102.0 → audit 105 > limit → FAIL
        result = verify_output_caching(ref, audit, threshold=0.02)
        assert result.passed is False


# ---------------------------------------------------------------------------
# OutputCachingAudit.plan_runs
# ---------------------------------------------------------------------------


class TestOutputCachingAuditPlanRuns:
    def _make_cfg(self, samples=None, audit_samples=None, sample_index=0):
        return OutputCachingTestConfig(
            test=AuditTestId.OUTPUT_CACHING_TEST,
            samples=samples,
            audit_samples=audit_samples,
            sample_index=sample_index,
        )

    @pytest.mark.unit
    def test_samples_is_required(self):
        # Audits need an explicit reference count so the per-phase completion
        # check has an independent target (see OutputCachingTestConfig.samples).
        with pytest.raises(pydantic.ValidationError):
            OutputCachingTestConfig(
                test=AuditTestId.OUTPUT_CACHING_TEST, sample_index=0
            )

    @pytest.mark.unit
    def test_plan_produces_two_specs(self):
        cfg = self._make_cfg(samples=500)
        specs = OutputCachingAudit().plan_runs(cfg)
        assert len(specs) == 2

    @pytest.mark.unit
    def test_reference_spec_uses_without_replacement(self):
        cfg = self._make_cfg(samples=500)
        specs = OutputCachingAudit().plan_runs(cfg)
        ref = specs[0]
        assert ref.label == "reference"
        assert ref.n_samples == 500
        assert ref.sample_order.fixed_index is None

    @pytest.mark.unit
    def test_audit_spec_uses_single_index(self):
        cfg = self._make_cfg(samples=500, sample_index=7)
        specs = OutputCachingAudit().plan_runs(cfg)
        audit = specs[1]
        assert audit.label == "output_caching"
        assert audit.sample_order.fixed_index == 7

    @pytest.mark.unit
    def test_audit_n_defaults_to_ref_n_when_not_set(self):
        cfg = self._make_cfg(samples=300)
        specs = OutputCachingAudit().plan_runs(cfg)
        assert specs[1].n_samples == 300

    @pytest.mark.unit
    def test_audit_n_overrides_when_audit_samples_set(self):
        cfg = self._make_cfg(samples=300, audit_samples=150)
        specs = OutputCachingAudit().plan_runs(cfg)
        assert specs[1].n_samples == 150


# ---------------------------------------------------------------------------
# OutputCachingAudit.verify — threshold plumbing + phase-count guard
# ---------------------------------------------------------------------------


class TestOutputCachingAuditVerify:
    def _cfg(self, threshold=0.10):
        return OutputCachingTestConfig(
            test=AuditTestId.OUTPUT_CACHING_TEST,
            samples=1000,
            audit_samples=1000,
            sample_index=0,
            threshold=threshold,
        )

    def _arts(self, qps, label, n_completed=1000, n_requested=1000):
        rep = MagicMock()
        rep.qps = qps
        rep.n_samples_completed = n_completed
        return AuditRunArtifacts(
            label=label, report_dir=Path("/tmp"), report=rep, n_requested=n_requested
        )

    @pytest.mark.unit
    def test_verify_honors_configured_threshold(self):
        # Regression: verify() must apply cfg.threshold, not the default 0.10.
        # audit 115 vs ref 100 → limit is 110 at 0.10 (FAIL) but 120 at 0.20 (PASS).
        ref = self._arts(100.0, label="reference")
        audit = self._arts(115.0, label="output_caching")

        v10 = OutputCachingAudit().verify([ref, audit], self._cfg(threshold=0.10))
        assert v10.passed is False

        v20 = OutputCachingAudit().verify([ref, audit], self._cfg(threshold=0.20))
        assert v20.passed is True
        assert v20.details["threshold"] == 0.20

    @pytest.mark.unit
    def test_verify_rejects_wrong_phase_count(self):
        ref = self._arts(100.0, label="reference")
        with pytest.raises(ValueError, match="exactly 2 phases"):
            OutputCachingAudit().verify([ref], self._cfg())


# ---------------------------------------------------------------------------
# AuditRunStats.from_report
# ---------------------------------------------------------------------------


class TestRunStats:
    @pytest.mark.unit
    def test_from_report_extracts_qps(self):
        mock_report = MagicMock()
        mock_report.qps = 42.5
        mock_report.n_samples_completed = 100
        stats = AuditRunStats.from_report(mock_report, n_requested=100)
        assert stats.qps == 42.5
        assert stats.n_completed == 100
        assert stats.n_requested == 100

    @pytest.mark.unit
    def test_from_report_raises_when_qps_is_none(self):
        mock_report = MagicMock()
        mock_report.qps = None
        with pytest.raises(ValueError, match="no duration"):
            AuditRunStats.from_report(mock_report, n_requested=100)

    @pytest.mark.unit
    def test_from_report_raises_when_qps_non_positive(self):
        # A zero-throughput run (no completions) can't anchor an output-caching ratio.
        mock_report = MagicMock()
        mock_report.qps = 0.0
        with pytest.raises(ValueError, match="non-positive throughput"):
            AuditRunStats.from_report(mock_report, n_requested=100)


# ---------------------------------------------------------------------------
# write_result — atomic disk write
# ---------------------------------------------------------------------------


class TestWriteResult:
    @pytest.mark.unit
    def test_writes_txt_and_json(self, tmp_path):
        result = AuditResult(
            test_id="output_caching_test",
            passed=True,
            details={"reason": "ok", "ref_qps": 100.0},
        )
        write_result(result, tmp_path)
        txt = (tmp_path / "verify_OUTPUT_CACHING_TEST.txt").read_text()
        assert "Performance check pass: True" in txt
        data = json.loads((tmp_path / "audit_result.json").read_text())
        assert data["passed"] is True
        assert data["test"] == "output_caching_test"

    @pytest.mark.unit
    def test_failed_result_writes_false(self, tmp_path):
        result = AuditResult(
            test_id="output_caching_test", passed=False, details={"reason": "fail"}
        )
        write_result(result, tmp_path)
        txt = (tmp_path / "verify_OUTPUT_CACHING_TEST.txt").read_text()
        assert "Performance check pass: False" in txt

    @pytest.mark.unit
    def test_json_contains_full_details(self, tmp_path):
        details = {
            "ref_qps": 100.0,
            "audit_qps": 80.0,
            "threshold": 0.10,
            "reason": "ok",
        }
        result = AuditResult(
            test_id="output_caching_test", passed=True, details=details
        )
        write_result(result, tmp_path)
        data = json.loads((tmp_path / "audit_result.json").read_text())
        assert data["ref_qps"] == 100.0
        assert data["threshold"] == 0.10


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    @pytest.mark.unit
    def test_get_audit_test_returns_output_caching(self):

        test = get_audit_test(AuditTestId.OUTPUT_CACHING_TEST)
        assert test.test_id == AuditTestId.OUTPUT_CACHING_TEST

    @pytest.mark.unit
    def test_get_audit_test_raises_on_unknown(self, monkeypatch):

        # Empty the registry so the id resolves to nothing.
        monkeypatch.setattr(compliance, "_REGISTRY", {})
        with pytest.raises(KeyError, match="No audit test registered"):
            compliance.get_audit_test(AuditTestId.OUTPUT_CACHING_TEST)


# ---------------------------------------------------------------------------
# run_audit orchestrator guards
# ---------------------------------------------------------------------------


class TestRunAuditGuards:
    def _audit_config(self) -> MagicMock:
        """A MagicMock BenchmarkConfig with a real output-caching audit block."""

        config = MagicMock()
        config.audit = OutputCachingTestConfig(
            test=AuditTestId.OUTPUT_CACHING_TEST, samples=4, sample_index=0
        )
        return config

    @pytest.mark.unit
    @pytest.mark.parametrize("pattern", _REJECTED_LOAD_PATTERNS, ids=lambda p: p.name)
    def test_rejects_paced_or_incompatible_load_pattern(self, tmp_path, pattern):
        """Only max_throughput / concurrency are valid; everything else is
        rejected before any phase runs (guards the fixed-index audit semantics)."""

        config = self._audit_config()
        config.settings.load_pattern.type = pattern
        with pytest.raises(SetupError, match="max_throughput, concurrency, or poisson"):
            run_audit(config, tmp_path)

    @pytest.mark.unit
    @pytest.mark.parametrize("pattern", ["MAX_THROUGHPUT", "CONCURRENCY", "POISSON"])
    def test_refuses_result_on_incomplete_phase(self, tmp_path, monkeypatch, pattern):
        """A phase whose Report.complete is False (drain timeout / interrupted)
        must abort with ExecutionError, never a certified result."""

        config = self._audit_config()
        config.settings.load_pattern.type = getattr(LoadPatternType, pattern)
        perf_ds = MagicMock()
        perf_ds.type.value = "performance"
        config.datasets = [perf_ds]
        config.with_updates.return_value = MagicMock()

        # The bounds-check reads num_samples from the first phase's loaded ctx.
        ctx = MagicMock()
        ctx.dataloader.num_samples.return_value = 100
        incomplete = MagicMock()
        incomplete.complete = False
        bench = MagicMock()
        bench.report = incomplete
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.setup_benchmark",
            lambda *a, **k: ctx,
        )
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.run_benchmark_async",
            lambda ctx: bench,
        )
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.finalize_benchmark",
            lambda ctx, b: None,
        )

        with pytest.raises(ExecutionError, match="did not complete cleanly"):
            run_audit(config, tmp_path)

    @pytest.mark.unit
    def test_rejects_reference_count_over_dataset_size(self, tmp_path, monkeypatch):
        """A without-replacement (reference) phase requesting more samples than
        the dataset holds must be rejected before any phase runs — otherwise the
        baseline repeats rows and becomes cacheable, masking output caching."""

        config = self._audit_config()  # reference samples=4, sample_index=0
        config.settings.load_pattern.type = LoadPatternType.MAX_THROUGHPUT
        perf_ds = MagicMock()
        perf_ds.type.value = "performance"
        config.datasets = [perf_ds]
        config.with_updates.return_value = MagicMock()

        ctx = MagicMock()
        ctx.dataloader.num_samples.return_value = 3  # < reference samples (4)
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.setup_benchmark",
            lambda *a, **k: ctx,
        )

        with pytest.raises(SetupError, match="exceeds dataset size"):
            run_audit(config, tmp_path)

    @pytest.mark.unit
    def test_strips_accuracy_datasets_from_phase_config(self, tmp_path, monkeypatch):
        """Each audit phase is performance-only: accuracy datasets must be
        stripped from the per-phase config so setup_benchmark never appends an
        ACCURACY phase (which would re-issue and re-score the accuracy set)."""

        config = self._audit_config()
        config.settings.load_pattern.type = LoadPatternType.MAX_THROUGHPUT
        perf_ds = MagicMock()
        perf_ds.type.value = "performance"
        acc_ds = MagicMock()
        acc_ds.type.value = "accuracy"
        config.datasets = [perf_ds, acc_ds]
        config.with_updates.return_value = MagicMock()

        ctx = MagicMock()
        ctx.dataloader.num_samples.return_value = 100
        bench = MagicMock()
        bench.report = None  # abort after the first phase's with_updates call
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.setup_benchmark",
            lambda *a, **k: ctx,
        )
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.run_benchmark_async",
            lambda ctx: bench,
        )
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.finalize_benchmark",
            lambda ctx, b: None,
        )

        with pytest.raises(ExecutionError):
            run_audit(config, tmp_path)

        # The per-phase config must carry performance datasets only.
        assert config.with_updates.call_args.kwargs["datasets"] == [perf_ds]

    @pytest.mark.unit
    def test_interrupted_phase_raises_keyboard_interrupt(self, tmp_path, monkeypatch):
        """A SIGINT/SIGTERM during an audit phase yields an 'interrupted' report;
        run_audit must propagate KeyboardInterrupt (CLI exit 130), not the
        generic ExecutionError (exit 4) used for a crashed/drain-timeout phase."""
        config = self._audit_config()
        config.settings.load_pattern.type = LoadPatternType.MAX_THROUGHPUT
        perf_ds = MagicMock()
        perf_ds.type.value = "performance"
        config.datasets = [perf_ds]
        config.with_updates.return_value = MagicMock()

        ctx = MagicMock()
        ctx.dataloader.num_samples.return_value = 100
        interrupted = MagicMock()
        interrupted.state = "interrupted"
        interrupted.complete = False
        bench = MagicMock()
        bench.report = interrupted
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.setup_benchmark", lambda *a, **k: ctx
        )
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.run_benchmark_async", lambda ctx: bench
        )
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.finalize_benchmark", lambda ctx, b: None
        )

        with pytest.raises(KeyboardInterrupt):
            run_audit(config, tmp_path)

    @pytest.mark.unit
    def test_rejects_out_of_range_sample_index(self, tmp_path, monkeypatch):
        """A fixed sample_index beyond the loaded dataset size is rejected before
        any phase runs."""
        config = MagicMock()
        config.audit = OutputCachingTestConfig(
            test=AuditTestId.OUTPUT_CACHING_TEST, samples=4, sample_index=100
        )
        config.settings.load_pattern.type = LoadPatternType.MAX_THROUGHPUT
        perf_ds = MagicMock()
        perf_ds.type.value = "performance"
        config.datasets = [perf_ds]
        config.with_updates.return_value = MagicMock()

        ctx = MagicMock()
        ctx.dataloader.num_samples.return_value = 10  # sample_index=100 out of range
        monkeypatch.setattr(
            "inference_endpoint.commands.audit.setup_benchmark", lambda *a, **k: ctx
        )

        with pytest.raises(SetupError, match="out of range"):
            run_audit(config, tmp_path)

    @pytest.mark.unit
    def test_rejects_when_no_performance_dataset(self, tmp_path):
        """An audit config with no performance dataset is rejected up front."""
        config = self._audit_config()
        config.settings.load_pattern.type = LoadPatternType.MAX_THROUGHPUT
        acc_ds = MagicMock()
        acc_ds.type.value = "accuracy"
        config.datasets = [acc_ds]

        with pytest.raises(SetupError, match="performance dataset"):
            run_audit(config, tmp_path)
