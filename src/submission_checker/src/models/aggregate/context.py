"""Model context — aggregate model-level validation (point count, coverage, consistency, accuracy)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ..regions import Regions
from ..results import CheckResult, err, ok, warn
from ..file.accuracy import AccuracyResult
from ..file.point_config import PointConfig
from ..file.point_summary import PointSummary
from ..file.system import SystemDescription

_MIN_POINTS = 7
_MAX_POINTS = 32


class ModelContext(BaseModel):
    """Aggregated data for one benchmark-model directory — carries all model-level validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    system_id: str
    system_desc: SystemDescription
    model_dir: Path
    regions: Regions
    points_dir: Path
    accuracy_dir: Path
    all_point_count: int
    valid_points: list[tuple[Path, PointConfig]]
    loaded_points: list[tuple[PointConfig, PointSummary]]
    accuracy_result: AccuracyResult | None = None

    @model_validator(mode="after")
    def _check_point_count(self) -> ModelContext:
        """§2, §8: submission must have 7–32 measurement points."""
        n = self.all_point_count
        if n < _MIN_POINTS:
            self._check_results.append(
                err(
                    "point-count",
                    f"Only {n} measurement point(s) — minimum {_MIN_POINTS} required",
                    self.points_dir,
                    "#2, #8",
                )
            )
        else:
            self._check_results.append(
                ok("point-count", f"Point count OK: {n}", self.points_dir, "#2, #8")
            )
        if n > _MAX_POINTS:
            self._check_results.append(
                err("point-cap", f"{n} points exceed the {_MAX_POINTS}-point cap", self.points_dir, "#2, #8")
            )
        return self

    @model_validator(mode="after")
    def _check_regional_coverage(self) -> ModelContext:
        """§3–6: at least one valid point must fall in each of the four concurrency regions."""
        concurrencies = [config.concurrency for _, config in self.valid_points]
        r = self.regions
        coverage_checks = [
            ("low-latency-coverage", "Low Latency", r.low_latency),
            ("low-throughput-coverage", "Low Throughput", r.low_throughput),
            ("med-throughput-coverage", "Medium Throughput", r.med_throughput),
            ("high-throughput-coverage", "High Throughput", r.high_throughput),
        ]
        for rule, label, bounds in coverage_checks:
            matching = [c for c in concurrencies if bounds.contains(c)]
            if matching:
                self._check_results.append(
                    ok(
                        rule,
                        f"{label} region covered: {sorted(matching)} (range {bounds})",
                        self.points_dir,
                        "#3–6",
                    )
                )
            else:
                self._check_results.append(
                    err(
                        rule,
                        f"No point in {label} region (concurrency {bounds})",
                        self.points_dir,
                        "#3–6",
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_config_consistency(self) -> ModelContext:
        """§16: all points must use the same dataset; directory name must match benchmark_model."""
        if not self.loaded_points:
            return self
        datasets = {config.dataset for config, _ in self.loaded_points}
        if len(datasets) > 1:
            self._check_results.append(
                err(
                    "config-consistency-dataset",
                    f"Inconsistent datasets across points: {datasets}",
                    self.model_dir,
                    "#16",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "config-consistency-dataset",
                    f"Dataset consistent: {next(iter(datasets))}",
                    self.model_dir,
                    "#16",
                )
            )

        if self.model_dir.name != self.system_desc.benchmark_model:
            self._check_results.append(
                warn(
                    "config-consistency-model",
                    f"Directory name '{self.model_dir.name}' ≠"
                    f" system_desc benchmark_model '{self.system_desc.benchmark_model}'",
                    self.model_dir,
                    "#16",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "config-consistency-model",
                    f"Benchmark model consistent: {self.model_dir.name}",
                    self.model_dir,
                    "#16",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_accuracy(self) -> ModelContext:
        """§15: accuracy score must meet or exceed the benchmark quality_target."""
        if self.accuracy_result is None:
            return self  # file missing/invalid already reported by checker.py
        accuracy = self.accuracy_result
        json_path = self.accuracy_dir / "accuracy_result.json"
        if accuracy.passed:
            self._check_results.append(
                ok(
                    "accuracy-gate",
                    f"Accuracy gate PASSED: {accuracy.metric} = {accuracy.score:.4f}"
                    f" ≥ target {accuracy.quality_target:.4f}",
                    json_path,
                    "#15",
                )
            )
        else:
            self._check_results.append(
                err(
                    "accuracy-gate",
                    f"Accuracy gate FAILED: {accuracy.metric} = {accuracy.score:.4f}"
                    f" < target {accuracy.quality_target:.4f}",
                    json_path,
                    "#15",
                )
            )
        return self
