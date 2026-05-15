"""Point result model — cross-file validation combining point config and result summary."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationInfo, model_validator

from ..regions import MIN_DURATION_MS, classify_concurrency
from ..results import CheckResult, err, ok, warn
from ..file.point_config import PointConfig
from ..file.point_summary import PointSummary

_TPS_TOLERANCE = 0.01  # 1% relative tolerance for stored-vs-derived comparisons

# Placeholder dataset → minimum completed query count (§6.4).
# Replace values with spec-ratified numbers before final release.
MIN_QUERY_COUNT: dict[str, int] = {
    "dataset-a": 1,
    "dataset-b": 10,
    "dataset-c": 100,
}


class PointResult(BaseModel):
    """Paired config and result summary for one measurement point."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    config: PointConfig
    summary: PointSummary
    yaml_path: Path

    @model_validator(mode="after")
    def _check_point_duration(self, info: ValidationInfo) -> PointResult:
        """§11: warn when measured duration is below the per-region minimum."""
        regions = (info.context or {}).get("regions")
        summary_path: Path | None = (info.context or {}).get("summary_path")
        if regions is None:
            return self
        c = self.config.concurrency
        region = classify_concurrency(c, regions)
        if region is None:
            return self  # already flagged by concurrency-in-range
        min_ms = MIN_DURATION_MS.get(region, 0)
        duration_ms = self.summary.duration_ms
        if duration_ms < min_ms:
            self._check_results.append(
                warn(
                    "point-duration",
                    f"Point {c} ({region}): duration {duration_ms:.0f} ms < minimum {min_ms} ms"
                    " (§6.2 values pending WG ratification)",
                    summary_path,
                    "#11",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "point-duration",
                    f"Point {c}: duration {duration_ms:.0f} ms meets minimum for {region}",
                    summary_path,
                    "#11",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_min_query_count(self, info: ValidationInfo) -> PointResult:
        """§12: n_samples_completed must meet the dataset's minimum query count (§6.4).

        Skipped when the dataset is not in MIN_QUERY_COUNT (unknown datasets are
        not yet mapped; add them as the spec is ratified).
        """
        summary_path: Path | None = (info.context or {}).get("summary_path")
        dataset = self.config.dataset
        min_queries = MIN_QUERY_COUNT.get(dataset)
        if min_queries is None:
            return self
        completed = self.summary.n_samples_completed
        if completed < min_queries:
            self._check_results.append(
                err(
                    "min-query-count",
                    f"Dataset '{dataset}': completed {completed} < minimum {min_queries} (§6.4)",
                    summary_path,
                    "#12",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "min-query-count",
                    f"Dataset '{dataset}': completed {completed} ≥ minimum {min_queries}",
                    summary_path,
                    "#12",
                )
            )
        return self

    # ------------------------------------------------------------------
    # Metric-consistency sub-checks (called from _check_metric_consistency)
    # ------------------------------------------------------------------

    def _check_duration_positive(self, s: PointSummary, path: Path | None) -> None:
        """Emit ok/err for the duration_ns > 0 invariant (§14)."""
        if s.duration_ns <= 0:
            self._check_results.append(
                err(
                    "metric-consistency-duration",
                    f"duration_ns is not positive: {s.duration_ns}",
                    path,
                    "#14",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "metric-consistency-duration",
                    f"duration_ns={s.duration_ns:.0f} ns",
                    path,
                    "#14",
                )
            )

    def _check_sample_accounting(self, s: PointSummary, path: Path | None) -> None:
        """Emit ok/err for the completed + failed == issued invariant (§14).

        Skipped when n_samples_issued is 0 (point tool did not track issued count).
        """
        if s.n_samples_issued > 0:
            accounted = s.n_samples_completed + s.n_samples_failed
            if accounted != s.n_samples_issued:
                self._check_results.append(
                    err(
                        "metric-consistency-accounting",
                        f"completed ({s.n_samples_completed}) + failed ({s.n_samples_failed})"
                        f" = {accounted} ≠ issued ({s.n_samples_issued})",
                        path,
                        "#14",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "metric-consistency-accounting",
                        f"Sample accounting consistent: {s.n_samples_issued} issued",
                        path,
                        "#14",
                    )
                )

    def _check_output_tokens_nonnegative(self, s: PointSummary, path: Path | None) -> None:
        """Emit ok/err for the total_output_tokens >= 0 invariant (§14)."""
        if s.total_output_tokens < 0:
            self._check_results.append(
                err(
                    "metric-consistency-output-tokens",
                    f"total_output_tokens is negative: {s.total_output_tokens}",
                    path,
                    "#14",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "metric-consistency-output-tokens",
                    f"total_output_tokens={s.total_output_tokens}",
                    path,
                    "#14",
                )
            )

    def _check_system_tps_derivable(self, s: PointSummary, path: Path | None) -> None:
        """§9.1: system_tps must equal total_output_tokens / elapsed_duration_seconds.

        If the result log also stores a system_tps field, verify it matches the
        derived value within _TPS_TOLERANCE. Always emits ok when consistent.
        """
        derived = s.system_tps
        stored = (s.model_extra or {}).get("system_tps")
        if stored is not None:
            rel_err = abs(float(stored) - derived) / max(abs(derived), 1e-9)
            if rel_err > _TPS_TOLERANCE:
                self._check_results.append(
                    err(
                        "metric-consistency-system-tps",
                        f"stored system_tps {stored:.3f} ≠ derived {derived:.3f}"
                        f" (rel err {rel_err:.1%})",
                        path,
                        "#9.1",
                    )
                )
                return
        self._check_results.append(
            ok(
                "metric-consistency-system-tps",
                f"system_tps={derived:.3f} tok/s"
                f" ({s.total_output_tokens} tokens / {s.elapsed_duration_seconds:.1f}s)",
                path,
                "#9.1",
            )
        )

    def _check_tps_per_user(self, s: PointSummary, concurrency: int, path: Path | None) -> None:
        """§9.1: tps_per_user must equal system_tps / concurrency.

        If the result log also stores a tps_per_user field, verify it matches the
        derived value within _TPS_TOLERANCE. Always emits ok when consistent.
        """
        if concurrency <= 0:
            self._check_results.append(
                err(
                    "metric-consistency-tps-per-user",
                    f"concurrency={concurrency} is not positive",
                    path,
                    "#9.1",
                )
            )
            return
        derived = s.system_tps / concurrency
        stored = (s.model_extra or {}).get("tps_per_user")
        if stored is not None:
            rel_err = abs(float(stored) - derived) / max(abs(derived), 1e-9)
            if rel_err > _TPS_TOLERANCE:
                self._check_results.append(
                    err(
                        "metric-consistency-tps-per-user",
                        f"stored tps_per_user {stored:.4f} ≠ derived system_tps/concurrency"
                        f" {derived:.4f} (rel err {rel_err:.1%})",
                        path,
                        "#9.1",
                    )
                )
                return
        self._check_results.append(
            ok(
                "metric-consistency-tps-per-user",
                f"tps_per_user={derived:.4f} tok/s/user"
                f" (system_tps={s.system_tps:.3f} / concurrency={concurrency})",
                path,
                "#9.1",
            )
        )

    @model_validator(mode="after")
    def _check_metric_consistency(self, info: ValidationInfo) -> PointResult:
        """§14 + §9.1: validate point-log accounting invariants and tps derivability."""
        summary_path: Path | None = (info.context or {}).get("summary_path")
        s = self.summary
        self._check_duration_positive(s, summary_path)
        self._check_sample_accounting(s, summary_path)
        self._check_output_tokens_nonnegative(s, summary_path)
        self._check_system_tps_derivable(s, summary_path)
        self._check_tps_per_user(s, self.config.concurrency, summary_path)
        return self
