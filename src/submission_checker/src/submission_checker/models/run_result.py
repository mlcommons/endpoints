"""Run result models — §8.3 result log schema, accuracy result, and per-run checks."""

from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    model_validator,
)

from .regions import MIN_DURATION_MS, classify_concurrency
from .results import CheckResult, err, ok, warn
from .run_config import RunConfig


class PercentileStats(BaseModel):
    """Summary statistics dict produced by the endpoints ``compute_summary()`` helper.

    Attributes:
        total: Sum across all samples (tokens, nanoseconds, etc.).
        percentiles: Mapping from percentile string (``"50"``, ``"95"``, …) to value.
    """

    model_config = ConfigDict(extra="allow")

    total: float = 0.0
    percentiles: dict[str, float] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Parsed contents of ``results/run_<N>/mlperf_endpoints_log_summary.json``.

    Follows the endpoints tool ``Report`` msgspec.Struct schema.  Raw timing values
    are in nanoseconds; derived millisecond / token-rate fields are computed
    properties so that checks can use them without unit conversions.

    Attributes:
        n_samples_issued: Total queries dispatched to the endpoint.
        n_samples_completed: Queries that returned a valid response.
        n_samples_failed: Queries that errored or timed out.
        duration_ns: Steady-state measurement window length in nanoseconds.
        ttft: Time-to-first-token statistics (nanoseconds).
        output_sequence_lengths: Output token count statistics.
    """

    model_config = ConfigDict(extra="allow")

    n_samples_issued: int = 0
    n_samples_completed: int
    n_samples_failed: int = 0
    duration_ns: float
    ttft: PercentileStats = Field(default_factory=PercentileStats)
    output_sequence_lengths: PercentileStats = Field(default_factory=PercentileStats)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> float:
        """Measurement duration in milliseconds."""
        return self.duration_ns / 1_000_000

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sample_count(self) -> int:
        """Alias for ``n_samples_completed``."""
        return self.n_samples_completed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_output_tokens(self) -> int:
        """Total output tokens from ``output_sequence_lengths.total``."""
        return int(self.output_sequence_lengths.total)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def elapsed_duration_seconds(self) -> float:
        """Measurement duration in seconds."""
        return self.duration_ns / 1e9

    @computed_field  # type: ignore[prop-decorator]
    @property
    def system_tps(self) -> float:
        """System-wide tokens per second: ``total_output_tokens / elapsed_s``."""
        elapsed_s = self.elapsed_duration_seconds
        return self.total_output_tokens / elapsed_s if elapsed_s > 0 else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ttft_p50_ms(self) -> float:
        """Median time to first token in milliseconds."""
        return self.ttft.percentiles.get("50", 0.0) / 1_000_000

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ttft_p95_ms(self) -> float:
        """95th-percentile time to first token in milliseconds."""
        return self.ttft.percentiles.get("95", 0.0) / 1_000_000


class AccuracyResult(BaseModel):
    """Parsed contents of ``accuracy/accuracy_result.json`` (§4.3, §6.6).

    Attributes:
        metric: Name of the accuracy metric (e.g., ``rouge1``).
        score: Achieved score on the accuracy dataset.
        quality_target: Minimum acceptable score defined by the benchmark.
        passed: Whether the score meets the quality target.
    """

    model_config = ConfigDict(extra="allow")

    metric: str
    score: float
    quality_target: float
    passed: bool


class RunResult(BaseModel):
    """Paired config and result summary for one measurement run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    config: RunConfig
    summary: RunSummary
    yaml_path: Path

    @model_validator(mode="after")
    def _check_run_duration(self, info: ValidationInfo) -> RunResult:
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
                    "run-duration",
                    f"Run {c} ({region}): duration {duration_ms:.0f} ms < minimum {min_ms} ms"
                    " (§6.2 values pending WG ratification)",
                    summary_path,
                    "#11",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "run-duration",
                    f"Run {c}: duration {duration_ms:.0f} ms meets minimum for {region}",
                    summary_path,
                    "#11",
                )
            )
        return self

    # ------------------------------------------------------------------
    # Metric-consistency sub-checks (called from _check_metric_consistency)
    # ------------------------------------------------------------------

    def _check_duration_positive(self, s: RunSummary, path: Path | None) -> None:
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

    def _check_sample_accounting(self, s: RunSummary, path: Path | None) -> None:
        """Emit ok/err for the completed + failed == issued invariant (§14).

        Skipped when n_samples_issued is 0 (run tool did not track issued count).
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

    def _check_output_tokens_nonnegative(self, s: RunSummary, path: Path | None) -> None:
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

    @model_validator(mode="after")
    def _check_metric_consistency(self, info: ValidationInfo) -> RunResult:
        """§14: validate three run-log accounting invariants via sub-check helpers."""
        summary_path: Path | None = (info.context or {}).get("summary_path")
        s = self.summary
        self._check_duration_positive(s, summary_path)
        self._check_sample_accounting(s, summary_path)
        self._check_output_tokens_nonnegative(s, summary_path)
        return self
