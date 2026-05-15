"""Point summary model — §8.3 result log schema for a single measurement point."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field


class PercentileStats(BaseModel):
    """Summary statistics dict produced by the endpoints ``compute_summary()`` helper.

    Attributes:
        total: Sum across all samples (tokens, nanoseconds, etc.).
        percentiles: Mapping from percentile string (``"50"``, ``"95"``, …) to value.
    """

    model_config = ConfigDict(extra="allow")

    total: float = 0.0
    percentiles: dict[str, float] = Field(default_factory=dict)


class PointSummary(BaseModel):
    """Parsed contents of ``results/point_<N>/mlperf_endpoints_log_summary.json``.

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
