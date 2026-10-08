# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    computed_field,
    field_validator,
)


class PercentileStats(BaseModel):
    """Summary statistics dict produced by the endpoints ``compute_summary()`` helper."""

    model_config = ConfigDict(extra="ignore")
    total: FiniteFloat = 0.0
    percentiles: dict[str, FiniteFloat] = Field(default_factory=dict)

    @field_validator("percentiles")
    @classmethod
    def _normalize_percentile_keys(cls, values: dict[str, float]) -> dict[str, float]:
        """Treat native decimal keys and integer keys as the same percentile."""
        normalized: dict[str, float] = {}
        for key, value in values.items():
            try:
                percentile = float(key)
            except ValueError:
                percentile = math.nan
            if not math.isfinite(percentile) or not 0 <= percentile <= 100:
                raise ValueError(f"{key!r} is not a percentile between 0 and 100")
            canonical = (
                str(int(percentile)) if percentile.is_integer() else str(percentile)
            )
            if canonical in normalized and normalized[canonical] != value:
                raise ValueError(
                    f"two keys for percentile {canonical} give different values"
                )
            normalized[canonical] = value
        return normalized


class FullRunLengths(BaseModel):
    output_sequence_lengths: LengthAverage


class LengthAverage(BaseModel):
    avg: FiniteFloat | None = None


class PointSummary(BaseModel):
    """Parsed contents of a measurement point's ``result_summary.json``."""

    model_config = ConfigDict(extra="ignore")
    git_sha: str | None = None
    system_tps_per_kw: FiniteFloat | None = None
    output_sequence_lengths_full_run: FullRunLengths | None = None
    tps_per_user: FiniteFloat | None = None
    reported_system_tps: FiniteFloat | None = Field(
        default=None, alias="system_tps", exclude=True
    )
    reported_e2e_avg_interactivity: FiniteFloat | None = Field(
        default=None, alias="e2e_avg_interactivity", exclude=True
    )
    n_samples_issued: int = 0
    n_samples_completed: int
    n_samples_failed: int = 0
    duration_ns: FiniteFloat
    latency: PercentileStats = Field(default_factory=PercentileStats)
    ttft: PercentileStats = Field(default_factory=PercentileStats)
    tpot: PercentileStats = Field(default_factory=PercentileStats)
    output_sequence_lengths: PercentileStats = Field(default_factory=PercentileStats)
    output_tokens_per_turn_total: FiniteFloat | None = None
    e2e_turn_time_seconds_total: FiniteFloat | None = None

    @computed_field  # type: ignore[misc]
    @property
    def duration_ms(self) -> float:
        """Measurement duration in milliseconds."""
        return self.duration_ns / 1000000

    @computed_field  # type: ignore[misc]
    @property
    def sample_count(self) -> int:
        """Alias for ``n_samples_completed``."""
        return self.n_samples_completed

    @computed_field  # type: ignore[misc]
    @property
    def total_output_tokens(self) -> int:
        """Total output tokens from ``output_sequence_lengths.total``."""
        return int(self.output_sequence_lengths.total)

    @computed_field  # type: ignore[misc]
    @property
    def elapsed_duration_seconds(self) -> float:
        """Measurement duration in seconds."""
        return self.duration_ns / 1000000000.0

    @computed_field  # type: ignore[misc]
    @property
    def system_tps(self) -> float:
        """System-wide tokens per second: ``total_output_tokens / elapsed_s``."""
        elapsed_s = self.elapsed_duration_seconds
        return self.total_output_tokens / elapsed_s if elapsed_s > 0 else 0.0

    @computed_field  # type: ignore[misc]
    @property
    def ttft_p50_ms(self) -> float:
        """Median time to first token in milliseconds."""
        return self.ttft.percentiles.get("50", 0.0) / 1000000

    @computed_field  # type: ignore[misc]
    @property
    def ttft_p90_ms(self) -> float:
        """90th-percentile time to first token in milliseconds (§4.1)."""
        return self.ttft.percentiles.get("90", 0.0) / 1000000

    @computed_field  # type: ignore[misc]
    @property
    def ttft_p95_ms(self) -> float:
        """95th-percentile time to first token in milliseconds."""
        return self.ttft.percentiles.get("95", 0.0) / 1000000

    @computed_field  # type: ignore[misc]
    @property
    def tpot_p90_ms(self) -> float | None:
        """90th-percentile time per output token in milliseconds (§9.1)."""
        raw = self.tpot.percentiles.get("90")
        return None if raw is None else raw / 1000000

    @computed_field  # type: ignore[misc]
    @property
    def e2e_avg_interactivity(self) -> float | None:
        """§4.1: ``sum(output_tokens_per_turn) / sum(e2e_turn_time_seconds)``."""
        tokens = self.output_tokens_per_turn_total
        seconds = self.e2e_turn_time_seconds_total
        if tokens is None and seconds is None and self.n_samples_failed == 0:
            tokens = self.output_sequence_lengths.total
            seconds = self.latency.total / 1e9
        if tokens is None or seconds is None or seconds <= 0:
            return None
        return tokens / seconds
