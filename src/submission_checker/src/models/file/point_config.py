"""Point configuration model — §8.3 measurement point YAML schema and per-point checks."""

from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    model_validator,
)

from ..regions import classify_concurrency
from ..results import CheckResult, err, ok, warn


_VALID_REGIONS = frozenset(
    {"low_latency", "low_throughput", "med_throughput", "high_throughput", "submitters_choice"}
)


class RuntimeSettings(BaseModel):
    """``runtime_settings`` block from ``points/point_<N>.yaml`` (§8.3).

    Attributes:
        load_pattern: Load pattern type — must be ``"concurrency"`` for submissions (§6.1).
        min_duration_ms: Minimum steady-state duration in milliseconds (§6.2).
        min_sample_count: Minimum completed queries required (§6.4). ``None`` = no override.
        stream_all_chunks: Must be ``True`` for all submission performance runs (§6.5).
    """

    model_config = ConfigDict(extra="allow")

    load_pattern: str = "concurrency"
    min_duration_ms: int = 600_000
    min_sample_count: int | None = None
    stream_all_chunks: bool = True


class PointConfig(BaseModel):
    """Parsed contents of ``points/point_<N>.yaml`` (§8.3).

    Attributes:
        concurrency: Target concurrency level for this measurement point.
        region: Spec region this point is claimed to satisfy (optional submitter hint).
        dataset: Dataset name used for this measurement point.
        runtime_settings: Load pattern, duration, and client settings.
    """

    model_config = ConfigDict(extra="allow")
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    concurrency: int
    region: str | None = None
    dataset: str = ""
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @model_validator(mode="after")
    def _check_load_pattern(self, info: ValidationInfo) -> PointConfig:
        """§10: load_pattern must be 'concurrency' with a positive concurrency level."""
        path: Path | None = (info.context or {}).get("yaml_path")
        lp = self.runtime_settings.load_pattern
        if lp != "concurrency":
            self._check_results.append(
                err(
                    "load-pattern",
                    f"Point {self.concurrency}: load_pattern '{lp}' ≠ 'concurrency'",
                    path,
                    "#10",
                )
            )
        elif self.concurrency <= 0:
            self._check_results.append(
                err(
                    "load-pattern",
                    f"concurrency must be positive, got {self.concurrency}",
                    path,
                    "#10",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "load-pattern",
                    f"Point {self.concurrency}: load pattern OK (concurrency)",
                    path,
                    "#10",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_streaming(self, info: ValidationInfo) -> PointConfig:
        """§13: stream_all_chunks must be True for all submission points."""
        path: Path | None = (info.context or {}).get("yaml_path")
        if not self.runtime_settings.stream_all_chunks:
            self._check_results.append(
                err(
                    "streaming-config",
                    f"Point {self.concurrency}: stream_all_chunks must be True",
                    path,
                    "#13",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "streaming-config",
                    f"Point {self.concurrency}: stream_all_chunks=True",
                    path,
                    "#13",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_concurrency_range(self, info: ValidationInfo) -> PointConfig:
        """§9: concurrency must not exceed the high_throughput upper bound (incl. 10% margin)."""
        path: Path | None = (info.context or {}).get("yaml_path")
        regions = (info.context or {}).get("regions")
        if regions is None:
            return self
        actual_region = classify_concurrency(self.concurrency, regions)
        if actual_region is None:
            self._check_results.append(
                err(
                    "concurrency-in-range",
                    f"Concurrency {self.concurrency} exceeds max valid range"
                    f" (max including 10% margin: {regions.high_throughput.end})",
                    path,
                    "#9",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "concurrency-in-range",
                    f"Concurrency {self.concurrency} valid ({actual_region})",
                    path,
                    "#9",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_region_declared(self, info: ValidationInfo) -> PointConfig:
        """§8.3: region must be a valid value; if declared, must match the computed region."""
        path: Path | None = (info.context or {}).get("yaml_path")
        region = self.region
        if region is None:
            return self  # optional field — absence is not an error
        if region not in _VALID_REGIONS:
            self._check_results.append(
                err(
                    "region-declared",
                    f"Invalid region '{region}': must be one of {sorted(_VALID_REGIONS)}",
                    path,
                    "#8.3",
                )
            )
            return self
        regions = (info.context or {}).get("regions")
        if regions is not None and region != "submitters_choice":
            computed = classify_concurrency(self.concurrency, regions)
            if computed is not None and computed != region:
                self._check_results.append(
                    warn(
                        "region-declared",
                        f"Declared region '{region}' ≠ computed region '{computed}'"
                        f" for concurrency {self.concurrency}",
                        path,
                        "#8.3",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "region-declared",
                        f"Declared region '{region}' consistent with concurrency {self.concurrency}",
                        path,
                        "#8.3",
                    )
                )
        else:
            self._check_results.append(
                ok("region-declared", f"region='{region}'", path, "#8.3")
            )
        return self
