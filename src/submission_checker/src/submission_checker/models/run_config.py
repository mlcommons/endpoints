"""Run configuration model — §8.3 measurement run YAML schema and per-run checks."""

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

from .regions import classify_concurrency
from .results import CheckResult, err, ok


class LoadPatternConfig(BaseModel):
    """Load pattern settings matching the endpoints tool ``LoadPattern`` schema.

    Attributes:
        type: Pattern type — must be ``"concurrency"`` for submissions (§6.1).
        target_concurrency: Fixed concurrency level for ``ConcurrencyScheduler`` runs.
        target_qps: Target queries-per-second for ``constant_qps`` runs (unused in submissions).
    """

    model_config = ConfigDict(extra="allow")

    type: str
    target_concurrency: int | None = None
    target_qps: float | None = None


class RuntimeConfig(BaseModel):
    """Runtime duration and sampling settings matching the endpoints ``RuntimeConfig`` schema.

    Attributes:
        min_duration_ms: Minimum steady-state duration in milliseconds (§6.2).
        max_duration_ms: Maximum duration cap (0 = unlimited).
        n_samples_to_issue: Fixed sample count override (``None`` = duration-based).
        scheduler_random_seed: RNG seed for the concurrency scheduler.
        dataloader_random_seed: RNG seed for dataset loading.
    """

    model_config = ConfigDict(extra="allow")

    min_duration_ms: int = 600_000
    max_duration_ms: int = 0
    n_samples_to_issue: int | None = None
    scheduler_random_seed: int = 42
    dataloader_random_seed: int = 42


class ClientConfig(BaseModel):
    """Client-side settings matching the endpoints ``ClientConfig`` schema.

    Attributes:
        stream_all_chunks: Must be ``True`` for all submission performance runs (§6.5).
    """

    model_config = ConfigDict(extra="allow")

    stream_all_chunks: bool = True


class SettingsConfig(BaseModel):
    """Top-level ``settings`` block from the endpoints ``BenchmarkConfig`` schema.

    Attributes:
        load_pattern: Load generation strategy.
        runtime: Duration and sampling constraints.
        client: Client-side streaming settings.
    """

    model_config = ConfigDict(extra="allow")

    load_pattern: LoadPatternConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)


class RunConfig(BaseModel):
    """Parsed contents of ``runs/run_<N>.yaml`` (§8.3).

    Follows the endpoints tool ``BenchmarkConfig`` schema.  The concurrency level
    and dataset name are surfaced as computed properties for use by checks.

    Attributes:
        type: Config type — ``"submission"`` or ``"development"``.
        settings: Nested load-pattern, runtime, and client settings.
        datasets: List of dataset descriptors (``[{"name": "..."}]``).
    """

    model_config = ConfigDict(extra="allow")
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    type: str = "submission"
    settings: SettingsConfig
    datasets: list[dict] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def concurrency(self) -> int:
        """Target concurrency level derived from ``settings.load_pattern.target_concurrency``."""
        return self.settings.load_pattern.target_concurrency or 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset(self) -> str:
        """Dataset name from the first entry in ``datasets``."""
        if self.datasets:
            return self.datasets[0].get("name", "")
        return ""

    @model_validator(mode="after")
    def _check_load_pattern(self, info: ValidationInfo) -> RunConfig:
        """§10: load_pattern.type must be 'concurrency' with target_concurrency set."""
        path: Path | None = (info.context or {}).get("yaml_path")
        lp = self.settings.load_pattern
        if lp.type != "concurrency":
            self._check_results.append(
                err(
                    "load-pattern",
                    f"Run {self.concurrency}: load_pattern.type '{lp.type}' ≠ 'concurrency'",
                    path,
                    "#10",
                )
            )
        elif lp.target_concurrency is None:
            self._check_results.append(
                err(
                    "load-pattern",
                    f"Run {self.concurrency}: load_pattern.type is 'concurrency'"
                    " but target_concurrency is not set",
                    path,
                    "#10",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "load-pattern",
                    f"Run {self.concurrency}: load pattern OK"
                    f" (type=concurrency, target={lp.target_concurrency})",
                    path,
                    "#10",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_streaming(self, info: ValidationInfo) -> RunConfig:
        """§13: stream_all_chunks must be True for all submission runs."""
        path: Path | None = (info.context or {}).get("yaml_path")
        if not self.settings.client.stream_all_chunks:
            self._check_results.append(
                err(
                    "streaming-config",
                    f"Run {self.concurrency}: stream_all_chunks must be True",
                    path,
                    "#13",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "streaming-config",
                    f"Run {self.concurrency}: stream_all_chunks=True",
                    path,
                    "#13",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_concurrency_range(self, info: ValidationInfo) -> RunConfig:
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
