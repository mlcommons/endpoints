"""Pydantic models for submission directory structure validation (§8.1)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, PrivateAttr, computed_field, model_validator

from .models import CheckResult, Division, err, ok


class SubmissionDir(BaseModel):
    """Validates the top-level submission directory: systems/ and pareto/ must exist."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def systems_dir(self) -> Path:
        return self.root / "systems"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pareto_dir(self) -> Path:
        return self.root / "pareto"

    @model_validator(mode="after")
    def _check_required_dirs(self) -> SubmissionDir:
        for name in ("systems", "pareto"):
            path = self.root / name
            if path.is_dir():
                self._check_results.append(
                    ok("required-dir", f"Found required directory: {name}/", path, "#1")
                )
            else:
                self._check_results.append(
                    err("required-dir", f"Missing required directory: {name}/", path, "#1")
                )
        return self


class SrcDir(BaseModel):
    """Validates src/ exists for Standardized division submissions (§2.2.1)."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path
    division: Division

    @model_validator(mode="after")
    def _check_src(self) -> SrcDir:
        if self.division != Division.STANDARDIZED:
            return self
        src_dir = self.root / "src"
        if src_dir.is_dir():
            self._check_results.append(
                ok("src-dir", "src/ present (required for Standardized division)", src_dir, "#1")
            )
        else:
            self._check_results.append(
                err(
                    "src-dir",
                    "Missing src/ directory (required for Standardized division)",
                    src_dir,
                    "#1",
                )
            )
        return self


class SystemPareto(BaseModel):
    """Validates pareto/<system_id>/ exists."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    pareto_dir: Path
    system_id: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def system_dir(self) -> Path:
        return self.pareto_dir / self.system_id

    @model_validator(mode="after")
    def _check_dir_exists(self) -> SystemPareto:
        path = self.pareto_dir / self.system_id
        if path.is_dir():
            self._check_results.append(
                ok("pareto-dir-exists", f"Found pareto/{self.system_id}/", path, "#1")
            )
        else:
            self._check_results.append(
                err("pareto-dir-exists", f"No pareto/{self.system_id}/ directory found", path, "#1")
            )
        return self


class ModelDir(BaseModel):
    """Validates runs/, results/, and accuracy/ exist under a benchmark-model directory."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path
    system_id: str
    benchmark_model: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accuracy_dir(self) -> Path:
        return self.root / "accuracy"

    @model_validator(mode="after")
    def _check_subdirs(self) -> ModelDir:
        for name in ("runs", "results", "accuracy"):
            path = self.root / name
            if path.is_dir():
                self._check_results.append(
                    ok(
                        "pareto-subdir",
                        f"Found {name}/ in pareto/{self.system_id}/{self.benchmark_model}/",
                        path,
                        "#1",
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "pareto-subdir",
                        f"Missing {name}/ in pareto/{self.system_id}/{self.benchmark_model}/",
                        path,
                        "#1",
                    )
                )
        return self
