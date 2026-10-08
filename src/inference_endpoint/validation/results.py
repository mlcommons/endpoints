# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Structured findings and submission reports."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .types import Severity


class CheckResult(BaseModel):
    """Result of a single automated check."""

    model_config = ConfigDict(frozen=True)
    rule: str
    message: str
    severity: Severity = Severity.ERROR
    path: Path | None = None
    spec_ref: str = ""
    key: str = ""
    title: str = ""
    fix: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def passed(self) -> bool:
        """True when the result is not an error."""
        return self.severity != Severity.ERROR


class Report(BaseModel):
    """Aggregated results from all checks against a submission."""

    submission_path: Path
    results: list[CheckResult] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def errors(self) -> list[CheckResult]:
        """All results with ERROR severity."""
        return [r for r in self.results if r.severity == Severity.ERROR]

    @computed_field  # type: ignore[misc]
    @property
    def warnings(self) -> list[CheckResult]:
        """All results with WARNING severity."""
        return [r for r in self.results if r.severity == Severity.WARNING]

    @computed_field  # type: ignore[misc]
    @property
    def passed(self) -> bool:
        """True when there are no errors."""
        return len(self.errors) == 0
