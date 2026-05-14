"""Check infrastructure — Severity, CheckResult, result helpers, and Report."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Severity(str, Enum):
    """Severity level for a check result."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class CheckResult(BaseModel):
    """Result of a single automated check.

    Attributes:
        rule: Short identifier matching a §9.1 check name.
        message: Human-readable description of the finding.
        severity: How critical the finding is.
        path: File or directory the finding applies to, if any.
    """

    model_config = ConfigDict(frozen=True)

    rule: str
    message: str
    severity: Severity = Severity.ERROR
    path: Path | None = None
    spec_ref: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """True when the result is not an error."""
        return self.severity != Severity.ERROR


def ok(rule: str, message: str, path: Path | None = None, spec_ref: str = "") -> CheckResult:
    """Return an INFO-severity :class:`CheckResult` — the check passed."""
    return CheckResult(
        rule=rule, message=message, severity=Severity.INFO, path=path, spec_ref=spec_ref
    )


def warn(rule: str, message: str, path: Path | None = None, spec_ref: str = "") -> CheckResult:
    """Return a WARNING-severity :class:`CheckResult` — notable but not a hard failure."""
    return CheckResult(
        rule=rule, message=message, severity=Severity.WARNING, path=path, spec_ref=spec_ref
    )


def err(rule: str, message: str, path: Path | None = None, spec_ref: str = "") -> CheckResult:
    """Return an ERROR-severity :class:`CheckResult` — the check failed."""
    return CheckResult(
        rule=rule, message=message, severity=Severity.ERROR, path=path, spec_ref=spec_ref
    )


class Report(BaseModel):
    """Aggregated results from all checks against a submission.

    Attributes:
        submission_path: Root directory that was checked.
        results: Individual check results in order of execution.
    """

    submission_path: Path
    results: list[CheckResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def errors(self) -> list[CheckResult]:
        """All results with ERROR severity."""
        return [r for r in self.results if r.severity == Severity.ERROR]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warnings(self) -> list[CheckResult]:
        """All results with WARNING severity."""
        return [r for r in self.results if r.severity == Severity.WARNING]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """True when there are no errors."""
        return len(self.errors) == 0
