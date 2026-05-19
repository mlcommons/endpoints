"""Utilities for loading and parsing submission artifact files."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import (
    AccuracyResult,
    CheckResult,
    PointConfig,
    PointSummary,
    Severity,
    SystemDescription,
)


def _load_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error in {path.name}: {exc}"
    except OSError as exc:
        return None, f"IO error reading {path.name}: {exc}"


def _load_yaml(path: Path) -> tuple[dict | None, str | None]:
    try:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            return None, f"Expected a YAML mapping in {path.name}"
        return data, None
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except yaml.YAMLError as exc:
        return None, f"YAML parse error in {path.name}: {exc}"
    except OSError as exc:
        return None, f"IO error reading {path.name}: {exc}"


def _validation_errors(exc: ValidationError, rule: str, path: Path) -> list[CheckResult]:
    return [
        CheckResult(
            rule=rule,
            message=f"Validation error in {path.name}: {e['loc']} — {e['msg']}",
            severity=Severity.ERROR,
            path=path,
        )
        for e in exc.errors()
    ]


def load_system_description(
    path: Path,
) -> tuple[SystemDescription | None, list[CheckResult]]:
    """Load and validate ``system_desc_id.json``.

    Returns:
        A ``(model, check_results)`` pair.  On success the model is not None and
        check_results is empty.  On failure the model is None and check_results
        contains one entry per validation error.
    """
    data, load_err = _load_json(path)
    if load_err:
        return None, [CheckResult(rule="system-description-valid", message=load_err,
                                  severity=Severity.ERROR, path=path)]
    try:
        return SystemDescription.model_validate(data), []
    except ValidationError as exc:
        return None, _validation_errors(exc, "system-description-valid", path)


def load_point_config(
    path: Path, context: dict | None = None
) -> tuple[PointConfig | None, list[CheckResult]]:
    """Load and validate a ``point_<N>.yaml`` measurement-point config.

    Returns:
        A ``(model, check_results)`` pair.  On success the model is not None and
        check_results contains the validator-produced CheckResult entries.
        On failure the model is None and check_results contains one entry per
        validation error.
    """
    data, load_err = _load_yaml(path)
    if load_err:
        return None, [CheckResult(rule="point-config-valid", message=load_err,
                                  severity=Severity.ERROR, path=path)]
    try:
        instance = PointConfig.model_validate(data, context=context or {})
        return instance, list(instance._check_results)
    except ValidationError as exc:
        return None, _validation_errors(exc, "point-config-valid", path)


def load_result_summary(path: Path) -> tuple[PointSummary | None, list[CheckResult]]:
    """Load and validate ``mlperf_endpoints_log_summary.json``.

    Returns:
        A ``(model, check_results)`` pair.  On success the model is not None and
        check_results is empty.  On failure the model is None and check_results
        contains one entry per validation error.
    """
    data, load_err = _load_json(path)
    if load_err:
        return None, [CheckResult(rule="result-file-valid", message=load_err,
                                  severity=Severity.ERROR, path=path)]
    try:
        return PointSummary.model_validate(data), []
    except ValidationError as exc:
        return None, _validation_errors(exc, "result-file-valid", path)


def load_accuracy_result(
    path: Path,
) -> tuple[AccuracyResult | None, list[CheckResult]]:
    """Load and validate ``accuracy_result.json``.

    Returns:
        A ``(model, check_results)`` pair.  On success the model is not None and
        check_results contains the validator-produced CheckResult entries.
        On failure the model is None and check_results contains one entry per
        validation error.
    """
    data, load_err = _load_json(path)
    if load_err:
        return None, [CheckResult(rule="accuracy-valid", message=load_err,
                                  severity=Severity.ERROR, path=path)]
    try:
        instance = AccuracyResult.model_validate(data, context={"json_path": path})
        return instance, list(instance._check_results)
    except ValidationError as exc:
        return None, _validation_errors(exc, "accuracy-valid", path)
