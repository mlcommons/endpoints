"""Utilities for loading and parsing submission artifact files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .models import (
    AccuracyResult,
    CheckResult,
    PointConfig,
    PointSummary,
    Severity,
    SystemDescription,
)

_T = TypeVar("_T", bound=BaseModel)


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


def _load_validated(data: dict, model_cls: type[_T], filename: str) -> tuple[_T | None, str | None]:
    try:
        return model_cls.model_validate(data), None
    except ValidationError as exc:
        first = exc.errors()[0]
        return None, f"Validation error in {filename}: {first['loc']} — {first['msg']}"


def load_system_description(path: Path) -> tuple[SystemDescription | None, str | None]:
    """Load and validate ``system_desc_id.json``.

    Returns:
        A ``(model, error)`` pair — exactly one of the two is ``None``.
    """
    data, err = _load_json(path)
    return _load_validated(data, SystemDescription, path.name) if data is not None else (None, err)


def load_point_config(
    path: Path, context: dict | None = None
) -> tuple[PointConfig | None, list[CheckResult]]:
    """Load and validate a ``point_<N>.yaml`` measurement-point config.

    Returns:
        A ``(model, check_results)`` pair.  On success the model is not None and
        check_results contains the validator-produced CheckResult entries.
        On failure the model is None and check_results contains a single ERROR entry.
    """
    data, load_err = _load_yaml(path)
    if load_err:
        return None, [
            CheckResult(
                rule="point-config-valid", message=load_err, severity=Severity.ERROR, path=path
            )
        ]
    try:
        instance = PointConfig.model_validate(data, context=context or {})
        return instance, list(instance._check_results)
    except ValidationError as exc:
        first = exc.errors()[0]
        return None, [
            CheckResult(
                rule="point-config-valid",
                message=f"Validation error in {path.name}: {first['loc']} — {first['msg']}",
                severity=Severity.ERROR,
                path=path,
            )
        ]


def load_result_summary(path: Path) -> tuple[PointSummary | None, str | None]:
    """Load and validate ``mlperf_endpoints_log_summary.json``.

    Returns:
        A ``(model, error)`` pair — exactly one of the two is ``None``.
    """
    data, err = _load_json(path)
    return _load_validated(data, PointSummary, path.name) if data is not None else (None, err)


def load_accuracy_result(path: Path) -> tuple[AccuracyResult | None, str | None]:
    """Load and validate ``accuracy_result.json``.

    Returns:
        A ``(model, error)`` pair — exactly one of the two is ``None``.
    """
    data, err = _load_json(path)
    return _load_validated(data, AccuracyResult, path.name) if data is not None else (None, err)
