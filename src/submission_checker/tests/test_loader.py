"""Tests for loader error paths (file-not-found, parse errors, validation errors)."""

from __future__ import annotations

import json
from unittest.mock import patch

import yaml

from submission_checker.loader import (
    load_accuracy_result,
    load_result_summary,
    load_point_config,
    load_system_description,
)
from submission_checker.models import Severity

# ---------------------------------------------------------------------------
# load_system_description
# ---------------------------------------------------------------------------


def test_load_system_description_missing_file(tmp_path):
    model, results = load_system_description(tmp_path / "missing.json")
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "File not found" in results[0].message


def test_load_system_description_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    model, results = load_system_description(p)
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "JSON parse error" in results[0].message


def test_load_system_description_schema_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"division": "Standardized"}))  # missing required fields
    model, results = load_system_description(p)
    assert model is None
    assert all(r.severity == Severity.ERROR for r in results)
    assert len(results) > 1  # all missing fields reported, not just the first
    assert any("Validation error" in r.message for r in results)


# ---------------------------------------------------------------------------
# load_point_config — now returns (PointConfig | None, list[CheckResult])
# ---------------------------------------------------------------------------


def test_load_point_config_missing_file(tmp_path):
    model, results = load_point_config(tmp_path / "missing.yaml")
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "File not found" in results[0].message


def test_load_point_config_invalid_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("key: [unclosed")
    model, results = load_point_config(p)
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "YAML parse error" in results[0].message


def test_load_point_config_non_dict_yaml(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- item1\n- item2\n")
    model, results = load_point_config(p)
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "Expected a YAML mapping" in results[0].message


def test_load_point_config_schema_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump({"dataset": "test"}))  # missing required concurrency
    model, results = load_point_config(p)
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "Validation error" in results[0].message


def test_load_point_config_valid_returns_check_results(tmp_path):
    p = tmp_path / "point_64.yaml"
    p.write_text(
        yaml.dump(
            {
                "concurrency": 64,
                "dataset": "mlperf-perf-dataset-v1",
                "runtime_settings": {"load_pattern": "concurrency"},
            }
        )
    )
    model, results = load_point_config(p, context={"yaml_path": p})
    assert model is not None
    # Should have validator-produced check results (load-pattern, streaming-config at minimum)
    rules = {r.rule for r in results}
    assert "load-pattern" in rules
    assert "streaming-config" in rules


# ---------------------------------------------------------------------------
# load_result_summary
# ---------------------------------------------------------------------------


def test_load_result_summary_missing_file(tmp_path):
    model, results = load_result_summary(tmp_path / "missing.json")
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "File not found" in results[0].message


def test_load_result_summary_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("}")
    model, results = load_result_summary(p)
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "JSON parse error" in results[0].message


def test_load_result_summary_schema_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"n_samples_completed": "not-a-number"}))
    model, results = load_result_summary(p)
    assert model is None
    assert len(results) >= 1
    assert all(r.severity == Severity.ERROR for r in results)
    assert any("Validation error" in r.message for r in results)


# ---------------------------------------------------------------------------
# load_accuracy_result
# ---------------------------------------------------------------------------


def test_load_accuracy_result_missing_file(tmp_path):
    model, results = load_accuracy_result(tmp_path / "missing.json")
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "File not found" in results[0].message


def test_load_accuracy_result_schema_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"metric": "rouge1"}))  # missing score, quality_target, passed
    model, results = load_accuracy_result(p)
    assert model is None
    assert len(results) > 1  # all missing fields reported, not just the first
    assert all(r.severity == Severity.ERROR for r in results)
    assert any("Validation error" in r.message for r in results)


# ---------------------------------------------------------------------------
# OS-level I/O errors (covers loader.py lines 31-32 and 45-46)
# ---------------------------------------------------------------------------


def test_load_json_os_error(tmp_path):
    """_load_json must surface an OSError (e.g. permission denied) as an error message."""
    p = tmp_path / "ok.json"
    payload = {"metric": "rouge1", "score": 0.5, "quality_target": 0.43, "passed": True}
    p.write_text(json.dumps(payload))
    with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
        model, results = load_accuracy_result(p)
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "IO error" in results[0].message


def test_load_yaml_os_error(tmp_path):
    """_load_yaml must surface an OSError (e.g. permission denied) as an error message."""
    p = tmp_path / "point_64.yaml"
    p.write_text(yaml.dump({"concurrency": 64, "runtime_settings": {"load_pattern": "concurrency"}}))
    with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
        model, results = load_point_config(p)
    assert model is None
    assert len(results) == 1
    assert results[0].severity == Severity.ERROR
    assert "IO error" in results[0].message
