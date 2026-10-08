# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural evidence parsing and policy/evidence separation."""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from inference_endpoint.validation import (
    bundled_policy_path,
    load_policy,
    validate_submission,
)
from inference_endpoint.validation.catalogs.seeds import (
    SeedSetError,
    parse_seed_catalog,
)
from inference_endpoint.validation.evidence.accuracy import AccuracyResult
from inference_endpoint.validation.evidence.loaders import (
    load_point_config,
    load_result_summary,
)
from inference_endpoint.validation.power.models import SourcedValue

pytestmark = pytest.mark.unit


def test_point_parsing_does_not_enforce_submission_policy(tmp_path):
    path = tmp_path / "point.yaml"
    path.write_text("concurrency: 1\nruntime_settings:\n  runtime: {}\n")
    model, findings = load_point_config(path)
    assert model is not None
    assert not findings
    assert not hasattr(model, "_check_results")


def test_native_accuracy_aliases_and_duplicate_datasets():
    entry = {
        "dataset_name": "aime25",
        "score": 90,
        "unit_samples": 30,
        "num_repeats": 2,
    }
    model = AccuracyResult.model_validate({"accuracy_scores": [entry]})
    assert model.root["aime25"]["num_samples"] == 30
    assert model.root["aime25"]["n_repeats"] == 2
    with pytest.raises(ValidationError, match="appears more than once"):
        AccuracyResult.model_validate({"accuracy_scores": [entry, entry]})


def test_empty_accuracy_is_rejected_by_selected_check(tmp_path):
    point = tmp_path / "results" / "system" / "llama3_1-8b" / "r16"
    point.mkdir(parents=True)
    (point / "accuracy_results.json").write_text("{}")
    policy = load_policy(bundled_policy_path())
    rule = next(rule for rule in policy.checks if rule.id == "accuracy-valid")
    report = validate_submission(tmp_path, policy=replace(policy, checks=(rule,)))
    assert not report.passed
    assert report.errors[0].rule == "accuracy-valid"
    assert "empty" in report.errors[0].message


def test_unreadable_encoding_becomes_structured_schema_error(tmp_path):
    path = tmp_path / "result_summary.json"
    path.write_bytes(b"\xff")
    model, findings = load_result_summary(path)
    assert model is None
    assert findings[0].key == "artifact-unreadable"


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_power_source_requires_finite_value(value):
    with pytest.raises(ValidationError):
        SourcedValue.model_validate(
            {
                "value_w": value,
                "source_type": "vendor_spec",
                "source": "https://example.com/power",
            }
        )


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_seed_catalog_rejects_lossy_integer_coercion(tmp_path, value):
    path = tmp_path / "seed_sets.yaml"
    path.write_text(
        json.dumps(
            {
                "seed_sets": [
                    {
                        "id": "A",
                        "scheduler_rng_seed": value,
                        "sample_index_rng_seed": 2,
                        "model_seed": 3,
                    }
                ]
            }
        )
    )
    with pytest.raises(SeedSetError, match="must define integer"):
        parse_seed_catalog(path)


def test_seed_catalog_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "seed_sets.yaml"
    entry = {
        "id": "A",
        "scheduler_rng_seed": 1,
        "sample_index_rng_seed": 2,
        "model_seed": 3,
    }
    path.write_text(json.dumps({"seed_sets": [entry, entry]}))
    with pytest.raises(SeedSetError, match="duplicate"):
        parse_seed_catalog(path)
