# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission boundary and metric calculation regressions."""

import json
import shutil

import pytest
import yaml
from pydantic import ValidationError

from inference_endpoint.validation import (
    load_policy,
    validate_submission,
)
from inference_endpoint.validation.evidence.loaders import (
    MAX_ARTIFACT_DEPTH,
    ParsedArtifact,
    supplied_fields,
)
from inference_endpoint.validation.evidence.point_config import PointConfig
from inference_endpoint.validation.evidence.point_summary import PointSummary
from inference_endpoint.validation.operations import SeedOperation
from inference_endpoint.validation.schemas.requirements_v1 import (
    SeedBindingRequirements,
)
from tests.unit.validation.helpers import selected_policy

pytestmark = pytest.mark.unit


def write_summary(root, summary):
    point = root / "results/system/gpt-oss-120b/r16"
    point.mkdir(parents=True)
    (point / "result_summary.json").write_text(
        json.dumps({"n_samples_completed": 100, "duration_ns": 1e9, **summary})
    )
    return point


def test_system_names_are_compared_within_each_system(
    standardized_submission, tmp_path
):
    shutil.copytree(standardized_submission, tmp_path / "submission")
    root = tmp_path / "submission"
    systems = root / "results"
    original = next(systems.iterdir())
    second = systems / "system_second"
    shutil.copytree(original, second)
    for path in second.glob("*/r*/system_desc.json"):
        data = json.loads(path.read_text())
        data["system_name"] = "system_second"
        path.write_text(json.dumps(data))
    policy = selected_policy("system-name-consistency")
    report = validate_submission(root, policy=policy)
    assert len(report.results) == 2
    assert report.passed
    path = next(second.glob("*/r*/system_desc.json"))
    data = json.loads(path.read_text())
    data["system_name"] = "inconsistent"
    path.write_text(json.dumps(data))
    report = validate_submission(root, policy=policy)
    assert len(report.errors) == 1
    assert report.errors[0].path == second


@pytest.mark.parametrize(
    "explicit,failed,expected",
    [
        ({}, 0, 100),
        ({}, 1, None),
        (
            {"output_tokens_per_turn_total": 200, "e2e_turn_time_seconds_total": 1},
            1,
            200,
        ),
        ({"output_tokens_per_turn_total": 200}, 0, None),
        ({"e2e_turn_time_seconds_total": 1}, 0, None),
        (
            {"output_tokens_per_turn_total": 200, "e2e_turn_time_seconds_total": 0},
            0,
            None,
        ),
    ],
)
def test_interactivity_derivation_preserves_population_and_precedence(
    explicit, failed, expected
):
    summary = PointSummary.model_validate(
        {
            "n_samples_completed": 100,
            "n_samples_failed": failed,
            "duration_ns": 1e9,
            "output_sequence_lengths": {"total": 1000},
            "latency": {"total": 1e10},
            **explicit,
        }
    )
    assert summary.e2e_avg_interactivity == expected


def test_client_latency_totals_execute_interactivity_check(tmp_path):
    write_summary(
        tmp_path,
        {
            "output_sequence_lengths": {"total": 1000},
            "latency": {"total": 1e10},
            "e2e_avg_interactivity": 100,
        },
    )
    report = validate_submission(
        tmp_path, policy=selected_policy("agentic-metric-consistency")
    )
    assert len(report.results) == 1
    assert report.passed


@pytest.mark.parametrize(
    "field",
    ["duration_ns", "output_tokens_per_turn_total", "e2e_turn_time_seconds_total"],
)
def test_nonfinite_summary_scalars_are_invalid(field):
    with pytest.raises(ValidationError, match="finite"):
        PointSummary.model_validate(
            {"n_samples_completed": 100, "duration_ns": 1e9, field: float("inf")}
        )


@pytest.mark.parametrize("stored", [None, 1])
def test_finite_inputs_with_overflowing_derived_throughput_fail(tmp_path, stored):
    data = {"duration_ns": 1e-299, "output_sequence_lengths": {"total": 1e100}}
    if stored is not None:
        data["system_tps"] = stored
    write_summary(tmp_path, data)
    report = validate_submission(
        tmp_path, policy=selected_policy("metric-consistency-system-tps")
    )
    assert len(report.errors) == 1
    assert "not finite" in report.errors[0].message


def test_overflowing_json_number_produces_invalid_artifact(tmp_path):
    point = write_summary(tmp_path, {})
    (point / "result_summary.json").write_text(
        '{"n_samples_completed": 13368, "duration_ns": 1e9, "output_sequence_lengths": {"total": 1e309}}'
    )
    report = validate_submission(tmp_path, policy=selected_policy("result-file-valid"))
    assert report.errors
    assert all(result.rule == "result-file-valid" for result in report.errors)
    assert any("finite" in result.message for result in report.errors)


def test_recursive_yaml_alias_is_invalid_artifact(tmp_path):
    point = write_summary(tmp_path, {})
    (point / "point.yaml").write_text(
        "concurrency: 16\nruntime_settings: {runtime: {}}\ncycle: &loop {self: *loop}\n"
    )
    report = validate_submission(tmp_path, policy=selected_policy("point-config-valid"))
    assert len(report.errors) == 1
    assert "recursive alias" in report.errors[0].message


def test_shared_alias_records_each_path():
    shared = {"leaf": 1}
    assert supplied_fields({"a": shared, "b": shared}) == frozenset(
        {"a", "a.leaf", "b", "b.leaf"}
    )


def test_cycle_through_list_is_invalid_artifact(tmp_path):
    cycle = []
    cycle.append(cycle)
    parsed = ParsedArtifact.from_json(
        {"unused": cycle}, PointConfig, tmp_path, "point-config-valid"
    )
    assert parsed.value is None
    assert "recursive alias" in parsed.errors[0].message


def test_excessive_nesting_is_invalid_artifact(tmp_path):
    value = {}
    for _ in range(MAX_ARTIFACT_DEPTH + 2):
        value = {"child": value}
    parsed = ParsedArtifact.from_json(
        value, PointConfig, tmp_path, "point-config-valid"
    )
    assert parsed.value is None
    assert "nesting exceeds" in parsed.errors[0].message


@pytest.mark.parametrize(
    "issued,completed,failed,passed",
    [
        (100, 99, 1, True),
        (100, 100, 1, False),
        (100, 100, 0, True),
        (100, 100, None, True),
    ],
)
def test_sample_accounting_includes_failures(
    tmp_path, issued, completed, failed, passed
):
    summary = {"n_samples_issued": issued, "n_samples_completed": completed}
    if failed is not None:
        summary["n_samples_failed"] = failed
    write_summary(tmp_path, summary)
    report = validate_submission(
        tmp_path, policy=selected_policy("metric-consistency-accounting")
    )
    assert len(report.results) == 1
    assert report.passed is passed


@pytest.mark.parametrize("override", [False, True])
def test_unknown_policy_mode_is_rejected_before_selection(policy_dir, override):
    directory = policy_dir
    path = directory / ("catalog.yaml" if override else "curve_checks.yaml")
    document = yaml.safe_load(path.read_text())
    if override:
        document["enrollment"]["overrides"].append(
            {
                "id": "bad-weights",
                "rule": "accuracy-gate",
                "reason": "test",
                "applies_to": {"models": ["gpt-oss-120b"]},
                "requirements": {"on_missing_weights": "not_an_implemented_mode"},
            }
        )
    else:
        document["checks"]["accuracy-gate"]["on_missing_weights"] = (
            "not_an_implemented_mode"
        )
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError, match="on_missing_weights"):
        load_policy(directory)


def test_requirement_sequences_and_aliases_are_immutable():
    aliases = {"scheduler_random_seed": "scheduler_rng_seed"}
    requirements = SeedBindingRequirements(
        operation=SeedOperation.LEGACY_NAMES, aliases=aliases
    )
    aliases["scheduler_random_seed"] = "changed"
    assert requirements.aliases["scheduler_random_seed"] == "scheduler_rng_seed"
    with pytest.raises(TypeError):
        requirements.aliases["scheduler_random_seed"] = "changed"
    assert requirements.with_updates({}).wire() == requirements.wire()


@pytest.mark.parametrize(
    "operands,issued,passed",
    [
        (
            [
                {
                    "kind": "sum",
                    "operands": [
                        {"kind": "constant", "value": 10},
                        {"kind": "constant", "value": 20},
                    ],
                },
                {"kind": "field", "field": "result_summary.n_samples_completed"},
            ],
            130,
            True,
        ),
        (
            [
                {"kind": "field", "field": "result_summary.tps_per_user", "default": 7},
                {"kind": "field", "field": "result_summary.n_samples_completed"},
            ],
            107,
            True,
        ),
        (
            [
                {"kind": "field", "field": "result_summary.tps_per_user"},
                {"kind": "field", "field": "result_summary.n_samples_completed"},
            ],
            100,
            False,
        ),
        (
            [
                {"kind": "constant", "value": 1e308},
                {"kind": "constant", "value": 1e308},
            ],
            100,
            False,
        ),
    ],
)
def test_sum_operands_support_nesting_defaults_and_missing_values(
    tmp_path, policy_dir, operands, issued, passed
):
    path = policy_dir / "point_checks.yaml"
    document = yaml.safe_load(path.read_text())
    document["checks"]["metric-consistency-accounting"]["left"] = {
        "kind": "sum",
        "operands": operands,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    policy = load_policy(policy_dir)
    policy = selected_policy("metric-consistency-accounting", policy=policy)
    write_summary(tmp_path, {"n_samples_issued": issued})
    report = validate_submission(tmp_path, policy=policy)
    assert len(report.results) == 1
    assert report.passed is passed
    assert all(
        "Evaluation could not complete" not in result.message
        for result in report.results
    )


@pytest.mark.parametrize("duration", [1e-320, 1e-299])
def test_tiny_positive_durations_never_crash_derivation(tmp_path, duration):
    write_summary(
        tmp_path,
        {
            "duration_ns": duration,
            "output_sequence_lengths": {"total": 1e100},
            "tpot": {"percentiles": {"90": duration}},
        },
    )
    report = validate_submission(
        tmp_path,
        policy=selected_policy(
            "metric-consistency-system-tps", "metric-consistency-tps-per-user"
        ),
    )
    assert report.errors
    assert all(
        "Evaluation could not complete" not in result.message
        for result in report.results
    )
