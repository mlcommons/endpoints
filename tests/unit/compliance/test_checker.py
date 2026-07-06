# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
from inference_endpoint.compliance import (
    check_accuracy,
    check_config_lock,
    check_perf_validity,
    check_submission,
)
from inference_endpoint.config.rulesets.mlcommons import models

GOLDEN = {"bfcl_overall_accuracy": 86.23, "bfcl_normalized_accuracy": 87.96}
FACTORS = {"bfcl_overall_accuracy": (0.97,), "bfcl_normalized_accuracy": (0.97,)}


def _passing_config() -> dict:
    return {
        "model_params": {"temperature": 0.0, "seed": 42},
        "settings": {
            "runtime": {"dataloader_random_seed": 42},
            "client": {"num_workers": 1, "max_connections": 1},
            "load_pattern": {"type": "agentic_inference", "target_concurrency": 1},
        },
    }


# A full, schema-valid config the checker can round-trip through
# BenchmarkConfig.from_yaml_file(); its values satisfy config-lock.
_VALID_CONFIG_YAML = """
name: "compliance-test"
type: "offline"
model_params:
  name: "qwen3.6-27b"
  temperature: 0.0
  seed: 42
datasets:
  - name: "bfcl_v4::function_calling"
    type: "accuracy"
    path: "bfcl.jsonl"
settings:
  runtime:
    dataloader_random_seed: 42
  load_pattern:
    type: "max_throughput"
  client:
    num_workers: 1
    max_connections: 1
endpoint_config:
  endpoints:
    - "http://localhost:8000"
"""


def _write_valid_config(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_VALID_CONFIG_YAML)


def _accuracy_results(
    overall: float | str, normalized: float | str, total: int | str
) -> dict:
    return {
        "accuracy_scores": {
            "bfcl_v4::function_calling": {
                "score": 0.8623,
                "breakdown": {
                    "overall_accuracy": overall,
                    "normalized_single_turn_score": normalized,
                    "total_samples": total,
                },
            }
        }
    }


@pytest.mark.unit
def test_config_lock_passes_on_compliant_config():
    checks = check_config_lock(_passing_config())
    assert {c.name for c in checks} == {"temperature==0", "seed==42", "single_stream"}
    assert all(c.passed for c in checks)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["model_params"].__setitem__("temperature", 0.7),
        lambda c: c["model_params"].__setitem__("seed", 7),
        lambda c: c["settings"]["client"].__setitem__("max_connections", 4),
        lambda c: c["settings"]["load_pattern"].__setitem__("target_concurrency", 8),
    ],
)
def test_config_lock_fails_on_violation(mutate):
    config = _passing_config()
    mutate(config)
    checks = check_config_lock(config)
    assert not all(c.passed for c in checks)


@pytest.mark.unit
def test_accuracy_gate_passes_at_reference():
    checks = check_accuracy(_accuracy_results(86.23, 87.96, 995), GOLDEN, FACTORS, 995)
    assert all(c.passed for c in checks)


@pytest.mark.unit
def test_accuracy_gate_passes_at_threshold_boundary():
    # 0.97 x 86.23 = 83.6431 ; 0.97 x 87.96 = 85.3212
    checks = check_accuracy(
        _accuracy_results(83.6431, 85.3212, 995), GOLDEN, FACTORS, 995
    )
    assert all(c.passed for c in checks)


@pytest.mark.unit
def test_accuracy_gate_fails_below_threshold():
    checks = check_accuracy(_accuracy_results(80.0, 85.4, 995), GOLDEN, FACTORS, 995)
    overall = next(c for c in checks if c.name == "accuracy:overall_accuracy")
    assert not overall.passed


@pytest.mark.unit
def test_accuracy_gate_fails_when_no_metric_applies():
    # A score is present but the golden/factor tables carry no key that
    # intersects _ACCURACY_METRIC_KEYS: the loop appends zero accuracy checks,
    # so without the explicit guard the gate would silently PASS. It must FAIL.
    checks = check_accuracy(_accuracy_results(86.23, 87.96, 995), {}, {}, None)
    applicable = next(c for c in checks if c.name == "accuracy_metric_applicable")
    assert not applicable.passed
    assert not any(c.name.startswith("accuracy:") for c in checks)


@pytest.mark.unit
def test_accuracy_gate_no_metric_applicable_check_absent_when_metrics_apply():
    checks = check_accuracy(_accuracy_results(86.23, 87.96, 995), GOLDEN, FACTORS, 995)
    assert not any(c.name == "accuracy_metric_applicable" for c in checks)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overall, normalized, total, expect_pass",
    [
        # Breakdown metrics are numeric, but older artifacts stored them as
        # ":.2f" strings; the gate must still coerce before comparing.
        ("86.23", "87.96", 995, True),
        ("83.65", "85.33", 995, True),  # just above 0.97x reference
        ("83.64", "87.96", 995, False),  # just below 0.97x86.23=83.6431
        ("80.00", "85.40", 995, False),
    ],
)
def test_accuracy_gate_string_contract(overall, normalized, total, expect_pass):
    # String-valued scores must not raise (str >= float / ":.2f" on str) — the
    # gate coerces them — and must produce the correct pass/fail.
    checks = check_accuracy(
        _accuracy_results(overall, normalized, total), GOLDEN, FACTORS, 995
    )
    assert all(c.passed for c in checks) is expect_pass


@pytest.mark.unit
def test_accuracy_gate_handles_string_total_samples():
    checks = check_accuracy(
        _accuracy_results("86.23", "87.96", "995"), GOLDEN, FACTORS, 995
    )
    samples = next(c for c in checks if c.name == "min_sample_count")
    assert samples.passed


@pytest.mark.unit
def test_check_submission_accuracy_dir_string_scores(tmp_path):
    # End-to-end with string-valued breakdown metrics (defensive coercion path).
    _write_valid_config(tmp_path)
    (tmp_path / "results.json").write_text(
        json.dumps(_accuracy_results("86.23", "87.96", 995))
    )
    report = check_submission(tmp_path)
    assert report.passed


@pytest.mark.unit
def test_accuracy_gate_fails_on_too_few_samples():
    checks = check_accuracy(_accuracy_results(90.0, 90.0, 500), GOLDEN, FACTORS, 995)
    samples = next(c for c in checks if c.name == "min_sample_count")
    assert not samples.passed


@pytest.mark.unit
def test_perf_validity_passes_when_observed_matches_expected():
    # issued (1007) > expected (1006): a complete run observes every *scorable*
    # turn but fewer than issued. Validating against expected must still pass.
    checks = check_perf_validity(
        {"turns": {"issued": 1007, "expected": 1006, "observed": 1006, "missing": 0}}
    )
    assert all(c.passed for c in checks)


@pytest.mark.unit
def test_perf_validity_fails_with_dropped_turns():
    checks = check_perf_validity(
        {
            "turns": {
                "issued": 1007,
                "expected": 1006,
                "observed": 969,
                "missing": 38,
            }
        }
    )
    assert not all(c.passed for c in checks)


@pytest.mark.unit
def test_check_submission_accuracy_dir(tmp_path):
    _write_valid_config(tmp_path)
    (tmp_path / "results.json").write_text(
        json.dumps(_accuracy_results(86.23, 87.96, 995))
    )
    report = check_submission(tmp_path)
    assert report.passed
    assert report.notes  # server-side attestation surfaced
    names = {c.name for c in report.checks}
    assert "accuracy:overall_accuracy" in names
    assert "min_sample_count" in names


@pytest.mark.unit
def test_check_submission_perf_dir(tmp_path):
    _write_valid_config(tmp_path)
    (tmp_path / "scores.json").write_text(
        json.dumps(
            {
                "turns": {
                    "issued": 1007,
                    "expected": 1006,
                    "observed": 1006,
                    "missing": 0,
                }
            }
        )
    )
    report = check_submission(tmp_path)
    assert report.passed
    assert "no_dropped_turns" in {c.name for c in report.checks}


@pytest.mark.unit
def test_check_submission_uses_ruleset_thresholds(tmp_path):
    # Real ruleset model golden/factors should yield the documented 83.64 gate.
    model = models.Qwen3_6_27B
    _, golden = model.golden_accuracy
    (factors,) = model.accuracy_target_settings
    threshold = golden["bfcl_overall_accuracy"] * factors["bfcl_overall_accuracy"][0]
    assert threshold == pytest.approx(83.6431)

    _write_valid_config(tmp_path)
    (tmp_path / "results.json").write_text(
        json.dumps(_accuracy_results(83.0, 87.0, 995))
    )
    report = check_submission(tmp_path)
    assert not report.passed


@pytest.mark.unit
def test_check_submission_missing_artifacts(tmp_path):
    report = check_submission(tmp_path)
    assert not report.passed
