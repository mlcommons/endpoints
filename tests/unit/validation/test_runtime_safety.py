# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime safety regressions for malformed evidence and repeated validation."""

import json

import pytest

from inference_endpoint.validation.evidence.loaders import read_artifact
from inference_endpoint.validation.evidence.point_summary import PointSummary
from inference_endpoint.validation.types import Severity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "field", ["tps_per_user", "system_tps", "e2e_avg_interactivity"]
)
@pytest.mark.parametrize(
    "value", [[], {}, "NaN", "Infinity", float("nan"), float("inf")]
)
def test_invalid_stored_metrics_are_structured_load_errors(tmp_path, field, value):
    path = tmp_path / "results_summary.json"
    path.write_text(
        json.dumps({"n_samples_completed": 1, "duration_ns": 1e9, field: value})
    )
    parsed = read_artifact(path, PointSummary, "result-file-valid")
    assert parsed.value is None
    assert parsed.errors
    assert all(result.severity == Severity.ERROR for result in parsed.errors)
