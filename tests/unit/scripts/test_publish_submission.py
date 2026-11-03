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

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "publish_submission.py"


def _load_publish_submission():
    """Load scripts/publish_submission.py as a module (it is not a package)."""
    if "publish_submission" in sys.modules:
        return sys.modules["publish_submission"]
    spec = importlib.util.spec_from_file_location("publish_submission", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["publish_submission"] = module
    spec.loader.exec_module(module)
    return module


class TestPercentile:
    def test_exact_key_match(self):
        ps = _load_publish_submission()
        metric = {"percentiles": {"99.0": 12.0}}
        assert ps._percentile(metric, "99.0") == 12.0

    def test_int_spelled_key_matches_float_request(self):
        ps = _load_publish_submission()
        metric = {"percentiles": {"99": 12.0}}
        assert ps._percentile(metric, "99.0") == 12.0

    def test_missing_fractional_percentile_does_not_fall_back_to_integer(self):
        """A request for 99.5 must not silently return the 99 bucket."""
        ps = _load_publish_submission()
        metric = {"percentiles": {"99.0": 12.0}}
        assert ps._percentile(metric, "99.5") is None

    def test_fractional_percentile_matches_when_present(self):
        ps = _load_publish_submission()
        metric = {"percentiles": {"99.5": 20.0, "99.0": 12.0}}
        assert ps._percentile(metric, "99.5") == 20.0

    def test_non_dict_metric_returns_none(self):
        ps = _load_publish_submission()
        assert ps._percentile(None, "99.0") is None


class TestVerifyAccuracy:
    def test_empty_scores_reports_missing(self, tmp_path):
        ps = _load_publish_submission()
        run = tmp_path / "acc_run"
        (run / "accuracy").mkdir(parents=True)
        (run / "accuracy" / "accuracy_results.json").write_text("{}", encoding="utf-8")
        findings = ps._verify_accuracy(run)
        assert any("MISSING" in f for f in findings)

    def test_populated_list_shape_reports_score_and_breakdown(self, tmp_path):
        """A populated list-shaped accuracy_scores must yield score/overall_accuracy
        findings without raising (the list shape has no ``.items()``)."""
        ps = _load_publish_submission()
        run = tmp_path / "acc_run"
        (run / "accuracy").mkdir(parents=True)
        results = {
            "accuracy_scores": [
                {
                    "dataset_name": "bfcl_v4::multi_turn",
                    "score": 0.83,
                    "breakdown": {"overall_accuracy": "83.00"},
                }
            ]
        }
        (run / "accuracy" / "accuracy_results.json").write_text(
            json.dumps(results), encoding="utf-8"
        )
        findings = ps._verify_accuracy(run)
        assert any(
            "accuracy_scores[bfcl_v4::multi_turn].score = 0.83" in f for f in findings
        )
        assert any(
            "accuracy_scores[bfcl_v4::multi_turn].breakdown.overall_accuracy = 83.00"
            in f
            for f in findings
        )


class TestMainExitCode:
    def _run(self, argv):
        ps = _load_publish_submission()
        old = sys.argv
        sys.argv = ["publish_submission.py", *argv]
        try:
            return ps.main()
        finally:
            sys.argv = old

    def test_empty_run_dir_returns_nonzero(self, tmp_path):
        """A run dir with none of the parseable artifacts must fail, not exit 0."""
        run = tmp_path / "empty_run"
        run.mkdir()
        rc = self._run(
            [
                "--run",
                str(run),
                "--output",
                str(tmp_path / "submission"),
                "--submitter",
                "NVIDIA",
                "--system",
                "AGX_Thor",
                "--benchmark",
                "qwen3.6-27b",
            ]
        )
        assert rc == 1

    def test_run_with_artifact_returns_zero(self, tmp_path):
        # A combined --run is verified as BOTH a perf and an accuracy run, so it
        # needs a perf artifact (result_summary.json) and an accuracy artifact
        # (accuracy/accuracy_results.json) for the tree to be complete.
        run = tmp_path / "good_run"
        (run / "performance").mkdir(parents=True)
        (run / "performance" / "result_summary.json").write_text("{}", encoding="utf-8")
        (run / "accuracy").mkdir(parents=True)
        (run / "accuracy" / "accuracy_results.json").write_text("{}", encoding="utf-8")
        rc = self._run(
            [
                "--run",
                str(run),
                "--output",
                str(tmp_path / "submission"),
                "--submitter",
                "NVIDIA",
                "--system",
                "AGX_Thor",
                "--benchmark",
                "qwen3.6-27b",
            ]
        )
        assert rc == 0
