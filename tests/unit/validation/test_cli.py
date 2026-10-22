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

"""CLI tests: `__main__.main`'s exit codes and report lines.

Severity-aware exit: a failing `warn` finding is reported and passes through
(exit 0). A failing `fail` finding blocks (exit 1). A refused cohort or upload
prints `refused:` to stderr and exits 2 before any check runs.
"""

from pathlib import Path

import pytest
from inference_endpoint.validation.__main__ import main
from submission_helpers import Run, write_cohort, write_upload

HOLD_CHECK = """
hold_check:
  description: A satisfied rule.
  rule: "Upload Rules > Holds"
  severity: fail
  assert: [{left: 1, op: eq, right: 1}]
"""

WARN_CHECK = """
warn_check:
  description: A violated advisory rule.
  rule: "Upload Rules > Advises"
  severity: warn
  assert: [{left: 1, op: eq, right: 2}]
"""

FAIL_CHECK = """
fail_check:
  description: A violated blocking rule.
  rule: "Upload Rules > Blocks"
  severity: fail
  assert: [{left: 1, op: eq, right: 2}]
"""
FAIL_WITH_FIX_CHECK = """
fail_fix_check:
  description: A violated blocking rule with a remedy.
  rule: "Upload Rules > Blocks"
  severity: fail
  fix: Raise the value to 2 and rerun.
  assert: [{left: 1, op: eq, right: 2}]
"""

GATED_OFF_CHECK = """
gated_off_check:
  description: A rule whose gate excludes every node.
  rule: "Upload Rules > Gated"
  severity: fail
  when: [{left: 1, op: eq, right: 2}]
  assert: [{left: 1, op: eq, right: 1}]
"""

MISSPELLED_WHEN_CHECK = """
misspelled_check:
  description: A gate naming a nonexistent fact.
  rule: "Upload Rules > Typo"
  severity: fail
  when:
    - left: point.config.ofline
      op: eq
      right: dedicated
  assert: [{left: 1, op: eq, right: 1}]
"""


def _upload(tmp_path: Path) -> Path:
    root = tmp_path / "upload"
    write_upload(root, {("sys", "model"): [Run(concurrency=16)]})
    return root


class TestExitCodes:
    """Exit 0 reports-only/warning failures, 1 blocking, 2 refusal."""

    @pytest.mark.unit
    def test_only_warn_findings_failing_exits_zero(self, tmp_path):
        cohort = write_cohort(tmp_path / "cohort", point=WARN_CHECK)
        assert main([str(cohort), str(_upload(tmp_path))]) == 0

    @pytest.mark.unit
    def test_a_failing_fail_finding_exits_one(self, tmp_path):
        cohort = write_cohort(tmp_path / "cohort", point=FAIL_CHECK)
        assert main([str(cohort), str(_upload(tmp_path))]) == 1

    @pytest.mark.unit
    def test_a_failing_warn_finding_adds_no_blocking_count(self, tmp_path, capsys):
        cohort = write_cohort(tmp_path / "cohort", point=WARN_CHECK)
        main([str(cohort), str(_upload(tmp_path))])
        assert "1 finding(s), 1 failing (0 blocking)" in capsys.readouterr().out

    @pytest.mark.unit
    def test_a_cohort_naming_a_misspelled_fact_is_refused(self, tmp_path, capsys):
        cohort = write_cohort(tmp_path / "cohort", point=MISSPELLED_WHEN_CHECK)
        assert main([str(cohort), str(_upload(tmp_path))]) == 2
        err = capsys.readouterr().err
        assert "refused:" in err
        assert "did you mean 'offline'?" in err

    @pytest.mark.unit
    def test_a_cohort_naming_an_unknown_ruleset_is_refused(self, tmp_path, capsys):
        cohort = write_cohort(tmp_path / "cohort", point=HOLD_CHECK)
        (cohort / "cohort.yaml").write_text(
            "ruleset: no-such-ruleset\n", encoding="utf-8"
        )
        assert main([str(cohort), str(_upload(tmp_path))]) == 2
        err = capsys.readouterr().err
        assert "refused:" in err
        assert "not found in registry" in err

    @pytest.mark.unit
    def test_a_cohort_pulls_its_ruleset_from_its_manifest(self, tmp_path):
        check = (
            "ruleset_row:\n"
            "  description: The packaged ruleset's models are visible to a row.\n"
            "  rule: The ruleset's models are the packaged data.\n"
            "  severity: fail\n"
            "  assert:\n"
            "    - left: ruleset.models\n"
            "      op: exists\n"
        )
        cohort = write_cohort(tmp_path / "cohort", point=check)
        (cohort / "cohort.yaml").write_text(
            "ruleset: mlperf-inference-v6.1\n", encoding="utf-8"
        )
        assert main([str(cohort), str(_upload(tmp_path))]) == 0


class TestReportLines:
    """Marks, per-check instance counts, and the Never-Ran disclosure."""

    @pytest.mark.unit
    def test_report_marks_are_uppercase_per_verdict(self, tmp_path, capsys):
        cohort = write_cohort(
            tmp_path / "cohort",
            point=HOLD_CHECK + WARN_CHECK + FAIL_CHECK,
        )
        assert main([str(cohort), str(_upload(tmp_path)), "--audit"]) == 1
        out = capsys.readouterr().out
        assert "OK hold_check [fail] " in out
        assert "WARN warn_check [warn] " in out
        assert "FAIL fail_check [fail] " in out
        assert "3 finding(s), 2 failing (1 blocking)" in out

    @pytest.mark.unit
    def test_a_check_that_ran_reports_its_instance_count(self, tmp_path, capsys):
        cohort = write_cohort(tmp_path / "cohort", point=HOLD_CHECK)
        upload = tmp_path / "upload"
        write_upload(
            upload,
            {("sys", "model"): [Run(concurrency=16), Run(concurrency=8192)]},
        )
        assert main([str(cohort), str(upload)]) == 0
        assert "hold_check ran at 2 instance(s)" in capsys.readouterr().out

    @pytest.mark.unit
    def test_a_check_gated_off_every_sole_node_says_never_ran(self, tmp_path, capsys):
        cohort = write_cohort(tmp_path / "cohort", point=GATED_OFF_CHECK)
        assert main([str(cohort), str(_upload(tmp_path))]) == 0
        captured = capsys.readouterr()
        err = captured.err
        assert "NEVER-RAN gated_off_check:" in err
        assert "no node was applicable" in err

    @pytest.mark.unit
    def test_control_characters_in_directory_names_stay_one_line(
        self, tmp_path, capsys
    ):
        cohort = write_cohort(tmp_path / "cohort", point=FAIL_CHECK)
        upload = tmp_path / "upload"
        write_upload(
            upload, {("evil\nFAIL forged\x1b[2K", "model"): [Run(concurrency=16)]}
        )
        assert main([str(cohort), str(upload)]) == 1
        out = capsys.readouterr().out
        assert "\x1b" not in out
        assert "evil\\nFAIL forged\\x1b[2K" in out

    @pytest.mark.unit
    def test_a_failing_line_labels_the_row_and_carries_its_rule(self, tmp_path, capsys):
        cohort = write_cohort(tmp_path / "cohort", point=FAIL_CHECK)
        assert main([str(cohort), str(_upload(tmp_path))]) == 1
        out = capsys.readouterr().out
        assert "1 eq 2 ->" in out
        assert "[Upload Rules > Blocks]" in out

    @pytest.mark.unit
    def test_a_declared_fix_rides_its_failing_line(self, tmp_path, capsys):
        cohort = write_cohort(tmp_path / "cohort", point=FAIL_WITH_FIX_CHECK)
        assert main([str(cohort), str(_upload(tmp_path))]) == 1
        assert "    fix: Raise the value to 2 and rerun." in capsys.readouterr().out

    @pytest.mark.unit
    def test_a_held_line_labels_its_row_without_rule_noise(self, tmp_path, capsys):
        cohort = write_cohort(tmp_path / "cohort", point=HOLD_CHECK)
        assert main([str(cohort), str(_upload(tmp_path)), "--audit"]) == 0
        out = capsys.readouterr().out
        assert "OK hold_check [fail] " in out
        assert "1 eq 1" in out
        assert "[Upload Rules > Holds]" not in out
