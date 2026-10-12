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

"""W4 layer tests: the `Check` model, row decoding and `Cohort.from_directory`.

Fixtures build tiny YAML strings on disk (`_cohort`), not stored files. Cases
assert the values each subject computes: `Check` its shape, `_decode_row` its
row-type selection, `Cohort.from_directory` its loaded set.
"""

from pathlib import Path

import pytest

from inference_endpoint.validation.check import Check, Cohort, Severity, _decode_row
from inference_endpoint.validation.compare import Aggregate, Compare
from inference_endpoint.validation.tree import RuleScopes

CHECK_ONE = """
id: concur_dur_round
description: Concurrency asked for must be the per-system maximum.
rule: "Upload Rules > Concurrency"
fix: Ask for max_sync concurrency in exec_scenario.
severity: fail
root: point
when:
  - over: sys.config.max_sync
    measure: count
    op: eq
    value: 1
assert:
  - left: point.request.concurrency_achieved
    op: eq
    right: point.config.max_sync
"""

CHECK_TWO = """
id: another_check
description: A different rule entirely.
rule: "Upload Rules > Other"
fix: Fix the other thing.
severity: warn
root: system
assert:
  - left: sys.config.is_public
    op: eq
    right: true
"""


def _cohort(tmp_path: Path, one: str = CHECK_ONE, two: str | None = None) -> Path:
    """A cohort directory with `one` (and `two`) written check YAMLs."""
    root = tmp_path / "cohort"
    root.mkdir()
    (root / "a.yaml").write_text(one)
    if two is not None:
        (root / "b.yaml").write_text(two)
    return root


class TestDecodeRow:
    """`_decode_row` selects the row model by what the YAML row holds."""

    def test_aggregate_row_by_measure(self):
        row = _decode_row({"over": [], "measure": "count", "op": "eq", "value": 0})
        assert isinstance(row, Aggregate)

    def test_compare_row_by_default(self):
        row = _decode_row({"left": 1, "op": "eq", "right": 1})
        assert type(row) is Compare

    def test_a_scalar_row_is_a_shape_error(self):
        with pytest.raises(ValueError):
            _decode_row("not-a-row")


class TestCheckShape:
    """`Check`: metadata binding and the `when`/`assert` row list typing."""

    def _yaml_dict(self) -> dict:
        return {
            "id": "check_id",
            "description": "What is observed.",
            "rule": "Where in the Rules this comes from.",
            "fix": "How to resolve it.",
            "severity": "fail",
            "root": "point",
            "assert": [{"left": 1, "op": "eq", "right": 1}],
        }

    def test_metadata_binds(self):
        check = Check.model_validate(self._yaml_dict())
        assert check.id == "check_id"
        assert check.severity is Severity.FAIL
        assert check.root is RuleScopes.POINT
        assert [type(row) for row in check.assert_] == [Compare]

    def test_when_rows_decode_too(self):
        check = Check.model_validate(
            self._yaml_dict()
            | {"when": [{"over": [], "measure": "count", "op": "eq", "value": 0}]}
        )
        assert isinstance(check.when[0], Aggregate)

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError):
            Check.model_validate(self._yaml_dict() | {"extra_key": 1})

    def test_without_when_or_assert_is_rejected(self):
        with pytest.raises(ValueError):
            Check.model_validate(
                {
                    "id": "check_id",
                    "description": "Nothing to say.",
                    "rule": "None",
                    "fix": "None",
                    "severity": "warn",
                    "root": "system",
                }
            )


class TestCohortFromDirectory:
    """`Cohort.from_directory`: loads one check per YAML, sorted by filename."""

    def test_loads_checks_in_order(self, tmp_path):
        cohort = Cohort.from_directory(_cohort(tmp_path, two=CHECK_TWO))
        assert [chunk.id for chunk in cohort.checks] == [
            "concur_dur_round",
            "another_check",
        ]

    def test_duplicate_id_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate check id"):
            Cohort.from_directory(_cohort(tmp_path, two=CHECK_ONE))

    def test_broken_yaml_names_the_file(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        (root / "bad.yaml").write_text("id: just_id\n")
        with pytest.raises(ValueError, match="bad.yaml"):
            Cohort.from_directory(root)

    def test_missing_directory_is_a_load_error(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            Cohort.from_directory(tmp_path / "nope")
