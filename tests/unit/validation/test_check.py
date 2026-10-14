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
import yaml
from inference_endpoint.validation.check import Check, Cohort, Severity, _decode_row
from inference_endpoint.validation.compare import Aggregate, Compare
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.runner import Finding
from inference_endpoint.validation.submission import Point, Submission
from inference_endpoint.validation.tree import RuleScopes, SubmissionNode

CHECK_ONE = """
offline_ordering:
  description: The dedicated Offline run meets Section 5.7.2's two clauses.
  rule: "Endpoints Rules Section 5.7.2, Relationship to Maximum Supported Concurrency."
  fix: Either raise the dedicated run's system_tps to 0.98x the C_max point's, or
    elect the C_max point as Offline.
  severity: warn
  when:
    - left: point.config.offline
      op: eq
      right: dedicated
  assert:
    - left: derived.offline_ordering
      op: eq
      right: true
"""

CHECK_TWO = """
another_check:
  description: A different rule entirely.
  rule: "Upload Rules > Other"
  fix: Fix the other thing.
  severity: warn
  assert:
    - left: system.config.is_public
      op: eq
      right: true
"""


def _cohort(tmp_path: Path, one: str = CHECK_ONE, two: str | None = None) -> Path:
    """A cohort directory with `one` in `point.yaml` and `two` in `system.yaml`."""
    root = tmp_path / "cohort"
    root.mkdir()
    (root / "point.yaml").write_text(one)
    if two is not None:
        (root / "system.yaml").write_text(two)
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

    def test_typed_rows_pass_through(self):
        compare = Compare.model_validate({"left": 1, "op": "eq", "right": 1})
        aggregate = Aggregate(over=[1], measure="count", op="eq", value=1)
        assert _decode_row(compare) is compare
        assert _decode_row(aggregate) is aggregate

    @pytest.mark.parametrize("row", [{2: 3}, {"left": 1}])
    def test_construction_failures_are_value_errors_naming_the_row(self, row):
        with pytest.raises(ValueError, match="bad row"):
            _decode_row(row)

    def test_broken_row_yaml_names_the_scope_file(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        (root / "point.yaml").write_text(
            "x:\n  description: d\n  rule: r\n  severity: warn\n"
            "  when:\n    - 2: 3\n  assert:\n    - left: 1\n      op: eq\n      right: 1\n"
        )
        with pytest.raises(ValueError, match="point.yaml"):
            Cohort.from_directory(root)


class TestCheckShape:
    """`Check`: metadata binding and the `when`/`assert` row list typing."""

    def _yaml_dict(self) -> dict:
        return {
            "id": "check_id",
            "description": "What is observed.",
            "rule": "Where in the Rules this comes from.",
            "fix": "How to resolve it.",
            "severity": "fail",
            "scope": "point",
            "assert": [{"left": 1, "op": "eq", "right": 1}],
        }

    def test_metadata_binds(self):
        check = Check.model_validate(self._yaml_dict())
        assert check.id == "check_id"
        assert check.severity is Severity.FAIL
        assert check.scope is RuleScopes.POINT
        assert [type(row) for row in check.assert_] == [Compare]

    def test_when_rows_decode_too(self):
        check = Check.model_validate(
            self._yaml_dict()
            | {"when": [{"over": [], "measure": "count", "op": "eq", "value": 0}]}
        )
        assert isinstance(check.when[0], Aggregate)

    def test_fix_is_optional_with_absent_grid(self):
        present = Check.model_validate(self._yaml_dict())
        assert present.fix == "How to resolve it."
        absent = Check.model_validate(
            {key: value for key, value in self._yaml_dict().items() if key != "fix"}
        )
        assert absent.fix is None

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError):
            Check.model_validate(self._yaml_dict() | {"extra_key": 1})

    def test_without_when_or_assert_is_rejected(self):
        bare = {
            key: value for key, value in self._yaml_dict().items() if key != "assert"
        }
        with pytest.raises(ValueError, match="at least one"):
            Check.model_validate(bare)


class TestStubCohort:
    """The 2 PR-1 stub checks decode through `Cohort.from_directory` against real rows."""

    def test_stub_cohort_loads_in_full(self):
        cohort = Cohort.from_directory(Path("tests/assets/cohorts/stub"))
        assert [check.id for check in cohort.checks] == [
            "concurrency_in_range",
            "offline_ordering",
        ]
        assert [check.severity for check in cohort.checks] == [Severity.WARN] * 2

    def test_stub_checks_evaluate_against_a_submission(self, tmp_path):
        # End-to-end over a real tree: the range stub holds at every in-band
        # point and flags r9013 (above the 10% margin); the offline stub reads
        # the dedicated point's declaration and reports its clause verdict.
        cohort = Cohort.from_directory(Path("tests/assets/cohorts/stub"))
        root = tmp_path / "acme"
        configs = {
            32: {"concurrency": 32},
            512: {"concurrency": 512, "offline": "dedicated"},
            8192: {"concurrency": 8192},
            9013: {"concurrency": 9013},
        }
        for concurrency, config in configs.items():
            point_dir = root / "results" / "sys" / "model" / f"r{concurrency}"
            point_dir.mkdir(parents=True)
            (point_dir / "point.yaml").write_text(yaml.dump(config))
        system_desc = {"max_supported_concurrency": 8192}
        tree = SubmissionNode.from_directory(
            root,
            regions=Regions.for_submission(system_desc, sorted(configs)),
        )
        nodes = {
            int(node.dir.name[1:]): node
            for node in tree.children[0].children[0].children
        }
        tps = {32: 1.0, 512: 5292.0, 8192: 5400.0, 9013: 1.0}
        submission = Submission(
            system_desc,
            tuple(
                Point(
                    "dedicated" if concurrency == 512 else None,
                    concurrency,
                    tps[concurrency],
                    nodes[concurrency],
                )
                for concurrency in sorted(configs)
            ),
        )
        findings = Finding.collect(cohort, submission)
        got = sorted(
            (f.check_id, f.node_directory.name, f.verdict.ok) for f in findings
        )
        assert got == [
            ("concurrency_in_range", "r32", True),
            ("concurrency_in_range", "r512", True),
            ("concurrency_in_range", "r8192", True),
            ("concurrency_in_range", "r9013", False),
            ("offline_ordering", "r512", False),
        ]


class TestCohortFromDirectory:
    """`Cohort.from_directory`: loads the four scope files as id-keyed blocks."""

    def test_checks_keep_their_mapping_order(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        (root / "point.yaml").write_text(
            "first_check:\n  description: First.\n  rule: R1\n  severity: warn\n"
            "  assert: [{left: 1, op: eq, right: 1}]\n"
            "second_check:\n  description: Second.\n  rule: R2\n  severity: warn\n"
            "  assert: [{left: 2, op: eq, right: 2}]\n"
        )
        cohort = Cohort.from_directory(root)
        assert [check.id for check in cohort.checks] == ["first_check", "second_check"]

    def test_scope_and_id_come_from_path_and_key(self, tmp_path):
        cohort = Cohort.from_directory(_cohort(tmp_path, two=CHECK_TWO))
        assert {c.id: c.scope.value for c in cohort.checks} == {
            "offline_ordering": "point",
            "another_check": "system",
        }

    def test_duplicate_id_across_scope_files_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate check id"):
            Cohort.from_directory(_cohort(tmp_path, two=CHECK_ONE))

    def test_a_check_does_not_declare_its_own_id_or_scope(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        (root / "point.yaml").write_text(
            CHECK_ONE.replace("severity: warn\n", "severity: warn\n  id: other\n")
        )
        with pytest.raises(ValueError, match="defines its own"):
            Cohort.from_directory(root)

    def test_broken_yaml_names_the_scope_file(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        (root / "point.yaml").write_text("bad: not-a-mapping\n")
        with pytest.raises(ValueError, match="point.yaml"):
            Cohort.from_directory(root)

    def test_parse_problems_name_the_scope_file_too(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        (root / "point.yaml").write_text("first: 1\nfirst: 2\n")
        with pytest.raises(ValueError, match="point.yaml"):
            Cohort.from_directory(root)

    def test_missing_directory_is_a_load_error(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            Cohort.from_directory(tmp_path / "nope")

    def test_an_empty_cohort_is_refused_not_a_clean_pass(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        with pytest.raises(ValueError, match="holds no checks"):
            Cohort.from_directory(root)

    def test_an_empty_scope_file_is_a_load_error(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        (root / "point.yaml").write_text("")
        with pytest.raises(ValueError, match="empty scope file"):
            Cohort.from_directory(root)

    def test_a_stray_file_names_what_it_found(self, tmp_path):
        root = tmp_path / "cohort"
        root.mkdir()
        (root / "almost.yaml.bak").write_text(CHECK_ONE)
        with pytest.raises(ValueError, match="almost.yaml.bak"):
            Cohort.from_directory(root)
