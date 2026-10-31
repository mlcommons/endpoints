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

"""Check-block tests: `Check` shape, row decode, and `Cohort.from_directory`.

Fixtures build tiny YAML strings on disk (`_cohort`), not stored files. Cases
assert the values each subject computes: `Check` its shape, `_decode_row` its
row-type selection, `Cohort.from_directory` its loaded set and its load-time
refusals of rows that name no resolvable fact.
"""

from pathlib import Path

import pytest
from inference_endpoint.validation.check import Check, Cohort, Severity, _decode_row
from inference_endpoint.validation.compare import Aggregate, Compare
from inference_endpoint.validation.refs import _Fact
from inference_endpoint.validation.runner import Finding
from inference_endpoint.validation.sentinels import MISSING
from inference_endpoint.validation.tree import RuleScopes
from submission_helpers import Run, write_cohort, write_upload

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
    - left: system.system_power.total
      op: ge
      right: 0
"""

DUP_CHECK = """
dup_check:
  description: A plain literal row.
  rule: "Upload Rules > Plain"
  severity: warn
  assert: [{left: 1, op: eq, right: 1}]
"""

LITERAL_CHECK = """
literal_check:
  description: One satisfied literal row.
  rule: "Upload Rules > Literal"
  severity: warn
  assert: [{left: 1, op: eq, right: 1}]
"""


def _cohort(tmp_path: Path, one: str = CHECK_ONE, two: str | None = None) -> Path:
    """A cohort directory with `one` in `point.yaml` and `two` in `system.yaml`."""
    scopes = {"point": one}
    if two is not None:
        scopes["system"] = two
    return write_cohort(tmp_path / "cohort", **scopes)


class TestDecodeRow:
    """`_decode_row` selects the row model by what the YAML row holds."""

    @pytest.mark.unit
    def test_aggregate_row_by_aggregate_key(self):
        row = _decode_row({"over": [], "aggregate": "count", "op": "eq", "value": 0})
        assert isinstance(row, Aggregate)

    @pytest.mark.unit
    def test_compare_row_by_default(self):
        row = _decode_row({"left": 1, "op": "eq", "right": 1})
        assert type(row) is Compare

    @pytest.mark.unit
    def test_a_scalar_row_is_a_shape_error(self):
        with pytest.raises(ValueError, match="expected a `Compare` or `Aggregate` row"):
            _decode_row("not-a-row")

    @pytest.mark.unit
    def test_typed_rows_pass_through(self):
        compare = Compare.model_validate({"left": 1, "op": "eq", "right": 1})
        aggregate = Aggregate(over=[1], name="count", op="eq", value=1)
        assert _decode_row(compare) is compare
        assert _decode_row(aggregate) is aggregate

    @pytest.mark.unit
    @pytest.mark.parametrize("row", [{2: 3}, {"left": 1}])
    def test_construction_failures_are_value_errors_naming_the_row(self, row):
        with pytest.raises(ValueError, match="bad row"):
            _decode_row(row)

    @pytest.mark.unit
    def test_broken_row_yaml_names_the_scope_file(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "x:\n  description: d\n  rule: r\n  severity: warn\n"
                "  when:\n    - 2: 3\n  assert:\n    - left: 1\n      op: eq\n"
                "      right: 1\n"
            ),
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

    @pytest.mark.unit
    def test_metadata_binds(self):
        check = Check.model_validate(self._yaml_dict())
        assert check.id == "check_id"
        assert check.severity is Severity.FAIL
        assert check.scope is RuleScopes.POINT
        assert [type(row) for row in check.assert_] == [Compare]

    @pytest.mark.unit
    def test_when_rows_decode_too(self):
        check = Check.model_validate(
            self._yaml_dict()
            | {"when": [{"over": [], "aggregate": "count", "op": "eq", "value": 0}]}
        )
        assert isinstance(check.when[0], Aggregate)

    @pytest.mark.unit
    def test_fix_is_optional_with_absent_grid(self):
        present = Check.model_validate(self._yaml_dict())
        assert present.fix == "How to resolve it."
        absent = Check.model_validate(
            {key: value for key, value in self._yaml_dict().items() if key != "fix"}
        )
        assert absent.fix is None

    @pytest.mark.unit
    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            Check.model_validate(self._yaml_dict() | {"extra_key": 1})

    @pytest.mark.unit
    def test_without_when_or_assert_is_rejected(self):
        bare = {
            key: value for key, value in self._yaml_dict().items() if key != "assert"
        }
        with pytest.raises(ValueError, match="at least one"):
            Check.model_validate(bare)


class TestRowRefValidation:
    """Every row `Ref` must name a fact resolvable at its check's scope."""

    @pytest.mark.unit
    def test_a_misspelled_when_key_is_refused_at_cohort_load(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "typo:\n  description: d\n  rule: r\n  severity: warn\n"
                "  when:\n    - left: point.config.ofline\n      op: eq\n"
                "      right: dedicated\n"
                "  assert: [{left: 1, op: eq, right: 1}]\n"
            ),
        )
        with pytest.raises(
            ValueError, match="names no fact; did you mean 'offline'\\?"
        ):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_an_over_deep_fact_path_is_refused_at_cohort_load(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "deep:\n  description: d\n  rule: r\n  severity: warn\n"
                "  assert:\n    - left: point.system_desc.cooling.detail\n"
                "      op: eq\n      right: 1\n"
            ),
        )
        with pytest.raises(ValueError, match="leaf fact"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_an_artifact_whole_ref_is_accepted_at_cohort_load(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "whole:\n  description: d\n  rule: r\n  severity: warn\n"
                "  assert:\n    - left: point.accuracy_results\n      op: exists\n"
            ),
        )
        assert [check.id for check in Cohort.from_directory(root).checks] == ["whole"]

    @pytest.mark.unit
    def test_an_open_dict_subtree_allows_deeper_names(self, tmp_path):
        # `result_summary.percentiles` is an open document (`dict`): its deeper
        # names resolve, unlike a name below a plain typed leaf.
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "deep_open:\n  description: d\n  rule: r\n  severity: warn\n"
                "  assert:\n    - left: point.result_summary.percentiles.p99\n"
                "      op: exists\n"
            ),
        )
        assert [check.id for check in Cohort.from_directory(root).checks] == [
            "deep_open"
        ]

    @pytest.mark.unit
    def test_checker_root_refs_load_unannotated_since_they_have_no_table_leaf(
        self, tmp_path
    ):
        # `derived`/`regions` rows load through the loader (their names are
        # checker-owned), and their `Ref`s carry no table policy notes.
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "root_rows:\n  description: d\n  rule: r\n  severity: warn\n"
                "  assert:\n"
                "    - left: derived.point_cap\n      op: exists\n"
                "    - left: point.config.concurrency\n      op: in\n"
                "      right: regions.coverage\n"
            ),
        )
        check = Cohort.from_directory(root).checks[0]
        derived, in_range = check.assert_
        assert derived.left.parts == ("derived", "point_cap")
        assert derived.left.required is False
        assert derived.left.default is MISSING
        coverage = in_range.operands()[1]
        assert coverage.parts == ("regions", "coverage")
        assert coverage.required is False
        assert coverage.default is MISSING

    @pytest.mark.unit
    def test_a_check_cannot_name_a_finer_scopes_facts(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            system=(
                "overreach:\n  description: d\n  rule: r\n  severity: warn\n"
                "  assert:\n    - left: point.config.concurrency\n      op: exists\n"
            ),
        )
        with pytest.raises(ValueError, match="cannot name"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_derivation_is_a_pareto_or_point_fact_only(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            system=(
                "misplaced:\n  description: d\n  rule: r\n  severity: warn\n"
                "  assert:\n    - left: derived.point_cap\n      op: exists\n"
            ),
        )
        with pytest.raises(ValueError, match="per-pareto checker namespace"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_an_unknown_derivation_name_is_refused(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "ghost:\n  description: d\n  rule: r\n  severity: warn\n"
                "  assert:\n    - left: derived.point_capp\n      op: exists\n"
            ),
        )
        with pytest.raises(ValueError, match="names no derivation"):
            Cohort.from_directory(root)


class TestCohortRefTable:
    """`valid_refs.yaml`: required beside the scope files, per-cohort by name.

    The table is the cohort's own declaration of what its rows may name and
    how an omitted fact reads (`missing` singular). The pins below show the
    load refusals and the per-cohort updatability that motivates it.
    """

    @pytest.mark.unit
    def test_a_cohort_without_a_ref_table_is_refused(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", refs=None, point=LITERAL_CHECK)
        with pytest.raises(ValueError, match="cohort needs valid_refs.yaml"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    @pytest.mark.parametrize("namespace", ["derived", "regions", "ruleset"])
    def test_checker_owned_namespaces_are_refused_inside_the_ref_table(
        self, tmp_path, namespace
    ):
        root = write_cohort(
            tmp_path / "cohort",
            refs={namespace: {"some": "int"}},
            point=LITERAL_CHECK,
        )
        with pytest.raises(ValueError, match="is not a node scope"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_an_unknown_type_name_is_refused(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            refs={"point": {"config": {"concurrency": "integer"}}},
            point=LITERAL_CHECK,
        )
        with pytest.raises(ValueError, match="is no type name"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_every_declared_type_name_has_a_value_test(self):
        # `VALID_TYPES` and the tests inlined in `_Fact.matches` are declared
        # separately, so this pin keeps the vocabulary and the behavior in step.
        for name in sorted(_Fact.VALID_TYPES - {"dict"}):
            assert isinstance(_Fact(type=name).matches(0), bool)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("int", "512", 512),
            ("int", "512.0", 512),
            ("int", " 12 ", 12),
            ("int", "1e3", 1000),
            ("int", "9007199254740993.0", 9007199254740993),
            ("int", "512.9999999999999999", "512.9999999999999999"),
            ("int", "512.5", "512.5"),
            ("int", "512 kw", "512 kw"),
            ("int", "1,000", "1,000"),
            ("int", "nan", "nan"),
            ("int", "inf", "inf"),
            ("int", "", ""),
            ("int", 512, 512),
            ("float", "12.61", 12.61),
            ("float", "1e3", 1000.0),
            ("float", "inf", "inf"),
            ("str", "512", "512"),
            ("bool", "true", "true"),
            ("list", "512", "512"),
        ],
    )
    def test_a_numeric_string_decodes_only_for_numeric_declarations(
        self, name, value, expected
    ):
        # A writer that stringifies a number must not cost the submission a
        # refusal: the declared type decides, and only exact spellings decode.
        assert _Fact(type=name).coerce(value) == expected

    @pytest.mark.unit
    def test_a_decoded_number_is_what_matches_then_accepts(self):
        fact = _Fact(type="int")
        for text in ("512", "512.0", " 12 ", "1e3"):
            assert fact.matches(fact.coerce(text))

    @pytest.mark.unit
    def test_a_scalar_where_a_block_or_type_name_belongs_is_refused(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", refs={"point": 5}, point=LITERAL_CHECK)
        with pytest.raises(ValueError, match="expected a type name"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    @pytest.mark.parametrize("allowlist", [[], "dedicated", [["a"]]])
    def test_an_allowlist_must_be_a_non_empty_list_of_values(self, tmp_path, allowlist):
        root = write_cohort(
            tmp_path / "cohort",
            refs={
                "point": {
                    "config": {
                        "offline": {
                            "type": "str",
                            "allowlist": allowlist,
                            "default": None,
                        }
                    }
                }
            },
            point=LITERAL_CHECK,
        )
        with pytest.raises(ValueError, match="needs a non-empty list"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_an_allowlist_on_a_dict_subtree_is_refused(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            refs={
                "point": {"config": {"window": {"type": "dict", "allowlist": ["a"]}}}
            },
            point=LITERAL_CHECK,
        )
        with pytest.raises(ValueError, match="needs a leaf type"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_non_null_default_must_live_in_its_allowlist(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            refs={
                "point": {
                    "config": {
                        "mood": {"type": "str", "allowlist": ["a"], "default": "b"}
                    }
                }
            },
            point=LITERAL_CHECK,
        )
        with pytest.raises(ValueError, match="is not in its allowlist"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_null_default_is_the_absent_state_and_exempt_from_its_allowlist(
        self, tmp_path
    ):
        # `default: null` IS the absent-state's own value ("absent or `none`"),
        # so `null` need not be listed.
        root = write_cohort(
            tmp_path / "cohort",
            refs={
                "point": {
                    "config": {
                        "offline": {
                            "type": "str",
                            "allowlist": ["dedicated", "elected", "none"],
                            "default": None,
                        }
                    }
                }
            },
            point=LITERAL_CHECK,
        )
        assert [check.id for check in Cohort.from_directory(root).checks] == [
            "literal_check"
        ]

    @pytest.mark.unit
    def test_required_and_default_contradict(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            refs={
                "point": {
                    "config": {
                        "offline": {
                            "type": "str",
                            "required": True,
                            "default": "none",
                        }
                    }
                }
            },
            point=LITERAL_CHECK,
        )
        with pytest.raises(ValueError, match="`required` and `default` contradict"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_default_must_match_its_declared_type(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            refs={
                "point": {
                    "config": {
                        "warmup": {"duration_s": {"type": "int", "default": "x"}}
                    }
                }
            },
            point=LITERAL_CHECK,
        )
        with pytest.raises(ValueError, match="does not match its type"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_an_allowlist_value_must_match_its_declared_type(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            refs={
                "point": {"config": {"mood": {"type": "int", "allowlist": ["a", 2]}}}
            },
            point=LITERAL_CHECK,
        )
        with pytest.raises(ValueError, match="allowlist values"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_custom_table_that_omits_a_field_refuses_rows_naming_it(self, tmp_path):
        # Per-cohort updatability: dropping `config.offline` retires the name,
        # so a row naming it dies at cohort load instead of silently switching
        # its check off against some submission.
        root = write_cohort(
            tmp_path / "cohort",
            refs={
                "point": {"config": {"concurrency": {"type": "int", "required": True}}}
            },
            point=(
                "retired:\n  description: d\n  rule: r\n  severity: warn\n"
                "  when:\n    - left: point.config.offline\n      op: eq\n"
                "      right: dedicated\n"
                "  assert: [{left: 1, op: eq, right: 1}]\n"
            ),
        )
        with pytest.raises(ValueError, match="names no fact"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_custom_table_that_adds_a_rule_field_accepts_rows_naming_it(
        self, tmp_path
    ):
        root = write_cohort(
            tmp_path / "cohort",
            refs={"point": {"config": {"new_rule_field": "dict"}}},
            point=(
                "fresh:\n  description: d\n  rule: r\n  severity: warn\n"
                "  assert:\n    - left: point.config.new_rule_field\n      op: exists\n"
            ),
        )
        assert [check.id for check in Cohort.from_directory(root).checks] == ["fresh"]


class TestRowAnnotations:
    """`Cohort.from_directory` fills each table leaf's policy notes onto its row."""

    @pytest.mark.unit
    def test_required_and_declared_default_land_on_the_row_operands(self, tmp_path):
        # `STANDARD_REFS` declares `concurrency` required and `offline`
        # `{default: null}`; those notes ride along on the operands the rows
        # resolve (`row.left`, reachable through `operands()`).
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "policy:\n  description: d\n  rule: r\n  severity: warn\n"
                "  when:\n    - left: point.config.offline\n      op: eq\n"
                "      right: dedicated\n"
                "  assert:\n    - left: point.config.concurrency\n      op: ge\n"
                "      right: 0\n"
            ),
        )
        check = Cohort.from_directory(root).checks[0]
        concurrency = check.assert_[0].left
        assert concurrency in check.assert_[0].operands()
        assert concurrency.required is True
        offline = check.when[0].left
        assert offline in check.when[0].operands()
        assert offline.required is False
        assert offline.default is None


class TestStubCohort:
    """The stub checks decode through `Cohort.from_directory` against real rows."""

    @pytest.mark.unit
    def test_stub_cohort_loads_in_full(self):
        cohort = Cohort.from_directory(Path("tests/assets/cohorts/stub"))
        assert [check.id for check in cohort.checks] == [
            "concurrency_in_range",
            "offline_ordering",
        ]
        assert [check.severity for check in cohort.checks] == [Severity.WARN] * 2

    @pytest.mark.unit
    def test_stub_checks_evaluate_against_a_submission(self, tmp_path):
        # End-to-end over a real tree: the range stub holds at every in-band
        # fixed-concurrency point and flags r9013 (above the 10% margin); the
        # dedicated run is exempt (§5.7.2); the offline stub reads the
        # dedicated point's declaration and reports its clause verdict.
        cohort = Cohort.from_directory(Path("tests/assets/cohorts/stub"))
        submission = write_upload(
            tmp_path / "acme",
            {
                ("sys", "model"): [
                    Run(concurrency=32),
                    Run(concurrency=512, offline="dedicated", system_tps=5292.0),
                    Run(concurrency=8192, system_tps=5400.0),
                    Run(concurrency=9013, system_tps=1.0),
                ]
            },
        )
        findings = Finding.collect(cohort, submission).findings
        got = sorted(
            (f.check_id, f.node_directory.name, f.verdict.ok) for f in findings
        )
        assert got == [
            ("concurrency_in_range", "r32", True),
            ("concurrency_in_range", "r8192", True),
            ("concurrency_in_range", "r9013", False),
            ("offline_ordering", "r512", False),
        ]

    @pytest.mark.unit
    def test_the_range_row_skips_a_dedicated_offline_run(self, tmp_path):
        # §5.7.2: a dedicated Offline run is not a fixed-concurrency point (its
        # concurrency is the performance dataset's cardinality), so a
        # conforming 200000 point draws no range finding against a C_max 8192.
        cohort = Cohort.from_directory(Path("tests/assets/cohorts/stub"))
        submission = write_upload(
            tmp_path / "acme",
            {
                ("sys", "model"): [
                    Run(concurrency=16),
                    Run(concurrency=8192, system_tps=5400.0),
                    Run(concurrency=200000, offline="dedicated", system_tps=7200.0),
                ]
            },
        )
        findings = Finding.collect(cohort, submission).findings
        got = sorted(
            (f.check_id, f.node_directory.name, f.verdict.ok) for f in findings
        )
        assert got == [
            ("concurrency_in_range", "r16", True),
            ("concurrency_in_range", "r8192", True),
            ("offline_ordering", "r200000", True),
        ]


class TestCohortFromDirectory:
    """`Cohort.from_directory`: loads the four scope files as id-keyed blocks."""

    @pytest.mark.unit
    def test_checks_keep_their_mapping_order(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            point=(
                "first_check:\n  description: First.\n  rule: R1\n  severity: warn\n"
                "  assert: [{left: 1, op: eq, right: 1}]\n"
                "second_check:\n  description: Second.\n  rule: R2\n  severity: warn\n"
                "  assert: [{left: 2, op: eq, right: 2}]\n"
            ),
        )
        cohort = Cohort.from_directory(root)
        assert [check.id for check in cohort.checks] == ["first_check", "second_check"]

    @pytest.mark.unit
    def test_scope_and_id_come_from_path_and_key(self, tmp_path):
        cohort = Cohort.from_directory(_cohort(tmp_path, two=CHECK_TWO))
        assert {c.id: c.scope.value for c in cohort.checks} == {
            "offline_ordering": "point",
            "another_check": "system",
        }

    @pytest.mark.unit
    def test_duplicate_id_across_scope_files_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate check id"):
            Cohort.from_directory(_cohort(tmp_path, one=DUP_CHECK, two=DUP_CHECK))

    @pytest.mark.unit
    def test_a_check_does_not_declare_its_own_id_or_scope(self, tmp_path):
        root = write_cohort(
            tmp_path / "cohort",
            point=CHECK_ONE.replace(
                "severity: warn\n", "severity: warn\n  id: other\n"
            ),
        )
        with pytest.raises(ValueError, match="defines its own"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_broken_yaml_names_the_scope_file(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point="bad: not-a-mapping\n")
        with pytest.raises(ValueError, match="point.yaml"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_parse_problems_name_the_scope_file_too(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point="first: 1\nfirst: 2\n")
        with pytest.raises(ValueError, match="point.yaml"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_missing_directory_is_a_load_error(self, tmp_path):
        with pytest.raises(
            NotADirectoryError, match="the cohort directory .* does not exist"
        ):
            Cohort.from_directory(tmp_path / "nope")

    @pytest.mark.unit
    def test_a_file_where_the_cohort_belongs_is_a_load_error(self, tmp_path):
        path = tmp_path / "cohort.yaml"
        path.write_text("holds no cohort\n", encoding="utf-8")
        with pytest.raises(NotADirectoryError, match="is not a directory"):
            Cohort.from_directory(path)

    @pytest.mark.unit
    def test_an_empty_cohort_is_refused_not_a_clean_pass(self, tmp_path):
        root = write_cohort(tmp_path / "cohort")
        with pytest.raises(ValueError, match="holds no checks"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_an_empty_scope_file_is_a_load_error(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point="")
        with pytest.raises(ValueError, match="empty scope file"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_stray_file_names_what_it_found(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point=LITERAL_CHECK)
        (root / "almost.yaml.bak").write_text(CHECK_ONE)
        with pytest.raises(ValueError, match="almost.yaml.bak"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_manifest_declares_the_cohorts_ruleset(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point=LITERAL_CHECK)
        (root / "cohort.yaml").write_text("ruleset: mlperf-inference-v6.1\n")
        assert Cohort.from_directory(root).ruleset == "mlperf-inference-v6.1"

    @pytest.mark.unit
    def test_a_cohort_without_a_manifest_has_no_ruleset(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point=LITERAL_CHECK)
        assert Cohort.from_directory(root).ruleset is None

    @pytest.mark.unit
    def test_a_manifest_with_unknown_keys_is_refused(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point=LITERAL_CHECK)
        (root / "cohort.yaml").write_text("ruleset: mlperf-inference-v6.1\npolicy: x\n")
        with pytest.raises(ValueError, match=r"unexpected \['policy'\]"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_manifest_ruleset_must_be_a_non_empty_string(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point=LITERAL_CHECK)
        (root / "cohort.yaml").write_text("ruleset: [a]\n")
        with pytest.raises(ValueError, match="non-empty string"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_manifest_may_bundle_the_cohorts_catalog(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point=LITERAL_CHECK)
        (root / "cohort.yaml").write_text("ruleset_dir: ruleset\n", encoding="utf-8")
        catalog = root / "ruleset"
        catalog.mkdir()
        (catalog / "models.yaml").write_text("m: {vocab_size: 1}\n", encoding="utf-8")
        (catalog / "datasets.yaml").write_text("d: {n: 1}\n", encoding="utf-8")
        cohort = Cohort.from_directory(root)
        assert cohort.ruleset is None
        assert cohort.ruleset_dir == catalog

    @pytest.mark.unit
    def test_a_manifest_cannot_declare_both_rule_sources(self, tmp_path):
        root = write_cohort(tmp_path / "cohort", point=LITERAL_CHECK)
        (root / "cohort.yaml").write_text(
            "ruleset: mlperf-inference-v6.1\nruleset_dir: ruleset\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="alternatives"):
            Cohort.from_directory(root)

    @pytest.mark.unit
    def test_a_manifest_ruleset_dir_must_be_a_directory_inside_the_cohort(
        self, tmp_path
    ):
        root = write_cohort(tmp_path / "cohort", point=LITERAL_CHECK)
        (root / "cohort.yaml").write_text("ruleset_dir: nope\n", encoding="utf-8")
        with pytest.raises(ValueError, match="is not a directory under"):
            Cohort.from_directory(root)
        (root / "cohort.yaml").write_text(
            "ruleset_dir: ../elsewhere\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="relative path inside"):
            Cohort.from_directory(root)
