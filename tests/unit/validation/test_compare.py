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

"""Compare-row tests: `Ref` resolution, operand decode, and `Verdict.evaluate`.

Each case asserts the values its subject computes: `Ref.from_string` decides
ref-vs-literal, `Ref.resolve` reports (found, value), `Verdict.evaluate` carries
each row's `ok` and `why`. Per-op arithmetic is tabled so the boilerplate is one
shape. Nothing here tests pydantic itself.
"""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from inference_endpoint.validation.compare import (
    Aggregate,
    Compare,
    Count,
    Verdict,
)
from inference_endpoint.validation.operations import (
    Eq,
    Matches,
    Op,
    Predicate,
    SameMultiset,
)
from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.refs import Ref, RefTable
from submission_helpers import STANDARD_REFS

# One `point` scope: present values, one stored null, one absent key.
SCOPES = {
    "point": {
        "result_summary": {
            "n_samples_issued": 100,
            "n_samples_completed": 100,
            "n_samples_failed": None,
            "tps": 12.5,
        },
        "accuracy": {"aime": {"score": 0.5}, "gpqa": {"score": None}},
    }
}


def _compare(**kwargs) -> Compare:
    return Compare.model_validate(kwargs)


def _aggregate(**kwargs) -> Aggregate:
    return Aggregate(**kwargs)


@pytest.fixture
def declared(tmp_path: Path) -> RefTable:
    """The fixture table every absent-state pin declares its policy against.

    Written from `STANDARD_REFS`: `offline` states §8.3's absent-state value
    (`{ type: str, default: null }`), `system_tps` is `required`, and
    `tpot_p90_ms` a plain optional (typed leaf, no annotations).
    """
    path = tmp_path / "valid_refs.yaml"
    path.write_text(STANDARD_REFS, encoding="utf-8")
    return RefTable.from_yaml(path)


def _annotated(row: Compare | Aggregate, table: RefTable) -> Compare | Aggregate:
    """`row` as a cohort load carries it: each `Ref` operand filled from `table`."""
    row.map_refs(table.annotate)
    return row


class TestFromString:
    """`Ref.from_string`: scoped dotted strings become refs, the rest are literals."""

    @pytest.mark.unit
    @pytest.mark.parametrize("value", [("point.result_summary.tps")])
    def test_scoped_strings_are_refs(self, value):
        assert Ref.from_string(value) == Ref(("point", "result_summary", "tps"))

    @pytest.mark.unit
    def test_unproducible_namespaces_never_register(self):
        # A literal: no cohort-params producer exists, so `cohort` is no namespace.
        assert Ref.from_string("cohort.metric_tolerance") is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "value", ["result_summary.tps", "pointish.thing", "", "literal", "point."]
    )
    def test_other_strings_are_literals(self, value):
        assert Ref.from_string(value) is None

    @pytest.mark.unit
    def test_the_pareto_scope_root_opens_a_reference(self):
        assert Ref.from_string("pareto.system_desc.c_min_override") == Ref(
            ("pareto", "system_desc", "c_min_override")
        )


class TestOperandForms:
    """Operand decode: root strings become refs, explicit forms beat the ambiguity."""

    @pytest.mark.unit
    def test_root_headed_dotted_string_is_a_ref(self):
        assert isinstance(_compare(op="exists", left="point.tps").left, Ref)

    @pytest.mark.unit
    def test_segment_list_addresses_a_dotted_key(self):
        row = _compare(op="exists", left=["ruleset", "models", "llama3.1-8b-v1"])
        assert row.left == Ref(("ruleset", "models", "llama3.1-8b-v1"))

    @pytest.mark.unit
    @pytest.mark.parametrize("value", ["submisson.system_desc.x", ["points", "tps"]])
    def test_root_typo_raises_instead_of_silently_becoming_a_literal(self, value):
        with pytest.raises(ValueError, match="misspelled"):
            _compare(op="exists", left=value)

    @pytest.mark.unit
    def test_literal_form_forces_a_namespace_looking_string(self):
        row = _compare(op="eq", left={"literal": "point.tps"}, right="point.tps")
        assert row.left == "point.tps"
        assert isinstance(row.right, Ref)

    @pytest.mark.unit
    @pytest.mark.parametrize("value", ["pointish.thing", "llama3.1-8b-v1", "blah.foo"])
    def test_far_dotted_strings_refuse_to_guess(self, value):
        with pytest.raises(ValueError, match="no ref namespace"):
            _compare(op="eq", left=value, right=value)

    @pytest.mark.unit
    @pytest.mark.parametrize("value", ["literal", "dedicated", "blah"])
    def test_scalars_without_dots_stay_literals(self, value):
        assert _compare(op="eq", left=value, right=value).left == value

    @pytest.mark.unit
    def test_far_dotted_values_need_the_literal_form(self):
        row = _compare(op="eq", left="point.dataset", right={"literal": "squad.v2.0"})
        assert row.right == "squad.v2.0"

    @pytest.mark.unit
    @pytest.mark.parametrize("op", ["exists", "absent", "not_null"])
    def test_presence_ops_need_a_reference_left(self, op):
        with pytest.raises(ValueError, match="presence"):
            _compare(op=op, left={"literal": "point.tps"})
        with pytest.raises(ValueError, match="presence"):
            _compare(op=op, left="tps")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "right",
        [
            "[0-9]+",
            r"\d+\.\d+",
            "system\\.model.*",
            ["point", "pattern"],
            5,
            True,
            [],
            {"literal": 5},
        ],
    )
    def test_matches_pattern_requires_the_explicit_literal_form(self, right):
        with pytest.raises(ValueError, match="must use the"):
            _compare(op="matches", left="point.dataset", right=right)

    @pytest.mark.unit
    def test_matches_pattern_decodes_from_the_explicit_literal_form(self):
        row = _compare(
            op="matches", left="point.dataset", right={"literal": r"\d+\.\d+"}
        )
        assert row.right == r"\d+\.\d+"


class TestMapRefs:
    """`map_refs` rewrites the `Ref` operands only. A literal operand is data."""

    @pytest.mark.unit
    def test_map_refs_rewrites_ref_operands_and_leaves_a_literal_alone(self):
        row = _compare(op="eq", left="point.config.concurrency", right=32)
        seen: list[Ref] = []

        def note(ref: Ref) -> Ref:
            seen.append(ref)
            return replace(ref, required=True)

        row.map_refs(note)
        assert seen == [Ref(("point", "config", "concurrency"))]
        assert row.left.required is True
        assert row.right == 32
        assert isinstance(row.right, int) and not isinstance(row.right, bool)
        assert row.operands() == (row.left, 32)


class TestResolve:
    """`Ref.resolve`: absent and null are different states. Resolution is mapping-only."""

    @pytest.mark.unit
    def test_found_value(self):
        assert Ref(("point", "result_summary", "tps")).resolve(SCOPES) == (True, 12.5)

    @pytest.mark.unit
    def test_absent_is_not_found(self):
        assert Ref(("point", "result_summary", "nope")).resolve(SCOPES) == (False, None)

    @pytest.mark.unit
    def test_stored_null_is_found(self):
        found, value = Ref(("point", "result_summary", "n_samples_failed")).resolve(
            SCOPES
        )
        assert (found, value) == (True, None)

    @pytest.mark.unit
    def test_a_lists_builtin_count_never_forges_a_present_fact(self):
        # `"errors": []` resolves `errors.count` ABSENT: the builtin method must
        # never answer as a present fact (a resolved leaf is returned as-is).
        scopes = {"point": {"result_summary": {"errors": []}}}
        assert Ref(("point", "result_summary", "errors", "count")).resolve(scopes) == (
            False,
            None,
        )

    @pytest.mark.unit
    def test_a_resolved_leaf_is_a_value_however_it_is_shaped(self):
        scopes = {
            "point": {"result_summary": {"errors": {"count": None}, "band": (1, None)}}
        }
        assert Ref(("point", "result_summary", "errors", "count")).resolve(scopes) == (
            True,
            None,
        )

    @pytest.mark.unit
    def test_the_threshold_may_be_a_ref(self):
        verdict = Verdict.evaluate(
            _aggregate(
                over=[1, 2],
                name="count",
                op="eq",
                value="point.result_summary.n_samples_issued",
            ),
            SCOPES,
        )
        assert verdict.ok is False
        assert "count of [1, 2] eq point.result_summary.n_samples_issued" in verdict.why


class TestDeclaredAbsentState:
    """`config.offline` judges its declared absent-state value (§8.3 "absent or `none`").

    Each row below says its policy in the fixture table (`declared`): the three
    absent states -- required (`system_tps`), declared default (`offline`), and
    the plain optional (`tpot_p90_ms`) -- are spelled in `STANDARD_REFS` and
    ride along via `_annotated`.
    """

    @pytest.mark.unit
    def test_exists_sees_only_the_offline_keys_raw_presence(self, declared):
        scopes_without = {"point": {"config": {"concurrency": 16}}}
        scopes_with_null = {"point": {"config": {"concurrency": 16, "offline": None}}}
        scopes_with_none_spelling = {
            "point": {"config": {"concurrency": 16, "offline": "none"}}
        }
        row = _annotated(_compare(op="exists", left="point.config.offline"), declared)
        omitted = Verdict.evaluate(row, scopes_without)
        stored_null = Verdict.evaluate(row, scopes_with_null)
        none_spelling = Verdict.evaluate(row, scopes_with_none_spelling)
        assert omitted.ok is False
        assert stored_null.ok is True
        assert none_spelling.ok is True

    @pytest.mark.unit
    def test_an_omitted_offline_judges_as_null_so_ne_dedicated_holds(self, declared):
        scopes_without = {"point": {"config": {"concurrency": 16}}}
        verdict = Verdict.evaluate(
            _annotated(
                _compare(op="ne", left="point.config.offline", right="dedicated"),
                declared,
            ),
            scopes_without,
        )
        assert verdict.ok is True
        assert verdict.broken is False

    @pytest.mark.unit
    def test_an_assert_row_over_an_absent_required_fact_is_a_broken_verdict(
        self, declared
    ):
        row = _annotated(
            _compare(op="eq", left="point.result_summary.system_tps", right=1),
            declared,
        )
        verdict = Verdict.evaluate(row, {"point": {"result_summary": {}}})
        assert verdict.broken is True
        assert verdict.ok is False
        assert "required fact point.result_summary.system_tps" in verdict.why
        assert "is absent from the upload" in verdict.why

    @pytest.mark.unit
    def test_an_absent_optional_fact_just_fails_the_row(self, declared):
        verdict = Verdict.evaluate(
            _annotated(
                _compare(op="ge", left="point.result_summary.tpot_p90_ms", right=0),
                declared,
            ),
            {"point": {"result_summary": {"tps": 1.0}}},
        )
        assert verdict.broken is False
        assert verdict.ok is False
        assert "missing" in verdict.why


class TestCompareShape:
    """Arity per op is enforced at decode, by field presence."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("kwargs", "left_is_ref"),
        [
            ({"op": "exists", "left": "point.x"}, True),
            ({"op": "eq", "left": 1, "right": 1}, False),
            (
                {
                    "op": "in",
                    "left": "point.result_summary.tps",
                    "right": {"lo": 0, "hi": 1},
                },
                True,
            ),
        ],
    )
    def test_valid_rows_decode(self, kwargs, left_is_ref):
        row = _compare(**kwargs)
        assert row.op == kwargs["op"]
        assert isinstance(row.left, Ref) is left_is_ref

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"op": "exists", "left": "point.x", "right": 1},
            {"op": "eq", "left": "point.x"},
        ],
    )
    def test_wrong_arity_is_rejected(self, kwargs):
        with pytest.raises(ValueError, match="takes `left`"):
            _compare(**kwargs)

    @pytest.mark.unit
    def test_matches_rejects_an_uncompilable_pattern_at_load(self):
        with pytest.raises(ValueError, match="does not compile"):
            _compare(op="matches", left="r16", right={"literal": "["})


class TestCompareVerdict:
    """`Verdict.evaluate` over `Compare`: each op's ok flag and the row-owned why."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("kwargs", "ok"),
        [
            (
                {
                    "op": "eq",
                    "left": "point.result_summary.n_samples_completed",
                    "right": "point.result_summary.n_samples_issued",
                },
                True,
            ),
            (
                {
                    "op": "eq",
                    "left": "point.result_summary.n_samples_completed",
                    "right": 101,
                },
                False,
            ),
            ({"op": "ge", "left": "point.result_summary.tps", "right": 10.0}, True),
            ({"op": "lt", "left": "point.result_summary.tps", "right": 10.0}, False),
            ({"op": "in", "left": "a", "right": ["a", "b"]}, True),
            ({"op": "in", "left": "k", "right": {"k": 1}}, True),
            (
                {
                    "op": "in",
                    "left": "point.result_summary.tps",
                    "right": {"lo": 0, "hi": 100},
                },
                True,
            ),
            (
                {
                    "op": "in",
                    "left": "point.result_summary.tps",
                    "right": {"lo": 0, "hi": 10},
                },
                False,
            ),
            (
                {
                    "op": "matches",
                    "left": "r16",
                    "right": {"literal": "r[0-9]+"},
                },
                True,
            ),
            (
                {
                    "op": "matches",
                    "left": 5,
                    "right": {"literal": "r[0-9]+"},
                },
                False,
            ),
            (
                {
                    "op": "eq",
                    "left": "point.result_summary.tps",
                    "right": 12.5000001,
                    "abs_tol": 1e-3,
                },
                True,
            ),
            (
                {
                    "op": "eq",
                    "left": "point.result_summary.n_samples_failed",
                    "right": None,
                },
                True,
            ),
            ({"op": "same_multiset", "left": [1, 2, 2], "right": [2, 1, 2]}, True),
            ({"op": "same_multiset", "left": [1], "right": [1, 1]}, False),
            ({"op": "same_multiset", "left": [[1], [2]], "right": [[2], [1]]}, True),
        ],
    )
    def test_ok_flag(self, kwargs, ok):
        assert Verdict.evaluate(_compare(**kwargs), SCOPES).ok is ok

    @pytest.mark.unit
    def test_why_names_the_got_values(self):
        verdict = Verdict.evaluate(
            _compare(op="eq", left="point.result_summary.tps", right=10.0), SCOPES
        )
        assert verdict.ok is False
        assert "eq" in verdict.why and "12.5" in verdict.why and "10.0" in verdict.why

    @pytest.mark.unit
    def test_incommensurable_ordering_pair_fails_with_why(self):
        verdict = Verdict.evaluate(_compare(op="gt", left=5, right=None), SCOPES)
        assert verdict.ok is False
        assert "incommensurable pair" in verdict.why
        assert "5" in verdict.why and "None" in verdict.why
        assert "TypeError" in verdict.why

    @pytest.mark.unit
    def test_incommensurable_aggregate_row_fails_with_why(self):
        verdict = Verdict.evaluate(
            _aggregate(over=[1, "a"], name="sum", op="eq", value=0), SCOPES
        )
        assert verdict.ok is False
        assert "sum of [1, 'a'] eq 0" in verdict.why
        assert "TypeError" in verdict.why

    @pytest.mark.unit
    def test_describe_labels_follow_the_row_shape(self):
        assert (
            _compare(op="in", left=1, right={"lo": 0, "hi": 2}).describe()
            == "1 in 0..2"
        )
        assert _compare(op="exists", left="point.x").describe() == "point.x exists"
        assert _compare(op="eq", left=1, right=2).describe() == "1 eq 2"

    @pytest.mark.unit
    def test_a_fail_loud_is_reported_as_itself_not_a_type_mismatch(self):
        class LoudDerived(Mapping):
            def __contains__(self, name):
                return True

            def __getitem__(self, name):
                raise ValueError("derived input missing")

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

        verdict = Verdict.evaluate(
            _compare(op="exists", left="derived.x"), {"derived": LoudDerived()}
        )
        assert verdict.ok is False
        assert "row failed to evaluate" in verdict.why
        assert "derived input missing" in verdict.why
        assert "incommensurable pair" not in verdict.why

    @pytest.mark.unit
    def test_null_right_failure_names_none_not_the_interval_form(self):
        verdict = Verdict.evaluate(_compare(op="eq", left=5, right=None), SCOPES)
        assert verdict.ok is False
        assert "5" in verdict.why and "None" in verdict.why and ".." not in verdict.why

    @pytest.mark.unit
    def test_big_distinct_ints_never_pass_at_zero_tolerance(self):
        huge_left = 2**53 + 1
        huge_right = 2**53 + 2
        verdict = Verdict.evaluate(
            _compare(op="eq", left=huge_left, right=huge_right), SCOPES
        )
        assert verdict.ok is False

    @pytest.mark.unit
    def test_nonzero_tolerance_on_unsupported_op_is_rejected(self):
        with pytest.raises(ValueError, match="tolerance semantics"):
            _compare(op="ge", left=1, right=1, rel_tol=0.02)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("kwargs", "ok"),
        [
            ({"op": "ne", "left": "point.result_summary.tps", "right": 12.5}, False),
            ({"op": "ne", "left": "point.result_summary.tps", "right": 10.0}, True),
            (
                {
                    "op": "ne",
                    "left": "point.result_summary.tps",
                    "right": 12.5001,
                    "rel_tol": 1e-3,
                },
                False,
            ),
            (
                {
                    "op": "ne",
                    "left": "point.result_summary.tps",
                    "right": 12.5001,
                    "abs_tol": 1e-6,
                },
                True,
            ),
            (
                {
                    "op": "eq",
                    "left": "point.result_summary.tps",
                    "right": 12.5001,
                    "rel_tol": 1e-3,
                },
                True,
            ),
            ({"op": "eq", "left": 100, "right": 101, "abs_tol": 1}, True),
            ({"op": "ne", "left": 100, "right": 101, "abs_tol": 1}, False),
            (
                {"op": "eq", "left": 2**53 + 1, "right": 2**53 + 3, "abs_tol": 1},
                False,
            ),
        ],
    )
    def test_ne_and_rel_tolerance_acceptance(self, kwargs, ok):
        assert Verdict.evaluate(_compare(**kwargs), SCOPES).ok is ok

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("op", "ok"),
        [("exists", False), ("absent", True), ("not_null", False)],
    )
    def test_missing_state_grid_for_presence_ops(self, op, ok):
        """A missing reference: exists false, absent true, not_null false."""
        assert (
            Verdict.evaluate(
                _compare(op=op, left="point.result_summary.nope"), SCOPES
            ).ok
            is ok
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("op", "ok"),
        [("exists", True), ("absent", False), ("not_null", False)],
    )
    def test_null_state_grid_for_presence_ops(self, op, ok):
        assert (
            Verdict.evaluate(
                _compare(op=op, left="point.result_summary.n_samples_failed"), SCOPES
            ).ok
            is ok
        )

    @pytest.mark.unit
    def test_missing_reference_fails_a_comparison_and_names_it(self):
        verdict = Verdict.evaluate(
            _compare(op="ge", left="point.result_summary.nope", right=0), SCOPES
        )
        assert verdict.ok is False
        assert "missing" in verdict.why and "nope" in verdict.why

    @pytest.mark.unit
    def test_presence_failure_bounds_a_huge_artifact_value(self):
        scopes = {"point": {"config": {"blob": "x" * 100_000}}}
        verdict = Verdict.evaluate(
            _compare(op="absent", left="point.config.blob"), scopes
        )
        assert verdict.ok is False
        assert len(verdict.why) < 150 and "..." in verdict.why


class TestAggregateVerdict:
    """`Verdict.evaluate` over `Aggregate`: the collection maths and the row-owned why."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("kwargs", "ok"),
        [
            (
                {
                    "over": "point.accuracy",
                    "aggregate": "count",
                    "op": "ge",
                    "value": 2,
                },
                True,
            ),
            (
                {
                    "over": "point.accuracy",
                    "aggregate": "count",
                    "op": "ge",
                    "value": 3,
                },
                False,
            ),
            ({"over": [1, 2, 3], "aggregate": "mean", "op": "eq", "value": 2}, True),
            ({"over": [1, 2, 3], "aggregate": "min", "op": "le", "value": 1}, True),
            ({"over": [1, 2, 3], "aggregate": "max", "op": "gt", "value": 2}, True),
            ({"over": [0, 0], "aggregate": "sum", "op": "eq", "value": 0}, True),
            (
                {"over": [True, False], "aggregate": "any", "op": "eq", "value": True},
                True,
            ),
            (
                {"over": [True, False], "aggregate": "all", "op": "eq", "value": True},
                False,
            ),
            ({"over": [], "aggregate": "count", "op": "eq", "value": 0}, True),
            ({"over": [], "aggregate": "any", "op": "eq", "value": False}, True),
            ({"over": [], "aggregate": "mean", "op": "eq", "value": 0}, False),
            ({"over": [], "aggregate": "all", "op": "eq", "value": True}, False),
            (
                {"over": {"a": 1, "b": 3}, "aggregate": "mean", "op": "eq", "value": 2},
                True,
            ),
            (
                {
                    "over": "point.result_summary.tps",
                    "aggregate": "count",
                    "op": "ge",
                    "value": 0,
                },
                False,
            ),
        ],
    )
    def test_ok_flag(self, kwargs, ok):
        assert Verdict.evaluate(_aggregate(**kwargs), SCOPES).ok is ok

    @pytest.mark.unit
    def test_constructs_its_aggregate_subclass(self):
        row = _aggregate(over=[1], name="count", op="eq", value=1)
        assert type(row).__name__ == "Count"

    @pytest.mark.unit
    def test_unknown_aggregate_raises(self):
        with pytest.raises(ValueError, match="unknown aggregate"):
            _aggregate(over=[1], name="bogus", op="eq", value=1)

    @pytest.mark.unit
    def test_missing_aggregate_is_a_value_error(self):
        with pytest.raises(ValueError, match="needs a"):
            Aggregate(over=[1], op="eq", value=1)

    @pytest.mark.unit
    def test_a_row_must_run_the_aggregate_it_declares(self):
        with pytest.raises(ValueError, match="declares aggregate"):
            Count(name="sum", over=[1, 2], op="eq", value=2)

    @pytest.mark.unit
    def test_aggregate_tolerances_are_guarded_like_compare(self):
        with pytest.raises(ValueError, match="tolerance semantics"):
            _aggregate(over=[], name="count", op="ge", value=1, abs_tol=2)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "items", "expected"),
        [
            ("count", [1, 2], 2),
            ("count", [], 0),
            ("sum", [1, 2, 3], 6),
            ("mean", [1, 3], 2.0),
            ("min", [2, 1, 3], 1),
            ("max", [2, 1, 3], 3),
            ("any", [0, 5], True),
            ("all", [0, 5], False),
        ],
    )
    def test_run_over_items(self, name, items, expected):
        row = _aggregate(over=[], name=name, op="eq", value=expected)
        assert row.run(items) == expected

    @pytest.mark.unit
    def test_why_names_the_aggregated_value(self):
        verdict = Verdict.evaluate(
            _aggregate(over="point.accuracy", name="count", op="ge", value=3),
            SCOPES,
        )
        assert verdict.ok is False
        assert "count" in verdict.why and "2" in verdict.why

    @pytest.mark.unit
    def test_empty_mean_says_no_values_in_why(self):
        verdict = Verdict.evaluate(
            _aggregate(over=[], name="mean", op="eq", value=0), SCOPES
        )
        assert "no values" in verdict.why

    @pytest.mark.unit
    def test_missing_collection_fails_and_names_it(self):
        verdict = Verdict.evaluate(
            _aggregate(over="point.results", name="count", op="ge", value=0), SCOPES
        )
        assert verdict.ok is False
        assert "missing" in verdict.why and "results" in verdict.why

    @pytest.mark.unit
    def test_non_collection_fails_with_why(self):
        verdict = Verdict.evaluate(
            _aggregate(over="point.result_summary.tps", name="count", op="ge", value=0),
            SCOPES,
        )
        assert "collection" in verdict.why


class TestOpConstruction:
    """`Op(name)` builds the registered op class. An unknown name raises."""

    @pytest.mark.unit
    def test_named_construction(self):
        op = Op("eq")
        assert isinstance(op, Eq)
        assert op.name == "eq"

    @pytest.mark.unit
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown op"):
            Op("bogus")


class TestEvaluateAll:
    """`Verdict.evaluate_all` keeps row order across row types."""

    @pytest.mark.unit
    def test_mixed_rows_in_order(self):
        rows = [
            _compare(op="eq", left=1, right=1),
            _aggregate(over=[1], name="count", op="eq", value=1),
            _compare(op="eq", left=1, right=2),
        ]
        assert [v.ok for v in Verdict.evaluate_all(rows, SCOPES)] == [True, True, False]


class TestHostileValues:
    """Crafted values refuse loudly instead of comparing quietly."""

    @pytest.mark.unit
    def test_matches_subject_over_the_cap_refuses(self):
        with pytest.raises(ValueError, match="`matches` subject"):
            Matches().holds(Predicate("x" * (64 * 1024 + 1), "x+"))

    @pytest.mark.unit
    def test_same_multiset_pair_over_the_cap_refuses(self):
        # Past the cap the pair refuses before either pass: the hashable
        # path's `Counter` work is bounded, not just the `==` scan.
        pair = [[1]] * 1025
        with pytest.raises(ValueError, match="caps at"):
            SameMultiset().holds(Predicate(pair, pair))
        with pytest.raises(ValueError, match="caps at"):
            SameMultiset().holds(Predicate(list(range(1025)), list(range(1025))))

    @pytest.mark.unit
    def test_a_non_finite_operand_is_broken_not_quietly_false(self):
        verdict = Verdict.evaluate(
            _compare(op="gt", left="point.tps", right=1),
            {"point": {"tps": float("nan")}},
        )
        assert verdict.broken
        assert "non-finite" in verdict.why

    @pytest.mark.unit
    def test_in_range_with_a_null_bound_is_broken_not_unbounded(self):
        verdict = Verdict.evaluate(
            _compare(op="in", left=1, right={"lo": 1, "hi": None}), SCOPES
        )
        assert verdict.broken
        assert "bounds" in verdict.why

    @pytest.mark.unit
    def test_in_over_an_interval_value_tests_membership(self):
        inside = _compare(op="in", left=5, right=Interval(0, 10))
        assert Verdict.evaluate(inside, SCOPES).ok is True
        outside = _compare(op="in", left=5, right=Interval(6))
        assert Verdict.evaluate(outside, SCOPES).ok is False

    @pytest.mark.unit
    def test_in_with_a_lo_hi_literal_is_a_range_not_a_key_set(self):
        # Exactly `{lo, hi}` is a range; any other mapping keeps key membership.
        literal = _compare(op="in", left="lo", right={"lo": 1, "hi": 2})
        assert Verdict.evaluate(literal, SCOPES).ok is False
        keys = _compare(op="in", left="lo", right={"lo": 1, "hi": 2, "x": 3})
        assert Verdict.evaluate(keys, SCOPES).ok is True

    @pytest.mark.unit
    def test_in_against_a_string_target_is_broken(self):
        verdict = Verdict.evaluate(
            _compare(op="in", left="point.name", right="abc"),
            {"point": {"name": "b"}},
        )
        assert verdict.broken
        assert "needs a list" in verdict.why
