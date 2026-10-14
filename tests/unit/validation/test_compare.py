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

"""W1+W3 layer tests: construction, resolution, and each row's own verdict.

Each case asserts the values its subject computes: `Ref.from_string` decides
ref-vs-literal, `Ref.resolve` reports (found, value), `Verdict.evaluate` carries
each row's `ok` and `why`. Per-op arithmetic is tabled so the boilerplate is one
shape. Nothing here tests pydantic itself.
"""

from collections.abc import Mapping

import pytest
from inference_endpoint.validation.compare import (
    Aggregate,
    Compare,
    Count,
    Ref,
    Verdict,
)
from inference_endpoint.validation.operations import Eq, Op

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


class TestFromString:
    """`Ref.from_string`: scoped dotted strings become refs, the rest are literals."""

    @pytest.mark.parametrize("value", [("point.result_summary.tps")])
    def test_scoped_strings_are_refs(self, value):
        assert Ref.from_string(value) == Ref(("point", "result_summary", "tps"))

    def test_unproducible_roots_never_register(self):
        # A literal: no cohort-params producer exists, so there is no cohort root.
        assert Ref.from_string("cohort.metric_tolerance") is None

    @pytest.mark.parametrize(
        "value", ["result_summary.tps", "pointish.thing", "", "literal", "point."]
    )
    def test_other_strings_are_literals(self, value):
        assert Ref.from_string(value) is None


class TestOperandForms:
    """Operand decode: root strings become refs, explicit forms beat the ambiguity."""

    def test_root_headed_dotted_string_is_a_ref(self):
        assert isinstance(_compare(op="exists", left="point.tps").left, Ref)

    def test_segment_list_addresses_a_dotted_key(self):
        row = _compare(op="exists", left=["ruleset", "models", "llama3.1-8b-v1"])
        assert row.left == Ref(("ruleset", "models", "llama3.1-8b-v1"))

    @pytest.mark.parametrize("value", ["submisson.system_desc.x", ["points", "tps"]])
    def test_root_typo_raises_instead_of_silently_becoming_a_literal(self, value):
        with pytest.raises(ValueError, match="misspelled"):
            _compare(op="exists", left=value)

    def test_literal_form_forces_a_root_looking_string(self):
        row = _compare(op="eq", left={"literal": "point.tps"}, right="point.tps")
        assert row.left == "point.tps"
        assert isinstance(row.right, Ref)

    @pytest.mark.parametrize("value", ["pointish.thing", "llama3.1-8b-v1", "literal"])
    def test_far_literals_stay_literals(self, value):
        assert _compare(op="eq", left=value, right=value).left == value


class TestResolve:
    """`Ref.resolve`: absent and null are different states."""

    def test_found_value(self):
        assert Ref(("point", "result_summary", "tps")).resolve(SCOPES) == (True, 12.5)

    def test_absent_is_not_found(self):
        assert Ref(("point", "result_summary", "nope")).resolve(SCOPES) == (False, None)

    def test_stored_null_is_found(self):
        found, value = Ref(("point", "result_summary", "n_samples_failed")).resolve(
            SCOPES
        )
        assert (found, value) == (True, None)

    def test_object_facts_read_like_mappings(self):
        scopes = {"point": type("P", (), {"tps": 1})()}
        assert Ref(("point", "tps")).resolve(scopes) == (True, 1)

    def test_private_attributes_are_not_facts(self):
        scopes = {"point": type("P", (), {"_x": 1})()}
        assert Ref(("point", "_x")).resolve(scopes) == (False, None)

    def test_the_threshold_may_be_a_ref(self):
        verdict = Verdict.evaluate(
            _aggregate(
                over=[1, 2],
                measure="count",
                op="eq",
                value="point.result_summary.n_samples_issued",
            ),
            SCOPES,
        )
        assert verdict.ok is False
        assert "count of [1, 2] eq point.result_summary.n_samples_issued" in verdict.why


class TestCompareShape:
    """Arity per op is enforced at decode, by field presence."""

    @pytest.mark.parametrize(
        ("kwargs", "left_is_ref"),
        [
            ({"op": "exists", "left": "point.x"}, True),
            ({"op": "eq", "left": 1, "right": 1}, False),
            (
                {
                    "op": "between",
                    "left": "point.result_summary.tps",
                    "lo": 0,
                    "hi": 1,
                },
                True,
            ),
        ],
    )
    def test_valid_rows_decode(self, kwargs, left_is_ref):
        row = _compare(**kwargs)
        assert row.op == kwargs["op"]
        assert isinstance(row.left, Ref) is left_is_ref

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"op": "exists", "left": "point.x", "right": 1},
            {"op": "eq", "left": "point.x"},
            {"op": "between", "left": "point.x", "lo": 0},
        ],
    )
    def test_wrong_arity_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            _compare(**kwargs)

    @pytest.mark.parametrize(
        "right", ["point.result_summary.pattern", ["point", "pattern"], 5]
    )
    def test_matches_needs_a_literal_regex(self, right):
        with pytest.raises(ValueError, match="literal regex"):
            _compare(op="matches", left="r16", right=right)

    def test_matches_rejects_an_uncompilable_pattern_at_load(self):
        with pytest.raises(ValueError, match="does not compile"):
            _compare(op="matches", left="r16", right="[")


class TestCompareVerdict:
    """`Verdict.evaluate` over `Compare`: each op's ok flag and the row-owned why."""

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
                    "op": "between",
                    "left": "point.result_summary.tps",
                    "lo": 0,
                    "hi": 100,
                },
                True,
            ),
            (
                {
                    "op": "between",
                    "left": "point.result_summary.tps",
                    "lo": "point.result_summary.n_samples_failed",
                    "hi": 20,
                },
                False,
            ),
            ({"op": "matches", "left": "r16", "right": "r[0-9]+"}, True),
            ({"op": "matches", "left": 5, "right": "r[0-9]+"}, False),
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

    def test_why_names_the_got_values(self):
        verdict = Verdict.evaluate(
            _compare(op="eq", left="point.result_summary.tps", right=10.0), SCOPES
        )
        assert verdict.ok is False
        assert "eq" in verdict.why and "12.5" in verdict.why and "10.0" in verdict.why

    def test_incommensurable_ordering_pair_fails_with_why(self):
        verdict = Verdict.evaluate(_compare(op="gt", left=5, right=None), SCOPES)
        assert verdict.ok is False
        assert "incommensurable pair" in verdict.why
        assert "5" in verdict.why and "None" in verdict.why
        assert "TypeError" in verdict.why

    def test_incommensurable_aggregate_row_fails_with_why(self):
        verdict = Verdict.evaluate(
            _aggregate(over=[1, "a"], measure="sum", op="eq", value=0), SCOPES
        )
        assert verdict.ok is False
        assert "sum of [1, 'a'] eq 0" in verdict.why
        assert "TypeError" in verdict.why

    def test_describe_labels_follow_the_row_shape(self):
        assert _compare(op="between", left=1, lo=0, hi=2).describe() == "1 between 0..2"
        assert _compare(op="exists", left="point.x").describe() == "point.x exists"
        assert _compare(op="eq", left=1, right=2).describe() == "1 eq 2"

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

    def test_null_right_failure_names_none_not_the_interval_form(self):
        verdict = Verdict.evaluate(_compare(op="eq", left=5, right=None), SCOPES)
        assert verdict.ok is False
        assert "5" in verdict.why and "None" in verdict.why and ".." not in verdict.why

    def test_big_distinct_ints_never_pass_at_zero_tolerance(self):
        huge_left = 2**53 + 1
        huge_right = 2**53 + 2
        verdict = Verdict.evaluate(
            _compare(op="eq", left=huge_left, right=huge_right), SCOPES
        )
        assert verdict.ok is False

    def test_nonzero_tolerance_on_unsupported_op_is_rejected(self):
        with pytest.raises(ValueError, match="tolerance semantics"):
            _compare(op="ge", left=1, right=1, rel_tol=0.02)

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

    def test_missing_reference_fails_a_comparison_and_names_it(self):
        verdict = Verdict.evaluate(
            _compare(op="ge", left="point.result_summary.nope", right=0), SCOPES
        )
        assert verdict.ok is False
        assert "missing" in verdict.why and "nope" in verdict.why

    def test_presence_failure_bounds_a_huge_artifact_value(self):
        scopes = {"point": {"config": {"blob": "x" * 100_000}}}
        verdict = Verdict.evaluate(
            _compare(op="absent", left="point.config.blob"), scopes
        )
        assert verdict.ok is False
        assert len(verdict.why) < 150 and "..." in verdict.why


class TestAggregateVerdict:
    """`Verdict.evaluate` over `Aggregate`: the collection maths and the row-owned why."""

    @pytest.mark.parametrize(
        ("kwargs", "ok"),
        [
            (
                {"over": "point.accuracy", "measure": "count", "op": "ge", "value": 2},
                True,
            ),
            (
                {"over": "point.accuracy", "measure": "count", "op": "ge", "value": 3},
                False,
            ),
            ({"over": [1, 2, 3], "measure": "mean", "op": "eq", "value": 2}, True),
            ({"over": [1, 2, 3], "measure": "min", "op": "le", "value": 1}, True),
            ({"over": [1, 2, 3], "measure": "max", "op": "gt", "value": 2}, True),
            ({"over": [0, 0], "measure": "sum", "op": "eq", "value": 0}, True),
            (
                {"over": [True, False], "measure": "any", "op": "eq", "value": True},
                True,
            ),
            (
                {"over": [True, False], "measure": "all", "op": "eq", "value": True},
                False,
            ),
            ({"over": [], "measure": "count", "op": "eq", "value": 0}, True),
            ({"over": [], "measure": "any", "op": "eq", "value": False}, True),
            ({"over": [], "measure": "mean", "op": "eq", "value": 0}, False),
            ({"over": [], "measure": "all", "op": "eq", "value": True}, False),
            (
                {"over": {"a": 1, "b": 3}, "measure": "mean", "op": "eq", "value": 2},
                True,
            ),
            (
                {
                    "over": "point.result_summary.tps",
                    "measure": "count",
                    "op": "ge",
                    "value": 0,
                },
                False,
            ),
        ],
    )
    def test_ok_flag(self, kwargs, ok):
        assert Verdict.evaluate(_aggregate(**kwargs), SCOPES).ok is ok

    def test_constructs_its_measure_subclass(self):
        row = _aggregate(over=[1], measure="count", op="eq", value=1)
        assert type(row).__name__ == "Count"

    def test_unknown_measure_raises(self):
        with pytest.raises(ValueError, match="unknown measure"):
            _aggregate(over=[1], measure="bogus", op="eq", value=1)

    def test_missing_measure_is_a_value_error(self):
        with pytest.raises(ValueError, match="needs a"):
            Aggregate(over=[1], op="eq", value=1)

    def test_a_row_must_run_the_measure_it_declares(self):
        with pytest.raises(ValueError, match="declares measure"):
            Count(measure="sum", over=[1, 2], op="eq", value=2)

    def test_aggregate_tolerances_are_guarded_like_compare(self):
        with pytest.raises(ValueError, match="tolerance semantics"):
            _aggregate(over=[], measure="count", op="ge", value=1, abs_tol=2)

    @pytest.mark.parametrize(
        ("measure", "items", "expected"),
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
    def test_run_over_items(self, measure, items, expected):
        row = _aggregate(over=[], measure=measure, op="eq", value=expected)
        assert row.run(items) == expected

    def test_why_names_the_aggregated_value(self):
        verdict = Verdict.evaluate(
            _aggregate(over="point.accuracy", measure="count", op="ge", value=3), SCOPES
        )
        assert verdict.ok is False
        assert "count" in verdict.why and "2" in verdict.why

    def test_empty_mean_says_no_values_in_why(self):
        verdict = Verdict.evaluate(
            _aggregate(over=[], measure="mean", op="eq", value=0), SCOPES
        )
        assert "no values" in verdict.why

    def test_missing_collection_fails_and_names_it(self):
        verdict = Verdict.evaluate(
            _aggregate(over="point.results", measure="count", op="ge", value=0), SCOPES
        )
        assert verdict.ok is False
        assert "missing" in verdict.why and "results" in verdict.why

    def test_non_collection_fails_with_why(self):
        verdict = Verdict.evaluate(
            _aggregate(
                over="point.result_summary.tps", measure="count", op="ge", value=0
            ),
            SCOPES,
        )
        assert "collection" in verdict.why


class TestOpConstruction:
    """`Op(name)` builds the registered op class; an unknown name raises."""

    def test_named_construction(self):
        op = Op("eq")
        assert isinstance(op, Eq)
        assert op.name == "eq"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown op"):
            Op("bogus")


class TestEvaluateAll:
    """`Verdict.evaluate_all` keeps row order across row types."""

    def test_mixed_rows_in_order(self):
        rows = [
            _compare(op="eq", left=1, right=1),
            _aggregate(over=[1], measure="count", op="eq", value=1),
            _compare(op="eq", left=1, right=2),
        ]
        assert [v.ok for v in Verdict.evaluate_all(rows, SCOPES)] == [True, True, False]
