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

"""Runner tests: `Finding.collect` over one walked upload.

A `Finding` is the outcome record of every *evaluated* `assert` row at its
applicable node (holds and failures alike -- the audit trail, formatted by a
renderer that joins `fix` tips where the catalog declares one). `needs`
gating suppresses only that check's rows.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from inference_endpoint.validation.check import Check, Cohort
from inference_endpoint.validation.runner import CheckRun, Finding
from inference_endpoint.validation.sentinels import MISSING
from inference_endpoint.validation.submission import Pareto, Point, Submission
from inference_endpoint.validation.tree import SubmissionNode
from submission_helpers import STANDARD_REFS, Run, write_cohort, write_run, write_upload

SYSTEM = ("sys", "model")


def _submission(
    tmp_path: Path,
    concurrencies: list[int],
    extras: Mapping[int, Mapping[str, Any]] | None = None,
) -> Submission:
    """An upload over `results/sys/model/r<concurrency>` subtrees.

    Args:
        tmp_path (Path): The root the upload is written under.
        concurrencies (list[int]): The concurrency of each written run.
        extras (Mapping[int, Mapping[str, Any]] | None): The `extras` mapping
            adds named optional facts to a run's `point.yaml` (e.g. `nodes_used`), so
            `when` rows can gate per point. (Default: None)
    """
    runs = [
        Run(concurrency=concurrency, config_extra=(extras or {}).get(concurrency, {}))
        for concurrency in concurrencies
    ]
    return write_upload(tmp_path / "acme", {SYSTEM: runs})


def _check(id: str, **kwargs) -> Check:
    base = {
        "id": id,
        "description": "What is observed.",
        "rule": "Where in the Rules this comes from.",
        "severity": "fail",
        "scope": "point",
    }
    return Check.model_validate(base | kwargs)


LITERAL_ROW = {"left": 1, "op": "eq", "right": 1}
NODES_USED_ONE = {"left": "point.config.nodes_used", "op": "eq", "right": 1}

OMITTED_OFFLINE_GATE = """
dedicated_only:
  description: A rule for the dedicated Offline run only.
  rule: "Upload Rules > Dedicated"
  severity: fail
  when:
    - left: point.config.offline
      op: eq
      right: dedicated
  assert: [{left: 0, op: eq, right: 1}]
"""

NE_DEDICATED = """
not_dedicated:
  description: A rule for every point not declared dedicated.
  rule: "Upload Rules > NotDedicated"
  severity: fail
  when:
    - left: point.config.offline
      op: ne
      right: dedicated
  assert: [{left: 1, op: eq, right: 1}]
"""

REQUIRED_WHEN = """
needs_system_desc:
  description: What is observed.
  rule: Where in the Rules this comes from.
  severity: fail
  when:
    - left: pareto.system_desc.max_supported_concurrency
      op: eq
      right: 8192
  assert: [{left: 1, op: eq, right: 1}]
"""

REQUIRED_ASSERT = """
needs_system_desc:
  description: What is observed.
  rule: Where in the Rules this comes from.
  severity: fail
  assert:
    - left: pareto.system_desc.max_supported_concurrency
      op: eq
      right: 8192
"""

PLAIN_CHECK = """
plain_check:
  description: One satisfied literal row.
  rule: "Upload Rules > Plain"
  severity: fail
  assert: [{left: 1, op: eq, right: 1}]
"""


def _declared_cohort(
    tmp_path: Path,
    *,
    refs: str | Mapping[str, Any] | None = STANDARD_REFS,
    **scopes: str,
) -> Cohort:
    """A loaded cohort: its rows carry the fixture table's policy notes."""
    return Cohort.from_directory(write_cohort(tmp_path / "cohort", refs=refs, **scopes))


class TestFindingsPerEvaluatedRow:
    """`Finding.collect` emits one `Finding` per `assert` row at each applicable node (holds too)."""

    @pytest.mark.unit
    def test_one_finding_per_assert_row_per_point(self, tmp_path):
        check = _check("cap", assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((check,)), _submission(tmp_path, [16, 8192])
        ).findings
        assert len(findings) == 2
        assert all(isinstance(f, Finding) for f in findings)
        assert [f.check_id for f in findings] == ["cap", "cap"]
        assert all(f.verdict.ok for f in findings)

    @pytest.mark.unit
    def test_failing_verdict_records_why(self, tmp_path):
        check = _check("failing", assert_=[{"left": 0, "op": "eq", "right": 1}])
        findings = Finding.collect(
            Cohort((check,)), _submission(tmp_path, [16])
        ).findings
        assert len(findings) == 1
        assert findings[0].verdict.ok is False
        assert findings[0].verdict.why == "expected 0 eq 1, got 0 vs 1"

    @pytest.mark.unit
    def test_when_gate_suppresses(self, tmp_path):
        check = _check(
            "gated",
            when=[{"left": 0, "op": "eq", "right": 1}],
            assert_=[LITERAL_ROW],
        )
        findings = Finding.collect(
            Cohort((check,)), _submission(tmp_path, [16])
        ).findings
        assert findings == ()


class TestWhenGateAbsentStates:
    """A `when` row over an absent fact: break loud, judge the default, or gate off.

    The policy says so in the fixture table (`STANDARD_REFS` via `write_cohort`):
    `config.offline` declares `{default: null}`,
    `system_desc.max_supported_concurrency` is `required`, and the table-free
    hand-built row carries neither note (the documented hand-built default).
    """

    @pytest.mark.unit
    def test_a_when_row_over_an_omitted_offline_gates_its_check_off(self, tmp_path):
        # `config.offline` absent is its declared absent-state (null), so
        # `eq dedicated` cleanly reads false: the check is not applicable and
        # nothing fails.
        cohort = _declared_cohort(tmp_path, point=OMITTED_OFFLINE_GATE)
        findings = Finding.collect(cohort, _submission(tmp_path, [16])).findings
        assert findings == ()

    @pytest.mark.unit
    def test_ne_dedicated_applies_identically_to_omitted_and_none_offline(
        self, tmp_path
    ):
        cohort = _declared_cohort(tmp_path, point=NE_DEDICATED)
        findings = Finding.collect(
            cohort,
            _submission(tmp_path, [16, 24], extras={24: {"offline": "none"}}),
        ).findings
        assert [(f.node_directory.name, f.verdict.ok) for f in findings] == [
            ("r16", True),
            ("r24", True),
        ]
        assert findings[0].verdict == findings[1].verdict

    @pytest.mark.unit
    def test_a_when_row_over_an_absent_required_fact_refuses_loud(self, tmp_path):
        cohort = _declared_cohort(tmp_path, pareto=REQUIRED_WHEN)
        with pytest.raises(
            ValueError,
            match=r"`when` row cannot be evaluated.*required fact "
            r"pareto\.system_desc\.max_supported_concurrency is absent from the upload",
        ):
            Finding.collect(cohort, _submission(tmp_path, [16]))

    @pytest.mark.unit
    def test_an_assert_row_over_an_absent_required_fact_is_recorded_as_a_broken_finding(
        self, tmp_path
    ):
        cohort = _declared_cohort(tmp_path, pareto=REQUIRED_ASSERT)
        findings = Finding.collect(cohort, _submission(tmp_path, [16])).findings
        assert len(findings) == 1
        assert findings[0].verdict.broken is True
        assert findings[0].verdict.ok is False
        assert "required fact" in findings[0].verdict.why
        assert "is absent from the upload" in findings[0].verdict.why

    @pytest.mark.unit
    def test_an_unannotated_hand_built_gate_gates_its_check_off(self, tmp_path):
        # The documented hand-built default (`Cohort((check,))`, no table):
        # the `Ref` carries neither note, so an omitted operand goes dark, the
        # `when` does not hold, and the check gates off instead of judging
        # its state or breaking.
        check = _check(
            "table_free",
            when=[{"left": "point.config.offline", "op": "eq", "right": "dedicated"}],
            assert_=[{"left": 0, "op": "eq", "right": 1}],
        )
        gate = check.when[0].left
        assert gate.required is False
        assert gate.default is MISSING
        findings = Finding.collect(
            Cohort((check,)), _submission(tmp_path, [16])
        ).findings
        assert findings == ()


class TestDeclaredValueChecking:
    """`Finding.collect` refuses a present artifact value its table will not carry.

    `RefTable.validate_values` runs per node (`_validate_declared_values`):
    `type` match and `allowlist` membership per declared leaf, block shape per
    declared block. An absent fact is checked by nothing (its absence reads
    per the `required`/`default` pins above), `dict` subtrees stay open and
    unchecked, undeclared keys are not walked, and `refs is None` skips the
    whole pass.
    """

    @pytest.mark.unit
    def test_a_present_leaf_value_must_match_its_declared_type(self, tmp_path):
        # A non-parser field (`warmup.duration_s`): `Point.from_node` refuses a
        # mistyped parser trio member earlier, at upload load.
        cohort = _declared_cohort(
            tmp_path,
            refs={"point": {"config": {"warmup": {"duration_s": {"type": "int"}}}}},
            point=PLAIN_CHECK,
        )
        with pytest.raises(
            ValueError,
            match=r"point\.config\.warmup\.duration_s: 'x' "
            r"does not match its declared type",
        ):
            Finding.collect(
                cohort,
                _submission(
                    tmp_path, [16], extras={16: {"warmup": {"duration_s": "x"}}}
                ),
            )

    @pytest.mark.unit
    def test_a_bool_value_fails_an_int_leaf(self, tmp_path):
        # The exact-type idiom: `bool` is not an `int`.
        cohort = _declared_cohort(
            tmp_path,
            refs={"point": {"config": {"warmup": {"duration_s": {"type": "int"}}}}},
            point=PLAIN_CHECK,
        )
        with pytest.raises(ValueError, match="does not match its declared type"):
            Finding.collect(
                cohort,
                _submission(
                    tmp_path, [16], extras={16: {"warmup": {"duration_s": True}}}
                ),
            )

    @pytest.mark.unit
    def test_a_float_leaf_accepts_an_int_value(self, tmp_path):
        cohort = _declared_cohort(
            tmp_path,
            refs={"point": {"config": {"warmup": {"duration_s": {"type": "float"}}}}},
            point=PLAIN_CHECK,
        )
        findings = Finding.collect(
            cohort,
            _submission(tmp_path, [16], extras={16: {"warmup": {"duration_s": 42}}}),
        ).findings
        assert [f.verdict.ok for f in findings] == [True]

    @pytest.mark.unit
    def test_a_present_leaf_value_must_be_in_its_allowlist(self, tmp_path):
        cohort = _declared_cohort(
            tmp_path,
            refs={
                "point": {
                    "config": {
                        "region": {
                            "type": "str",
                            "allowlist": [
                                "low_latency",
                                "low_concurrency",
                                "med_concurrency",
                                "high_concurrency",
                                "submitters_choice",
                            ],
                        }
                    }
                }
            },
            point=PLAIN_CHECK,
        )
        with pytest.raises(
            ValueError,
            match=r"point\.config\.region: 'bogus' is not one of",
        ):
            Finding.collect(
                cohort, _submission(tmp_path, [16], extras={16: {"region": "bogus"}})
            )

    @pytest.mark.unit
    def test_a_value_must_be_a_block_where_the_table_declares_one(self, tmp_path):
        cohort = _declared_cohort(
            tmp_path,
            refs={"point": {"config": {"warmup": {"duration_s": "int"}}}},
            point=PLAIN_CHECK,
        )
        with pytest.raises(
            ValueError,
            match=r"point\.config\.warmup: 'foo' is no block",
        ):
            Finding.collect(
                cohort, _submission(tmp_path, [16], extras={16: {"warmup": "foo"}})
            )

    @pytest.mark.unit
    def test_a_dict_subtree_value_is_never_checked(self, tmp_path):
        # `percentiles` is an open document: junk at any depth is the
        # submitter's own, unchecked by the table.
        cohort = _declared_cohort(tmp_path, point=PLAIN_CHECK)
        submission = write_upload(
            tmp_path / "acme",
            {
                SYSTEM: [
                    Run(
                        concurrency=16,
                        summary_extra={"percentiles": {"p99": {"deep": ["junk"]}}},
                    )
                ]
            },
        )
        findings = Finding.collect(cohort, submission).findings
        assert [f.verdict.ok for f in findings] == [True]

    @pytest.mark.unit
    def test_an_undeclared_key_is_not_checked(self, tmp_path):
        # The walk visits the table's declared entries only; an undeclared key
        # (with whatever subtree) is not the table's to judge.
        cohort = _declared_cohort(
            tmp_path,
            refs={"point": {"config": {"concurrency": "int"}}},
            point=PLAIN_CHECK,
        )
        findings = Finding.collect(
            cohort,
            _submission(tmp_path, [16], extras={16: {"mystery": {"any": ["shape"]}}}),
        ).findings
        assert [f.verdict.ok for f in findings] == [True]

    @pytest.mark.unit
    def test_the_multi_node_walk_refuses_on_the_offending_node(self, tmp_path):
        # Only the later run carries the unmappable value: the per-node walk
        # still refuses, and the message names the offending node's
        # `point.<document>.<field>` path.
        cohort = _declared_cohort(
            tmp_path,
            refs={"point": {"config": {"warmup": {"duration_s": {"type": "int"}}}}},
            point=PLAIN_CHECK,
        )
        submission = write_upload(
            tmp_path / "acme",
            {
                SYSTEM: [
                    Run(concurrency=16),
                    Run(
                        concurrency=8192,
                        config_extra={"warmup": {"duration_s": "x"}},
                    ),
                ]
            },
        )
        with pytest.raises(
            ValueError,
            match=r"point\.config\.warmup\.duration_s: 'x' "
            r"does not match its declared type",
        ):
            Finding.collect(cohort, submission)


class TestNeedsGate:
    """`needs` gates the dependent's rows at each node on its upstream's state there."""

    @pytest.mark.unit
    def test_failed_needs_gate_suppresses_only_the_dependent(self, tmp_path):
        first = _check("first", assert_=[{"left": 0, "op": "eq", "right": 1}])
        second = _check("second", needs=["first"], assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((first, second)), _submission(tmp_path, [16])
        ).findings
        assert [f.check_id for f in findings] == ["first"]
        assert findings[0].verdict.ok is False

    @pytest.mark.unit
    def test_holding_needs_gate_allows_both(self, tmp_path):
        first = _check("first", assert_=[LITERAL_ROW])
        second = _check("second", needs=["first"], assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((first, second)), _submission(tmp_path, [16])
        ).findings
        assert [f.check_id for f in findings] == ["first", "second"]

    @pytest.mark.unit
    def test_optional_needs_is_not_a_field(self):
        with pytest.raises(ValueError, match="optional_needs"):
            Check.model_validate(
                {
                    "id": "second",
                    "description": "What is observed.",
                    "rule": "Where in the Rules this comes from.",
                    "severity": "fail",
                    "scope": "point",
                    "optional_needs": ["first"],
                    "assert": [LITERAL_ROW],
                }
            )


class TestNeedsGatePerNode:
    """`needs` speaks per node: one node's upstream state does not bleed into another's."""

    @pytest.mark.unit
    def test_a_when_gated_upstream_vouches_where_it_speaks(self, tmp_path):
        # A system-scoped dependent with a point-scoped `when`-gated upstream
        # vouches via the points where the upstream claims and holds --
        # NOT_APPLICABLE neighbors neither vouch nor disavow.
        upstream = _check("upstream", when=[NODES_USED_ONE], assert_=[LITERAL_ROW])
        dependent = _check(
            "dependent",
            scope="system",
            needs=["upstream"],
            assert_=[LITERAL_ROW],
        )
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(tmp_path, [16, 512], extras={16: {"nodes_used": 1}}),
        ).findings
        got = [(f.check_id, f.node_directory.name, f.verdict.ok) for f in findings]
        assert sorted(got) == [
            ("dependent", "sys", True),
            ("upstream", "r16", True),
        ]

    @pytest.mark.unit
    def test_a_silenced_upstream_silences_through_its_dependents(self, tmp_path):
        bottom = _check("bottom", when=[NODES_USED_ONE], assert_=[LITERAL_ROW])
        middle = _check("middle", needs=["bottom"], assert_=[LITERAL_ROW])
        top = _check("top", needs=["middle"], assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((bottom, middle, top)),
            _submission(tmp_path, [16], extras={16: {"nodes_used": 2}}),
        ).findings
        assert findings == ()

    @pytest.mark.unit
    def test_an_ancestor_scoped_upstream_vouches_and_dedups(self, tmp_path):
        # Two points under one system: the system-scoped upstream emits once
        # (shared-ancestor dedup) and its claim vouches both points.
        upstream = _check("sys_up", scope="system", assert_=[LITERAL_ROW])
        dependent = _check("dep", needs=["sys_up"], assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((upstream, dependent)), _submission(tmp_path, [16, 8192])
        ).findings
        assert [(f.check_id, f.verdict.ok) for f in findings] == [
            ("sys_up", True),
            ("dep", True),
            ("dep", True),
        ]

    @pytest.mark.unit
    def test_rows_emit_in_order_holds_and_failures_alike(self, tmp_path):
        check = _check(
            "rows",
            assert_=[{"left": 0, "op": "eq", "right": 1}, LITERAL_ROW],
        )
        findings = Finding.collect(
            Cohort((check,)), _submission(tmp_path, [16])
        ).findings
        assert [f.verdict.ok for f in findings] == [False, True]

    @pytest.mark.unit
    def test_upstream_failure_silences_only_its_own_nodes(self, tmp_path):
        upstream = _check("upstream", assert_=[NODES_USED_ONE])
        dependent = _check("dependent", needs=["upstream"], assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(
                tmp_path,
                [16, 512],
                extras={16: {"nodes_used": 1}, 512: {"nodes_used": 2}},
            ),
        ).findings
        got = sorted(
            (f.check_id, f.node_directory.name, f.verdict.ok) for f in findings
        )
        assert got == [
            ("dependent", "r16", True),
            ("upstream", "r16", True),
            ("upstream", "r512", False),
        ]

    @pytest.mark.unit
    def test_one_failing_claim_disavows_a_coarser_dependent(self, tmp_path):
        # The claims list mixes HOLDS (r16) and FAILS (r512): one failing claim
        # disavows, so the system-scoped dependent must stay silent while both
        # upstream findings remain.
        upstream = _check("upstream", assert_=[NODES_USED_ONE])
        dependent = _check(
            "dependent",
            scope="system",
            needs=["upstream"],
            assert_=[LITERAL_ROW],
        )
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(
                tmp_path,
                [16, 512],
                extras={16: {"nodes_used": 1}, 512: {"nodes_used": 2}},
            ),
        ).findings
        assert [(f.check_id, f.verdict.ok) for f in findings] == [
            ("upstream", True),
            ("upstream", False),
        ]

    @pytest.mark.unit
    def test_an_upstream_that_makes_no_claim_at_a_node_does_not_vouch(self, tmp_path):
        upstream = _check("upstream", when=[NODES_USED_ONE], assert_=[LITERAL_ROW])
        dependent = _check("dependent", needs=["upstream"], assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(
                tmp_path,
                [16, 512],
                extras={16: {"nodes_used": 2}, 512: {"nodes_used": 1}},
            ),
        ).findings
        got = [(f.check_id, f.node_directory.name) for f in findings]
        assert got == [("upstream", "r512"), ("dependent", "r512")]

    @pytest.mark.unit
    def test_an_upstream_that_ran_nowhere_never_vouches(self, tmp_path):
        upstream = _check("upstream", when=[NODES_USED_ONE], assert_=[LITERAL_ROW])
        dependent = _check("dependent", needs=["upstream"], assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(
                tmp_path,
                [16, 512],
                extras={16: {"nodes_used": 2}, 512: {"nodes_used": 2}},
            ),
        ).findings
        assert findings == ()

    @pytest.mark.unit
    def test_a_silenced_claim_at_one_node_disavows_a_coarser_dependent(self, tmp_path):
        # `bottom` fails at r512, `middle` is SILENCED there and holds at r16;
        # the system-scoped `top` must not run off middle's r16 hold -- an
        # indirect failure cannot hide behind a silenced check.
        bottom = _check("bottom", assert_=[NODES_USED_ONE])
        middle = _check("middle", needs=["bottom"], assert_=[LITERAL_ROW])
        top = _check("top", scope="system", needs=["middle"], assert_=[LITERAL_ROW])
        findings = Finding.collect(
            Cohort((bottom, middle, top)),
            _submission(
                tmp_path,
                [16, 512],
                extras={16: {"nodes_used": 1}, 512: {"nodes_used": 2}},
            ),
        ).findings
        got = [(f.check_id, f.node_directory.name, f.verdict.ok) for f in findings]
        assert got == [
            ("bottom", "r16", True),
            ("bottom", "r512", False),
            ("middle", "r16", True),
        ]


class TestNeedsFailLoud:
    """Unknown referenced ids and `needs` cycles are load mistakes, not report items."""

    @pytest.mark.unit
    def test_unknown_need_raises(self, tmp_path):
        check = _check("check", needs=["ghost"], assert_=[LITERAL_ROW])
        with pytest.raises(ValueError, match="unknown check"):
            Finding.collect(Cohort((check,)), _submission(tmp_path, [16]))

    @pytest.mark.unit
    def test_needs_cycle_raises(self, tmp_path):
        first = _check("first", needs=["second"], assert_=[LITERAL_ROW])
        second = _check("second", needs=["first"], assert_=[LITERAL_ROW])
        with pytest.raises(ValueError, match="cycle"):
            Finding.collect(Cohort((first, second)), _submission(tmp_path, [16]))

    @pytest.mark.unit
    def test_a_cycle_is_refused_whatever_the_when_rows_decide(self, tmp_path):
        # The cycle sits behind a `when` that excludes every node: the load-time
        # graph check must not care that the walk would never reach the edge.
        first = _check(
            "first",
            needs=["second"],
            when=[{"left": 0, "op": "eq", "right": 1}],
            assert_=[LITERAL_ROW],
        )
        second = _check("second", needs=["first"], assert_=[LITERAL_ROW])
        with pytest.raises(ValueError, match="cycle"):
            Finding.collect(Cohort((first, second)), _submission(tmp_path, [16]))


class TestCollectFailsLoud:
    """A submission `Finding.collect` cannot walk is an error, never a zero-finding report."""

    @pytest.mark.unit
    def test_no_points_raises(self, tmp_path):
        root = tmp_path / "acme"
        root.mkdir()
        tree = SubmissionNode.from_directory(root)
        with pytest.raises(ValueError, match="no measurement points"):
            Finding.collect(
                Cohort((_check("cap", assert_=[LITERAL_ROW]),)), Submission(tree, ())
            )

    @pytest.mark.unit
    def test_point_without_its_node_raises(self, tmp_path):
        root = tmp_path / "acme"
        root.mkdir()
        tree = SubmissionNode.from_directory(root)
        system_desc = {"max_supported_concurrency": 8192}
        point = Point(
            offline=None,
            concurrency=512,
            system_tps=1.0,
            system_desc=system_desc,
        )
        submission = Submission(tree, (Pareto(system_desc, [point]),))
        with pytest.raises(ValueError, match="point node"):
            Finding.collect(Cohort((_check("cap", assert_=[LITERAL_ROW]),)), submission)

    @pytest.mark.unit
    def test_an_unreadable_artifact_never_reads_as_absent(self, tmp_path):
        run_dir = tmp_path / "acme" / "results" / "sys" / "model" / "r16"
        write_run(run_dir, Run(concurrency=16))
        (run_dir / "config.yaml").write_text("{bad", encoding="utf-8")
        submission = Submission.from_directory(tmp_path / "acme")
        check = _check("cap", assert_=[LITERAL_ROW])
        with pytest.raises(ValueError, match="unreadable artifacts"):
            Finding.collect(Cohort((check,)), submission)

    @pytest.mark.unit
    def test_a_cohort_naming_the_ruleset_root_needs_one(self, tmp_path):
        check = _check(
            "catalogued", assert_=[{"left": "ruleset.models.gpt", "op": "exists"}]
        )
        with pytest.raises(ValueError, match="without ruleset"):
            Finding.collect(Cohort((check,)), _submission(tmp_path, [16]))

    @pytest.mark.unit
    def test_a_when_row_that_cannot_be_judged_is_refused(self, tmp_path):
        gated = _check(
            "gated",
            when=[{"over": [1, "a"], "aggregate": "mean", "op": "eq", "value": 0}],
            assert_=[LITERAL_ROW],
        )
        with pytest.raises(ValueError, match="cannot be evaluated"):
            Finding.collect(Cohort((gated,)), _submission(tmp_path, [16]))

    @pytest.mark.unit
    def test_an_unlisted_run_directory_is_refused(self, tmp_path):
        submission = _submission(tmp_path, [16, 8192])
        pareto = submission.paretos[0]
        truncated = Submission(
            submission.tree,
            (Pareto(pareto.system_desc, pareto.points[:1], node=pareto.node),),
        )
        check = _check("cap", assert_=[LITERAL_ROW])
        with pytest.raises(ValueError, match="no carried measurement point"):
            Finding.collect(Cohort((check,)), truncated)

    @pytest.mark.unit
    def test_two_points_sharing_a_run_directory_are_refused(self, tmp_path):
        submission = _submission(tmp_path, [16])
        point = submission.points[0]
        twin = Point(
            offline=None,
            concurrency=40,
            system_tps=2.0,
            system_desc=point.system_desc,
            node=point.node,
        )
        pareto = submission.paretos[0]
        twins = Submission(
            submission.tree,
            (Pareto(pareto.system_desc, [point, twin], node=pareto.node),),
        )
        check = _check("cap", assert_=[LITERAL_ROW])
        with pytest.raises(ValueError, match="share one run directory"):
            Finding.collect(Cohort((check,)), twins)

    @pytest.mark.unit
    def test_a_foreign_tree_point_is_refused(self, tmp_path):
        here = _submission(tmp_path, [16])
        other_root = tmp_path / "elsewhere"
        (other_root / "results" / "sys" / "model" / "r40").mkdir(parents=True)
        foreign = SubmissionNode.from_directory(other_root)
        foreign_point = Point(
            offline=None,
            concurrency=40,
            system_tps=1.0,
            system_desc={"max_supported_concurrency": 8192},
            node=foreign.children[0].children[0].children[0],
        )
        here_point = here.points[0]
        mixed = Submission(
            here.tree,
            (
                Pareto(
                    here_point.system_desc,
                    [here_point, foreign_point],
                    node=here.paretos[0].node,
                ),
            ),
        )
        check = _check("cap", assert_=[LITERAL_ROW])
        with pytest.raises(ValueError, match="outside this submission's tree"):
            Finding.collect(Cohort((check,)), mixed)


class TestDerivedExposureLink:
    """`derived.<name>` rows resolve through the owning pareto, lazily."""

    @pytest.mark.unit
    def test_derived_fact_row_resolves(self, tmp_path):
        check = _check(
            "derived_row",
            assert_=[{"left": "derived.point_cap", "op": "eq", "right": True}],
        )
        findings = Finding.collect(
            Cohort((check,)), _submission(tmp_path, [16])
        ).findings
        assert len(findings) == 1
        assert findings[0].verdict.ok, findings[0].verdict.why

    @pytest.mark.unit
    def test_derived_out_of_range_points_counts_per_point_node(self, tmp_path):
        check = _check(
            "flagged",
            assert_=[
                {
                    "over": "derived.out_of_range_points",
                    "aggregate": "count",
                    "op": "eq",
                    "value": 1,
                }
            ],
        )
        findings = Finding.collect(
            Cohort((check,)), _submission(tmp_path, [16, 9013])
        ).findings
        assert [f.verdict.ok for f in findings] == [True, True]


class TestCheckRunRecords:
    """`CheckRun` counts the states a check met, not the findings it emitted."""

    @pytest.mark.unit
    def test_a_when_only_check_ran_everywhere(self, tmp_path):
        # A `when`-only check is a legal shape: it holds at every applicable
        # node and emits no findings -- the run record is its only witness.
        check = _check("when_only", when=[{"left": 1, "op": "eq", "right": 1}])
        collected = Finding.collect(Cohort((check,)), _submission(tmp_path, [16, 8192]))
        assert collected.findings == ()
        assert collected.runs["when_only"] == CheckRun(
            ran_at=2, gated_off=0, silenced=0
        )

    @pytest.mark.unit
    def test_gated_and_silenced_nodes_are_counted(self, tmp_path):
        upstream = _check("upstream", when=[NODES_USED_ONE], assert_=[LITERAL_ROW])
        dependent = _check("dependent", needs=["upstream"], assert_=[LITERAL_ROW])
        collected = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(tmp_path, [16, 512], extras={16: {"nodes_used": 1}}),
        )
        assert collected.runs["upstream"] == CheckRun(ran_at=1, gated_off=1, silenced=0)
        assert collected.runs["dependent"] == CheckRun(
            ran_at=1, gated_off=0, silenced=1
        )

    @pytest.mark.unit
    def test_the_run_key_set_is_the_cohort(self, tmp_path):
        check = _check("cap", assert_=[LITERAL_ROW])
        collected = Finding.collect(Cohort((check,)), _submission(tmp_path, [16]))
        assert set(collected.runs) == {"cap"}
