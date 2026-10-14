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

"""W6 tests: the runner's Finding-per-row shape, `needs` gating and cycle-freedom.

A `Finding` is the outcome record of every *evaluated* `assert` row at its
applicable node (holds and failures alike -- the audit trail, formatted by a
renderer that joins `fix` tips where the catalog declares one). `needs`
gating suppresses only that check's rows.
"""

from pathlib import Path

import pytest
import yaml
from inference_endpoint.validation.check import Check, Cohort
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.runner import Finding
from inference_endpoint.validation.submission import Point, Submission
from inference_endpoint.validation.tree import SubmissionNode


def _submission(
    tmp_path: Path, concurrencies: list[int], configs: dict[int, dict] | None = None
) -> Submission:
    """A submission over real `results/sys/model/r<concurrency>` subtrees.

    `configs` writes each named point's `point.yaml`; a point without one is an
    empty run directory, so rows read `point.config` as absent there.
    """
    root = tmp_path / "acme"
    for concurrency in concurrencies:
        point_dir = root / "results" / "sys" / "model" / f"r{concurrency}"
        point_dir.mkdir(parents=True)
        config = {"concurrency": concurrency, **(configs or {}).get(concurrency, {})}
        (point_dir / "point.yaml").write_text(yaml.dump(config))
    tree = SubmissionNode.from_directory(root)
    points_by_concurrency = {
        int(node.dir.name[1:]): node for node in tree.children[0].children[0].children
    }
    points = tuple(
        Point(
            offline=None,
            concurrency=concurrency,
            system_tps=1.0,
            node=points_by_concurrency[concurrency],
        )
        for concurrency in concurrencies
    )
    return Submission({"max_supported_concurrency": 8192}, points)


def _check(id: str, **kwargs) -> Check:
    base = {
        "id": id,
        "description": "What is observed.",
        "rule": "Where in the Rules this comes from.",
        "severity": "fail",
        "scope": "point",
    }
    return Check.model_validate(base | kwargs)


class TestFindingsPerEvaluatedRow:
    """`Finding.collect` emits one `Finding` per `assert` row at each applicable node (holds too)."""

    def test_one_finding_per_assert_row_per_point(self, tmp_path):
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        findings = Finding.collect(Cohort((check,)), _submission(tmp_path, [512, 8192]))
        assert len(findings) == 2
        assert all(isinstance(f, Finding) for f in findings)
        assert [f.check_id for f in findings] == ["cap", "cap"]
        assert all(f.verdict.ok for f in findings)

    def test_failing_verdict_records_why(self, tmp_path):
        check = _check("failing", assert_=[{"left": 0, "op": "eq", "right": 1}])
        findings = Finding.collect(Cohort((check,)), _submission(tmp_path, [512]))
        assert len(findings) == 1
        assert findings[0].verdict.ok is False
        assert findings[0].verdict.why == "expected 0 eq 1, got 0 vs 1"

    def test_when_gate_suppresses(self, tmp_path):
        check = _check(
            "gated",
            when=[{"left": 0, "op": "eq", "right": 1}],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        assert Finding.collect(Cohort((check,)), _submission(tmp_path, [512])) == ()


class TestNeedsGate:
    """`needs` gates the dependent's rows at each node on its upstream's state there."""

    def test_failed_needs_gate_suppresses_only_the_dependent(self, tmp_path):
        first = _check("first", assert_=[{"left": 0, "op": "eq", "right": 1}])
        second = _check(
            "second", needs=["first"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        findings = Finding.collect(
            Cohort((first, second)), _submission(tmp_path, [512])
        )
        assert [f.check_id for f in findings] == ["first"]
        assert findings[0].verdict.ok is False

    def test_holding_needs_gate_allows_both(self, tmp_path):
        first = _check("first", assert_=[{"left": 1, "op": "eq", "right": 1}])
        second = _check(
            "second", needs=["first"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        findings = Finding.collect(
            Cohort((first, second)), _submission(tmp_path, [512])
        )
        assert [f.check_id for f in findings] == ["first", "second"]

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
                    "assert": [{"left": 1, "op": "eq", "right": 1}],
                }
            )


class TestNeedsGatePerNode:
    """`needs` speaks per node: one node's upstream state does not bleed into another's."""

    def test_a_when_gated_upstream_vouches_where_it_speaks(self, tmp_path):
        # The README's cross-scope shape: a system-scoped dependent with a
        # point-scoped `when`-gated upstream vouches via the points where the
        # upstream claims and holds -- NOT_APPLICABLE neighbors neither vouch
        # nor disavow.
        upstream = _check(
            "upstream",
            when=[{"left": "point.config.x", "op": "eq", "right": 1}],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        dependent = _check(
            "dependent",
            scope="system",
            needs=["upstream"],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(tmp_path, [512, 8192], configs={512: {"x": 1}, 8192: {"x": 2}}),
        )
        got = [(f.check_id, f.node_directory.name, f.verdict.ok) for f in findings]
        assert sorted(got) == [
            ("dependent", "sys", True),
            ("upstream", "r512", True),
        ]

    def test_a_silenced_upstream_silences_through_its_dependents(self, tmp_path):
        bottom = _check(
            "bottom",
            when=[{"left": "point.config.x", "op": "eq", "right": 1}],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        middle = _check(
            "middle", needs=["bottom"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        top = _check(
            "top", needs=["middle"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        findings = Finding.collect(
            Cohort((bottom, middle, top)),
            _submission(tmp_path, [512], configs={512: {"x": 2}}),
        )
        assert findings == ()

    def test_an_ancestor_scoped_upstream_vouches_and_dedups(self, tmp_path):
        # Two points under one system: the system-scoped upstream emits once
        # (shared-ancestor dedup) and its claim vouches both points.
        upstream = _check(
            "sys_up",
            scope="system",
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        dependent = _check(
            "dep", needs=["sys_up"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        findings = Finding.collect(
            Cohort((upstream, dependent)), _submission(tmp_path, [512, 8192])
        )
        assert [(f.check_id, f.verdict.ok) for f in findings] == [
            ("sys_up", True),
            ("dep", True),
            ("dep", True),
        ]

    def test_rows_emit_in_order_holds_and_failures_alike(self, tmp_path):
        check = _check(
            "rows",
            assert_=[
                {"left": 0, "op": "eq", "right": 1},
                {"left": 1, "op": "eq", "right": 1},
            ],
        )
        findings = Finding.collect(Cohort((check,)), _submission(tmp_path, [512]))
        assert [f.verdict.ok for f in findings] == [False, True]

    def test_upstream_failure_silences_only_its_own_nodes(self, tmp_path):
        upstream = _check(
            "upstream",
            assert_=[{"left": "point.config.x", "op": "eq", "right": 1}],
        )
        dependent = _check(
            "dependent",
            needs=["upstream"],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(tmp_path, [512, 8192], configs={512: {"x": 1}, 8192: {"x": 2}}),
        )
        got = sorted(
            (f.check_id, f.node_directory.name, f.verdict.ok) for f in findings
        )
        assert got == [
            ("dependent", "r512", True),
            ("upstream", "r512", True),
            ("upstream", "r8192", False),
        ]

    def test_one_failing_claim_disavows_a_coarser_dependent(self, tmp_path):
        # The claims list mixes HOLDS (r512) and FAILS (r8192): one failing
        # claim disavows, so the system-scoped dependent must stay silent while
        # both upstream findings remain.
        upstream = _check(
            "upstream",
            assert_=[{"left": "point.config.x", "op": "eq", "right": 1}],
        )
        dependent = _check(
            "dependent",
            scope="system",
            needs=["upstream"],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(tmp_path, [512, 8192], configs={512: {"x": 1}, 8192: {"x": 2}}),
        )
        assert [(f.check_id, f.verdict.ok) for f in findings] == [
            ("upstream", True),
            ("upstream", False),
        ]

    def test_an_upstream_that_makes_no_claim_at_a_node_does_not_vouch(self, tmp_path):
        upstream = _check(
            "upstream",
            when=[{"left": "point.config.x", "op": "eq", "right": 1}],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        dependent = _check(
            "dependent",
            needs=["upstream"],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(tmp_path, [512, 8192], configs={512: {"x": 2}, 8192: {"x": 1}}),
        )
        got = [(f.check_id, f.node_directory.name) for f in findings]
        assert got == [("upstream", "r8192"), ("dependent", "r8192")]

    def test_an_upstream_that_ran_nowhere_never_vouches(self, tmp_path):
        upstream = _check(
            "upstream",
            when=[{"left": "point.config.x", "op": "eq", "right": 1}],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        dependent = _check(
            "dependent",
            needs=["upstream"],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        findings = Finding.collect(
            Cohort((upstream, dependent)),
            _submission(tmp_path, [512, 8192], configs={512: {"x": 2}, 8192: {"x": 2}}),
        )
        assert findings == ()


class TestNeedsFailLoud:
    """Unknown referenced ids and `needs` cycles are load mistakes, not report items."""

    def test_unknown_need_raises(self, tmp_path):
        check = _check(
            "check", needs=["ghost"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        with pytest.raises(ValueError, match="unknown check"):
            Finding.collect(Cohort((check,)), _submission(tmp_path, [512]))

    def test_needs_cycle_raises(self, tmp_path):
        first = _check(
            "first", needs=["second"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        second = _check(
            "second", needs=["first"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        with pytest.raises(ValueError, match="cycle"):
            Finding.collect(Cohort((first, second)), _submission(tmp_path, [512]))

    def test_a_cycle_is_refused_whatever_the_when_rows_decide(self, tmp_path):
        # The cycle sits behind a `when` that excludes every node: the load-time
        # graph check must not care that the walk would never reach the edge.
        first = _check(
            "first",
            needs=["second"],
            when=[{"left": 0, "op": "eq", "right": 1}],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        second = _check(
            "second", needs=["first"], assert_=[{"left": 1, "op": "eq", "right": 1}]
        )
        with pytest.raises(ValueError, match="cycle"):
            Finding.collect(Cohort((first, second)), _submission(tmp_path, [512]))


class TestCollectFailsLoud:
    """A submission `Finding.collect` cannot walk is an error, never a zero-finding report."""

    def test_no_points_raises(self, tmp_path):
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        with pytest.raises(ValueError, match="no measurement points"):
            Finding.collect(
                Cohort((check,)),
                Submission({"max_supported_concurrency": 8192}, ()),
            )

    def test_point_without_its_node_raises(self, tmp_path):
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        with pytest.raises(ValueError, match="point node"):
            Finding.collect(
                Cohort((check,)),
                Submission(
                    {"max_supported_concurrency": 8192}, (Point(None, 512, 1.0, None),)
                ),
            )

    def test_an_unreadable_artifact_never_reads_as_absent(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r512"
        point_dir.mkdir(parents=True)
        (point_dir / "point.yaml").write_text("{bad")
        node = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        submission = Submission(
            {"max_supported_concurrency": 8192}, (Point(None, 512, 1.0, node),)
        )
        with pytest.raises(ValueError, match="unreadable artifacts"):
            Finding.collect(Cohort((check,)), submission)

    def test_hand_carried_facts_must_agree_with_the_tree(self, tmp_path):
        submission = _submission(
            tmp_path, [512], configs={512: {"concurrency": 512, "offline": "dedicated"}}
        )
        disagreement = Submission(
            submission.system_desc,
            (Point(None, 768, 1.0, submission.points[0].node),),
        )
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        with pytest.raises(ValueError, match="declares"):
            Finding.collect(Cohort((check,)), disagreement)

    def test_a_cohort_naming_a_missing_fold_is_a_load_error(self, tmp_path):
        check = _check("ranged", assert_=[{"left": "regions.coverage", "op": "exists"}])
        with pytest.raises(ValueError, match="without regions"):
            Finding.collect(Cohort((check,)), _submission(tmp_path, [512]))

    def test_a_cohort_naming_the_ruleset_fold_needs_one(self, tmp_path):
        check = _check(
            "catalogued", assert_=[{"left": "ruleset.models.gpt", "op": "exists"}]
        )
        with pytest.raises(ValueError, match="without ruleset"):
            Finding.collect(Cohort((check,)), _submission(tmp_path, [512]))

    def test_a_when_row_that_cannot_be_judged_is_refused(self, tmp_path):
        gated = _check(
            "gated",
            when=[{"over": [1, "a"], "measure": "mean", "op": "eq", "value": 0}],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        with pytest.raises(ValueError, match="cannot be evaluated"):
            Finding.collect(Cohort((gated,)), _submission(tmp_path, [512]))

    def test_the_offline_none_spelling_is_no_declaration(self, tmp_path):
        # Section 8.3's three-state field: `offline: none` normalizes to the
        # carried `None`, so a compliant upload never hits an unsatisfiable
        # disagreement.
        submission = _submission(tmp_path, [512], configs={512: {"offline": "none"}})
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        assert Finding.collect(Cohort((check,)), submission)

    def test_a_missing_declaration_disagrees_too(self, tmp_path):
        submission = _submission(tmp_path, [512, 8192])
        by_name = {p.node.dir.name: p.node for p in submission.points}
        disagreement = Submission(
            submission.system_desc,
            (
                Point("dedicated", 512, 1.0, by_name["r512"]),
                Point(None, 8192, 2.0, by_name["r8192"]),
            ),
        )
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        with pytest.raises(ValueError, match="carries offline"):
            Finding.collect(Cohort((check,)), disagreement)

    def test_an_offline_token_outside_the_domain_is_refused(self):
        with pytest.raises(ValueError, match="offline must be"):
            Point("dedicatd", 512, 1.0, None)

    def test_an_uncarried_run_directory_is_refused(self, tmp_path):
        root = tmp_path / "acme"
        for concurrency in (512, 8192):
            (root / "results" / "sys" / "model" / f"r{concurrency}").mkdir(parents=True)
        tree = SubmissionNode.from_directory(root)
        node = next(
            n for n in tree.children[0].children[0].children if n.dir.name == "r512"
        )
        submission = Submission(
            {"max_supported_concurrency": 8192}, (Point(None, 512, 1.0, node),)
        )
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        with pytest.raises(ValueError, match="no carried measurement point"):
            Finding.collect(Cohort((check,)), submission)

    def test_two_points_sharing_a_run_directory_are_refused(self, tmp_path):
        submission = _submission(tmp_path, [512])
        node = submission.points[0].node
        twins = Submission(
            submission.system_desc,
            (Point(None, 512, 1.0, node), Point(None, 40, 2.0, node)),
        )
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        with pytest.raises(ValueError, match="share one run directory"):
            Finding.collect(Cohort((check,)), twins)

    def test_a_foreign_tree_point_is_refused(self, tmp_path):
        here = _submission(tmp_path, [512, 8192])
        by_name = {p.node.dir.name: p.node for p in here.points}
        other_root = tmp_path / "elsewhere"
        (other_root / "results" / "sys" / "model" / "r40").mkdir(parents=True)
        foreign = SubmissionNode.from_directory(other_root)
        foreign_node = foreign.children[0].children[0].children[0]
        mixed = Submission(
            here.system_desc,
            (
                Point(None, 512, 1.0, by_name["r512"]),
                Point(None, 40, 1.0, foreign_node),
            ),
        )
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        with pytest.raises(ValueError, match="outside this submission's tree"):
            Finding.collect(Cohort((check,)), mixed)

    def test_carried_system_desc_must_agree_with_the_artifact(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results" / "sys" / "model" / "r512").mkdir(parents=True)
        (root / "system_desc.json").write_text('{"max_supported_concurrency": 4096}')
        node = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        submission = Submission(
            {"max_supported_concurrency": 8192}, (Point(None, 512, 1.0, node),)
        )
        check = _check("cap", assert_=[{"left": 1, "op": "eq", "right": 1}])
        with pytest.raises(ValueError, match="system_desc.json"):
            Finding.collect(Cohort((check,)), submission)


class TestDerivedExposureLink:
    """`derived.<name>` rows resolve lazily; an unnamed derivation never runs."""

    def test_derived_fact_row_resolves(self, tmp_path):
        check = _check(
            "derived_row",
            assert_=[{"left": "derived.point_cap", "op": "eq", "right": True}],
        )
        findings = Finding.collect(Cohort((check,)), _submission(tmp_path, [512]))
        assert len(findings) == 1
        assert findings[0].verdict.ok, findings[0].verdict.why

    def test_region_derived_fact_row_resolves_from_the_tree_fold(self, tmp_path):
        root = tmp_path / "acme"
        for concurrency in (24, 9013):
            (root / "results" / "sys" / "model" / f"r{concurrency}").mkdir(parents=True)
        tree = SubmissionNode.from_directory(
            root,
            regions=Regions.for_submission(
                {"max_supported_concurrency": 8192}, [24, 9013]
            ),
        )
        nodes = {
            int(node.dir.name[1:]): node
            for node in tree.children[0].children[0].children
        }
        submission = Submission(
            {"max_supported_concurrency": 8192},
            tuple(Point(None, c, 1.0, nodes[c]) for c in (24, 9013)),
        )
        check = _check(
            "flagged",
            assert_=[
                {
                    "over": "derived.out_of_range_points",
                    "measure": "count",
                    "op": "eq",
                    "value": 1,
                }
            ],
        )
        findings = Finding.collect(Cohort((check,)), submission)
        assert [f.verdict.ok for f in findings] == [True, True]

    def test_unnamed_derivation_never_runs(self, tmp_path):
        # A regions-less tree makes `derived.out_of_range_points` fail loud
        # when named; a row that names no such derivation must never reach it.
        check = _check(
            "plain",
            assert_=[{"left": "derived.point_cap", "op": "eq", "right": True}],
        )
        findings = Finding.collect(Cohort((check,)), _submission(tmp_path, [512]))
        assert [f.verdict.ok for f in findings] == [True]

    def test_a_named_derivation_gap_reports_as_itself(self, tmp_path):
        # The tree-builtin fold needs check fires first at load; the derivation
        # layer's own loud (`needs the submission's regions`) is pinned on
        # `Submission.derived` in test_submission.
        check = _check(
            "flagged",
            assert_=[{"left": "derived.out_of_range_points", "op": "exists"}],
        )
        with pytest.raises(ValueError, match="without regions"):
            Finding.collect(Cohort((check,)), _submission(tmp_path, [512]))
