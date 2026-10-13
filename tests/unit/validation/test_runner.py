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
from inference_endpoint.validation.check import Check, Cohort
from inference_endpoint.validation.runner import Finding
from inference_endpoint.validation.submission import Point, Submission
from inference_endpoint.validation.tree import SubmissionNode


def _submission(tmp_path: Path, concurrencies: list[int]) -> Submission:
    """A submission over real `results/sys/model/r<concurrency>` subtrees."""
    root = tmp_path / "acme"
    for concurrency in concurrencies:
        (root / "results" / "sys" / "model" / f"r{concurrency}").mkdir(parents=True)
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
    """`needs` gates only the gated check's rows; `optional_needs` never gate."""

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

    def test_optional_needs_never_gates(self, tmp_path):
        first = _check("first", assert_=[{"left": 0, "op": "eq", "right": 1}])
        second = _check(
            "second",
            optional_needs=["first"],
            assert_=[{"left": 1, "op": "eq", "right": 1}],
        )
        findings = Finding.collect(
            Cohort((first, second)), _submission(tmp_path, [512])
        )
        assert [f.check_id for f in findings] == ["first", "second"]


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

    def test_unnamed_derivation_never_runs(self, tmp_path):
        # Two `offline: dedicated` points make `derived.offline_ordering` raise
        # "exactly one point may carry an `offline` declaration"; a row that
        # names no derivation must never reach it.
        root = tmp_path / "acme"
        for concurrency in (512, 8192):
            (root / "results" / "sys" / "model" / f"r{concurrency}").mkdir(parents=True)
        tree = SubmissionNode.from_directory(root)
        nodes = {
            int(node.dir.name[1:]): node
            for node in tree.children[0].children[0].children
        }
        submission = Submission(
            {"max_supported_concurrency": 8192},
            tuple(Point("dedicated", c, 1.0, nodes[c]) for c in (512, 8192)),
        )
        check = _check("plain", assert_=[{"left": 1, "op": "eq", "right": 1}])
        findings = Finding.collect(Cohort((check,)), submission)
        assert [f.verdict.ok for f in findings] == [True, True]
