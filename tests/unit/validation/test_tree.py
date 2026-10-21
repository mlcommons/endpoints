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

"""W2 layer tests: tree walking, artifact loading, and the scopes mapping.

Fixtures are one small builder (`_tree`) writing tiny artifacts on demand, not
stored files. Cases assert what each subject computes: `from_directory` the
walk shape and node classes, `read` its (value, error) with `MISSING`, `scopes`
its mapping.
"""

import json
import os

import pytest
from inference_endpoint.validation.operations import Absent, Predicate
from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.refs import Ref
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.sentinels import MISSING
from inference_endpoint.validation.tree import (
    Node,
    ParetoNode,
    PointNode,
    RuleScopes,
    SubmissionNode,
    SystemNode,
)


def _tree(tmp_path, point_artifacts=None, system_artifacts=None):
    """One org-less submission: results/sys/model/r16 (+ r8), optional artifacts."""
    root = tmp_path / "acme"
    point = root / "results" / "sys" / "model" / "r16"
    point.mkdir(parents=True)
    (root / "results" / "sys" / "model" / "r8").mkdir()
    for name, value in (point_artifacts or {}).items():
        _write(point / name, value)
    for name, value in (system_artifacts or {}).items():
        _write(root / "results" / "sys" / name, value)
    return SubmissionNode.from_directory(root), root


def _write(path, value):
    """String values are written verbatim (for malformed content). Others are
    written as JSON."""
    path.write_text(value if isinstance(value, str) else json.dumps(value))


class TestWalk:
    """`SubmissionNode.from_directory`: kinds, classes, order and parentage."""

    @pytest.mark.unit
    def test_walk_shape(self, tmp_path):
        node, _ = _tree(tmp_path)
        assert isinstance(node, SubmissionNode)
        system = node.children[0]
        assert type(system) is SystemNode
        assert type(system.children[0]) is ParetoNode
        points = system.children[0].children
        assert [type(p) for p in points] == [PointNode, PointNode]
        assert [p.dir.name for p in points] == ["r16", "r8"]
        assert points[1].parent is system.children[0]
        assert points[1].parent.parent.parent is node

    @pytest.mark.unit
    def test_node_construction_follows_the_kind(self):
        """`Node(kind, ...)` yields the registered class. The registry is keyed on kind."""
        assert Node.registry == {
            RuleScopes.SUBMISSION: SubmissionNode,
            RuleScopes.SYSTEM: SystemNode,
            RuleScopes.PARETO: ParetoNode,
            RuleScopes.POINT: PointNode,
        }

    @pytest.mark.unit
    def test_non_r_directory_is_not_a_point(self, tmp_path):
        _, root = _tree(tmp_path)
        (root / "results" / "sys" / "model" / "notes").mkdir()
        node = SubmissionNode.from_directory(root)
        assert [p.dir.name for p in node.children[0].children[0].children] == [
            "r16",
            "r8",
        ]

    @pytest.mark.unit
    def test_symlinked_directory_out_of_the_submission_is_skipped(self, tmp_path):
        _, root = _tree(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "results" / "sys" / "model" / "r99").symlink_to(
            outside, target_is_directory=True
        )
        node = SubmissionNode.from_directory(root)
        assert "r99" not in [p.dir.name for p in node.children[0].children[0].children]

    @pytest.mark.unit
    def test_symlinked_directory_inside_the_submission_is_skipped_too(self, tmp_path):
        _, root = _tree(tmp_path)
        model = root / "results" / "sys" / "model"
        (model / "r9").symlink_to(model / "r8", target_is_directory=True)
        node = SubmissionNode.from_directory(root)
        # One real point stays one point: symlinks must not multiply nodes (or
        # inflate `point_cap`/counts the way a YAML alias would).
        assert [p.dir.name for p in node.children[0].children[0].children] == [
            "r16",
            "r8",
        ]

    @pytest.mark.unit
    def test_missing_root_is_a_load_error(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            SubmissionNode.from_directory(tmp_path / "nope")


class TestRead:
    """`Node.read`: (value, error) over the eagerly loaded artifacts."""

    def _point(self, tmp_path, artifacts=None):
        node, _ = _tree(tmp_path, artifacts)
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    def test_json_artifact_reads(self, tmp_path):
        point = self._point(tmp_path, {"result_summary.json": {"n": 1}})
        assert point.read("result_summary.json") == ({"n": 1}, None)

    @pytest.mark.unit
    def test_absent_artifact_is_missing(self, tmp_path):
        point = self._point(tmp_path)
        value, error = point.read("result_summary.json")
        assert value is MISSING
        assert error is None

    @pytest.mark.unit
    def test_missing_sentinel_is_the_shared_one_for_presence_ops(self, tmp_path):
        point = self._point(tmp_path)
        value, _ = point.read("result_summary.json")
        assert Absent().holds(Predicate(value, None), None, None) is True
        assert Absent().holds(Predicate(None, None), None, None) is False

    @pytest.mark.unit
    def test_present_null_is_none_not_missing(self, tmp_path):
        point = self._point(tmp_path, {"result_summary.json": None})
        assert point.read("result_summary.json") == (None, None)

    @pytest.mark.unit
    def test_artifacts_are_loaded_once_not_reread(self, tmp_path):
        point = self._point(tmp_path, {"result_summary.json": {"n": 1}})
        (point.dir / "result_summary.json").write_text(json.dumps({"n": 2}))
        assert point.read("result_summary.json") == ({"n": 1}, None)

    @pytest.mark.unit
    def test_duplicate_key_yaml_is_a_parse_error(self, tmp_path):
        point = self._point(tmp_path, {"point.yaml": "a: 1\na: 2\n"})
        value, error = point.read("point.yaml")
        assert value is None
        assert "duplicate key" in error

    @pytest.mark.unit
    def test_yaml_aliases_are_a_parse_error(self, tmp_path):
        point = self._point(tmp_path, {"point.yaml": "base: &b {x: 1}\np: *b\n"})
        value, error = point.read("point.yaml")
        assert value is None
        assert "aliases are not allowed" in error

    @pytest.mark.unit
    def test_yaml_merge_keys_are_a_parse_error(self, tmp_path):
        point = self._point(tmp_path, {"point.yaml": "p: {<<: {x: 1}, y: 2}\n"})
        value, error = point.read("point.yaml")
        assert value is None
        assert "merge keys are not allowed" in error

    @pytest.mark.unit
    def test_unparsable_json_reports_the_error(self, tmp_path):
        point = self._point(tmp_path, {"result_summary.json": "{bad"})
        value, error = point.read("result_summary.json")
        assert value is None
        assert "parse error" in error


class TestReadChannel:
    """`_read_artifact`: bounded, symlink-safe reads. Nothing here aborts the run."""

    def _point_with(self, tmp_path, name: str, data: bytes):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / name).write_bytes(data)
        node = SubmissionNode.from_directory(root)
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    def test_oversized_artifact_reports_too_large(self, tmp_path, monkeypatch):
        monkeypatch.setattr("inference_endpoint.validation.tree._MAX_ARTIFACT_BYTES", 8)
        point = self._point_with(tmp_path, "result_summary.json", b'{"n": 12}')
        value, error = point.read("result_summary.json")
        assert value is None
        assert "too large" in error

    @pytest.mark.unit
    def test_non_utf8_artifact_reports_a_parse_error(self, tmp_path):
        point = self._point_with(tmp_path, "result_summary.json", b"\xff\xfe{")
        value, error = point.read("result_summary.json")
        assert value is None
        assert "parse error" in error

    @pytest.mark.unit
    def test_a_fifo_artifact_is_refused_not_awaited(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        os.mkfifo(point_dir / "point.yaml")
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        value, error = point.read("point.yaml")
        assert value is None
        assert "not a regular file" in error

    @pytest.mark.unit
    def test_a_directory_where_an_artifact_belongs_is_refused(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        (point_dir / "result_summary.json").mkdir(parents=True)
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        value, error = point.read("result_summary.json")
        assert value is None
        assert "not a regular file" in error

    @pytest.mark.unit
    def test_json_duplicate_keys_are_a_parse_error(self, tmp_path):
        point = self._point_with(tmp_path, "result_summary.json", b'{"n": 1, "n": 2}')
        value, error = point.read("result_summary.json")
        assert value is None
        assert "duplicate key" in error

    @pytest.mark.unit
    def test_yaml_duplicate_null_keys_are_a_parse_error(self, tmp_path):
        point = self._point_with(tmp_path, "point.yaml", b"null: 1\nnull: 2\n")
        value, error = point.read("point.yaml")
        assert value is None
        assert "duplicate key" in error

    @pytest.mark.unit
    @pytest.mark.parametrize("payload", [b"", b"   \n", b"just a scalar\n", b"- 1\n"])
    def test_an_unusable_document_is_a_parse_error(self, tmp_path, payload):
        point = self._point_with(tmp_path, "point.yaml", payload)
        value, error = point.read("point.yaml")
        assert value is None
        assert "parse error" in error

    @pytest.mark.unit
    def test_symlinked_artifact_out_of_the_node_is_refused(self, tmp_path):
        outside = tmp_path / "outside.json"
        outside.write_text('{"n": 42}')
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / "result_summary.json").symlink_to(outside)
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        value, error = point.read("result_summary.json")
        assert value is None  # never the outside target's content
        assert "parse error" in error

    @pytest.mark.unit
    def test_symlinked_artifact_inside_the_node_is_a_parse_error(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / "other.json").write_text('{"n": 1}')
        (point_dir / "result_summary.json").symlink_to(point_dir / "other.json")
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        value, error = point.read("result_summary.json")
        assert value is None
        assert "parse error" in error


class TestScopes:
    """`Node.scopes`: one mapping per node up to the submission. Fact names are keys."""

    def _point(self, tmp_path):
        node, _ = _tree(
            tmp_path,
            {"result_summary.json": {"n": 1}, "point.yaml": "concurrency: 16\n"},
            {"system_power.json": None},
        )
        return node.children[0].children[0].children[0]

    @pytest.mark.unit
    def test_keys_are_fact_names_per_kind(self, tmp_path):
        scopes = self._point(tmp_path).scopes()
        assert set(scopes) == {"submission", "system", "pareto", "point"}
        assert set(scopes["point"]) == {"result_summary", "config"}
        assert set(scopes["system"]) == {"system_power"}

    @pytest.mark.unit
    def test_absent_artifact_is_an_absent_key(self, tmp_path):
        scopes = self._point(tmp_path).scopes()
        assert "system_desc" not in scopes["point"]

    @pytest.mark.unit
    def test_stored_null_is_a_present_key(self, tmp_path):
        scopes = self._point(tmp_path).scopes()
        assert scopes["system"]["system_power"] is None

    @pytest.mark.unit
    def test_corrupt_artifact_reads_as_absent_but_keeps_its_error(self, tmp_path):
        root = tmp_path / "acme"
        point_dir = root / "results" / "sys" / "model" / "r8"
        point_dir.mkdir(parents=True)
        (point_dir / "result_summary.json").write_text("{bad")
        (point_dir / "point.yaml").write_text("concurrency: 8\n")
        point = SubmissionNode.from_directory(root).children[0].children[0].children[0]
        assert "result_summary" not in point.scopes()["point"]
        assert set(point.scopes()["point"]) == {"config"}
        value, error = point.read("result_summary.json")
        assert value is None and "parse error" in error

    @pytest.mark.unit
    def test_root_walks_to_the_submission(self, tmp_path):
        point = self._point(tmp_path)
        assert isinstance(point.root(), SubmissionNode)

    @pytest.mark.unit
    def test_scopes_serve_ref_resolution(self, tmp_path):
        scopes = self._point(tmp_path).scopes()
        assert Ref(("point", "result_summary", "n")).resolve(scopes) == (True, 1)
        assert Ref(("point", "result_summary", "nope")).resolve(scopes) == (
            False,
            None,
        )


class TestRegionsRoot:
    """`Node.scopes` folds `Regions` into the `regions` root of `Ref`."""

    @pytest.mark.unit
    def test_regions_root_answers_coverage_lookups(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results").mkdir(parents=True)
        scopes = SubmissionNode.from_directory(
            root, regions=Regions.for_bounds(c_min=32, c_max=8192)
        ).scopes()
        assert Ref(("regions", "coverage")).resolve(scopes) == (
            True,
            Interval(1, 9012),
        )
        # The fold's `Interval` values are leaves: `Ref` returns one as-is and
        # never walks its attributes (`margin.lo` would be attribute traversal).
        assert Ref(("regions", "by_kind", "margin")).resolve(scopes) == (
            True,
            Interval(8193, 9012),
        )

    @pytest.mark.unit
    def test_without_regions_the_root_stays_away(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results").mkdir(parents=True)
        assert "regions" not in SubmissionNode.from_directory(root).scopes()
