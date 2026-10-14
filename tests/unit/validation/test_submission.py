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

"""W5f tests: `Point`/`Submission` hold the carry-set facts without drift."""

from pathlib import Path

import pytest
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.submission import Point, Submission
from inference_endpoint.validation.tree import SubmissionNode


def _point(tmp_path: Path, concurrency: int, offline=None, system_tps: float = 1.0):
    root = tmp_path / "acme"
    point_dir = root / "results" / "sys" / "model" / f"r{concurrency}"
    point_dir.mkdir(parents=True, exist_ok=True)
    tree = SubmissionNode.from_directory(root)
    point_node = next(
        node
        for node in tree.children[0].children[0].children
        if node.dir.name == f"r{concurrency}"
    )
    return Point(
        offline=offline,
        concurrency=concurrency,
        system_tps=system_tps,
        node=point_node,
    )


class TestOfflineOrdering:
    """`Submission.offline_ordering`: 5.7.2's pair of clauses, or None when n/a."""

    def _submission(self, tmp_path, offline_off=None, offline_tps=5292.0):
        system_desc = {"max_supported_concurrency": 8192}
        c_max_point = _point(tmp_path, 8192, system_tps=5400.0)
        points = [c_max_point]
        if offline_off is not None:
            points.append(
                _point(tmp_path, 8192, offline=offline_off, system_tps=offline_tps)
            )
        return Submission(system_desc, points)

    def test_dedicated_at_the_ratio_boundary_passes(self, tmp_path):
        assert (
            self._submission(tmp_path, offline_off="dedicated").offline_ordering()
            is True
        )

    def test_dedicated_below_the_ratio_fails(self, tmp_path):
        assert (
            self._submission(
                tmp_path, offline_off="dedicated", offline_tps=5291.0
            ).offline_ordering()
            is False
        )

    def test_dedicated_below_c_max_concurrency_fails(self, tmp_path):
        system_desc = {"max_supported_concurrency": 8192}
        submission = Submission(
            system_desc,
            [
                _point(tmp_path, 8192, system_tps=5400.0),
                _point(tmp_path, 512, offline="dedicated", system_tps=5292.0),
            ],
        )
        assert submission.offline_ordering() is False

    def test_elected_offline_is_not_applicable(self, tmp_path):
        assert (
            self._submission(tmp_path, offline_off="elected").offline_ordering() is None
        )

    def test_no_offline_declaration_is_not_applicable(self, tmp_path):
        assert self._submission(tmp_path).offline_ordering() is None

    def test_mixed_declarations_raise(self, tmp_path):
        system_desc = {"max_supported_concurrency": 8192}
        with pytest.raises(ValueError, match="offline"):
            Submission(
                system_desc,
                [
                    _point(tmp_path, 8192, offline="elected", system_tps=5400.0),
                    _point(tmp_path, 512, offline="dedicated", system_tps=5292.0),
                ],
            )

    def test_missing_c_max_point_raises(self, tmp_path):
        system_desc = {"max_supported_concurrency": 8192}
        with pytest.raises(ValueError, match="C_max"):
            Submission(
                system_desc,
                [_point(tmp_path, 512, offline="dedicated", system_tps=5292.0)],
            )


class TestPointCarry:
    """`Point` keeps the three Section 8.3 facts and its node link as declared."""

    def test_offline_declaration_three_states(self, tmp_path):
        assert _point(tmp_path, 32).offline is None
        assert _point(tmp_path, 32, offline="dedicated").offline == "dedicated"
        assert _point(tmp_path, 32, offline="elected").offline == "elected"


class TestPointCap:
    """`Submission.point_cap`: Section 5.6's "<=32 total points" cap, Offline counted."""

    def test_cap_boundary_32_passes(self, tmp_path):
        points = [_point(tmp_path, i + 1) for i in range(32)]
        submission = Submission({"max_supported_concurrency": 8192}, points)
        assert submission.point_cap() is True

    def test_point_33_fails(self, tmp_path):
        points = [_point(tmp_path, i + 1) for i in range(33)]
        submission = Submission({"max_supported_concurrency": 8192}, points)
        assert submission.point_cap() is False


class TestOutOfRangePoints:
    """`Submission.out_of_range_points`: the flagged Points per Section 5.4's row."""

    def test_points_above_the_margin_flag(self, tmp_path):
        regions = Regions.for_bounds(8192, 32)
        submission = Submission(
            {"max_supported_concurrency": 8192},
            [_point(tmp_path, 512), _point(tmp_path, 9013)],
        )
        assert submission.out_of_range_points(regions) == (submission.points[1],)

    def test_in_band_points_flag_nothing(self, tmp_path):
        regions = Regions.for_bounds(8192, 32)
        submission = Submission(
            {"max_supported_concurrency": 8192},
            [_point(tmp_path, 32), _point(tmp_path, 52)],
        )
        assert submission.out_of_range_points(regions) == ()

    def test_points_below_1_flag_too(self, tmp_path):
        regions = Regions.for_bounds(8192, 32)
        submission = Submission(
            {"max_supported_concurrency": 8192}, [_point(tmp_path, 0)]
        )
        assert submission.out_of_range_points(regions) == (submission.points[0],)


class TestPointPowerKw:
    """`Point.point_power_kw`: the Section 4.5.3 per-point denominator golden rows."""

    def _point(self, concurrency: int = 32):
        return Point(
            offline=None,
            concurrency=concurrency,
            system_tps=6000.0,
            node=None,
        )

    def test_golden_row_partial_engagement(self):
        assert self._point().point_power_kw(152.10, 16, 18) == 135.2

    def test_golden_row_full_engagement_reduces_to_provisioned(self):
        assert self._point().point_power_kw(152.10, 18, 18) == 152.1


class TestThresholdsPerShape:
    """`point_count_min` / `accuracy_run_count_requirement`: §5.3/§4.3-derived counts."""

    def test_dedicated_offline_wants_8_and_5(self, tmp_path):
        submission = Submission(
            {"max_supported_concurrency": 8192},
            [_point(tmp_path, 512, offline="dedicated"), _point(tmp_path, 8192)],
        )
        assert submission.point_count_min() == 8
        assert submission.accuracy_run_count_requirement() == 5

    def test_elected_offline_wants_7_and_5(self, tmp_path):
        submission = Submission(
            {"max_supported_concurrency": 8192},
            [_point(tmp_path, 512), _point(tmp_path, 8192, offline="elected")],
        )
        assert submission.point_count_min() == 7
        assert submission.accuracy_run_count_requirement() == 5

    def test_agentic_wants_7_and_4(self, tmp_path):
        submission = Submission(
            {"max_supported_concurrency": 8192}, [_point(tmp_path, 512)]
        )
        assert submission.point_count_min() == 7
        assert submission.accuracy_run_count_requirement() == 4


class TestDerivedExposure:
    """`Submission.derived`: the named derivations as `derived.<name>` fields."""

    def test_out_of_range_points_exposes_through_the_fold(self, tmp_path):
        submission = Submission(
            {"max_supported_concurrency": 8192},
            [_point(tmp_path, 512), _point(tmp_path, 9013)],
        )
        derived = submission.derived(regions=Regions.for_bounds(8192, 32))
        assert "out_of_range_points" in derived
        assert derived["out_of_range_points"] == (submission.points[1],)

    def test_region_derivation_without_regions_raises(self, tmp_path):
        submission = Submission(
            {"max_supported_concurrency": 8192}, [_point(tmp_path, 512)]
        )
        derived = submission.derived()
        with pytest.raises(ValueError, match="regions"):
            derived["out_of_range_points"]

    def test_unknown_derivation_is_absent(self, tmp_path):
        derived = Submission(
            {"max_supported_concurrency": 8192}, [_point(tmp_path, 512)]
        ).derived()
        with pytest.raises(KeyError):
            derived["nope"]

    def test_the_vocabulary_is_the_named_derivations(self, tmp_path):
        derived = Submission(
            {"max_supported_concurrency": 8192}, [_point(tmp_path, 512)]
        ).derived()
        assert set(derived) == {
            "offline_ordering",
            "point_cap",
            "point_count_min",
            "accuracy_run_count_requirement",
            "out_of_range_points",
        }


class TestFromDirectory:
    """`Submission.from_directory`: the typed view read from one uploaded tree."""

    def _upload(self, tmp_path: Path) -> Path:
        root = tmp_path / "upload"
        (root / "results" / "sys" / "model" / "r16").mkdir(parents=True)
        (root / "system_desc.json").write_text('{"max_supported_concurrency": 64}')
        return root

    def test_reads_the_facts_the_rows_read(self, tmp_path):
        root = self._upload(tmp_path)
        (root / "results" / "sys" / "model" / "r16" / "point.yaml").write_text(
            "concurrency: 16\noffline: none\nsystem_tps: 7.5\n"
        )
        submission = Submission.from_directory(root)
        assert submission.system_desc == {"max_supported_concurrency": 64}
        assert [
            (p.concurrency, p.offline, p.system_tps) for p in submission.points
        ] == [(16, None, 7.5)]

    def test_latches_the_regions_fold_onto_the_tree_root(self, tmp_path):
        root = self._upload(tmp_path)
        (root / "results" / "sys" / "model" / "r16" / "point.yaml").write_text(
            "concurrency: 16\nsystem_tps: 7.5\n"
        )
        node = Submission.from_directory(root).points[0].node
        assert node is not None
        assert node.root().regions is not None

    def test_a_point_without_system_tps_is_refused(self, tmp_path):
        root = self._upload(tmp_path)
        (root / "results" / "sys" / "model" / "r16" / "point.yaml").write_text(
            "concurrency: 16\n"
        )
        with pytest.raises(ValueError, match="system_tps"):
            Submission.from_directory(root)

    def test_a_missing_system_desc_is_refused(self, tmp_path):
        root = self._upload(tmp_path)
        (root / "system_desc.json").unlink()
        with pytest.raises(ValueError, match="system_desc.json"):
            Submission.from_directory(root)


class TestValidateShapeArms:
    """`Submission._validate_shape`: each refusal arm, matched to its own message."""

    @pytest.mark.parametrize("system_desc", [{}, {"max_supported_concurrency": "8192"}])
    def test_a_non_integer_max_supported_concurrency(self, tmp_path, system_desc):
        with pytest.raises(ValueError, match="integer max_supported_concurrency"):
            Submission(system_desc, [_point(tmp_path, 512)])

    def test_two_declarations_of_one_kind(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one point may carry"):
            Submission(
                {"max_supported_concurrency": 8192},
                [
                    _point(tmp_path, 512, offline="dedicated"),
                    _point(tmp_path, 8192, offline="dedicated"),
                ],
            )

    def test_two_c_max_rivals(self, tmp_path):
        with pytest.raises(ValueError, match="C_max"):
            Submission(
                {"max_supported_concurrency": 8192},
                [
                    _point(tmp_path, 512, offline="dedicated"),
                    _point(tmp_path, 8192),
                    _point(tmp_path, 8192, system_tps=2.0),
                ],
            )


class TestSubmissionCarry:
    """`Submission` pairs system fields with the declared points."""

    def test_regions_agree_with_the_trio_fields(self, tmp_path):
        system_desc = {"max_supported_concurrency": 8192}
        submission = Submission(
            system_desc, [_point(tmp_path, 32), _point(tmp_path, 40, system_tps=2.0)]
        )
        regions = Regions.for_submission(
            submission.system_desc,
            [p.concurrency for p in submission.points],
        )
        assert regions == Regions.for_bounds(8192, 32)
        assert submission.points[1].system_tps == 2.0
