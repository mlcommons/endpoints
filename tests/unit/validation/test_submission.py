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

"""Point/Pareto/Submission tests: the parsed facts and the named derivations.

Tests prefer `Submission.from_directory` over hand-built objects: each `r<N>`
run directory carries its `point.yaml`, `system_desc.json` and
`result_summary.json` (via `submission_helpers.write_upload`) and the suite
asserts what each subject computes on the walked upload. Hand-built `Point`/
`Pareto` appear only where a derivation needs facts no conforming upload can
declare (e.g. a concurrency below 1).
"""

from pathlib import Path

import pytest
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.submission import (
    DERIVATION_NAMES,
    OFFLINE_TPS_RATIO,
    Pareto,
    Point,
    Submission,
)
from submission_helpers import OMITTED, Run, write_run, write_upload

SYSTEM = ("sys", "model")


def _pareto(tmp_path: Path, runs: list[Run]) -> Pareto:
    """The one pareto an upload of `runs` under `tmp_path` derives."""
    return write_upload(tmp_path / "acme", {SYSTEM: runs}).paretos[0]


class TestPointFromNode:
    """`Point.from_node`: one run directory's named facts, enforced at load."""

    @pytest.mark.unit
    def test_point_facts_come_from_the_three_run_artifacts(self, tmp_path):
        submission = write_upload(
            tmp_path / "acme",
            {
                SYSTEM: [
                    Run(
                        concurrency=16,
                        system_tps=7.5,
                        max_supported_concurrency=64,
                    )
                ]
            },
        )
        point = submission.points[0]
        assert (point.concurrency, point.system_tps) == (16, 7.5)
        assert point.system_desc == {"max_supported_concurrency": 64}
        assert point.node is not None

    @pytest.mark.unit
    def test_the_offline_declaration_keeps_its_two_named_states_and_null(
        self, tmp_path
    ):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=24, offline="none"),
                Run(concurrency=512, offline="dedicated", system_tps=5292.0),
                Run(concurrency=8192, system_tps=5400.0),
            ],
        )
        by_concurrency = {point.concurrency: point.offline for point in pareto.points}
        assert by_concurrency == {
            16: None,
            24: None,  # the `none` spelling is "no declaration" (§8.3)
            512: "dedicated",
            8192: None,
        }

    @pytest.mark.unit
    def test_an_elected_offline_point_is_kept_as_declared(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16), Run(concurrency=8192, offline="elected")],
        )
        assert pareto.points[1].offline == "elected"

    @pytest.mark.unit
    def test_an_offline_token_outside_the_three_states_is_refused(self):
        with pytest.raises(ValueError, match="offline must be"):
            Point(offline="dedicatd", concurrency=512, system_tps=1.0, system_desc={})

    @pytest.mark.unit
    def test_an_offline_token_outside_the_three_states_is_refused_at_upload_load(
        self, tmp_path
    ):
        # Loader-minimum precedence: `Point`'s own domain check speaks at
        # `Submission.from_directory`, before any collect-time table value
        # checking could judge `offline`.
        with pytest.raises(ValueError, match="offline must be"):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, offline="maybe")]},
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("system_tps", [None, OMITTED, "nan", True])
    def test_a_result_summary_without_numeric_system_tps_is_refused(
        self, tmp_path, system_tps
    ):
        with pytest.raises(ValueError, match="needs a numeric system_tps"):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, system_tps=system_tps)]},
            )

    @pytest.mark.unit
    def test_a_stringified_system_tps_reads_as_its_number(self, tmp_path):
        submission = write_upload(
            tmp_path / "acme",
            {SYSTEM: [Run(concurrency=16, system_tps="7.5")]},
        )
        assert [point.system_tps for point in submission.points] == [7.5]

    @pytest.mark.unit
    def test_a_stringified_concurrency_reads_as_its_number(self, tmp_path):
        # Writers stringify numbers: the parser decodes them the way the ref
        # table reads its declared leaves (`refs.decode_number`).
        submission = write_upload(
            tmp_path / "acme",
            {SYSTEM: [Run(concurrency="16", dir_name="r16")]},
        )
        assert [point.concurrency for point in submission.points] == [16]

    @pytest.mark.unit
    def test_a_concurrency_that_spells_no_number_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="needs an integer concurrency"):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency="sixteen", dir_name="r16")]},
            )

    @pytest.mark.unit
    def test_a_system_desc_without_max_supported_concurrency_is_refused(self, tmp_path):
        with pytest.raises(
            ValueError, match="needs an integer max_supported_concurrency"
        ):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, max_supported_concurrency=OMITTED)]},
            )

    @pytest.mark.unit
    def test_a_run_without_a_result_summary_file_is_refused(self, tmp_path):
        run_dir = tmp_path / "acme" / "results" / "sys" / "model" / "r16"
        write_run(run_dir, Run(concurrency=16))
        (run_dir / "result_summary.json").unlink()
        with pytest.raises(ValueError, match="cannot carry its facts"):
            write_upload(tmp_path / "acme", {SYSTEM: []})


class TestParetoShape:
    """`Pareto._validate_shape`'s refusal arms (Endpoints Policy §9.1 rows)."""

    @pytest.mark.unit
    def test_two_offline_declarations_are_refused(self, tmp_path):
        runs = [
            Run(concurrency=16, offline="dedicated", system_tps=5292.0),
            Run(concurrency=8192, offline="elected"),
        ]
        with pytest.raises(ValueError, match="exactly one point may carry"):
            write_upload(tmp_path / "acme", {SYSTEM: runs})

    @pytest.mark.unit
    def test_a_dedicated_run_needs_exactly_one_c_max_point(self, tmp_path):
        runs = [
            Run(concurrency=16, offline="dedicated", system_tps=5292.0),
            Run(concurrency=512),
        ]
        with pytest.raises(ValueError, match="needs exactly one C_max point"):
            write_upload(tmp_path / "acme", {SYSTEM: runs})

    @pytest.mark.unit
    def test_two_c_max_points_beside_a_dedicated_run_are_refused(self, tmp_path):
        runs = [
            Run(concurrency=16, offline="dedicated", system_tps=5292.0),
            Run(concurrency=8192, system_tps=5400.0),
            Run(concurrency=8192, dir_name="r8193", system_tps=5200.0),
        ]
        with pytest.raises(ValueError, match="needs exactly one C_max point"):
            write_upload(tmp_path / "acme", {SYSTEM: runs})

    @pytest.mark.unit
    def test_an_elected_offline_point_not_at_c_max_is_refused(self, tmp_path):
        runs = [
            Run(concurrency=16),
            Run(concurrency=512, offline="elected"),
            Run(concurrency=8192, system_tps=5400.0),
        ]
        with pytest.raises(ValueError, match="Offline point present"):
            write_upload(tmp_path / "acme", {SYSTEM: runs})

    @pytest.mark.unit
    def test_the_points_must_agree_on_max_supported_concurrency(self, tmp_path):
        runs = [
            Run(concurrency=16, max_supported_concurrency=8192),
            Run(concurrency=8192, max_supported_concurrency=4096),
        ]
        with pytest.raises(ValueError, match="must declare the same"):
            write_upload(tmp_path / "acme", {SYSTEM: runs})

    @pytest.mark.unit
    def test_the_points_must_agree_on_a_declared_c_min_override(self, tmp_path):
        runs = [
            Run(concurrency=16, c_min_override=16),
            Run(concurrency=8192),
        ]
        with pytest.raises(ValueError, match="C_min override"):
            write_upload(tmp_path / "acme", {SYSTEM: runs})

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "system_desc",
        [
            {},
            {"max_supported_concurrency": "8192"},
            {"max_supported_concurrency": True},
        ],
    )
    def test_a_pareto_needs_one_integer_max_supported_concurrency(self, system_desc):
        point = Point(
            offline=None, concurrency=16, system_tps=1.0, system_desc=system_desc
        )
        with pytest.raises(ValueError, match="one integer max_supported_concurrency"):
            Pareto(system_desc, [point])


class TestFromDirectory:
    """`Submission.from_directory`: the typed view read from one uploaded tree."""

    @pytest.mark.unit
    def test_one_pareto_is_walked_per_system_model_node(self, tmp_path):
        submission = write_upload(
            tmp_path / "acme",
            {
                ("sys", "modelA"): [Run(concurrency=16), Run(concurrency=8192)],
                ("sys", "modelB"): [Run(concurrency=24)],
            },
        )
        assert [p.node.dir.name for p in submission.paretos] == ["modelA", "modelB"]
        assert [point.concurrency for point in submission.points] == [16, 8192, 24]

    @pytest.mark.unit
    def test_pareto_at_locates_the_pareto_of_a_node_only(self, tmp_path):
        submission = write_upload(
            tmp_path / "acme",
            {
                ("sys", "modelA"): [Run(concurrency=16)],
                ("sys", "modelB"): [Run(concurrency=24)],
            },
        )
        first, second = submission.paretos
        assert submission.pareto_at(first.node) is first
        assert submission.pareto_at(first.points[0].node) is first
        assert submission.pareto_at(second.points[0].node) is second
        assert submission.pareto_at(submission.tree) is None

    @pytest.mark.unit
    def test_the_clients_tps_loads_as_the_required_system_tps(self, tmp_path):
        submission = write_upload(
            tmp_path / "acme",
            {SYSTEM: [Run(system_tps=OMITTED, summary_extra={"tps": 6000.0})]},
        )
        assert submission.points[0].system_tps == 6000.0

    @pytest.mark.unit
    def test_a_null_client_tps_is_still_refused(self, tmp_path):
        # A run that recorded no OSL writes `tps: null`; the policy metric is
        # not there to read, so the load still refuses.
        with pytest.raises(ValueError, match="needs a numeric system_tps"):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(system_tps=OMITTED, summary_extra={"tps": None})]},
            )

    @pytest.mark.unit
    def test_a_pareto_without_run_directories_is_refused(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results" / "sys" / "model").mkdir(parents=True)
        with pytest.raises(ValueError, match="has no run directories"):
            write_upload(root, {})


class TestMultiPareto:
    """One upload, many paretos: each keeps its own declarations and derivations."""

    @pytest.mark.unit
    def test_each_pareto_keeps_one_dedicated_run_and_counts_its_own_points(
        self, tmp_path
    ):
        # 35 run directories across the two modelA/modelB paretos: the two
        # dedicated Offline runs never trip the one-declaration arm across
        # paretos, and the point cap judges each pareto on its own count.
        model_a = [Run(concurrency=i, system_tps=float(i)) for i in range(1, 32)] + [
            Run(concurrency=8192, system_tps=5400.0),
            Run(concurrency=200000, offline="dedicated", system_tps=5291.0),
        ]
        model_b = [
            Run(concurrency=16),
            Run(concurrency=8192, system_tps=5400.0),
            Run(concurrency=200000, offline="dedicated", system_tps=5292.0),
        ]
        submission = write_upload(
            tmp_path / "acme",
            {("sys", "modelA"): model_a, ("sys", "modelB"): model_b},
        )
        pareto_a, pareto_b = submission.paretos
        assert len(submission.points) == 36
        assert pareto_a.point_cap() is False
        assert pareto_b.point_cap() is True
        assert pareto_a.offline_ordering() is False
        assert pareto_b.offline_ordering() is True


class TestOfflineOrdering:
    """`Pareto.offline_ordering`: 5.7.2's pair of clauses, or None when n/a."""

    @pytest.mark.unit
    def test_dedicated_at_the_ratio_boundary_passes(self, tmp_path):
        assert OFFLINE_TPS_RATIO == 0.98
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=8192, system_tps=5400.0),
                Run(concurrency=8193, offline="dedicated", system_tps=5292.0),
            ],
        )
        assert pareto.offline_ordering() is True

    @pytest.mark.unit
    def test_dedicated_below_the_ratio_fails(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=8192, system_tps=5400.0),
                Run(concurrency=8193, offline="dedicated", system_tps=5291.0),
            ],
        )
        assert pareto.offline_ordering() is False

    @pytest.mark.unit
    def test_dedicated_below_c_max_concurrency_fails(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=512, offline="dedicated", system_tps=5292.0),
                Run(concurrency=8192, system_tps=5400.0),
            ],
        )
        assert pareto.offline_ordering() is False

    @pytest.mark.unit
    def test_elected_offline_is_not_applicable(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16), Run(concurrency=8192, offline="elected")],
        )
        assert pareto.offline_ordering() is None

    @pytest.mark.unit
    def test_no_offline_declaration_is_not_applicable(self, tmp_path):
        pareto = _pareto(tmp_path, [Run(concurrency=16), Run(concurrency=8192)])
        assert pareto.offline_ordering() is None


class TestPointCap:
    """`Pareto.point_cap`: Section 5.6's "<= 32 points per pareto" cap."""

    @pytest.mark.unit
    def test_cap_boundary_32_passes(self, tmp_path):
        pareto = _pareto(tmp_path, [Run(concurrency=i) for i in range(1, 33)])
        assert pareto.point_cap() is True

    @pytest.mark.unit
    def test_point_33_fails(self, tmp_path):
        pareto = _pareto(tmp_path, [Run(concurrency=i) for i in range(1, 34)])
        assert pareto.point_cap() is False


class TestOutOfRangePoints:
    """`Pareto.out_of_range_points`: §9.1's Concurrency-in-range row per point."""

    @pytest.mark.unit
    def test_a_dedicated_offline_run_is_exempt_but_a_fixed_point_above_the_margin_is_flagged(
        self, tmp_path
    ):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=8192, system_tps=5400.0),
                Run(concurrency=9013),
                Run(concurrency=200000, offline="dedicated", system_tps=5292.0),
            ],
        )
        flagged = pareto.out_of_range_points(pareto.regions())
        assert tuple(point.concurrency for point in flagged) == (9013,)

    @pytest.mark.unit
    def test_in_band_points_flag_nothing(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16), Run(concurrency=40), Run(concurrency=8192)],
        )
        assert pareto.out_of_range_points(pareto.regions()) == ()

    @pytest.mark.unit
    def test_points_below_one_flag_too(self, tmp_path):
        # No conforming upload declares a concurrency below 1 (C_min would not
        # derive); the derivation itself still judges one out of range.
        point = Point(
            offline=None,
            concurrency=0,
            system_tps=1.0,
            system_desc={"max_supported_concurrency": 8192},
        )
        pareto = Pareto({"max_supported_concurrency": 8192}, [point])
        flagged = pareto.out_of_range_points(Regions.for_bounds(c_min=32, c_max=8192))
        assert flagged == (point,)

    @pytest.mark.unit
    def test_the_c_min_override_still_covers_the_points_below_it(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16, c_min_override=32),
                Run(concurrency=32, c_min_override=32),
                Run(concurrency=64, c_min_override=32),
                Run(concurrency=8192, c_min_override=32, system_tps=5400.0),
            ],
        )
        regions = pareto.regions()
        assert regions == Regions.for_bounds(c_min=32, c_max=8192)
        assert regions.low_concurrency.lo == 33
        assert 16 in regions.coverage()
        assert (
            Regions.for_pareto(
                {"max_supported_concurrency": 8192, "c_min_override": 32},
                [16, 32, 64, 8192],
            )
            == regions
        )
        assert pareto.out_of_range_points(regions) == ()


class TestThresholdsPerShape:
    """`point_count_min` / `accuracy_run_count_requirement`: §5.3/§4.3 counts."""

    @pytest.mark.unit
    def test_dedicated_offline_wants_8_and_5(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=512, offline="dedicated", system_tps=5292.0),
                Run(concurrency=8192, system_tps=5400.0),
            ],
        )
        assert pareto.point_count_min() == 8
        assert pareto.accuracy_run_count_requirement() == 5

    @pytest.mark.unit
    def test_elected_offline_wants_7_and_5(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16), Run(concurrency=8192, offline="elected")],
        )
        assert pareto.point_count_min() == 7
        assert pareto.accuracy_run_count_requirement() == 5

    @pytest.mark.unit
    def test_agentic_wants_7_and_4(self, tmp_path):
        pareto = _pareto(tmp_path, [Run(concurrency=16), Run(concurrency=8192)])
        assert pareto.point_count_min() == 7
        assert pareto.accuracy_run_count_requirement() == 4


class TestDerivedExposure:
    """`Pareto.derived`: the `derived.<name>` fields, resolved from the pareto itself."""

    @pytest.mark.unit
    def test_the_derivation_names_are_the_five_policy_derivations(self):
        assert set(DERIVATION_NAMES) == {
            "offline_ordering",
            "point_cap",
            "point_count_min",
            "accuracy_run_count_requirement",
            "out_of_range_points",
        }

    @pytest.mark.unit
    def test_every_named_derivation_resolves_for_one_pareto(self, tmp_path):
        pareto = _pareto(tmp_path, [Run(concurrency=16), Run(concurrency=8192)])
        derived = pareto.derived()
        assert set(derived) == set(DERIVATION_NAMES)
        assert derived["point_count_min"] == 7
        assert derived["out_of_range_points"] == ()

    @pytest.mark.unit
    def test_out_of_range_points_derivation_uses_the_paretos_own_regions(
        self, tmp_path
    ):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16), Run(concurrency=9013), Run(concurrency=8192)],
        )
        derived = pareto.derived()
        assert "out_of_range_points" in derived
        assert tuple(point.concurrency for point in derived["out_of_range_points"]) == (
            9013,
        )

    @pytest.mark.unit
    def test_unknown_derivation_is_absent(self, tmp_path):
        derived = _pareto(tmp_path, [Run(concurrency=16)]).derived()
        with pytest.raises(KeyError, match="nope"):
            _ = derived["nope"]


class TestPointPowerKw:
    """`Point.point_power_kw`: the Section 4.5.3 per-point denominator golden rows."""

    def _point(self) -> Point:
        return Point(
            offline=None,
            concurrency=32,
            system_tps=6000.0,
            system_desc={},
        )

    @pytest.mark.unit
    def test_golden_row_partial_engagement(self):
        assert self._point().point_power_kw(152.10, 16, 18) == 135.2

    @pytest.mark.unit
    def test_golden_row_full_engagement_reduces_to_provisioned(self):
        assert self._point().point_power_kw(152.10, 18, 18) == 152.1


class TestNumericGuards:
    """Upload-controlled numbers no arithmetic can carry are refused at load."""

    @pytest.mark.unit
    def test_a_non_finite_system_tps_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="needs a finite value"):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, system_tps=float("inf"))]},
            )

    @pytest.mark.unit
    def test_a_system_tps_outside_float_range_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="outside float range"):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, system_tps=10**400)]},
            )

    @pytest.mark.unit
    def test_a_c_max_outside_float_range_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="outside float range"):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, max_supported_concurrency=10**400)]},
            )

    @pytest.mark.unit
    def test_a_concurrency_outside_float_range_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="outside float range"):
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=10**400, dir_name="r16")]},
            )


class TestSharedFactsAgreement:
    """Every point's descriptor is compared by presence and typed value."""

    @pytest.mark.unit
    def test_an_explicit_null_c_min_override_does_not_agree_with_an_omission(
        self, tmp_path
    ):
        # Run 2's `c_min_override: null` is §5.4's invalid declaration; a
        # `get`-based comparison once let it read equal to run 1's omission.
        runs = [Run(concurrency=16), Run(concurrency=8192, c_min_override=None)]
        with pytest.raises(ValueError, match="must declare the same c_min_override"):
            write_upload(tmp_path / "acme", {SYSTEM: runs})

    @pytest.mark.unit
    def test_an_int_and_a_float_c_min_override_do_not_agree(self, tmp_path):
        runs = [
            Run(concurrency=16, c_min_override=16),
            Run(concurrency=32, c_min_override=16.0),
            Run(concurrency=8192),
        ]
        with pytest.raises(ValueError, match="must declare the same c_min_override"):
            write_upload(tmp_path / "acme", {SYSTEM: runs})


class TestEmptySystems:
    """A system directory with no model directories is refused at load."""

    @pytest.mark.unit
    def test_a_system_without_model_directories_is_refused(self, tmp_path):
        root = tmp_path / "acme"
        write_run(root / "results" / "sys" / "model" / "r16", Run(concurrency=16))
        (root / "results" / "empty_sys").mkdir(parents=True)
        with pytest.raises(ValueError, match="has no model directories"):
            Submission.from_directory(root)
