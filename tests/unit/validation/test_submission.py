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

import json
from pathlib import Path

import pytest
from inference_endpoint.validation.catalog import Catalog
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.submission import (
    DERIVATION_NAMES,
    OFFLINE_TPS_RATIO,
    Pareto,
    Point,
    Submission,
)
from submission_helpers import OMITTED, Run, refused_messages, write_run, write_upload

SYSTEM = ("sys", "model")


def _pareto(tmp_path: Path, runs: list[Run]) -> Pareto:
    """The one pareto an upload of `runs` under `tmp_path` derives."""
    return write_upload(tmp_path / "acme", {SYSTEM: runs}).paretos[0]


def _gated_pareto(tmp_path: Path, *, score: float, unit_samples: int = 4395) -> Pareto:
    """A one-point gpt-oss-120b pareto whose run ships one accuracy entry."""
    run = Run(concurrency=16, config_extra={"model_name": "gpt-oss-120b"})
    run_dir = tmp_path / "upload" / "results" / SYSTEM[0] / SYSTEM[1] / run.name()
    write_run(run_dir, run)
    # Written before the walk, so the run's node carries it.
    (run_dir / "accuracy_results.json").write_text(
        json.dumps(
            {
                "accuracy_scores": [
                    {
                        "dataset_name": "mlperf_gpt_oss_accuracy",
                        "score": score,
                        "unit_samples": unit_samples,
                        "total_samples": unit_samples,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    catalog = Catalog(
        models={
            "gpt-oss-120b": {
                "accuracy_targets": {"exact_match": {"lower": 83.13, "factor": 0.99}},
                "min_queries": 4395,
            }
        },
        datasets={},
    )
    return write_upload(tmp_path / "upload", {SYSTEM: [run]}, ruleset=catalog).paretos[
        0
    ]


def _pareto_with_accuracy(
    tmp_path: Path, runs: list[Run], carrying: set[int]
) -> Pareto:
    """A pareto whose `carrying`-concurrency runs also ship an accuracy result."""
    root = tmp_path / "upload"
    for run in runs:
        run_dir = root / "results" / SYSTEM[0] / SYSTEM[1] / run.name()
        write_run(run_dir, run)
        if run.concurrency in carrying:
            (run_dir / "accuracy_results.json").write_text(
                json.dumps(
                    {
                        "accuracy_scores": [
                            {
                                "dataset_name": "cnn_dailymail",
                                "score": 0.9,
                                "unit_samples": 13368,
                                "total_samples": 13368,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
    return write_upload(root, {SYSTEM: runs}).paretos[0]


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
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, offline="maybe")]},
            )
        )
        assert any("offline must be" in message for message in refused)

    @pytest.mark.unit
    @pytest.mark.parametrize("system_tps", [None, OMITTED, "nan", True])
    def test_a_result_summary_without_numeric_system_tps_is_refused(
        self, tmp_path, system_tps
    ):
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, system_tps=system_tps)]},
            )
        )
        assert any("needs a numeric system_tps" in message for message in refused)

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
    def test_a_stringified_maximum_is_carried_as_its_number(self, tmp_path):
        # The pareto shape check and the region bounds read the carried
        # descriptor, so the decoded maximum has to ride on `Point`.
        submission = write_upload(
            tmp_path / "acme",
            {SYSTEM: [Run(concurrency=16, max_supported_concurrency="8192")]},
        )
        assert submission.paretos[0].system_desc["max_supported_concurrency"] == 8192

    @pytest.mark.unit
    def test_a_stringified_override_is_carried_as_its_number(self, tmp_path):
        submission = write_upload(
            tmp_path / "acme",
            {SYSTEM: [Run(concurrency=16, c_min_override="16")]},
        )
        assert submission.paretos[0].system_desc["c_min_override"] == 16

    @pytest.mark.unit
    def test_a_concurrency_that_spells_no_number_is_refused(self, tmp_path):
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency="sixteen", dir_name="r16")]},
            )
        )
        assert any("needs an integer concurrency" in message for message in refused)

    @pytest.mark.unit
    def test_a_system_desc_without_max_supported_concurrency_is_refused(self, tmp_path):
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, max_supported_concurrency=OMITTED)]},
            )
        )
        assert any(
            "needs an integer max_supported_concurrency" in message
            for message in refused
        )

    @pytest.mark.unit
    def test_a_run_without_a_result_summary_file_is_refused(self, tmp_path):
        run_dir = tmp_path / "acme" / "results" / "sys" / "model" / "r16"
        write_run(run_dir, Run(concurrency=16))
        (run_dir / "result_summary.json").unlink()
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: []}))
        assert any("cannot carry its facts" in message for message in refused)


class TestParetoShape:
    """`Pareto._validate_shape`'s refusal arms (Endpoints Policy §9.1 rows)."""

    @pytest.mark.unit
    def test_two_offline_declarations_are_refused(self, tmp_path):
        runs = [
            Run(concurrency=16, offline="dedicated", system_tps=5292.0),
            Run(concurrency=8192, offline="elected"),
        ]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any("exactly one point may carry" in message for message in refused)

    @pytest.mark.unit
    def test_a_dedicated_run_needs_exactly_one_c_max_point(self, tmp_path):
        runs = [
            Run(concurrency=16, offline="dedicated", system_tps=5292.0),
            Run(concurrency=512),
        ]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any("needs exactly one C_max point" in message for message in refused)

    @pytest.mark.unit
    def test_two_c_max_points_beside_a_dedicated_run_are_refused(self, tmp_path):
        runs = [
            Run(concurrency=16, offline="dedicated", system_tps=5292.0),
            Run(concurrency=8192, system_tps=5400.0),
            Run(concurrency=8192, dir_name="r8193", system_tps=5200.0),
        ]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any("needs exactly one C_max point" in message for message in refused)

    @pytest.mark.unit
    def test_two_points_with_the_same_concurrency_are_refused(self, tmp_path):
        runs = [
            Run(concurrency=16),
            Run(concurrency=16, dir_name="r016", system_tps=5400.0),
        ]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any("must be distinct" in message for message in refused)

    @pytest.mark.unit
    def test_a_dedicated_offline_run_may_share_a_concurrency(self, tmp_path):
        runs = [
            Run(concurrency=16, offline="dedicated", system_tps=5292.0),
            Run(concurrency=16, dir_name="r016"),
            Run(concurrency=8192, system_tps=5400.0),
        ]
        pareto = _pareto(tmp_path, runs)
        assert pareto.point_count() == 3

    @pytest.mark.unit
    def test_an_elected_offline_point_not_at_c_max_is_refused(self, tmp_path):
        runs = [
            Run(concurrency=16),
            Run(concurrency=512, offline="elected"),
            Run(concurrency=8192, system_tps=5400.0),
        ]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any("Offline point present" in message for message in refused)

    @pytest.mark.unit
    def test_the_points_must_agree_on_max_supported_concurrency(self, tmp_path):
        runs = [
            Run(concurrency=16, max_supported_concurrency=8192),
            Run(concurrency=8192, max_supported_concurrency=4096),
        ]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any("must declare the same" in message for message in refused)

    @pytest.mark.unit
    def test_the_points_must_agree_on_a_declared_c_min_override(self, tmp_path):
        runs = [
            Run(concurrency=16, c_min_override=16),
            Run(concurrency=8192),
        ]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any("C_min override" in message for message in refused)

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
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(system_tps=OMITTED, summary_extra={"tps": None})]},
            )
        )
        assert any("needs a numeric system_tps" in message for message in refused)

    @pytest.mark.unit
    def test_a_pareto_without_run_directories_is_refused(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results" / "sys" / "model").mkdir(parents=True)
        refused = refused_messages(write_upload(root, {}))
        assert any("has no run directories" in message for message in refused)


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
    """`point_count_min`: Section 5.3's minimum point count per shape."""

    @pytest.mark.unit
    def test_dedicated_offline_wants_8(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=512, offline="dedicated", system_tps=5292.0),
                Run(concurrency=8192, system_tps=5400.0),
            ],
        )
        assert pareto.point_count_min() == 8

    @pytest.mark.unit
    def test_elected_offline_wants_7(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16), Run(concurrency=8192, offline="elected")],
        )
        assert pareto.point_count_min() == 7

    @pytest.mark.unit
    def test_agentic_wants_7(self, tmp_path):
        pareto = _pareto(tmp_path, [Run(concurrency=16), Run(concurrency=8192)])
        assert pareto.point_count_min() == 7

    @pytest.mark.unit
    def test_a_cataloged_non_agentic_model_without_offline_reads_8(self, tmp_path):
        # The declared model keys the catalog's flag, so a missing Offline
        # declaration cannot silently re-read the pareto as agentic.
        upload = write_upload(
            tmp_path / "acme",
            {
                SYSTEM: [
                    Run(
                        concurrency=16,
                        config_extra={"model_name": "gpt-oss-120b"},
                    )
                ]
            },
            ruleset=Catalog(models={"gpt-oss-120b": {"agentic": False}}, datasets={}),
        )
        pareto = upload.paretos[0]
        assert pareto.agentic() is False
        assert pareto.point_count_min() == 8

    @pytest.mark.unit
    def test_a_cataloged_agentic_model_reads_7(self, tmp_path):
        upload = write_upload(
            tmp_path / "acme",
            {SYSTEM: [Run(concurrency=16, config_extra={"model_name": "kimi-k3"})]},
            ruleset=Catalog(models={"kimi-k3": {"agentic": True}}, datasets={}),
        )
        pareto = upload.paretos[0]
        assert pareto.agentic() is True
        assert pareto.point_count_min() == 7

    @pytest.mark.unit
    def test_a_cataloged_model_name_is_admitted(self, tmp_path):
        upload = write_upload(
            tmp_path / "acme",
            {SYSTEM: [Run(concurrency=16, config_extra={"model_name": "kimi-k3"})]},
            ruleset=Catalog(models={"kimi-k3": {"agentic": True}}, datasets={}),
        )
        assert upload.paretos[0].model_name_admitted() is True

    @pytest.mark.unit
    def test_a_model_name_outside_the_catalog_is_not_admitted(self, tmp_path):
        upload = write_upload(
            tmp_path / "acme",
            {
                SYSTEM: [
                    Run(
                        concurrency=16,
                        config_extra={"model_name": "qwen3-coder-480b"},
                    )
                ]
            },
            ruleset=Catalog(models={"kimi-k3": {"agentic": True}}, datasets={}),
        )
        assert upload.paretos[0].model_name_admitted() is False

    @pytest.mark.unit
    def test_an_unknown_model_falls_back_to_the_declared_shape(self, tmp_path):
        upload = write_upload(
            tmp_path / "acme",
            {
                SYSTEM: [
                    Run(concurrency=16),
                    Run(
                        concurrency=512,
                        offline="dedicated",
                        config_extra={"model_name": "unlisted-model"},
                    ),
                    Run(concurrency=8192),
                ]
            },
            ruleset=Catalog(models={"other": {"agentic": True}}, datasets={}),
        )
        pareto = upload.paretos[0]
        assert pareto.agentic() is False
        assert pareto.point_count_min() == 8


class TestCoverageAndCounts:
    """`uncovered_regions` / `offline_points` / the accuracy coverage facts."""

    @pytest.mark.unit
    def test_every_mandatory_region_can_be_covered(self, tmp_path):
        # c_min 16, c_max 8192: Low is 17..36, Medium is 37..422.
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=24),
                Run(concurrency=100),
                Run(concurrency=8192, system_tps=5400.0),
            ],
        )
        assert pareto.uncovered_regions() == ()

    @pytest.mark.unit
    def test_a_region_with_no_point_is_named(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16), Run(concurrency=100), Run(concurrency=8192)],
        )
        assert pareto.uncovered_regions() == ("low_concurrency",)

    @pytest.mark.unit
    def test_a_dedicated_offline_point_covers_nothing(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(concurrency=24),
                Run(concurrency=8192, system_tps=5400.0),
                Run(concurrency=50, offline="dedicated"),
            ],
        )
        # 50 sits in Medium, but §5.7.2 exempts the dedicated run from coverage.
        assert pareto.uncovered_regions() == ("med_concurrency",)

    @pytest.mark.unit
    def test_a_curve_above_the_ultra_low_region_names_both_low_bands(self, tmp_path):
        # Every point above 32: C_min clamps to 32 (the round's checker reads
        # `min(lowest, 32)`), so Ultra Low (`low_latency` on the wire) and Low
        # are both uncovered -- the checker's own two coverage errors.
        pareto = _pareto(tmp_path, [Run(concurrency=64), Run(concurrency=8192)])
        assert pareto.uncovered_regions() == ("low_latency", "low_concurrency")

    @pytest.mark.unit
    def test_offline_points_lists_the_declaration(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16), Run(concurrency=8192, offline="elected")],
        )
        assert tuple(p.offline for p in pareto.offline_points()) == ("elected",)

    @pytest.mark.unit
    def test_accuracy_coverage_counts_only_carrying_points(self, tmp_path):
        root = tmp_path / "acme"
        # c_min 16, c_max 8192: Ultra Low is 16, Low is 17..36. The Low point
        # carries no accuracy result, so the curve covers Low and its
        # validation runs do not.
        for name, carries in (
            ("r16", True),
            ("r24", False),
            ("r100", True),
            ("r8192", True),
        ):
            run_dir = write_run(
                root / "results" / "sys" / "model" / name,
                Run(concurrency=int(name[1:])),
            )
            if carries:
                (run_dir / "accuracy_results.json").write_text(
                    '{"accuracy_scores": [{"score": 1.0}]}', encoding="utf-8"
                )
        pareto = Submission.from_directory(root).paretos[0]
        assert pareto.uncovered_regions() == ()
        assert pareto.accuracy_uncovered_regions() == ("low_concurrency",)

    @pytest.mark.unit
    def test_an_offline_point_without_an_accuracy_result_is_named(self, tmp_path):
        root = tmp_path / "acme"
        run_dir = write_run(
            root / "results" / "sys" / "model" / "r8192",
            Run(concurrency=8192, offline="elected"),
        )
        write_run(root / "results" / "sys" / "model" / "r16", Run(concurrency=16))
        pareto = Submission.from_directory(root).paretos[0]
        assert pareto.offline_points_missing_accuracy() == (8192,)
        (run_dir / "accuracy_results.json").write_text(
            '{"accuracy_scores": [{"score": 1.0}]}', encoding="utf-8"
        )
        walked = Submission.from_directory(root).paretos[0]
        assert walked.offline_points_missing_accuracy() == ()


class TestCurveConsistency:
    """The per-curve agreement derivations (Sections 8.5/9.1)."""

    @pytest.mark.unit
    def test_a_dedicated_offline_run_is_left_out_of_the_type_vote(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    config_extra={"runtime_settings": {"load_pattern": "concurrency"}},
                ),
                Run(
                    concurrency=50,
                    offline="dedicated",
                    config_extra={
                        "runtime_settings": {"load_pattern": "max_throughput"}
                    },
                ),
                Run(
                    concurrency=8192,
                    config_extra={"runtime_settings": {"load_pattern": "concurrency"}},
                ),
            ],
        )
        assert pareto.conflicting_load_patterns() == ()

    @pytest.mark.unit
    def test_disagreeing_fixed_points_name_their_patterns(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    config_extra={"runtime_settings": {"load_pattern": "concurrency"}},
                ),
                Run(
                    concurrency=32,
                    config_extra={
                        "runtime_settings": {"load_pattern": "agentic_inference"}
                    },
                ),
                Run(
                    concurrency=8192,
                    config_extra={"runtime_settings": {"load_pattern": "concurrency"}},
                ),
            ],
        )
        assert pareto.conflicting_load_patterns() == (
            "agentic_inference",
            "concurrency",
        )

    @pytest.mark.unit
    def test_disagreeing_datasets_include_the_omitted_default(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16, config_extra={"dataset": "mlperf-perf-dataset-v1"}),
                Run(concurrency=8192),
            ],
        )
        assert pareto.conflicting_datasets() == ("", "mlperf-perf-dataset-v1")

    @pytest.mark.unit
    def test_disagreeing_seed_sets_skip_the_absent_declarations(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16, config_extra={"seed_set": "A"}),
                Run(concurrency=32, config_extra={"seed_set": "B"}),
                Run(concurrency=8192),
            ],
        )
        assert pareto.conflicting_seed_sets() == ("A", "B")

    @pytest.mark.unit
    def test_only_the_identity_fields_make_a_description_inconsistent(self, tmp_path):
        identity = {"node_sets": [{"system_node_ensemble_id": 0, "nodes": 1}]}
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16, system_desc_extra={"cooling": "air", **identity}),
                Run(
                    concurrency=8192,
                    system_desc_extra={"cooling": "liquid", **identity},
                ),
            ],
        )
        assert pareto.inconsistent_system_desc_keys() == ("cooling",)
        per_point = _pareto(
            tmp_path / "second",
            [
                Run(concurrency=16, system_desc_extra={"batch": 1, **identity}),
                Run(concurrency=8192, system_desc_extra={"batch": 8, **identity}),
            ],
        )
        assert per_point.inconsistent_system_desc_keys() == ()

    @pytest.mark.unit
    def test_mismatched_sample_counts_are_named(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    summary_extra={"n_samples_issued": 10, "n_samples_completed": 10},
                ),
                Run(
                    concurrency=64,
                    summary_extra={"n_samples_issued": 10, "n_samples_completed": 9},
                ),
            ],
        )
        assert pareto.accounting_mismatch_points() == (64,)

    @pytest.mark.unit
    def test_consistent_counts_and_throughput_hold(self, tmp_path):
        summary = {
            "n_samples_issued": 10,
            "n_samples_completed": 10,
            "n_samples_failed": 0,
            "duration_ns": 2_000_000_000,
            "output_sequence_lengths": {"total": 2000},
        }
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16, system_tps=1000.0, summary_extra=summary)],
        )
        assert pareto.accounting_mismatch_points() == ()
        assert pareto.system_tps_mismatch_points() == ()

    @pytest.mark.unit
    def test_a_stored_throughput_off_its_derivation_is_named(self, tmp_path):
        summary = {
            "duration_ns": 2_000_000_000,
            "output_sequence_lengths": {"total": 2000},
        }
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16, system_tps=1000.0, summary_extra=summary),
                Run(concurrency=64, system_tps=500.0, summary_extra=summary),
            ],
        )
        assert pareto.system_tps_mismatch_points() == (64,)

    @pytest.mark.unit
    def test_a_declared_region_off_its_band_is_named(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16, config_extra={"region": "low_latency"}),
                Run(concurrency=50, config_extra={"region": "low_latency"}),
                Run(concurrency=64, config_extra={"region": "submitters_choice"}),
            ],
        )
        assert pareto.misplaced_region_points() == (50,)

    @pytest.mark.unit
    def test_a_declared_utilization_off_its_share_is_named(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    system_tps=1000.0,
                    summary_extra={
                        "duration_ns": 2_000_000_000,
                        "output_sequence_lengths": {"total": 2000},
                    },
                    system_desc_extra={"tps_utilization": 1.0},
                ),
                Run(
                    concurrency=64,
                    system_tps=500.0,
                    summary_extra={
                        "duration_ns": 2_000_000_000,
                        "output_sequence_lengths": {"total": 1000},
                    },
                    system_desc_extra={"tps_utilization": 1.0},
                ),
            ],
        )
        assert pareto.tps_utilization_mismatch_points() == (64,)

    @pytest.mark.unit
    def test_the_throughput_band_and_the_whole_run_reading(self, tmp_path):
        measured = {
            "duration_ns": 2_000_000_000,
            "output_sequence_lengths": {"total": 2000},
        }
        pareto = _pareto(
            tmp_path,
            [
                # 0.9% off its derivation: inside the 1% band.
                Run(concurrency=16, system_tps=1009.0, summary_extra=measured),
                # 1.5% off: the blocking row's finding.
                Run(concurrency=32, system_tps=1015.0, summary_extra=measured),
                # The client's whole-run `tps` is the stored side, not the
                # steady-window alias `system_tps` would carry (500).
                Run(
                    concurrency=50,
                    system_tps=OMITTED,
                    summary_extra={
                        **measured,
                        "tps": 1000.0,
                        "steady_state": {"window": {"tps": {"system": 500.0}}},
                    },
                ),
            ],
        )
        assert pareto.system_tps_mismatch_points() == (32,)

    @pytest.mark.unit
    def test_unsafe_summary_numbers_are_named_not_raised(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                # A NaN duration derives no throughput (and never agrees).
                Run(
                    concurrency=16,
                    system_tps=1000.0,
                    summary_extra={
                        "duration_ns": float("nan"),
                        "output_sequence_lengths": {"total": 2000},
                    },
                ),
                # A subnormal duration rounds to zero seconds: no division.
                Run(
                    concurrency=32,
                    system_tps=1000.0,
                    summary_extra={
                        "duration_ns": 5e-324,
                        "output_sequence_lengths": {"total": 2000},
                    },
                ),
                # A stringified total decodes like every other numeric fact.
                Run(
                    concurrency=50,
                    system_tps=1000.0,
                    summary_extra={
                        "duration_ns": 2_000_000_000,
                        "output_sequence_lengths": {"total": "2000.0"},
                    },
                ),
            ],
        )
        # A NaN or subnormal duration is unverifiable arithmetic evidence: the
        # row names it (fail-closed) rather than crashing or passing it.
        assert pareto.system_tps_mismatch_points() == (16, 32)

    @pytest.mark.unit
    def test_a_conforming_accuracy_entry_holds_its_gate(self, tmp_path):
        pareto = _gated_pareto(tmp_path, score=0.8320)
        assert pareto.accuracy_gate_misses() == ()

    @pytest.mark.unit
    def test_a_low_score_and_a_short_count_are_named(self, tmp_path):
        pareto = _gated_pareto(tmp_path, score=0.70, unit_samples=10)
        misses = pareto.accuracy_gate_misses()
        assert any("exact_match" in miss for miss in misses)
        assert any("accuracy samples" in miss for miss in misses)

    @pytest.mark.unit
    def test_a_dedicated_carrier_covers_its_band(self, tmp_path):
        pareto = _pareto_with_accuracy(
            tmp_path,
            [
                Run(concurrency=8, max_supported_concurrency=2048),
                Run(
                    concurrency=16, max_supported_concurrency=2048, offline="dedicated"
                ),
                Run(concurrency=64, max_supported_concurrency=2048),
                Run(concurrency=1200, max_supported_concurrency=2048),
                Run(concurrency=2048, max_supported_concurrency=2048),
            ],
            carrying={16, 64, 1200},
        )
        assert "low_concurrency" not in pareto.accuracy_uncovered_regions()

    @pytest.mark.unit
    def test_a_dedicated_point_without_a_result_is_named(self, tmp_path):
        pareto = _pareto_with_accuracy(
            tmp_path,
            [
                Run(concurrency=8, max_supported_concurrency=2048),
                Run(
                    concurrency=16, max_supported_concurrency=2048, offline="dedicated"
                ),
                Run(concurrency=64, max_supported_concurrency=2048),
                Run(concurrency=1200, max_supported_concurrency=2048),
                Run(concurrency=2048, max_supported_concurrency=2048),
            ],
            carrying={64, 1200},
        )
        assert pareto.offline_points_missing_accuracy() == (16,)

    @pytest.mark.unit
    def test_a_margin_carrier_covers_no_band(self, tmp_path):
        pareto = _pareto_with_accuracy(
            tmp_path,
            [
                Run(concurrency=8, max_supported_concurrency=2048),
                Run(concurrency=64, max_supported_concurrency=2048),
                Run(concurrency=2048, max_supported_concurrency=2048),
                Run(concurrency=2200, max_supported_concurrency=2048),
            ],
            carrying={8, 64, 2200},
        )
        assert "high_concurrency" in pareto.accuracy_uncovered_regions()

    @pytest.mark.unit
    def test_a_stringified_count_reads_as_its_number(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    summary_extra={
                        "n_samples_issued": "10",
                        "n_samples_completed": "9",
                    },
                )
            ],
        )
        assert pareto.accounting_mismatch_points() == (16,)

    @pytest.mark.unit
    def test_a_stringified_number_is_the_same_declaration(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(
                    concurrency=32,
                    system_desc_extra={"max_supported_concurrency": "8192"},
                ),
            ],
        )
        assert pareto.inconsistent_system_desc_keys() == ()


class TestDerivedExposure:
    """`Pareto.derived`: the `derived.<name>` fields, resolved from the pareto itself."""

    @pytest.mark.unit
    def test_the_derivation_names_are_the_policy_derivations(self):
        assert set(DERIVATION_NAMES) == {
            "agentic",
            "model_name_admitted",
            "accuracy_uncovered_regions",
            "offline_points_missing_accuracy",
            "offline_ordering",
            "offline_points",
            "out_of_range_points",
            "point_cap",
            "point_count",
            "point_count_min",
            "uncovered_regions",
            "conflicting_load_patterns",
            "conflicting_model_names",
            "conflicting_datasets",
            "conflicting_seed_sets",
            "inconsistent_system_desc_keys",
            "accounting_mismatch_points",
            "system_tps_mismatch_points",
            "misplaced_region_points",
            "tps_utilization_mismatch_points",
            "accuracy_gate_misses",
            "structure_misses",
            "entryless_accuracy_points",
            "input_shape",
            "model_dir_name_mismatches",
            "nodes_used_problems",
            "maximal_engagement_problems",
            "tps_per_kw_mismatches",
            "tps_per_user_mismatches",
            "unresolved_shared_paths",
            "unapproved_drafter_points",
        }

    @pytest.mark.unit
    def test_the_tree_structure_and_shared_pointers_are_checked(self, tmp_path):
        root = tmp_path / "upload"
        runs = [
            Run(
                concurrency=16,
                config_extra={"shared_src": "src/impl", "shared_docs": "docs"},
            )
        ]
        pareto = write_upload(root, {SYSTEM: runs}).paretos[0]
        assert pareto.structure_misses() == (
            "docs/ is missing beside results/",
            "src/ is missing",
        )
        assert pareto.unresolved_shared_paths() == (
            "shared_docs=docs",
            "shared_src=src/impl",
        )
        (root / "docs").mkdir()
        (root / "src" / "impl").mkdir(parents=True)
        (root / "src" / "impl" / "README.md").write_text("x", encoding="utf-8")
        assert pareto.structure_misses() == ()
        assert pareto.unresolved_shared_paths() == ()

    @pytest.mark.unit
    def test_an_escaping_pointer_does_not_resolve(self, tmp_path):
        root = tmp_path / "upload"
        runs = [Run(concurrency=16, config_extra={"shared_src": "../outside"})]
        pareto = write_upload(root, {SYSTEM: runs}).paretos[0]
        assert pareto.unresolved_shared_paths() == ("shared_src=../outside",)

    @pytest.mark.unit
    def test_a_symlinked_pointer_must_resolve_below_the_root(self, tmp_path):
        root = tmp_path / "upload"
        runs = [Run(concurrency=16, config_extra={"shared_src": "src/impl"})]
        pareto = write_upload(root, {SYSTEM: runs}).paretos[0]
        inner = root / "src" / "real"
        inner.mkdir(parents=True)
        link = root / "src" / "impl"
        link.symlink_to(inner, target_is_directory=True)
        assert pareto.unresolved_shared_paths() == ()
        outside = tmp_path / "outside"
        outside.mkdir()
        link.unlink()
        link.symlink_to(outside, target_is_directory=True)
        assert pareto.unresolved_shared_paths() == ("shared_src=src/impl",)

    @pytest.mark.unit
    def test_a_drafter_must_be_approved_and_old_enough(self, tmp_path):
        weight = "git-sha1:" + "a" * 40
        target = "git-sha1:" + "b" * 40
        catalog = Catalog(
            models={
                "gpt-oss-120b": {
                    "approved_drafters": [
                        {
                            "approved_cohort": "2026-09-C1",
                            "model_id": "acme/head",
                            "weight_checksum": weight,
                        },
                        {
                            "approved_cohort": "2026-09-C1",
                            "target_checksum": target,
                            "configuration": {"exit_layer": 24},
                        },
                    ]
                }
            },
            datasets={},
        )
        runs = [
            # A declaration naming no approved checksum.
            Run(
                concurrency=8,
                config_extra={
                    "model_name": "gpt-oss-120b",
                    "target_cohort": "2026-10-C1",
                    "speculative_decoding": {"weight_checksum": "git-sha1:unknown"},
                },
            ),
            # The weight-identified entry, at the earliest cohort it may be used.
            Run(
                concurrency=16,
                config_extra={
                    "model_name": "gpt-oss-120b",
                    "target_cohort": "2026-10-C1",
                    "speculative_decoding": {"weight_checksum": weight},
                },
            ),
            # Approved, but targeted before its two-cohort lead time elapsed.
            Run(
                concurrency=32,
                config_extra={
                    "model_name": "gpt-oss-120b",
                    "target_cohort": "2026-09-C1",
                    "speculative_decoding": {"weight_checksum": weight},
                },
            ),
            # The configuration-identified entry, matched by checksum + block.
            Run(
                concurrency=64,
                config_extra={
                    "model_name": "gpt-oss-120b",
                    "target_cohort": "2026-11-C1",
                    "speculative_decoding": {
                        "target_checksum": target,
                        "configuration": {"exit_layer": 24},
                    },
                },
            ),
        ]
        pareto = write_upload(
            tmp_path / "upload", {SYSTEM: runs}, ruleset=catalog
        ).paretos[0]
        misses = pareto.unapproved_drafter_points()
        assert any("matches no approved drafter" in miss for miss in misses)
        assert any("2026-09-C1 precedes" in miss for miss in misses)
        assert len(misses) == 2

    @pytest.mark.unit
    def test_every_named_derivation_resolves_for_one_pareto(self, tmp_path):
        pareto = _pareto(tmp_path, [Run(concurrency=16), Run(concurrency=8192)])
        derived = pareto.derived()
        assert set(derived) == set(DERIVATION_NAMES)
        assert derived["agentic"] is True
        assert derived["point_count"] == 2
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
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, system_tps=float("inf"))]},
            )
        )
        assert any("needs a finite value" in message for message in refused)

    @pytest.mark.unit
    def test_a_system_tps_outside_float_range_is_refused(self, tmp_path):
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, system_tps=10**400)]},
            )
        )
        assert any("outside float range" in message for message in refused)

    @pytest.mark.unit
    def test_a_c_max_outside_float_range_is_refused(self, tmp_path):
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=16, max_supported_concurrency=10**400)]},
            )
        )
        assert any("outside float range" in message for message in refused)

    @pytest.mark.unit
    def test_a_concurrency_outside_float_range_is_refused(self, tmp_path):
        refused = refused_messages(
            write_upload(
                tmp_path / "acme",
                {SYSTEM: [Run(concurrency=10**400, dir_name="r16")]},
            )
        )
        assert any("outside float range" in message for message in refused)


class TestSharedFactsAgreement:
    """Every point's descriptor is compared by presence and typed value."""

    @pytest.mark.unit
    def test_an_explicit_null_c_min_override_does_not_agree_with_an_omission(
        self, tmp_path
    ):
        # Run 2's `c_min_override: null` is §5.4's invalid declaration; a
        # `get`-based comparison once let it read equal to run 1's omission.
        runs = [Run(concurrency=16), Run(concurrency=8192, c_min_override=None)]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any(
            "must declare the same c_min_override" in message for message in refused
        )

    @pytest.mark.unit
    def test_an_int_and_a_float_c_min_override_do_not_agree(self, tmp_path):
        runs = [
            Run(concurrency=16, c_min_override=16),
            Run(concurrency=32, c_min_override=16.0),
            Run(concurrency=8192),
        ]
        refused = refused_messages(write_upload(tmp_path / "acme", {SYSTEM: runs}))
        assert any(
            "must declare the same c_min_override" in message for message in refused
        )


class TestEmptySystems:
    """A system directory with no model directories is refused at load."""

    @pytest.mark.unit
    def test_a_system_without_model_directories_is_refused(self, tmp_path):
        root = tmp_path / "acme"
        write_run(root / "results" / "sys" / "model" / "r16", Run(concurrency=16))
        (root / "results" / "empty_sys").mkdir(parents=True)
        refused = refused_messages(Submission.from_directory(root))
        assert any("has no model directories" in message for message in refused)

    @pytest.mark.unit
    def test_a_pool_root_holding_no_paretos_is_refused(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results").mkdir(parents=True)
        with pytest.raises(ValueError, match="holds no paretos"):
            Submission.from_directory(root)


class TestAccuracyGateEdges:
    """The accuracy gate's and utilization check's arithmetic edges."""

    @pytest.mark.unit
    def test_a_zero_sample_count_still_fails_the_gate(self, tmp_path):
        pareto = _gated_pareto(tmp_path, score=0.95, unit_samples=0)
        misses = pareto.accuracy_gate_misses()
        assert any("accuracy samples" in miss for miss in misses)

    @pytest.mark.unit
    def test_a_non_finite_score_is_named_not_silently_passed(self, tmp_path):
        pareto = _gated_pareto(tmp_path, score=float("nan"))
        misses = pareto.accuracy_gate_misses()
        assert any("non-finite" in miss for miss in misses)

    @pytest.mark.unit
    def test_the_utilization_peak_covers_undeclared_points(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    system_tps=100.0,
                    summary_extra={
                        "duration_ns": 1_000_000_000,
                        "output_sequence_lengths": {"total": 100},
                    },
                    system_desc_extra={"tps_utilization": 0.5},
                ),
                Run(
                    concurrency=32,
                    system_tps=200.0,
                    summary_extra={
                        "duration_ns": 1_000_000_000,
                        "output_sequence_lengths": {"total": 200},
                    },
                ),
            ],
        )
        assert pareto.tps_utilization_mismatch_points() == ()

    @pytest.mark.unit
    def test_a_numeric_string_utilization_is_decoded(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    system_tps=1000.0,
                    summary_extra={
                        "duration_ns": 1_000_000_000,
                        "output_sequence_lengths": {"total": 1000},
                    },
                    system_desc_extra={"tps_utilization": "0.5"},
                ),
            ],
        )
        assert pareto.tps_utilization_mismatch_points() == (16,)

    @pytest.mark.unit
    def test_an_underivable_throughput_is_named(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    system_tps=1000.0,
                    summary_extra={
                        "duration_ns": None,
                        "output_sequence_lengths": {"total": 2000},
                    },
                ),
            ],
        )
        assert pareto.system_tps_mismatch_points() == (16,)


class TestLoadPatternAbsence:
    """The declared absent state reads the same for both spellings of absence."""

    @pytest.mark.unit
    def test_a_missing_runtime_settings_block_still_votes_concurrency(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(concurrency=16),
                Run(
                    concurrency=32,
                    config_extra={"runtime_settings": {"load_pattern": "poisson"}},
                ),
            ],
        )
        assert pareto.conflicting_load_patterns() == ("concurrency", "poisson")


class TestInputShape:
    """The structure rows know whether the walk root is a submission root."""

    @pytest.mark.unit
    def test_a_system_directory_input_reads_as_the_system_shape(self, tmp_path):
        root = tmp_path / "upload"
        upload = write_upload(root, {SYSTEM: [Run(concurrency=16)]})
        system = upload.paretos[0].node
        assert system is not None
        grouped = Submission.from_directory(Path(system.dir).parent)
        assert grouped.paretos[0].input_shape() == "system"
        assert upload.paretos[0].input_shape() == "submission"


class TestModelDirNameConsistency:
    """The `<model>` directory must be named after the declared model_name (§8.1)."""

    @pytest.mark.unit
    def test_a_mismatched_directory_name_is_named(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16, config_extra={"model_name": "gpt-oss-120b"})],
        )
        assert pareto.model_dir_name_mismatches() == ("gpt-oss-120b != model",)

    @pytest.mark.unit
    def test_a_matching_directory_name_holds_its_row(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16, config_extra={"model_name": "model"})],
        )
        assert pareto.model_dir_name_mismatches() == ()

    @pytest.mark.unit
    def test_an_undeclared_name_leaves_the_row_to_the_disclosure_checks(self, tmp_path):
        pareto = _pareto(tmp_path, [Run(concurrency=16)])
        assert pareto.model_dir_name_mismatches() == ()


class TestPointPower:
    """Appendix E / §9.1: node declarations, maximal engagement, the denominator."""

    @staticmethod
    def _pareto(tmp_path, runs, power):
        root = tmp_path / "upload"
        write_upload(root, {SYSTEM: runs})
        system_dir = root / "results" / SYSTEM[0]
        system_dir.mkdir(parents=True, exist_ok=True)
        (system_dir / "system_power.json").write_text(
            json.dumps(power), encoding="utf-8"
        )
        # The walk reads artifacts eagerly: the power descriptor must exist
        # before the tree is built for a row to see it.
        return Submission.from_directory(root).paretos[0]

    @pytest.mark.unit
    def test_a_valid_node_declaration_holds_every_row(self, tmp_path):
        power = {
            "provisioned_power_kw": 100.0,
            "node_sets": [
                {
                    "system_node_ensemble_id": "ensemble-a",
                    "nodes": 4,
                    "accelerators_per_node": 8,
                }
            ],
        }
        pareto = self._pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    system_tps=1000.0,
                    system_desc_extra={
                        "tensor_parallel": 2,
                        "data_parallel": 2,
                    },
                    config_extra={
                        "nodes_used": [
                            {"system_node_ensemble_id": "ensemble-a", "nodes": 2}
                        ],
                        "dp_shortfall": {
                            "dp_actual": 2,
                            "dp_formula": 16,
                            "reason": "memory",
                        },
                    },
                    summary_extra={"system_tps_per_kw": 20.0},
                )
            ],
            power,
        )
        assert pareto.nodes_used_problems() == ()
        assert pareto.maximal_engagement_problems() == ()
        assert pareto.tps_per_kw_mismatches() == ()

    @pytest.mark.unit
    def test_an_unknown_ensemble_is_named(self, tmp_path):
        power = {
            "provisioned_power_kw": 100.0,
            "node_sets": [{"system_node_ensemble_id": "ensemble-a", "nodes": 4}],
        }
        pareto = self._pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    config_extra={
                        "nodes_used": [
                            {"system_node_ensemble_id": "ensemble-z", "nodes": 1}
                        ]
                    },
                )
            ],
            power,
        )
        assert pareto.nodes_used_problems() == ("c16: ensemble-z names 0 node sets",)

    @pytest.mark.unit
    def test_overflowing_the_provisioned_nodes_is_named(self, tmp_path):
        power = {
            "provisioned_power_kw": 100.0,
            "node_sets": [{"system_node_ensemble_id": "ensemble-a", "nodes": 4}],
        }
        pareto = self._pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    config_extra={
                        "nodes_used": [
                            {"system_node_ensemble_id": "ensemble-a", "nodes": 9}
                        ]
                    },
                )
            ],
            power,
        )
        assert pareto.nodes_used_problems() == ("c16: ensemble-a engages 9 of 4 nodes",)

    @pytest.mark.unit
    def test_a_short_capacity_is_named(self, tmp_path):
        power = {
            "provisioned_power_kw": 100.0,
            "node_sets": [
                {
                    "system_node_ensemble_id": "ensemble-a",
                    "nodes": 2,
                    "accelerators_per_node": 8,
                }
            ],
        }
        pareto = self._pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    system_desc_extra={"tensor_parallel": 8, "data_parallel": 8},
                    config_extra={
                        "nodes_used": [
                            {"system_node_ensemble_id": "ensemble-a", "nodes": 2}
                        ]
                    },
                )
            ],
            power,
        )
        problems = pareto.nodes_used_problems()
        assert len(problems) == 1
        assert "cannot hold" in problems[0]

    @pytest.mark.unit
    def test_engaging_past_the_maximum_dp_is_named(self, tmp_path):
        power = {
            "provisioned_power_kw": 100.0,
            "node_sets": [
                {
                    "system_node_ensemble_id": "ensemble-a",
                    "nodes": 2,
                    "accelerators_per_node": 8,
                }
            ],
        }
        pareto = self._pareto(
            tmp_path,
            [Run(concurrency=16, system_desc_extra={"data_parallel": 20})],
            power,
        )
        assert pareto.maximal_engagement_problems() == (
            "c16: DP 20 engages 20 of 16 provisioned accelerators (max DP 16)",
        )

    @pytest.mark.unit
    def test_a_tps_per_kw_outside_one_percent_is_named(self, tmp_path):
        power = {
            "provisioned_power_kw": 100.0,
            "node_sets": [
                {
                    "system_node_ensemble_id": "ensemble-a",
                    "nodes": 4,
                    "accelerators_per_node": 8,
                }
            ],
        }
        pareto = self._pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    system_tps=1000.0,
                    config_extra={
                        "nodes_used": [
                            {"system_node_ensemble_id": "ensemble-a", "nodes": 2}
                        ]
                    },
                    summary_extra={"system_tps_per_kw": 25.0},
                )
            ],
            power,
        )
        mismatches = pareto.tps_per_kw_mismatches()
        assert len(mismatches) == 1
        assert "against 20.0" in mismatches[0]

    @pytest.mark.unit
    def test_a_missing_power_declaration_skips_the_denominator(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16, system_tps=1000.0)],
        )
        assert pareto.tps_per_kw_mismatches() == ()


class TestTpsPerUser:
    """§9.1: `tps_per_user = 1000 / tpot_p90_ms`."""

    @pytest.mark.unit
    def test_a_matching_p90_implied_rate_holds_its_row(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    summary_extra={
                        "tpot": {"percentiles": {"90.0": 5_500_000.0}},
                        "tps_per_user": 1000.0 / 5.5,
                    },
                )
            ],
        )
        assert pareto.tps_per_user_mismatches() == ()

    @pytest.mark.unit
    def test_a_stored_rate_off_the_p90_is_named(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [
                Run(
                    concurrency=16,
                    summary_extra={
                        "tpot": {"percentiles": {"90": 10_000_000.0}},
                        "tps_per_user": 250.0,
                    },
                )
            ],
        )
        mismatches = pareto.tps_per_user_mismatches()
        assert len(mismatches) == 1
        assert "against 100.0" in mismatches[0]

    @pytest.mark.unit
    def test_a_summary_without_a_p90_has_nothing_to_check(self, tmp_path):
        pareto = _pareto(
            tmp_path,
            [Run(concurrency=16, summary_extra={"tps_per_user": 250.0})],
        )
        assert pareto.tps_per_user_mismatches() == ()
