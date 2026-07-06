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

import json

import pytest
from inference_endpoint.metrics.results_plots import (
    Distribution,
    extract_accuracy,
    extract_distribution,
    extract_turn_scores,
    generate_plots,
    load_run,
    plot_distribution,
)


def _accuracy_results() -> dict:
    # Current contract: scalar score + a numeric breakdown block. A couple of
    # values are left as strings to exercise the extractor's defensive coercion
    # (older artifacts serialized percentages as strings).
    return {
        "accuracy_scores": {
            "bfcl_v4::function_calling": {
                "score": 0.8623,
                "breakdown": {
                    "overall_accuracy": 86.23,
                    "normalized_single_turn_score": 87.96,
                    "category_scores": {
                        "non_live": 82.59,
                        "live": 84.12,
                        "hallucination": "97.16",
                    },
                    "subset_scores": {"simple_python": 92.74, "simple_java": "46.77"},
                    "total_samples": 995,
                },
            }
        }
    }


def _perf_scores() -> dict:
    return {
        "score": 0.613,
        "turns": {"issued": 1007, "observed": 968, "missing": 38, "scored": 1006},
        "per_turn": [
            {"score": 0.2857},
            {"score": 1.0},
            {"score": 0.3333},
            {"score": "0.5"},
        ],
    }


def _result_summary() -> dict:
    return {
        "ttft": {
            "min": 750485004,
            "max": 19540631274,
            "median": 2078687662.0,
            "avg": 2551060382.97,
            "percentiles": {"50.0": 2078687662.0, "99.0": 8425474865.0},
            "histogram": {
                "buckets": [[750485004.0, 836620633.0], [836620633.0, 932000000.0]],
                "counts": [10, 5],
            },
        },
        "tpot": {},
        "latency": {
            "min": 3074,
            "max": 111556566205,
            "median": 6397624496.0,
            "avg": 9950193051.35,
            "percentiles": {"50.0": 6397624496.0, "99.0": 68694660297.0},
            "histogram": {"buckets": [[3074.0, 5491.0]], "counts": [1]},
        },
        "output_sequence_lengths": {},
    }


@pytest.mark.unit
def test_extract_accuracy_coerces_strings():
    b = extract_accuracy(_accuracy_results())
    assert b is not None
    assert b.overall == pytest.approx(86.23)
    assert b.normalized == pytest.approx(87.96)
    assert b.total_samples == 995
    assert b.category_scores["hallucination"] == pytest.approx(97.16)
    assert b.subset_scores["simple_java"] == pytest.approx(46.77)


@pytest.mark.unit
def test_extract_accuracy_returns_none_without_scores():
    assert extract_accuracy({"accuracy_scores": {}}) is None
    assert extract_accuracy({}) is None


@pytest.mark.unit
def test_extract_turn_scores_handles_str_and_float():
    scores = extract_turn_scores(_perf_scores())
    assert scores == pytest.approx([0.2857, 1.0, 0.3333, 0.5])


@pytest.mark.unit
def test_extract_distribution_scales_ns_to_seconds():
    dist = extract_distribution(_result_summary(), "ttft")
    assert dist is not None
    assert dist.unit == "s"
    assert dist.minimum == pytest.approx(0.750485004)
    assert dist.percentiles[50.0] == pytest.approx(2.078687662)
    # Histogram bucket edges scaled too.
    lo, hi = dist.hist_buckets[0]
    assert lo == pytest.approx(0.750485004)
    assert dist.hist_counts == [10, 5]


@pytest.mark.unit
def test_extract_distribution_skips_empty_blocks():
    summary = _result_summary()
    assert extract_distribution(summary, "tpot") is None
    assert extract_distribution(summary, "output_sequence_lengths") is None
    assert extract_distribution(summary, "missing_key") is None


@pytest.mark.unit
def test_load_run_assembles_artifacts(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(_accuracy_results()))
    (tmp_path / "scores.json").write_text(json.dumps(_perf_scores()))
    (tmp_path / "result_summary.json").write_text(json.dumps(_result_summary()))

    run = load_run(tmp_path)
    assert run.accuracy is not None
    assert len(run.turn_scores) == 4
    assert run.turn_summary["missing"] == 38
    assert run.inline_score == pytest.approx(0.613)
    # Only populated distributions are kept (tpot/osl were empty).
    assert set(run.distributions) == {"ttft", "latency"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "buckets, counts",
    [
        ([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], [10, 5]),  # more buckets than counts
        ([(0.0, 1.0)], [10, 5, 1]),  # more counts than buckets
    ],
)
def test_plot_distribution_tolerates_bucket_count_mismatch(tmp_path, buckets, counts):
    # buckets and counts are extracted independently; a length mismatch must be
    # sliced to a common length rather than raising inside matplotlib's bar().
    pytest.importorskip("matplotlib")
    dist = Distribution(
        name="ttft",
        unit="s",
        minimum=0.0,
        maximum=3.0,
        median=1.0,
        avg=1.0,
        percentiles={50.0: 1.0},
        hist_buckets=buckets,
        hist_counts=counts,
    )
    out = plot_distribution(dist, tmp_path / "dist.png")
    assert out is not None and out.exists()


@pytest.mark.unit
def test_generate_plots_writes_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    (tmp_path / "results.json").write_text(json.dumps(_accuracy_results()))
    (tmp_path / "scores.json").write_text(json.dumps(_perf_scores()))
    (tmp_path / "result_summary.json").write_text(json.dumps(_result_summary()))

    written = generate_plots(tmp_path)
    names = {p.name for p in written}
    assert "accuracy.png" in names
    assert "perf_turns.png" in names
    assert "perf_ttft.png" in names
    assert "perf_latency.png" in names
    for p in written:
        assert p.exists() and p.stat().st_size > 0
