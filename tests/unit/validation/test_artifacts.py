# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact construction, field presence, and input ownership."""

import json
from collections import Counter
from pathlib import Path

import pytest

from inference_endpoint.config.schema import LoadPatternType
from inference_endpoint.validation import (
    bundled_policy_path,
    load_policy,
)
from inference_endpoint.validation.artifacts import PointArtifacts, load_artifacts

pytestmark = pytest.mark.unit


@pytest.fixture
def documents():
    return (
        {"concurrency": 16, "runtime_settings": {"runtime": {}}},
        {
            "n_samples_completed": 4,
            "duration_ns": 2e9,
            "output_sequence_lengths": {"total": 100},
            "system_tps": 999,
        },
        {},
    )


def test_point_does_not_retain_input_documents(tmp_path, documents):
    config, summary, system = documents
    config["checkpoint"] = {"repository": "model", "revision": "a" * 40}
    config["unused_payload"] = {"large": [1, 2, 3]}
    point = PointArtifacts.from_json(tmp_path / "model" / "r16", *documents)
    config["checkpoint"]["revision"] = "b" * 40
    summary["output_sequence_lengths"]["total"] = 10000
    system["system_name"] = "changed"
    assert point.config.checkpoint.revision == "a" * 40
    assert point.throughput == 50
    assert not point.config.model_extra
    assert not hasattr(point.config, "unused_payload")
    assert not any(
        hasattr(point, name) for name in ("raw", "summary_raw", "system_raw")
    )


def test_supplied_fields_distinguish_omissions_from_defaults(tmp_path, documents):
    config, _, _ = documents
    config["runtime_settings"]["runtime"]["scheduler_random_seed"] = 123
    point = PointArtifacts.from_json(tmp_path, *documents)
    assert point.config.runtime_settings.load_pattern is LoadPatternType.CONCURRENCY
    assert point.context.load_pattern is None
    assert point.get_config("runtime_settings.stream_all_chunks") is None
    assert point.config.runtime_settings.runtime.scheduler_rng_seed == 123
    assert (
        "runtime_settings.runtime.scheduler_rng_seed"
        not in point.evidence.config.fields
    )
    assert (
        "runtime_settings.runtime.scheduler_random_seed" in point.evidence.config.fields
    )
    config["runtime_settings"].update(
        load_pattern="concurrency", stream_all_chunks=False
    )
    explicit = PointArtifacts.from_json(tmp_path, *documents)
    assert explicit.context.load_pattern is LoadPatternType.CONCURRENCY
    assert explicit.get_config("runtime_settings.stream_all_chunks") is False


def test_reported_metrics_do_not_resolve_to_calculated_values(tmp_path, documents):
    point = PointArtifacts.from_json(tmp_path, *documents)
    assert point.summary.system_tps == 50
    assert point.get_summary("system_tps") == 999
    del documents[1]["system_tps"]
    omitted = PointArtifacts.from_json(tmp_path, *documents)
    assert omitted.summary.system_tps == 50
    assert omitted.get_summary("system_tps") is None


@pytest.mark.parametrize("standalone_accuracy", [False, True])
def test_point_input_files_are_read_once(
    tmp_path, documents, monkeypatch, standalone_accuracy
):
    point = tmp_path / "results" / "system" / "gpt-oss-120b" / "r16"
    point.mkdir(parents=True)
    for name, data in zip(
        ("point.yaml", "result_summary.json", "system_desc.json"),
        documents,
        strict=True,
    ):
        (point / name).write_text(json.dumps(data))
    accuracy = {"aime25": {"num_samples": 30, "score": 90}}
    accuracy_name = "accuracy_results.json" if standalone_accuracy else "results.json"
    (point / accuracy_name).write_text(
        json.dumps(accuracy if standalone_accuracy else {"accuracy_scores": accuracy})
    )
    reads = Counter()
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        reads[path] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    artifacts = load_artifacts(tmp_path, load_policy(bundled_policy_path()))
    assert (
        artifacts.index.points[str(point)].accuracy.root["aime25"]["num_samples"] == 30
    )
    assert {
        path.name: count for path, count in reads.items() if path.parent == point
    } == {
        "point.yaml": 1,
        "result_summary.json": 1,
        "system_desc.json": 1,
        accuracy_name: 1,
    }


@pytest.mark.parametrize("field", ["duration_ns", "output_sequence_lengths"])
def test_throughput_requires_supplied_operands(tmp_path, documents, field):
    del documents[1][field]
    point = PointArtifacts.from_json(tmp_path, *documents)
    assert point.throughput is None
