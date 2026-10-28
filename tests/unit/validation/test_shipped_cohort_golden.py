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

"""The shipped 2026-10-C1 cohort against client-written artifacts.

One golden upload's summaries come from the client's own writers
(`Report.to_json`, `BenchmarkConfig.to_yaml_file`, `write_accuracy_results`),
and a conforming 8-point pareto must exit 0 against the packaged cohort. The
fixture carries the shapes only real output has -- a null
`e2e_avg_interactivity` on the point with a failed sample, the dedicated
Offline run's empty `tpot`, the recorded warmup salt -- so cohort edits and
client schema changes surface here instead of on a real upload.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import inference_endpoint
import pytest
import yaml
from inference_endpoint.commands.benchmark.accuracy import write_accuracy_results
from inference_endpoint.config.schema import BenchmarkConfig
from inference_endpoint.config.schema import TestType as BenchmarkTestType
from inference_endpoint.metrics.report import Report
from inference_endpoint.validation.__main__ import main

from tests.test_helpers import DEFAULT_SOURCE

COHORT = Path(inference_endpoint.__file__).parent / "config" / "cohorts" / "2026-10-C1"
SYSTEM = "sys"
MODEL = "gpt-oss-120b"
DATASET = "cnn_dailymail"
FLOOR = 13368
MAX_SUPPORTED = 2048
DURATION_NS = 1_300_000_000_000

# c_min 8, c_max 2048: one point per mandatory region (8 ultra-low, 16 low,
# 64 medium, 2048 high), three submitter's-choice points, and the dedicated
# Offline run reporting the dataset's cardinality (§5.7.1).
FIXED_POINTS = (8, 16, 64, 2048, 32, 256, 1024)
OFFLINE_CONCURRENCY = FLOOR
# Accuracy results sit at the five §5.3 mandatory points only; the three
# submitter's-choice points get non-blocking `point_accuracy_present` warns.
ACCURACY_POINTS = (8, 16, 64, 2048, OFFLINE_CONCURRENCY)
FAILED_SAMPLE_POINT = 64
SET_A = {
    "scheduler_random_seed": 10487924139932647040,
    "dataloader_random_seed": 586478644936801402,
}


def _summary(run_dir: Path, *, failed: int, with_tpot: bool) -> None:
    """Write `result_summary.json` through `Report.to_json`."""
    fake_out = 4096
    latency_ns = 20_000_000_000
    metrics: list[dict[str, Any]] = [
        {"type": "counter", "name": "tracked_duration_ns", "value": DURATION_NS},
        {"type": "counter", "name": "tracked_samples_issued", "value": FLOOR},
        {"type": "counter", "name": "tracked_samples_completed", "value": FLOOR},
        {"type": "counter", "name": "tracked_samples_failed", "value": failed},
        {
            "type": "series",
            "name": "osl",
            "count": FLOOR,
            "total": FLOOR * fake_out,
            "min": fake_out,
            "max": fake_out,
            "sum_sq": FLOOR * fake_out * fake_out,
            "percentiles": {"50.0": fake_out, "90.0": fake_out, "99.0": fake_out},
            "histogram": [[[fake_out, fake_out], FLOOR]],
        },
        {
            "type": "series",
            "name": "sample_latency_ns",
            "count": FLOOR,
            "total": FLOOR * latency_ns,
            "min": latency_ns,
            "max": latency_ns,
            "sum_sq": FLOOR * latency_ns * latency_ns,
            "percentiles": {
                "50.0": latency_ns,
                "90.0": latency_ns,
                "99.0": latency_ns,
            },
            "histogram": [[[latency_ns, latency_ns], FLOOR]],
        },
    ]
    if with_tpot:
        tpot_ns = 2_000_000
        metrics.append(
            {
                "type": "series",
                "name": "tpot_ns",
                "count": FLOOR,
                "total": FLOOR * tpot_ns,
                "min": tpot_ns,
                "max": tpot_ns,
                "sum_sq": FLOOR * tpot_ns * tpot_ns,
                "percentiles": {"50.0": tpot_ns, "90.0": tpot_ns, "99.0": tpot_ns},
                "histogram": [[[tpot_ns, tpot_ns], FLOOR]],
            }
        )
    snapshot = {"state": "complete", "n_pending_tasks": 0, "metrics": metrics}
    report = Report.from_snapshot(
        snapshot,
        token_count_source=DEFAULT_SOURCE,
        run_config={"load_pattern": "concurrency"},
    )
    report.to_json(save_to=run_dir / "result_summary.json")


def _point_yaml(
    run_dir: Path,
    concurrency: int,
    *,
    offline: str | None,
    load_pattern: str,
    api_key: str,
) -> None:
    """Write `point.yaml` (§8.3) with the disclosures the cohort checks."""
    config: dict[str, Any] = {
        "concurrency": concurrency,
        "dataset": "mlperf-perf-dataset-v1",
        "dataset_link": "https://example.com/dataset",
        "dataset_name": DATASET,
        "dataset_type": "Performance",
        "division": "Standardized",
        "link_to_model": "https://example.com/model",
        "link_to_model_transformation": None,
        "load_pattern": load_pattern,
        "max_supported_concurrency": MAX_SUPPORTED,
        "model_name": MODEL,
        "model_precision": "FP16",
        "runtime_settings": {
            "runtime": dict(SET_A),
            "load_pattern": {"type": load_pattern},
            "client": {"stream_all_chunks": True, "api_key": api_key},
            "warmup": {"enabled": True, "n_requests": 160, "salt": True},
        },
        "seed_set": "A",
        "shared_docs": "docs",
        "shared_src": "src",
        "target_cohort": "2026-10-C1",
        "warmup": {
            "concurrency": 16,
            "data_source": "mlperf-perf-dataset-v1 validation split",
            "duration_s": 60,
            "initialization_steps": ["model loaded", "kv-cache warmed"],
            "logs_retained": True,
            "requests_completed": 160,
            "requests_issued": 160,
        },
    }
    if offline is not None:
        config["offline"] = offline
    (run_dir / "point.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )


def _config_yaml(path: Path) -> None:
    """Write `config.yaml` through the client's own serializer."""
    BenchmarkConfig(
        type=BenchmarkTestType.OFFLINE,
        model_params={"name": MODEL, "max_new_tokens": 1024},
        endpoint_config={
            "endpoints": ["http://localhost:8000"],
            "api_type": "openai",
        },
        datasets=[{"name": "perf", "type": "performance", "path": "dataset.jsonl"}],
    ).to_yaml_file(path)


def _accuracy(run_dir: Path) -> None:
    """Write this run's `accuracy_results.json` through the client's writer."""
    write_accuracy_results(
        run_dir,
        [
            {
                "dataset_name": DATASET,
                "score": 0.99,
                "unit_samples": FLOOR,
                "total_samples": FLOOR,
            }
        ],
    )
    written = run_dir / "accuracy" / "accuracy_results.json"
    written.replace(run_dir / "accuracy_results.json")
    (run_dir / "accuracy").rmdir()


def _build_upload(root: Path, *, api_key: str = "<redacted>") -> None:
    """One conforming 8-point pareto plus its system-level power descriptor."""
    results = root / "results" / SYSTEM
    model_dir = results / MODEL
    for concurrency in (*FIXED_POINTS, OFFLINE_CONCURRENCY):
        offline = "dedicated" if concurrency == OFFLINE_CONCURRENCY else None
        run_dir = model_dir / f"r{concurrency}"
        run_dir.mkdir(parents=True)
        _point_yaml(
            run_dir,
            concurrency,
            offline=offline,
            load_pattern="max_throughput" if offline else "concurrency",
            api_key=api_key,
        )
        (run_dir / "system_desc.json").write_text(
            json.dumps({"max_supported_concurrency": MAX_SUPPORTED}),
            encoding="utf-8",
        )
        _summary(
            run_dir,
            failed=1 if concurrency == FAILED_SAMPLE_POINT else 0,
            with_tpot=offline is None,
        )
        _config_yaml(run_dir / "config.yaml")
        if concurrency in ACCURACY_POINTS:
            _accuracy(run_dir)
    (results / "system_power.json").write_text(
        json.dumps(
            {
                "provisioned_power_kw": 1000.0,
                "node_sets": [{"system_node_ensemble_id": "ensemble-a", "nodes": 1}],
                "mlc_estimated_power": False,
            }
        ),
        encoding="utf-8",
    )


class TestShippedCohortGolden:
    """The packaged cohort decides a client-written upload end to end."""

    @pytest.mark.unit
    def test_a_conforming_upload_exits_zero(self, tmp_path):
        root = tmp_path / "upload"
        _build_upload(root)
        assert main([str(COHORT), str(root)]) == 0

    @pytest.mark.unit
    def test_dropping_the_offline_run_fails_the_cohort(self, tmp_path, capsys):
        root = tmp_path / "upload"
        _build_upload(root)
        shutil.rmtree(root / "results" / SYSTEM / MODEL / f"r{OFFLINE_CONCURRENCY}")
        assert main([str(COHORT), str(root)]) == 1
        out = capsys.readouterr().out
        assert "pareto_offline_present" in out
        assert "pareto_point_count_min" in out

    @pytest.mark.unit
    def test_a_live_api_key_fails_the_cohort(self, tmp_path, capsys):
        root = tmp_path / "upload"
        _build_upload(root, api_key="sk-live-abc123")
        assert main([str(COHORT), str(root)]) == 1
        assert "point_api_key_redacted" in capsys.readouterr().out
