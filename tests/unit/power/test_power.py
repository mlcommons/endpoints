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

"""Unit tests for vendor-agnostic power monitoring."""

from __future__ import annotations

import sys
import time

import pytest
from inference_endpoint.config.schema import PowerConfig
from inference_endpoint.power.collector import PowerCollector
from inference_endpoint.power.parse import parse_trace
from inference_endpoint.power.sources import ResolvedSource, power_source, resolve
from inference_endpoint.power.window import build_power_report

pytestmark = pytest.mark.unit


def _jsonl_source(value_kind: str = "power_w") -> ResolvedSource:
    return ResolvedSource(
        argv=["true"],
        fmt="jsonl",
        value_kind=value_kind,
        ts_field="ts",
        value_field="value",
        label_field="label",
        csv_header=False,
    )


# --------------------------------------------------------------------------- #
# Config + sources
# --------------------------------------------------------------------------- #
def test_source_requires_its_options():
    # Each source validates its own options at build time (vendor-neutral core).
    with pytest.raises(ValueError, match="options.url"):
        resolve(PowerConfig(source="prometheus"))
    with pytest.raises(ValueError, match="options.argv"):
        resolve(PowerConfig(source="command"))
    # nvidia_smi needs nothing extra
    resolve(PowerConfig(source="nvidia_smi"))


def test_disabled_power_has_no_config_footprint():
    # source=None must serialize to an empty mapping (frictionless when off).
    assert PowerConfig().model_dump() == {}
    assert PowerConfig(source="nvidia_smi").model_dump()["source"] == "nvidia_smi"


def test_resolve_nvidia_smi_argv():
    r = resolve(
        PowerConfig(
            source="nvidia_smi", options={"gpu_indices": [0, 3]}, interval_s=0.5
        )
    )
    assert "nvidia-smi" in r.argv
    assert "-lms" in r.argv and "500" in r.argv
    assert r.argv[-2:] == ["-i", "0,3"]
    assert r.value_kind == "power_w" and r.fmt == "csv"


def test_custom_source_plugin_registration():
    # A user registers their own source with the decorator + cfg.options.
    @power_source("test_custom_src")
    def _build(cfg: PowerConfig) -> ResolvedSource:
        return ResolvedSource(
            argv=["echo", cfg.options["tag"]],
            fmt="jsonl",
            value_kind="power_w",
            ts_field="ts",
            value_field="value",
            label_field="label",
            csv_header=False,
        )

    r = resolve(PowerConfig(source="test_custom_src", options={"tag": "hi"}))
    assert r.argv == ["echo", "hi"]


def test_resolve_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown power source"):
        resolve(PowerConfig(source="nope"))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_jsonl_drops_malformed(tmp_path):
    f = tmp_path / "t.log"
    f.write_text(
        '{"ts": 1000.0, "value": 100.0, "label": "gpu0"}\n'
        "not json\n"
        '{"ts": 1001.0, "value": 110.0, "label": "gpu0"}\n'
        '{"ts": 1002.0}\n'  # missing value
    )
    res = parse_trace(f, _jsonl_source())
    assert len(res.samples) == 2
    assert res.dropped == 2
    assert res.samples[0].label == "gpu0"


def test_parse_csv_nvidia_format(tmp_path):
    f = tmp_path / "smi.csv"
    f.write_text(
        "2026/05/28 05:51:22.131, 0, 259.16\n" "2026/05/28 05:51:22.131, 1, 260.00\n"
    )
    res = parse_trace(f, resolve(PowerConfig(source="nvidia_smi")))
    assert res.dropped == 0
    labels = {s.label for s in res.samples}
    assert labels == {"gpu0", "gpu1"}
    assert res.samples[0].value == pytest.approx(259.16)


def test_parse_missing_file_is_empty(tmp_path):
    res = parse_trace(tmp_path / "nope.log", _jsonl_source())
    assert res.samples == [] and res.dropped == 0


# --------------------------------------------------------------------------- #
# Windowing + integration
# --------------------------------------------------------------------------- #
def test_trapezoid_energy_constant_power(tmp_path):
    # Constant 100 W for 10 s -> 1000 J.
    f = tmp_path / "p.log"
    lines = [
        f'{{"ts": {1000.0 + i}, "value": 100.0, "label": "gpu0"}}' for i in range(11)
    ]
    f.write_text("\n".join(lines) + "\n")
    rep = build_power_report(
        resolved=_jsonl_source(),
        trace_path=f,
        window_start_epoch_s=1000.0,
        window_end_epoch_s=1010.0,
        output_tokens=2000,
        token_window_basis="performance_phase_tracked",
        consistent_with_window=True,
        collector_status="ok",
        collector_error=None,
        interval_s=1.0,
    )
    assert rep["status"] == "ok"
    assert rep["totals"]["energy_j"] == pytest.approx(1000.0)
    assert rep["totals"]["mean_power_w"] == pytest.approx(100.0)
    # 1000 J / 2000 tokens = 0.5 J/token
    assert rep["totals"]["energy_per_output_token_j"] == pytest.approx(0.5)


def test_energy_counter_kind_uses_delta(tmp_path):
    # Cumulative joule counter: 5000 -> 9000 over window => 4000 J.
    f = tmp_path / "c.log"
    f.write_text(
        '{"ts": 1000.0, "value": 5000.0, "label": "node"}\n'
        '{"ts": 1005.0, "value": 7000.0, "label": "node"}\n'
        '{"ts": 1010.0, "value": 9000.0, "label": "node"}\n'
    )
    rep = build_power_report(
        resolved=_jsonl_source(value_kind="energy_j"),
        trace_path=f,
        window_start_epoch_s=1000.0,
        window_end_epoch_s=1010.0,
        output_tokens=None,
        token_window_basis="performance_phase_tracked",
        consistent_with_window=True,
        collector_status="ok",
        collector_error=None,
        interval_s=5.0,
    )
    assert rep["totals"]["energy_j"] == pytest.approx(4000.0)


def test_epot_suppressed_when_inconsistent(tmp_path):
    f = tmp_path / "p.log"
    f.write_text(
        '{"ts": 1000.0, "value": 100.0, "label": "gpu0"}\n'
        '{"ts": 1010.0, "value": 100.0, "label": "gpu0"}\n'
    )
    rep = build_power_report(
        resolved=_jsonl_source(),
        trace_path=f,
        window_start_epoch_s=1000.0,
        window_end_epoch_s=1010.0,
        output_tokens=2000,
        token_window_basis="global_run",
        consistent_with_window=False,  # denominators would mix
        collector_status="ok",
        collector_error=None,
        interval_s=10.0,
    )
    assert rep["totals"]["energy_per_output_token_j"] is None
    assert "suppressed" in rep["totals"]["energy_per_output_token_note"]


def test_status_no_data_when_window_empty(tmp_path):
    f = tmp_path / "p.log"
    f.write_text('{"ts": 5.0, "value": 100.0, "label": "gpu0"}\n')  # outside window
    rep = build_power_report(
        resolved=_jsonl_source(),
        trace_path=f,
        window_start_epoch_s=1000.0,
        window_end_epoch_s=1010.0,
        output_tokens=10,
        token_window_basis="performance_phase_tracked",
        consistent_with_window=True,
        collector_status="ok",
        collector_error=None,
        interval_s=1.0,
    )
    assert rep["status"] == "no_data"
    assert rep["totals"]["energy_j"] is None


def test_collector_failed_status_propagates(tmp_path):
    rep = build_power_report(
        resolved=_jsonl_source(),
        trace_path=tmp_path / "missing.log",
        window_start_epoch_s=1000.0,
        window_end_epoch_s=1010.0,
        output_tokens=10,
        token_window_basis="performance_phase_tracked",
        consistent_with_window=True,
        collector_status="failed",
        collector_error="boom",
        interval_s=1.0,
    )
    assert rep["status"] == "failed"


# --------------------------------------------------------------------------- #
# End-to-end collector with a fake command source
# --------------------------------------------------------------------------- #
def test_collector_end_to_end(tmp_path):
    # A fake source: emit 5 JSONL samples at ~50 ms spacing, then exit.
    script = (
        "import json,sys,time\n"
        "for i in range(5):\n"
        "    print(json.dumps({'ts': time.time(), 'value': 200.0, 'label': 'gpu0'}), flush=True)\n"
        "    time.sleep(0.05)\n"
    )
    cfg = PowerConfig(
        source="command",
        options={"argv": [sys.executable, "-c", script]},
        interval_s=0.05,
    )
    t0 = time.time()
    collector = PowerCollector(cfg, tmp_path / "power")
    collector.start()
    assert collector.status == "ok"
    time.sleep(0.6)  # let it finish
    collector.stop()
    t1 = time.time()

    assert collector.trace_path.exists()
    rep = build_power_report(
        resolved=collector.resolved,
        trace_path=collector.trace_path,
        window_start_epoch_s=t0,
        window_end_epoch_s=t1,
        output_tokens=1000,
        token_window_basis="performance_phase_tracked",
        consistent_with_window=True,
        collector_status=collector.status,
        collector_error=collector.error,
        interval_s=cfg.interval_s,
    )
    assert rep["status"] == "ok"
    assert rep["sources"][0]["sample_count"] >= 3
    assert rep["sources"][0]["power_w"]["mean"] == pytest.approx(200.0)
    assert rep["totals"]["energy_j"] is not None
