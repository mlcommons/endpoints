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

import importlib.util
import json
from pathlib import Path

import msgspec.json
import pytest
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)

_CLI = Path(__file__).resolve().parents[4] / "scripts" / "steady_state_from_events.py"
_spec = importlib.util.spec_from_file_location("steady_state_cli", _CLI)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

_ENC = msgspec.json.Encoder(enc_hook=EventType.encode_hook)


def _write(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(_ENC.encode(r))
            f.write(b"\n")


@pytest.mark.unit
def test_cli_writes_json(tmp_path):
    p = tmp_path / "events.jsonl"
    recs = [
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0
        )
    ]
    t = 100
    for i in range(
        8
    ):  # dataset_size=2, concurrency=2 -> 2/super-pass -> 4 super-passes
        u = f"s{i}"
        recs += [
            EventRecord(
                event_type=SampleEventType.ISSUED, sample_uuid=u, timestamp_ns=t
            ),
            EventRecord(
                event_type=SampleEventType.RECV_FIRST,
                sample_uuid=u,
                timestamp_ns=t + 10,
            ),
            EventRecord(
                event_type=SampleEventType.COMPLETE, sample_uuid=u, timestamp_ns=t + 50
            ),
        ]
        t += 1_000_000_000
    recs.append(
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=t
        )
    )
    _write(p, recs)

    out = tmp_path / "steady.json"
    cli.main(
        [
            str(p),
            "--dataset-size",
            "2",
            "--concurrency",
            "2",
            "--warmup",
            "1",
            "--k",
            "2",
            "--cov-window",
            "2",
            "--cov-bound",
            "0.5",
            "--json",
            str(out),
        ]
    )
    doc = json.loads(out.read_text())
    assert "total" in doc and "steady_state" in doc and "rules" in doc
    assert doc["total"]["sp_start"] == 0
    assert doc["status"] == "windowable" and doc["windowable"] is True


@pytest.mark.unit
def test_cli_partial_dataset_best_effort(tmp_path):
    # 8 samples issued but --dataset-size 100 => < 1 full pass => partial_dataset.
    # Must NOT crash: emit a flagged best-effort steady_state.
    p = tmp_path / "events.jsonl"
    recs = [
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0
        )
    ]
    t = 100
    for i in range(8):
        u = f"s{i}"
        recs += [
            EventRecord(
                event_type=SampleEventType.ISSUED, sample_uuid=u, timestamp_ns=t
            ),
            EventRecord(
                event_type=SampleEventType.RECV_FIRST,
                sample_uuid=u,
                timestamp_ns=t + 10,
            ),
            EventRecord(
                event_type=SampleEventType.COMPLETE, sample_uuid=u, timestamp_ns=t + 50
            ),
        ]
        t += 1_000_000_000
    recs.append(
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=t
        )
    )
    _write(p, recs)

    out = tmp_path / "steady.json"
    doc = cli.main(
        [str(p), "--dataset-size", "100", "--concurrency", "8", "--json", str(out)]
    )
    assert doc["status"] == "partial_dataset"
    assert doc["windowable"] is False
    assert doc["n_issued"] == 8
    # best-effort steady_state is still emitted (not null)
    assert doc["steady_state"]["n_samples"] > 0
