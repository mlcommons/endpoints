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

import msgspec.json
import pytest
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.metrics.steady_state.series import (
    build_super_pass_series,
    super_pass_size,
)

_ENC = msgspec.json.Encoder(enc_hook=EventType.encode_hook)


def _write_events(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(_ENC.encode(r))
            f.write(b"\n")


def _sample(uuid, issue_ns, first_ns, complete_ns):
    return [
        EventRecord(
            event_type=SampleEventType.ISSUED, sample_uuid=uuid, timestamp_ns=issue_ns
        ),
        EventRecord(
            event_type=SampleEventType.RECV_FIRST,
            sample_uuid=uuid,
            timestamp_ns=first_ns,
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid=uuid,
            timestamp_ns=complete_ns,
        ),
    ]


@pytest.mark.unit
def test_super_pass_size_rounds_up():
    # dataset_size 4, concurrency 10 -> S=ceil(10/4)=3 -> 12 samples/super-pass
    assert super_pass_size(4, 10) == 12
    # exact multiple
    assert super_pass_size(5, 10) == 10


@pytest.mark.unit
def test_buckets_by_issue_order(tmp_path):
    # dataset_size=2, concurrency=2 -> S=1 -> 2 samples per super-pass.
    p = tmp_path / "events.jsonl"
    recs = [
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0
        )
    ]
    # 4 issued samples -> super-pass 0 = {s0,s1}, super-pass 1 = {s2,s3}
    for i in range(4):
        recs += _sample(
            f"s{i}", issue_ns=100 + i, first_ns=200 + i, complete_ns=500 + i
        )
    recs.append(
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=999
        )
    )
    _write_events(p, recs)

    series = build_super_pass_series(str(p), dataset_size=2, concurrency=2)

    assert [sp.index for sp in series] == [0, 1]
    assert series[0].n_issued == 2
    assert series[0].first_issue_ns == 100 and series[0].last_issue_ns == 101
    assert series[0].ttft_ns == [100.0, 100.0]  # first_ns - issue_ns = 100 each
    assert series[1].n_issued == 2
    assert series[1].first_issue_ns == 102


@pytest.mark.unit
def test_untracked_issued_excluded(tmp_path):
    p = tmp_path / "events.jsonl"
    # one sample issued BEFORE tracking starts (warmup) must be dropped
    recs = _sample("warm", 1, 2, 3)
    recs.append(
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=10
        )
    )
    recs += _sample("s0", 100, 200, 500)
    recs += _sample("s1", 101, 201, 501)
    recs.append(
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=999
        )
    )
    _write_events(p, recs)

    series = build_super_pass_series(str(p), dataset_size=2, concurrency=2)
    assert len(series) == 1
    assert series[0].n_issued == 2  # warmup sample excluded


@pytest.mark.unit
def test_retry_does_not_advance_issue_counter(tmp_path):
    # dataset_size=2, concurrency=2 -> S=1 -> 2 samples per super-pass.
    # A retry (second ISSUED for s0) must NOT advance the issue counter, so
    # s2/s3 must not shift into a phantom 3rd super-pass, and s0's issue ts
    # must be refreshed to the retry timestamp.
    p = tmp_path / "events.jsonl"
    recs = [
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED, sample_uuid="s0", timestamp_ns=100
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED, sample_uuid="s1", timestamp_ns=101
        ),
        # retry: second ISSUED for s0 with a later timestamp
        EventRecord(
            event_type=SampleEventType.ISSUED, sample_uuid="s0", timestamp_ns=150
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED, sample_uuid="s2", timestamp_ns=200
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED, sample_uuid="s3", timestamp_ns=201
        ),
        # complete s0 to inspect its refreshed-issue-ts latency
        EventRecord(
            event_type=SampleEventType.RECV_FIRST, sample_uuid="s0", timestamp_ns=300
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE, sample_uuid="s0", timestamp_ns=500
        ),
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=999
        ),
    ]
    _write_events(p, recs)

    series = build_super_pass_series(str(p), dataset_size=2, concurrency=2)

    # exactly two super-passes; retry did not create a 5th slot / shift s2,s3
    assert len(series) == 2
    assert series[0].n_issued == 2
    assert series[1].n_issued == 2
    # s0 (super-pass 0) latency uses the refreshed issue ts: 500 - 150 = 350
    assert series[0].latency_ns == [350.0]


@pytest.mark.unit
def test_count_tokens_populates_tpot_and_out_tokens(tmp_path):
    # streaming output with a non-empty text_after_first_chunk:
    # output=("hello ", "world") with no reasoning -> skip first chunk -> "world"
    p = tmp_path / "events.jsonl"
    output = TextModelOutput(output=("hello ", "world"))
    assert output.text_after_first_chunk() == "world"
    recs = [
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED, sample_uuid="s0", timestamp_ns=100
        ),
        EventRecord(
            event_type=SampleEventType.RECV_FIRST, sample_uuid="s0", timestamp_ns=200
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="s0",
            timestamp_ns=500,
            data=output,
        ),
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=999
        ),
    ]
    _write_events(p, recs)

    series = build_super_pass_series(
        str(p),
        dataset_size=2,
        concurrency=2,
        count_tokens=lambda texts: [7] * len(texts),
    )
    assert len(series) == 1
    # tpot = (complete - recv_first) / tokens = (500 - 200) / 7
    assert series[0].tpot_ns == [300.0 / 7]
    assert series[0].out_tokens == 7

    # without a tokenizer the same events yield no TPOT / out_tokens
    no_tok = build_super_pass_series(str(p), dataset_size=2, concurrency=2)
    assert no_tok[0].tpot_ns == []
    assert no_tok[0].out_tokens == 0


@pytest.mark.unit
def test_completion_after_stop_tracking_still_counts(tmp_path):
    # a row that exists at issue time keeps counting even if RECV_FIRST/COMPLETE
    # arrive after STOP_PERFORMANCE_TRACKING.
    p = tmp_path / "events.jsonl"
    recs = [
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED, sample_uuid="s0", timestamp_ns=100
        ),
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=150
        ),
        # completion arrives after tracking stopped
        EventRecord(
            event_type=SampleEventType.RECV_FIRST, sample_uuid="s0", timestamp_ns=200
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE, sample_uuid="s0", timestamp_ns=500
        ),
    ]
    _write_events(p, recs)

    series = build_super_pass_series(str(p), dataset_size=2, concurrency=2)
    assert len(series) == 1
    assert series[0].n_issued == 1
    assert series[0].ttft_ns == [100.0]  # 200 - 100
    assert series[0].latency_ns == [400.0]  # 500 - 100
