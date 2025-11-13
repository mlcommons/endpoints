# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math

import pytest
from inference_endpoint.load_generator.events import SessionEvent
from inference_endpoint.metrics.recorder import sqlite3_cursor
from inference_endpoint.metrics.reporter import (
    MetricsReporter,
    RollupQueryTable,
    TPOTReportingMode,
)


def test_sample_counting(events_db):
    with MetricsReporter(events_db) as reporter:
        stats = reporter.get_sample_statuses()
        assert stats["completed"] == 2
        assert stats["in_flight"] == 1


def test_error_counting(events_db):
    with MetricsReporter(events_db) as reporter:
        assert reporter.get_error_count() == 3


def test_derive_ttft(events_db, sample_uuids):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    with MetricsReporter(events_db) as reporter:
        ttft_rows = reporter.derive_TTFT()
    assert len(ttft_rows) == 2
    assert ttft_rows[0].metric_type == "ttft"
    assert ttft_rows[1].metric_type == "ttft"
    assert ttft_rows.filter_uuid(uuid1, only_first=True) == 10
    assert ttft_rows.filter_uuid(uuid2, only_first=True) == 187
    assert ttft_rows.filter_uuid("asdf", only_first=True) is None
    assert ttft_rows.filter_uuid("asdf", only_first=False) == ()


def test_derive_tpot(events_db, sample_uuids, fake_outputs, tokenizer):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    with MetricsReporter(events_db) as reporter:
        reporter.outputs_path = fake_outputs.path
        tpot_rows = reporter.derive_TPOT(
            tokenizer, reporting_mode=TPOTReportingMode.TOKEN_WEIGHTED
        )

    # From test_derive_sample_latency and ttft:
    expected_tpot1 = (10211 - 10000 - 10) / len(fake_outputs[uuid1][1])
    expected_tpot2 = (10219 - 10003 - 187) / len(fake_outputs[uuid2][1])

    tpot1 = tpot_rows.filter_uuid(uuid1, only_first=False)
    tpot2 = tpot_rows.filter_uuid(uuid2, only_first=False)
    assert len(tpot1) == len(fake_outputs[uuid1][1])
    assert len(tpot2) == len(fake_outputs[uuid2][1])
    assert all(tpot == expected_tpot1 for tpot in tpot1)
    assert all(tpot == expected_tpot2 for tpot in tpot2)


def test_derive_sample_latency(events_db, sample_uuids):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    with MetricsReporter(events_db) as reporter:
        sample_latency_rows = reporter.derive_sample_latency()

    assert len(sample_latency_rows) == 2
    latency1, latency2 = tuple(sorted(sample_latency_rows, key=lambda x: x.sample_uuid))
    assert latency1.metric_type == "sample_latency"
    assert latency1.sample_uuid == uuid1
    assert latency1.metric_value == 10211 - 10000

    assert latency2.metric_type == "sample_latency"
    assert latency2.sample_uuid == uuid2
    assert latency2.metric_value == 10219 - 10003


def test_derive_duration(events_db):
    with MetricsReporter(events_db) as reporter:
        duration = reporter.derive_duration()
    assert duration == (10300 - 5000)


def test_derive_duration_malformed(tmp_path):
    test_db_path = str(tmp_path / "bad_events.db")
    with sqlite3_cursor(test_db_path) as (cursor, _):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns) VALUES (?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000),
                ("", SessionEvent.TEST_ENDED.value, 10300),
                ("", SessionEvent.TEST_STARTED.value, 11000),
                ("", SessionEvent.TEST_ENDED.value, 12000),
            ],
        )

    with MetricsReporter(test_db_path) as reporter:
        with pytest.raises(
            RuntimeError, match="Multiple TEST_STARTED or TEST_ENDED events found"
        ):
            reporter.derive_duration()


def test_tpot_to_histogram(
    events_db_reporter_with_fake_outputs, fake_outputs, tokenizer, sample_uuids
):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    expected = [
        {
            "tpot": (10211 - 10000 - 10) / len(fake_outputs[uuid1][1]),
            "count": len(fake_outputs[uuid1][1]),
        },
        {
            "tpot": (10219 - 10003 - 187) / len(fake_outputs[uuid2][1]),
            "count": len(fake_outputs[uuid2][1]),
        },
    ]
    expected.sort(key=lambda x: x["tpot"])

    bucket_boundaries = [
        expected[0]["tpot"] - 1,
        (expected[0]["tpot"] + expected[1]["tpot"]) / 2,
        expected[1]["tpot"] + 1,
    ]

    reporter = events_db_reporter_with_fake_outputs
    tpot_rows = reporter.derive_TPOT(
        tokenizer, reporting_mode=TPOTReportingMode.TOKEN_WEIGHTED
    )

    # This isn't documented since it's an internal detail and should not be relied on, but `n_buckets`
    # is passed directly to np.histogram, so we can specify exact buckets to use
    buckets, counts = tpot_rows.to_histogram(n_buckets=bucket_boundaries)
    assert len(buckets) == 2
    assert len(counts) == 2

    assert buckets[0] == (bucket_boundaries[0], bucket_boundaries[1])
    assert buckets[1] == (bucket_boundaries[1], bucket_boundaries[2])
    assert counts[0] == expected[0]["count"]
    assert counts[1] == expected[1]["count"]


def test_percentile():
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    table = RollupQueryTable(
        metric_type="test", from_query="", rows=[(0, v) for v in values]
    )
    assert table.percentile(50) == 5
    assert table.percentile([50, 75]) == {50: 5, 75: 7.5}
    assert table.percentile(90) == 9
    with pytest.raises(TypeError):
        table.percentile("10")
    with pytest.raises(ValueError):
        table.percentile(101)
    with pytest.raises(ValueError):
        table.percentile(-1)


def test_rollup_summarize(events_db):
    with MetricsReporter(events_db) as reporter:
        latencies = reporter.derive_sample_latency()
    summary = latencies.summarize()
    values = [10211 - 10000, 10219 - 10003]
    assert summary["total"] == sum(values)
    assert summary["min"] == min(values)
    assert summary["max"] == max(values)
    assert summary["median"] == (values[0] + values[1]) / 2
    assert summary["avg"] == (values[0] + values[1]) / 2

    deviations_squared = [(value - summary["avg"]) ** 2 for value in values]

    assert math.isclose(
        summary["std_dev"],
        math.sqrt(sum(deviations_squared) / len(values)),
        rel_tol=1e-3,
    )

    for percentile in [99.9, 99, 95, 90, 80, 75, 50, 25, 10, 5, 1]:
        s = str(percentile)
        assert s in summary["percentiles"]
        assert summary["percentiles"][s] == latencies.percentile(percentile)


def test_reporter_create_report(events_db_reporter_with_fake_outputs, tokenizer):
    reporter = events_db_reporter_with_fake_outputs

    report = reporter.create_report(tokenizer)

    # Expected
    ttft_rollup = reporter.derive_TTFT()
    sample_latency_rollup = reporter.derive_sample_latency()
    tpot_rollup = reporter.derive_TPOT(
        tokenizer,
        ttft_rollup=ttft_rollup,
        sample_latency_rollup=sample_latency_rollup,
    )

    assert report.n_samples_issued == 3
    assert report.n_samples_completed == 2
    assert report.duration_ns == (10300 - 5000)

    for k, expected in ttft_rollup.summarize().items():
        assert k in report.ttft
        assert report.ttft[k] == expected
    for k, expected in tpot_rollup.summarize().items():
        assert k in report.tpot
        assert report.tpot[k] == expected
    for k, expected in sample_latency_rollup.summarize().items():
        assert k in report.latency
        assert report.latency[k] == expected
    for k, expected in tpot_rollup.summarize().items():
        assert k in report.tpot
        assert report.tpot[k] == expected

    expected_e2e_latency_ns = (10211 - 10000) + (10219 - 10003)
    # QPS should be: completed_samples / (duration_ns / 1e9)
    expected_qps = report.n_samples_completed / (report.duration_ns / 1e9)
    assert report.qps == expected_qps
    assert report.e2e_sample_latency_sec == expected_e2e_latency_ns / 1e9


def test_reporter_json(events_db):
    with MetricsReporter(events_db) as reporter:
        report = reporter.create_report()

    json_str = report.to_json()

    json_dict = json.loads(json_str)

    expected_keys = [
        "n_samples_issued",
        "n_samples_completed",
        "duration_ns",
        "ttft",
        "tpot",
        "latency",
        "output_sequence_lengths",
        "tpot_reporting_mode",
        "qps",
        "e2e_sample_latency_sec",
    ]
    assert set(json_dict.keys()) == set(expected_keys)
    assert json_dict["n_samples_issued"] == report.n_samples_issued
    assert json_dict["n_samples_completed"] == report.n_samples_completed
    assert json_dict["duration_ns"] == report.duration_ns
    assert json_dict["qps"] == report.qps
    assert json_dict["e2e_sample_latency_sec"] == report.e2e_sample_latency_sec

    # For ttft, tpot, and latency, JSON decode will only decode as lists, not tuples
    # This only matters in the histogram
    def _assert_rollup_summary_equal(json_dict, summary_dict):
        if summary_dict is None:
            assert json_dict is None
            return

        for k in summary_dict.keys():
            if k == "histogram":
                continue
            assert json_dict[k] == summary_dict[k]

        assert json_dict["histogram"]["buckets"] == [
            list(bucket) for bucket in summary_dict["histogram"]["buckets"]
        ]
        assert json_dict["histogram"]["counts"] == summary_dict["histogram"]["counts"]

    _assert_rollup_summary_equal(json_dict["ttft"], report.ttft)
    _assert_rollup_summary_equal(json_dict["tpot"], report.tpot)
    _assert_rollup_summary_equal(json_dict["latency"], report.latency)
    _assert_rollup_summary_equal(
        json_dict["output_sequence_lengths"], report.output_sequence_lengths
    )


def test_display_report(events_db):
    with MetricsReporter(events_db) as reporter:
        report = reporter.create_report()

    import io

    buf = io.StringIO()

    def _write_with_newline(s):
        buf.write(s + "\n")

    report.display(fn=_write_with_newline)
    s = buf.getvalue()
    lines = s.splitlines()

    assert "- Summary -" in lines[0]
    assert lines[1].startswith("Total samples issued:")
