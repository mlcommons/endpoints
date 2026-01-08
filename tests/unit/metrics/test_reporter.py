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
import math

import orjson
import pytest
from inference_endpoint.load_generator.events import SampleEvent, SessionEvent
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
    with sqlite3_cursor(test_db_path) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
                ("", SessionEvent.TEST_STARTED.value, 11000, b""),
                ("", SessionEvent.TEST_ENDED.value, 12000, b""),
            ],
        )
        conn.commit()

    with pytest.raises(
        RuntimeError, match=r"Multiple .*TEST_.* events found - 2 events"
    ):
        with MetricsReporter(test_db_path) as reporter:
            reporter.derive_duration()


def test_derive_duration_multiple_starts_check_malformed_false(tmp_path):
    """Test that derive_duration doesn't raise error for multiple TEST_STARTED when check_malformed=False."""
    test_db_path = str(tmp_path / "multiple_starts.db")
    with sqlite3_cursor(test_db_path) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_STARTED.value, 6000, b""),  # Duplicate start
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    # Should not raise when check_malformed=False
    with MetricsReporter(test_db_path) as reporter:
        duration = reporter.derive_duration(check_malformed=False)

    # Should use max(TEST_STARTED) which is 6000
    assert duration == 10300 - 6000


def test_derive_duration_multiple_ends_check_malformed_false(tmp_path):
    """Test that derive_duration doesn't raise error for multiple TEST_ENDED when check_malformed=False."""
    test_db_path = str(tmp_path / "multiple_ends.db")
    with sqlite3_cursor(test_db_path) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
                ("", SessionEvent.TEST_ENDED.value, 12000, b""),  # Duplicate end
            ],
        )
        conn.commit()

    # Should not raise when check_malformed=False
    with MetricsReporter(test_db_path) as reporter:
        duration = reporter.derive_duration(check_malformed=False)

    # Should use max(timestamp_ns) which is 12000
    assert duration == 12000 - 5000


def test_derive_duration_test_ended_not_last_check_malformed_false(tmp_path):
    """Test that derive_duration doesn't raise error when TEST_ENDED is not max timestamp and check_malformed=False."""
    test_db_path = str(tmp_path / "test_ended_not_last.db")
    with sqlite3_cursor(test_db_path) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
                (
                    "some_uuid",
                    SampleEvent.COMPLETE.value,
                    15000,
                    b"",
                ),  # Event after TEST_ENDED
            ],
        )
        conn.commit()

    # Should raise when check_malformed=True (default)
    with pytest.raises(
        RuntimeError,
        match=r"TEST_ENDED exists .* but is not the maximum timestamp in database",
    ):
        with MetricsReporter(test_db_path) as reporter:
            reporter.derive_duration(check_malformed=True)

    # Should not raise when check_malformed=False
    with MetricsReporter(test_db_path) as reporter:
        duration = reporter.derive_duration(check_malformed=False)

    # Should use max(timestamp_ns) which is 15000
    assert duration == 15000 - 5000


def test_tpot_to_histogram(events_db, fake_outputs, tokenizer, sample_uuids):
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

    with MetricsReporter(events_db) as reporter:
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


def test_reporter_create_report(events_db, fake_outputs, tokenizer):
    with MetricsReporter(events_db) as reporter:
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

    # QPS should be: completed_samples / (duration_ns / 1e9)
    expected_qps = report.n_samples_completed / (report.duration_ns / 1e9)
    assert report.qps == expected_qps

    expected_total_tokens = 0
    for output in fake_outputs.values():
        for chunk in output:
            expected_total_tokens += len(tokenizer.tokenize(chunk))
    expected_tps = expected_total_tokens / ((10300 - 5000) / 1e9)
    assert report.tps == expected_tps


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
        "tps",
    ]
    assert set(json_dict.keys()) == set(expected_keys)
    assert json_dict["n_samples_issued"] == report.n_samples_issued
    assert json_dict["n_samples_completed"] == report.n_samples_completed
    assert json_dict["duration_ns"] == report.duration_ns
    assert json_dict["qps"] == report.qps
    assert json_dict["tps"] == report.tps

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


def test_stop_performance_tracking_timestamp_property(tmp_path, sample_uuids):
    """Test that stop_performance_tracking_timestamp_ns returns correct value when event exists."""
    test_db = str(tmp_path / "test_stop_perf_tracking.db")
    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10100, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        assert reporter.stop_performance_tracking_timestamp_ns == 10100


def test_stop_performance_tracking_timestamp_missing(tmp_path):
    """Test that stop_performance_tracking_timestamp_ns returns infinity when event is missing."""
    test_db = str(tmp_path / "test_no_stop_perf_tracking.db")
    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        assert reporter.stop_performance_tracking_timestamp_ns == float("inf")
        assert reporter.derive_duration() == 10300 - 5000


def test_derive_ttft_with_stop_performance_tracking(tmp_path, sample_uuids):
    """Test that derive_TTFT excludes samples issued after STOP_PERFORMANCE_TRACKING."""
    test_db = str(tmp_path / "test_ttft_stop_perf.db")
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
                (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
                (uuid1, SampleEvent.FIRST_CHUNK.value, 10010, b""),
                (uuid2, SampleEvent.FIRST_CHUNK.value, 10190, b""),
                (
                    "",
                    SessionEvent.STOP_PERFORMANCE_TRACKING.value,
                    10150,
                    b"",
                ),  # Before uuid3 issued
                (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
                (uuid3, SampleEvent.FIRST_CHUNK.value, 10220, b""),
                (uuid1, SampleEvent.COMPLETE.value, 10211, b""),
                (uuid2, SampleEvent.COMPLETE.value, 10219, b""),
                (uuid3, SampleEvent.COMPLETE.value, 10250, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        ttft_rows = reporter.derive_TTFT()

    # Should only include uuid1 and uuid2, not uuid3 (issued after STOP_PERFORMANCE_TRACKING)
    assert len(ttft_rows) == 2
    assert ttft_rows.filter_uuid(uuid1, only_first=True) == 10
    assert ttft_rows.filter_uuid(uuid2, only_first=True) == 187
    assert ttft_rows.filter_uuid(uuid3, only_first=True) is None


def test_derive_sample_latency_with_stop_performance_tracking(tmp_path, sample_uuids):
    """Test that derive_sample_latency excludes samples issued after STOP_PERFORMANCE_TRACKING."""
    test_db = str(tmp_path / "test_latency_stop_perf.db")
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
                (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
                (uuid1, SampleEvent.COMPLETE.value, 10211, b""),
                (uuid2, SampleEvent.COMPLETE.value, 10219, b""),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
                (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
                (uuid3, SampleEvent.COMPLETE.value, 10250, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        latency_rows = reporter.derive_sample_latency()

    # Should only include uuid1 and uuid2
    assert len(latency_rows) == 2
    assert latency_rows.filter_uuid(uuid1, only_first=True) == 10211 - 10000
    assert latency_rows.filter_uuid(uuid2, only_first=True) == 10219 - 10003
    assert latency_rows.filter_uuid(uuid3, only_first=True) is None


def test_derive_duration_with_stop_performance_tracking_no_samples(tmp_path):
    """Test that derive_duration uses STOP_PERFORMANCE_TRACKING timestamp when present."""
    test_db = str(tmp_path / "test_duration_stop_perf.db")
    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10100, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        duration = reporter.derive_duration()

    # Should use STOP_PERFORMANCE_TRACKING - TEST_STARTED (not TEST_ENDED - TEST_STARTED)
    assert duration is None  # Default behavior - No perf test run.


def test_derive_duration_with_stop_performance_tracking(tmp_path, sample_uuids):
    """Test that derive_duration uses STOP_PERFORMANCE_TRACKING timestamp when present."""
    test_db = str(tmp_path / "test_duration_stop_perf.db")
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
                (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
                (uuid1, SampleEvent.COMPLETE.value, 10211, b""),
                (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10213, b""),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10216, b""),
                (
                    uuid2,
                    SampleEvent.COMPLETE.value,
                    10250,
                    b"",
                ),  # Intentionally out of order for test.
                (uuid3, SampleEvent.COMPLETE.value, 10219, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        duration = reporter.derive_duration()

    # Should use timestamp of uuid2's COMPLETE event - timestamp of TEST_STARTED event
    assert duration == 10250 - 5000


def test_derive_duration_all_samples_complete_after_stop_performance_tracking(
    tmp_path, sample_uuids
):
    """Test derive_duration when samples are issued before but all complete after STOP_PERFORMANCE_TRACKING."""
    test_db = str(tmp_path / "test_duration_all_complete_after_stop.db")
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (
                    uuid1,
                    SessionEvent.LOADGEN_ISSUE_CALLED.value,
                    10000,
                    b"",
                ),  # Issued before stop
                (
                    uuid2,
                    SessionEvent.LOADGEN_ISSUE_CALLED.value,
                    10003,
                    b"",
                ),  # Issued before stop
                (
                    uuid3,
                    SessionEvent.LOADGEN_ISSUE_CALLED.value,
                    10010,
                    b"",
                ),  # Issued before stop
                (
                    "",
                    SessionEvent.STOP_PERFORMANCE_TRACKING.value,
                    10100,
                    b"",
                ),  # STOP marker
                # All completions happen AFTER stop_ts
                (
                    uuid1,
                    SampleEvent.COMPLETE.value,
                    10211,
                    b"",
                ),  # Complete after stop
                (
                    uuid2,
                    SampleEvent.COMPLETE.value,
                    10252,
                    b"",
                ),  # Complete after stop
                (
                    uuid3,
                    SampleEvent.COMPLETE.value,
                    10219,
                    b"",
                ),  # Complete after stop
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        duration = reporter.derive_duration()

    # Should use timestamp of the last COMPLETE event (uuid2 at 10250) - TEST_STARTED
    # Since all samples were issued before STOP_PERFORMANCE_TRACKING, they all count,
    # and duration is measured until the last one completes
    assert duration == 10252 - 5000


def test_derive_duration_without_stop_performance_tracking(tmp_path):
    """Test that derive_duration uses TEST_ENDED when STOP_PERFORMANCE_TRACKING is absent."""
    test_db = str(tmp_path / "test_duration_no_stop_perf.db")
    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        duration = reporter.derive_duration()

    # Should use TEST_ENDED - TEST_STARTED
    assert duration == 10300 - 5000


def test_get_sample_statuses_with_stop_performance_tracking(tmp_path, sample_uuids):
    """Test that get_sample_statuses excludes samples issued after STOP_PERFORMANCE_TRACKING.

    This test verifies:
    1. Samples issued before stop_ts are counted in total_sent
    2. Samples issued after stop_ts are NOT counted in total_sent
    3. Completed samples are only counted if they were issued before stop_ts
    4. Samples issued before stop_ts but completing after stop_ts ARE still counted as completed
    """
    test_db = str(tmp_path / "test_statuses_stop_perf.db")
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (
                    uuid1,
                    SessionEvent.LOADGEN_ISSUE_CALLED.value,
                    10000,
                    b"",
                ),  # Issued before stop_ts
                (
                    uuid2,
                    SessionEvent.LOADGEN_ISSUE_CALLED.value,
                    10003,
                    b"",
                ),  # Issued before stop_ts
                (
                    uuid1,
                    SampleEvent.COMPLETE.value,
                    10100,
                    b"",
                ),  # Completed before stop_ts
                (
                    "",
                    SessionEvent.STOP_PERFORMANCE_TRACKING.value,
                    10150,
                    b"",
                ),  # STOP marker
                (
                    uuid3,
                    SessionEvent.LOADGEN_ISSUE_CALLED.value,
                    10200,
                    b"",
                ),  # Issued AFTER stop_ts
                (
                    uuid2,
                    SampleEvent.COMPLETE.value,
                    10219,
                    b"",
                ),  # Issued before but completed AFTER stop_ts
                (uuid3, SampleEvent.COMPLETE.value, 10250, b""),  # Issued after stop_ts
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        stats = reporter.get_sample_statuses()
        assert reporter.stop_performance_tracking_timestamp_ns == 10150

    # Should only count uuid1 and uuid2 as issued (uuid3 issued after cutoff)
    # Both uuid1 and uuid2 should be counted as completed even though uuid2 completed after stop_ts
    assert stats["total_sent"] == 2
    assert (
        stats["completed"] == 2
    )  # uuid1 and uuid2 (uuid3 not counted because it was issued after stop_ts)
    assert stats["in_flight"] == 0


def test_get_sample_statuses_excludes_late_issued_completions(tmp_path, sample_uuids):
    """Test that completed samples issued after STOP_PERFORMANCE_TRACKING are not counted.

    This specifically tests the edge case where a sample is issued after stop_ts and completes,
    ensuring it's not included in the completed count.
    """
    test_db = str(tmp_path / "test_late_completion.db")
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)
    uuid4 = sample_uuids(4)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
                (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
                (uuid1, SampleEvent.COMPLETE.value, 10100, b""),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
                # uuid2 still in flight when stop_ts happens
                (
                    uuid3,
                    SessionEvent.LOADGEN_ISSUE_CALLED.value,
                    10200,
                    b"",
                ),  # Issued after stop_ts
                (
                    uuid4,
                    SessionEvent.LOADGEN_ISSUE_CALLED.value,
                    10205,
                    b"",
                ),  # Issued after stop_ts
                (
                    uuid3,
                    SampleEvent.COMPLETE.value,
                    10250,
                    b"",
                ),  # Completes but was issued after stop_ts
                (
                    uuid4,
                    SampleEvent.COMPLETE.value,
                    10260,
                    b"",
                ),  # Completes but was issued after stop_ts
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        stats = reporter.get_sample_statuses()

    # Only uuid1 and uuid2 should be counted (issued before stop_ts)
    assert stats["total_sent"] == 2
    # Only uuid1 completed (uuid2 never completed, uuid3 and uuid4 don't count)
    assert stats["completed"] == 1
    assert stats["in_flight"] == 1  # uuid2 is still in flight


def test_get_output_sequence_lengths_with_stop_performance_tracking(
    tmp_path, sample_uuids, tokenizer
):
    """Test that get_output_sequence_lengths excludes samples issued after STOP_PERFORMANCE_TRACKING."""
    test_db = str(tmp_path / "test_osl_stop_perf.db")
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
                (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
                (
                    uuid1,
                    SampleEvent.COMPLETE.value,
                    10211,
                    orjson.dumps({"output": ["Hello, ", "world"]}),
                ),
                (
                    uuid2,
                    SampleEvent.COMPLETE.value,
                    10219,
                    orjson.dumps({"output": ["And ", "goodbye."]}),
                ),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
                (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
                (
                    uuid3,
                    SampleEvent.COMPLETE.value,
                    10250,
                    orjson.dumps({"output": ["Extra ", "sample"]}),
                ),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        osl_rollup = reporter.get_output_sequence_lengths(tokenizer)

    # Should only include uuid1 and uuid2 (uuid3 issued after STOP_PERFORMANCE_TRACKING)
    assert len(osl_rollup) == 2
    assert uuid1 in osl_rollup
    assert uuid2 in osl_rollup
    assert uuid3 not in osl_rollup


def test_create_report_with_stop_performance_tracking(
    tmp_path, sample_uuids, tokenizer
):
    """Test that create_report respects STOP_PERFORMANCE_TRACKING for all metrics."""
    test_db = str(tmp_path / "test_report_stop_perf.db")
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
                (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
                (uuid1, SampleEvent.FIRST_CHUNK.value, 10010, b""),
                (uuid2, SampleEvent.FIRST_CHUNK.value, 10190, b""),
                (
                    uuid1,
                    SampleEvent.COMPLETE.value,
                    10211,
                    orjson.dumps({"output": ["Hello, ", "world"]}),
                ),
                (
                    uuid2,
                    SampleEvent.COMPLETE.value,
                    10219,
                    orjson.dumps({"output": ["And ", "goodbye."]}),
                ),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10150, b""),
                (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10200, b""),
                (uuid3, SampleEvent.FIRST_CHUNK.value, 10220, b""),
                (
                    uuid3,
                    SampleEvent.COMPLETE.value,
                    10250,
                    orjson.dumps({"output": ["Extra ", "sample"]}),
                ),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        report = reporter.create_report(tokenizer)

    # Verify that only uuid1 and uuid2 are counted
    assert report.n_samples_issued == 2  # Only uuid1 and uuid2
    assert report.n_samples_completed == 2
    assert (
        report.duration_ns == 10219 - 5000
    )  # timestamp of uuid2's COMPLETE event - timestamp of TEST_STARTED event

    # Verify QPS is based on the truncated duration
    expected_qps = 2 / ((10219 - 5000) / 1e9)
    assert report.qps == expected_qps

    # Verify latency includes only uuid1 and uuid2
    assert report.latency["total"] == (10211 - 10000) + (10219 - 10003)


def test_create_report_with_zero_samples_before_stop_performance_tracking(tmp_path):
    """Test that create_report shows 'Duration: N/A' when 0 samples issued before STOP_PERFORMANCE_TRACKING."""
    test_db = str(tmp_path / "test_zero_samples_stop_perf.db")

    with sqlite3_cursor(test_db) as (cursor, conn):
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER, data BLOB)"
        )
        cursor.executemany(
            "INSERT INTO events (sample_uuid, event_type, timestamp_ns, data) VALUES (?, ?, ?, ?)",
            [
                ("", SessionEvent.TEST_STARTED.value, 5000, b""),
                ("", SessionEvent.STOP_PERFORMANCE_TRACKING.value, 10100, b""),
                ("", SessionEvent.TEST_ENDED.value, 10300, b""),
            ],
        )
        conn.commit()

    with MetricsReporter(test_db) as reporter:
        report = reporter.create_report()

    # Verify report values
    assert report.n_samples_issued == 0
    assert report.n_samples_completed == 0
    assert report.duration_ns is None

    # Verify display shows 'Duration: N/A'
    import io

    buf = io.StringIO()

    def _write_with_newline(s):
        buf.write(s + "\n")

    report.display(fn=_write_with_newline)
    display_output = buf.getvalue()

    assert "Duration: N/A" in display_output
    assert "(no performance samples were issued)" in display_output
