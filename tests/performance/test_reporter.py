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

import math
import random
import time

import pytest
from inference_endpoint.load_generator.events import SampleEvent, SessionEvent
from inference_endpoint.metrics.recorder import EventRecorder
from inference_endpoint.metrics.reporter import MetricsReporter, TPOTReportingMode
from pympler import asizeof


def get_EventRecorder(*args, **kwargs):
    # Set requirement to 128MB for testing
    return EventRecorder(*args, min_memory_req_bytes=128 * 1024 * 1024, **kwargs)


class CharTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def __call__(
        self, texts: list[str], **kwargs: object
    ) -> dict[str, list[list[int]]]:
        return {"input_ids": [list(range(len(t))) for t in texts]}


def time_fn(fn, *args, **kwargs):
    start_time = time.monotonic_ns()
    result = fn(*args, **kwargs)
    end_time = time.monotonic_ns()
    return result, end_time - start_time


@pytest.mark.skip(reason="Only used to manually test TPOT performance")
@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
def test_tpot_performance(cleanup_connections):
    # Generate a very large number of events, and see if queries return within a threshold
    n_events = int(1e6)
    n_queries = n_events // 4

    with get_EventRecorder() as rec:
        conn_name = rec.connection_name
        cleanup_connections["delete"].append(conn_name)
        cleanup_connections["delete"].append(rec.outputs_path)

        for sample_uuid in range(n_queries):
            rec.record_event(
                SessionEvent.LOADGEN_ISSUE_CALLED,
                time.monotonic_ns(),
                sample_uuid=str(sample_uuid + 1),
            )

        order = list(range(n_queries))
        random.shuffle(order)
        for sample_uuid in order:
            rec.record_event(
                SampleEvent.FIRST_CHUNK,
                time.monotonic_ns(),
                sample_uuid=str(sample_uuid + 1),
                output="a",
            )

        random.shuffle(order)
        for sample_uuid in order:
            rec.record_event(
                SampleEvent.NON_FIRST_CHUNK,
                time.monotonic_ns(),
                sample_uuid=str(sample_uuid + 1),
            )

        random.shuffle(order)
        for sample_uuid in order:
            rec.record_event(
                SampleEvent.COMPLETE,
                time.monotonic_ns(),
                sample_uuid=str(sample_uuid + 1),
                output=["a", "a" * 128],
            )

        rec.wait_for_writes(force_commit=True)

    with MetricsReporter(conn_name) as reporter:
        # Precompute rollups to avoid recomputing them for each test
        ttft_rollup = reporter.derive_TTFT()
        sample_latency_rollup = reporter.derive_sample_latency()

        tpot_condensed, condensed_duration_ns = time_fn(
            reporter.derive_TPOT,
            CharTokenizer(),
            ttft_rollup=ttft_rollup,
            sample_latency_rollup=sample_latency_rollup,
            condense_table=True,
            reporting_mode=TPOTReportingMode.TOKEN_WEIGHTED,
        )

        tpot_full, full_duration_ns = time_fn(
            reporter.derive_TPOT,
            CharTokenizer(),
            ttft_rollup=ttft_rollup,
            sample_latency_rollup=sample_latency_rollup,
            condense_table=False,
            reporting_mode=TPOTReportingMode.TOKEN_WEIGHTED,
        )

    condensed_size = asizeof.asizeof(tpot_condensed)
    full_size = asizeof.asizeof(tpot_full)
    print(f"Condensed TPOT table size: {condensed_size} bytes")
    print(f"Condensed TPOT table calculated in {condensed_duration_ns} ns")
    print(f"Full TPOT table size: {full_size} bytes")
    print(f"Full TPOT table calculated in {full_duration_ns} ns")
    assert condensed_size <= (full_size / 5)
    assert condensed_duration_ns <= full_duration_ns * 1.1

    condensed_summary, condensed_summary_duration_ns = time_fn(
        tpot_condensed.summarize,
    )
    full_summary, full_summary_duration_ns = time_fn(
        tpot_full.summarize,
    )
    print(f"Condensed TPOT summary calculated in {condensed_summary_duration_ns} ns")
    print(f"Full TPOT summary calculated in {full_summary_duration_ns} ns")
    assert condensed_summary_duration_ns < (full_summary_duration_ns / 40)

    # These should definitely be the same
    for k in [
        "total",
        "histogram",
        "min",
        "max",
        "avg",
    ]:
        assert condensed_summary[k] == full_summary[k]

    for k in [
        "median",
        "std_dev",
    ]:
        assert math.isclose(condensed_summary[k], full_summary[k], rel_tol=0.01)

    for percentile in [99.9, 99, 95, 90, 80, 75, 50, 25, 10, 5, 1]:
        assert math.isclose(
            condensed_summary["percentiles"][str(percentile)],
            full_summary["percentiles"][str(percentile)],
            rel_tol=0.01,
        )
