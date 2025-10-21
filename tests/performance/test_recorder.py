import os
import random
import time
from dataclasses import dataclass, fields
from pathlib import Path

import pytest
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.metrics.recorder import EventRecorder, MetricsReporter
from inference_endpoint.profiling.line_profiler import ENV_VAR_ENABLE_LINE_PROFILER


def get_EventRecorder(*args, **kwargs):
    # Set requirement to 128MB for testing
    return EventRecorder(*args, min_memory_req_bytes=128 * 1024 * 1024, **kwargs)


@pytest.fixture
def cleanup_connections():
    to_cleanup = {
        "close": [],
        "delete": [],
    }
    yield to_cleanup

    for obj in to_cleanup["close"]:
        obj.close()
    for obj in to_cleanup["delete"]:
        Path(obj).unlink()


class TimingLog:
    def __init__(self, log_file: Path | str | None = None):
        if log_file is None:
            log_file = Path("/tmp/recorder_timing_log.txt")
        self.log_file = Path(log_file)
        self.f_obj = None

    def __enter__(self):
        if self.f_obj is not None:
            raise ValueError("TimingLog already open")
        self.f_obj = self.log_file.open("a")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f_obj.close()
        self.f_obj = None

    def log(self, key: str, duration_sec: float, variant: str = "default"):
        self.f_obj.write(f"[{key}] {variant}: {duration_sec} sec.\n")


@pytest.fixture
def timing_log():
    with TimingLog() as log:
        yield log


@pytest.fixture
def check_time_fn(timing_log):
    def check_time(fn, thresh, log_key: str, variant: str = "default", rel_tol=0.05):
        start_time = time.monotonic_ns()
        r = fn()
        end_time = time.monotonic_ns()
        duration_sec = (end_time - start_time) / 1e9
        timing_log.log(log_key, duration_sec, variant=variant)

        upper_limit = thresh * (1 + rel_tol)
        assert duration_sec <= upper_limit
        return r

    yield check_time


@dataclass
class ReporterTimeThresholds:
    write: float
    ttft: float
    tpot: float
    sample_statuses: float

    def __post_init__(self):
        if os.environ.get(ENV_VAR_ENABLE_LINE_PROFILER, "0") == "1":
            profile_overhead_factor = 5
        else:
            profile_overhead_factor = 1

        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) * profile_overhead_factor)


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize(
    "client_type,time_thresholds",
    [
        (
            "duckdb",
            ReporterTimeThresholds(write=20e-6, ttft=0.1, tpot=1, sample_statuses=0.15),
        ),
        (
            "sqlite",
            ReporterTimeThresholds(write=20e-6, ttft=0.3, tpot=2, sample_statuses=0.3),
        ),
    ],
)
def test_many_chunk_performance(
    client_type, time_thresholds, cleanup_connections, check_time_fn
):
    # Generate a very large number of events, and see if queries return within a threshold
    n_samples = 10
    n_chunks = int(1e6)

    with get_EventRecorder() as rec:
        conn_name = rec.connection_name
        cleanup_connections["delete"].append(conn_name)

        start_time = time.monotonic_ns()
        for sample_uuid in range(n_samples):
            rec.record_event(
                SampleEvent.REQUEST_SENT,
                time.monotonic_ns(),
                sample_uuid=sample_uuid + 1,
            )

        for sample_uuid in range(n_samples):
            rec.record_event(
                SampleEvent.FIRST_CHUNK,
                time.monotonic_ns(),
                sample_uuid=sample_uuid + 1,
            )

        for _ in range(n_chunks):
            rec.record_event(
                SampleEvent.NON_FIRST_CHUNK,
                time.monotonic_ns(),
                sample_uuid=random.randint(1, n_samples),
            )

        for sample_uuid in range(n_samples):
            rec.record_event(
                SampleEvent.COMPLETE, time.monotonic_ns(), sample_uuid=sample_uuid + 1
            )
        rec.wait_for_writes(force_commit=True)
        end_time = time.monotonic_ns()

    assert (
        end_time - start_time
    ) / 1e9 <= n_chunks * time_thresholds.write  # Cap at ~20 microseconds per event

    with MetricsReporter(
        conn_name, client_type=client_type, intermediate_chunks_logged=True
    ) as reporter:
        variant = f"{client_type}_{n_chunks}rows_{n_samples}samples"
        assert check_time_fn(
            reporter.get_sample_statuses,
            time_thresholds.sample_statuses,
            log_key="many_chunk_completed",
            variant=variant,
        ) == {"total_sent": n_samples, "completed": n_samples, "in_flight": 0}
        check_time_fn(
            reporter.derive_TTFT,
            time_thresholds.ttft,
            log_key="many_chunk_ttft",
            variant=variant,
        )
        check_time_fn(
            reporter.derive_TPOT,
            time_thresholds.tpot,
            log_key="many_chunk_tpot",
            variant=variant,
        )


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize(
    "client_type,time_thresholds",
    [
        (
            "duckdb",
            ReporterTimeThresholds(
                write=20e-6, ttft=0.3, tpot=0.3, sample_statuses=0.15
            ),
        ),
        (
            "sqlite",
            ReporterTimeThresholds(
                write=20e-6, ttft=0.6, tpot=0.6, sample_statuses=0.3
            ),
        ),
    ],
)
def test_2_chunk_per_query_performance(
    client_type, time_thresholds, cleanup_connections, check_time_fn
):
    # Generate a very large number of events, and see if queries return within a threshold
    n_events = int(1e6)
    n_queries = n_events // 4

    with get_EventRecorder() as rec:
        conn_name = rec.connection_name
        cleanup_connections["delete"].append(conn_name)

        start_time = time.monotonic_ns()
        for sample_uuid in range(n_queries):
            rec.record_event(
                SampleEvent.REQUEST_SENT,
                time.monotonic_ns(),
                sample_uuid=sample_uuid + 1,
            )

        order = list(range(n_queries))
        random.shuffle(order)
        for sample_uuid in order:
            rec.record_event(
                SampleEvent.FIRST_CHUNK,
                time.monotonic_ns(),
                sample_uuid=sample_uuid + 1,
            )

        random.shuffle(order)
        for sample_uuid in order:
            rec.record_event(
                SampleEvent.NON_FIRST_CHUNK,
                time.monotonic_ns(),
                sample_uuid=sample_uuid + 1,
            )

        random.shuffle(order)
        for sample_uuid in order:
            rec.record_event(
                SampleEvent.COMPLETE, time.monotonic_ns(), sample_uuid=sample_uuid + 1
            )

        rec.wait_for_writes(force_commit=True)
        end_time = time.monotonic_ns()

    assert (
        end_time - start_time
    ) / 1e9 <= n_events * time_thresholds.write  # Cap at ~20 microseconds per event

    with MetricsReporter(
        conn_name, client_type=client_type, intermediate_chunks_logged=False
    ) as reporter:
        variant = f"{client_type}_{n_events}events"
        assert check_time_fn(
            reporter.get_sample_statuses,
            time_thresholds.sample_statuses,
            log_key="2_chunk_per_query_completed",
            variant=variant,
        ) == {"total_sent": n_queries, "completed": n_queries, "in_flight": 0}
        check_time_fn(
            reporter.derive_TTFT,
            time_thresholds.ttft,
            log_key="2_chunk_per_query_ttft",
            variant=variant,
        )
        check_time_fn(
            reporter.derive_TPOT,
            time_thresholds.tpot,
            log_key="2_chunk_per_query_tpot",
            variant=variant,
        )


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
def test_db_write_performance(cleanup_connections, check_time_fn):
    with get_EventRecorder() as rec:
        cleanup_connections["delete"].append(rec.connection_name)

        n_events = int(1e6)

        def bulk_write():
            for i in range(n_events):
                rec.record_event(
                    SampleEvent.REQUEST_SENT, time.monotonic_ns(), sample_uuid=i + 1
                )
            rec.wait_for_writes(force_commit=True)

        check_time_fn(
            bulk_write,
            n_events * 10e-6,
            log_key="bulk_write",
            variant=f"{n_events}events",
        )
