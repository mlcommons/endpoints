import random
from pathlib import Path

import inference_endpoint.metrics as metrics
import pytest
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.load_generator.sample import Sample
from inference_endpoint.load_generator.scheduler import (
    MaxThroughputScheduler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.session import BenchmarkSession
from inference_endpoint.metrics.recorder import MetricsReporter

from tests.test_helpers import DummyDataLoader, PooledSampleIssuer

# The following are tests for PooledSampleIssuer in test_helpers.py. If these tests pass
# The following are tests for PooledSampleIssuer in test_helpers.py. If these tests pass
# but session.py tests fail, it's probably not the PooledSampleIssuer's fault.


def noop(*args, **kwargs):
    pass


empty_callbacks = {
    SampleEvent.REQUEST_SENT: noop,
    SampleEvent.FIRST_CHUNK: noop,
    SampleEvent.NON_FIRST_CHUNK: noop,
    SampleEvent.COMPLETE: noop,
}


def test_pooled_issuer_exception_propagation():
    """Test that exceptions in worker threads are properly propagated to the main thread."""

    def failing_compute(sample):
        raise ValueError("Worker thread error!")

    issuer = PooledSampleIssuer(compute_func=failing_compute, n_workers=2)

    sample1 = Sample(
        uuid="1",
        callbacks=empty_callbacks,
        get_bytes=lambda: b"sample1",
    )
    sample2 = Sample(
        uuid="2",
        callbacks=empty_callbacks,
        get_bytes=lambda: b"sample2",
    )

    # Submit some work that will fail
    issuer.issue(sample1)
    issuer.issue(sample2)

    # Shutdown should raise the exception from the worker thread
    with pytest.raises(ValueError, match="Worker thread error!"):
        issuer.shutdown()


def test_pooled_issuer_futures_cleanup():
    """Test that completed futures are cleaned up to prevent memory leaks."""
    import time

    def slow_compute(sample):
        time.sleep(0.01)  # Small delay
        return sample

    issuer = PooledSampleIssuer(compute_func=slow_compute, n_workers=4)

    # Submit 250 samples (should trigger cleanup at 100 and 200)
    for i in range(250):
        issuer.issue(
            Sample(
                uuid=f"sample{i}",
                callbacks=empty_callbacks,
                get_bytes=lambda: b"sample",
            )
        )

    # Let some complete
    time.sleep(0.5)

    # Manually check errors to trigger cleanup
    issuer.check_errors()

    # The futures list should have been cleaned up
    # With 4 workers and small delays, most should be complete
    assert len(issuer.futures) < 250, "Completed futures were not cleaned up"

    issuer.shutdown()

    # After shutdown, all futures should be cleared
    assert len(issuer.futures) == 0, "Futures not cleared after shutdown"


# session.py tests


def test_session_start():
    rt_settings = RuntimeSettings(
        metrics.Throughput(5000),
        [metrics.Throughput(5000)],
        min_duration_ms=1000,
        max_duration_ms=10_000,
        n_samples_from_dataset=100,
        n_samples_to_issue=10_000,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
    )

    def digits_of_square_iter(n: int):
        yield from str(n**2)

    dl = DummyDataLoader(n_samples=100)
    sample_issuer = PooledSampleIssuer(digits_of_square_iter)
    sched = MaxThroughputScheduler(rt_settings, WithoutReplacementSampleOrder)
    sess = BenchmarkSession.start(
        rt_settings, dl, sample_issuer, sched, name="pytest_test_session_start"
    )
    events_db_path = sess.event_recorder.connection_name
    sess.wait_for_test_end()

    # Shutdown the sample issuer to ensure proper cleanup and error propagation
    sample_issuer.shutdown()

    assert Path(events_db_path).exists()
    with MetricsReporter(events_db_path) as reporter:
        stats = reporter.get_sample_statuses()
        assert stats["total_sent"] == 10_000
        assert stats["completed"] == 10_000
        assert stats["in_flight"] == 0
