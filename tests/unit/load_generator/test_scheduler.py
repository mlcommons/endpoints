import random

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.core.types import QueryResult
from inference_endpoint.load_generator.scheduler import (
    MaxThroughputScheduler,
    SampleEvent,
    SampleFactory,
    WithoutReplacementSampleOrder,
    WithReplacementSampleOrder,
)

from tests.test_helpers import DummyDataLoader


def test_sample_factory():
    completed = []

    # Create dataloader instance
    dummy_dataloader = DummyDataLoader(n_samples=100)

    class TestingFactory(SampleFactory):
        @staticmethod
        def sample_complete_callback(output, sid=None):
            completed.append(sid)

    factory = TestingFactory(dummy_dataloader)

    indices = list(range(dummy_dataloader.n_samples))
    random.shuffle(indices)

    uuids = set()
    for idx in indices:
        sample = factory(idx)
        assert sample.uuid not in uuids, "UUIDs should be unique but found duplicate"
        uuids.add(sample.uuid)

        obj = sample.get_bytes()
        assert obj == idx, "Sample 'bytes' should be equal to index for DummyDataLoader"

        result = QueryResult(id=sample.uuid, response_output=None)
        sample.callbacks[SampleEvent.COMPLETE](result)
        assert len(completed) == len(
            uuids
        ), "Completed callback should be called for each sample"
        assert (
            completed[-1] == idx
        ), "Completed callback should be called with correct sample index"


@pytest.mark.parametrize(
    "n_samples_to_issue, expected_total_samples_to_issue, metric_target",
    [
        (
            1001,
            1001,
            metrics.Throughput(12345),
        ),  # n_samples_to_issue is set, so total_samples_to_issue should be 1024
        (
            None,
            1358,
            metrics.Throughput(12345),
        ),  # n_samples_to_issue is None, so total_samples_to_issue should be calculated based on the metric target
        (
            1010,
            1010,
            metrics.QueryLatency(15),
        ),  # n_samples_to_issue is set, so total_samples_to_issue should be 1024
        (
            None,
            8,
            metrics.QueryLatency(15),
        ),  # n_samples_to_issue is None, so total_samples_to_issue should be calculated based on the metric target
    ],
)
def test_num_samples_to_issue(
    n_samples_to_issue, expected_total_samples_to_issue, metric_target
):
    """Test that total_samples_to_issue is set correctly when n_samples_to_issue is provided.
    If not provided, it should be calculated based on the metric target.
    For throughput, it should be calculated as ceil(target * (min_duration_ms / 1000) * PADDING_FACTOR)
    For query latency, it should be calculated as ceil(min_duration_ms / target * PADDING_FACTOR)
    """
    dummy_dataloader = DummyDataLoader(n_samples=100)

    class NoOpFactory(SampleFactory):
        @staticmethod
        def sample_complete_callback(output, sid=None):
            pass

    rt_settings = RuntimeSettings(
        metric_target=metric_target,
        reported_metrics=[metric_target],
        min_duration_ms=100,
        max_duration_ms=100,
        n_samples_from_dataset=100,
        n_samples_to_issue=n_samples_to_issue,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
    )

    scheduler = MaxThroughputScheduler(
        runtime_settings=rt_settings,
        dataloader=dummy_dataloader,
        sample_factory_cls=NoOpFactory,
        sample_order_cls=WithoutReplacementSampleOrder,
    )

    assert (
        scheduler.total_samples_to_issue == expected_total_samples_to_issue
    ), "Total samples to issue should be equal to n_samples_to_issue"


def test_without_replacement_sample_order():
    ordering = WithoutReplacementSampleOrder(
        total_samples_to_issue=12345, n_samples_in_dataset=100
    )
    indices = list(iter(ordering))
    for i in range(0, 12345, 100):
        assert len(set(indices[i : i + 100])) == min(
            100, 12345 - i
        ), "Indices should be unique, and occur at least once"

    # Assert that order is different in each pass of the dataset
    assert (
        indices[:100] != indices[100:200]
    ), "Order should be different in each pass of the dataset"


def test_with_replacement_sample_order():
    ordering = WithReplacementSampleOrder(
        total_samples_to_issue=12345, n_samples_in_dataset=100, rng=random.Random(42)
    )
    indices = list(iter(ordering))

    # With Python random.Random(42), the order can be deterministic
    assert indices[:10] == [
        81,
        14,
        3,
        94,
        35,
        31,
        28,
        17,
        94,
        13,
    ], "Order does not match expected deterministic order"
    # Note with this specific seed and order, 94 occurs twice in the first 10 indices
    assert indices[:10].count(94) == 2, "94 should occur twice in the first 10 indices"


def test_max_throughput_scheduler():
    """Test that the MaxThroughputScheduler issues the correct number of samples and delays."""

    expected_delays = [
        96645.35356921388,
        44073.259917535266,
        749.1470058587191,
        91097.59624491242,
        93926.8997363764,
        58222.75730589491,
        67156.34814879851,
        8393.822683708397,
        76648.09327917964,
        23680.977536311777,
        3081.4021726609963,
        78877.27172362835,
        34608.89655971231,
        62328.14750391685,
        61581.56951036152,
        14855.463870828755,
        18309.064740993163,
        11441.296968868764,
        1461.8780486909122,
        48675.154060475834,
        96490.15609162157,
        6456.228097718608,
        54108.81855511302,
        46589.85590083095,
        60146.344956105146,
        8892.882999066233,
        57900.26861873665,
        26958.550381944824,
        55643.25605562156,
        64463.42341782827,
        48103.63713665184,
        35523.91474429834,
        24915.212136120903,
        93351.54980423467,
        45338.80194764936,
        53016.12069115903,
        1929.9566309716854,
        50810.19257797922,
    ]
    num_unique_samples = 100
    dummy_dataloader = DummyDataLoader(n_samples=num_unique_samples)

    class NoOpFactory(SampleFactory):
        @staticmethod
        def sample_complete_callback(output, sid=None):
            pass

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(12345),
        reported_metrics=[metrics.Throughput(12345)],
        min_duration_ms=100,
        max_duration_ms=100,
        n_samples_from_dataset=num_unique_samples,
        n_samples_to_issue=1024,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
    )
    scheduler = MaxThroughputScheduler(
        runtime_settings=rt_settings,
        dataloader=dummy_dataloader,
        sample_factory_cls=NoOpFactory,
        sample_order_cls=WithoutReplacementSampleOrder,
    )
    indices, delays = [], []
    for sample, delay in iter(scheduler):
        indices.append(sample.get_bytes.args[1])
        delays.append(delay)
    assert (
        len(indices) == scheduler.total_samples_to_issue
    ), "Number of indices should be equal to total_samples_to_issue"
    assert (
        len(set(indices)) == num_unique_samples
    ), "Number of unique indices should be equal to n_samples_in_dataset"
    assert (
        delays[: len(expected_delays)] == expected_delays
    ), "Delays should be equal to expected delays for deterministic scheduler"
