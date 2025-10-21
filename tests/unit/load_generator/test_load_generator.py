import random
from collections import defaultdict
from unittest.mock import patch

import inference_endpoint.metrics as metrics
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.load_generator.load_generator import (
    SampleIssuer,
    SchedulerBasedLoadGenerator,
)
from inference_endpoint.load_generator.scheduler import (
    MaxThroughputScheduler,
    SampleOrder,
    WithoutReplacementSampleOrder,
)

from tests.test_helpers import (
    DummyDataLoader,
    NoEventRecordingSample,
    SerialSampleIssuer,
)


class FibonacciSampleOrder(SampleOrder):
    """Sample order where the corresponding value for a sample index is that number value in
    the Fibonacci sequence.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = 0
        self.b = 1

    def next_sample_index(self) -> int:
        retval = self.a
        c = self.a + self.b
        self.a = self.b
        self.b = c
        return retval


@patch("inference_endpoint.load_generator.load_generator.EventRecorder.record_event")
@patch(
    "inference_endpoint.load_generator.load_generator.LoadGenerator.load_sample_data"
)
def test_load_generator(load_sample_data_mock, event_recorder_mock, runtime_settings):
    load_sample_data_mock.side_effect = lambda index, _uuid: index**2
    event_recorder_mock.return_value = True

    class ListAppendIssuer(SampleIssuer):
        def __init__(self):
            self.issued = []

        def issue(self, sample):
            self.issued.append(sample)

    fake_sample_issuer = ListAppendIssuer()

    load_generator = SchedulerBasedLoadGenerator(
        fake_sample_issuer,
        NoEventRecordingSample,
        None,  # No Dataloader to set, we're using Mock to prevent accessing the
        scheduler=MaxThroughputScheduler(
            runtime_settings,
            FibonacciSampleOrder,
        ),
    )
    a = 0
    b = 1
    for i, issued_sample in enumerate(load_generator):
        assert issued_sample.sample.data == a**2
        assert issued_sample.sample == fake_sample_issuer.issued[i]
        assert len(fake_sample_issuer.issued) == i + 1

        c = a + b
        a = b
        b = c


@patch("inference_endpoint.metrics.recorder.EventRecorder.record_event")
def test_full_run(record_event_mock):
    record_event_mock.return_value = None

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

    def compute_digits_of_square(n: int):
        yield from str(n**2)

    sample_issuer = SerialSampleIssuer(compute_digits_of_square)
    load_generator = SchedulerBasedLoadGenerator(
        sample_issuer,
        NoEventRecordingSample,
        DummyDataLoader(100),
        scheduler=MaxThroughputScheduler(
            rt_settings,
            WithoutReplacementSampleOrder,
        ),
    )

    sent_hist = defaultdict(int)
    sent_uuids = defaultdict(list)
    seen_uuids = set()
    for issued_sample in load_generator:
        # The test issuer is serial, so we can confirm that a sample is completed before the next
        # is issued.
        expected = str(issued_sample.index**2)
        assert issued_sample.sample.first_chunk == expected[0]
        assert len(issued_sample.sample.non_first_chunks) == len(expected) - 1
        assert "".join(issued_sample.sample.non_first_chunks) == expected[1:]
        assert "".join(issued_sample.sample.complete_all_chunks) == expected

        sent_hist[issued_sample.index] += 1
        sent_uuids[issued_sample.index].append(issued_sample.sample.uuid)
        seen_uuids.add(issued_sample.sample.uuid)

    # WithoutReplacementSampleOrder should ensure that as long as total # of samples issued is a multiple of dataset size,
    # the number of issues per sample is the same
    target_issues = rt_settings.n_samples_to_issue // rt_settings.n_samples_from_dataset
    for index, n_sent in sent_hist.items():
        assert (
            n_sent == target_issues
        ), f"Sample {index} should have been issued {target_issues} times, but was issued {n_sent} times"

        # Check uuid uniqueness
        n_distinct_uuids = len(set(sent_uuids[index]))
        assert (
            n_distinct_uuids == n_sent
        ), f"Sample {index} should have {n_sent} unique uuids, but has {n_distinct_uuids}"

    # Check that ALL uuids are unique
    assert (
        len(seen_uuids) == rt_settings.n_samples_to_issue
    ), f"Should have seen {rt_settings.n_samples_to_issue} unique uuids, but saw {len(seen_uuids)}"
