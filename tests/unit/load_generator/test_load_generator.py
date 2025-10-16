import random

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

from tests.test_helpers import HistogramSampleFactory, SerialSampleIssuer


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


def test_load_generator(runtime_settings):
    class ListAppendIssuer(SampleIssuer):
        def __init__(self):
            self.issued = []

        def issue(self, sample):
            self.issued.append(sample)

    def fake_sample_factory(s_idx):
        return s_idx**2

    fake_sample_issuer = ListAppendIssuer()

    load_generator = SchedulerBasedLoadGenerator(
        fake_sample_issuer,
        fake_sample_factory,
        scheduler=MaxThroughputScheduler(
            runtime_settings,
            FibonacciSampleOrder,
        ),
    )
    a = 0
    b = 1
    for i, (sample, _) in enumerate(load_generator):
        assert sample == a**2
        assert sample == fake_sample_issuer.issued[i]
        assert len(fake_sample_issuer.issued) == i + 1
        c = a + b
        a = b
        b = c


def test_full_run():
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

    sample_factory = HistogramSampleFactory()
    sample_issuer = SerialSampleIssuer(digits_of_square_iter)
    load_generator = SchedulerBasedLoadGenerator(
        sample_issuer,
        sample_factory,
        scheduler=MaxThroughputScheduler(
            rt_settings,
            WithoutReplacementSampleOrder,
        ),
    )

    for sample, _ in load_generator:
        # The test issuer is serial, so we can confirm that a sample is completed before the next
        # is issued.
        assert "".join(sample_factory.completed_chunks[sample.uuid]) == str(
            sample_factory.uuid_to_idx[sample.uuid] ** 2
        )

    # WithoutReplacementSampleOrder should ensure that as long as total # of samples issued is a multiple of dataset size,
    # the number of issues per sample is the same
    target_issues = rt_settings.n_samples_to_issue // rt_settings.n_samples_from_dataset
    for sid, n_sent in sample_factory.sent_hist.items():
        assert (
            n_sent == target_issues
        ), f"Sample {sid} should have been issued {target_issues} times, but was issued {n_sent} times"
    for sid, n_completed in sample_factory.completed_hist.items():
        assert (
            n_completed == target_issues
        ), f"Sample {sid} should have been completed {target_issues} times, but was completed {n_completed} times"
