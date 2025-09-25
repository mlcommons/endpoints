import random
import time
from collections import defaultdict

from inference_endpoint import metrics
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.load_generator import LoadGenerator, SampleIssuer
from inference_endpoint.load_generator.scheduler import (
    MaxThroughputScheduler,
    SampleFactory,
    WithoutReplacementSampleOrder,
)


class DummySampleFactory(SampleFactory):
    _output_histogram = defaultdict(int)

    @staticmethod
    def sample_complete_callback(output, sid=None):
        assert isinstance(output, list)
        assert len(output) == 1
        assert output[0] == (
            sid**2
        )  # DummyDataloader returns sample index as data, model is x^2
        DummySampleFactory._output_histogram[sid] += 1


def DummyXSquaredModel(x):
    return x**2


class DummySampleIssuer(SampleIssuer):
    def process_sample_data(self, s_uuid: int, sample_data: bytes):
        output = DummyXSquaredModel(sample_data)
        self.push_response_chunk(s_uuid, output)
        self.stop_response_thread(s_uuid)


def test_load_generator_full_run(dummy_dataloader):
    # Reset for test
    DummySampleFactory._output_histogram.clear()

    rt_settings = RuntimeSettings(
        metrics.Throughput(5000),
        [metrics.Throughput(5000)],
        min_duration_ms=1000,
        max_duration_ms=3000,
        n_samples_from_dataset=100,
        n_samples_to_issue=10_000,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
    )
    scheduler = MaxThroughputScheduler(
        rt_settings, dummy_dataloader, DummySampleFactory, WithoutReplacementSampleOrder
    )
    sample_issuer = DummySampleIssuer()
    load_generator = LoadGenerator(scheduler, sample_issuer)
    sess = load_generator.start_test()

    sess.wait_for_test_end()
    end_time_ns = time.monotonic_ns()
    assert sess.start_time_ns is not None
    assert (
        sess.min_end_time_ns - sess.start_time_ns == rt_settings.min_duration_ms * 1e6
    )
    assert (
        sess.max_end_time_ns - sess.start_time_ns == rt_settings.max_duration_ms * 1e6
    )
    duration_ns = end_time_ns - sess.start_time_ns
    assert duration_ns >= rt_settings.min_duration_ms * 1e6
    assert duration_ns <= rt_settings.max_duration_ms * 1e6

    # WithoutReplacementSampleOrder should ensure that as long as total # of samples issued is a multiple of dataset size,
    # the number of issues per sample is the same
    target_issues = rt_settings.n_samples_to_issue // rt_settings.n_samples_from_dataset
    for sid, issues in DummySampleFactory._output_histogram.items():
        assert (
            issues == target_issues
        ), f"Sample {sid} should have been issued {target_issues} times, but was issued {issues} times"
