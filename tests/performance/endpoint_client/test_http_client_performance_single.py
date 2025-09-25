"""Performance tests for HTTPEndpointClient with single worker using echo server."""

import logging
import random

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.dataset_manager.dataloader import DataLoader
from inference_endpoint.endpoint_client.loadgen import HttpClientSampleIssuer
from inference_endpoint.load_generator import LoadGenerator
from inference_endpoint.load_generator.scheduler import (
    MaxThroughputScheduler,
    WithoutReplacementSampleOrder,
)

from tests.performance.utils import MetricsSampleFactory

logger = logging.getLogger(__name__)

# =============================================================================
# PERFORMANCE TARGETS - Single worker configuration
# =============================================================================

PERFORMANCE_CONFIG = {
    # Common test parameters
    "warmup_requests": 100,
    "test_duration_ms": 5000,  # 5 seconds in milliseconds
    "default_message_size": 1000 * 4,  # 4k characters
    # Mode-specific target QPS
    "streaming_target_qps": 1500,
    "offline_target_qps": 1500,
    # Test to maintain peak throughput at these message sizes
    "message_sizes": [100 * 4, 500 * 4, 1000 * 4, 2000 * 4],
    # Assertions (pass/fail criteria)
    "required_success_rate_percent": 100,
    "target_qps_tolerance": 0.90,  # measurement noise tolerance
    "p99_degradation_tolerance": 1.10,  # across message sizes
}


# Helper dataloader for Query objects
class QueryDataLoader(DataLoader):
    """Dataloader that extracts .data from Query objects."""

    def __init__(self, queries):
        super().__init__(None)
        self.queries = queries
        self.n_samples = len(queries)

    def load(self):
        pass

    def num_samples(self) -> int:
        return self.n_samples

    def load_sample(self, sample_index: int):
        assert 0 <= sample_index < self.n_samples
        return self.queries[sample_index].data


@pytest.mark.timeout(0)  # Disable timeout for all performance tests
class TestHTTPClientPerformanceSingleWorker:
    """Performance tests for HTTPEndpointClient with single worker."""

    # =========================================================================
    # 1. BASELINE PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    def test_streaming_baseline_performance(
        self,
        http_client,
        create_test_query,
    ):
        """Test baseline streaming performance at target QPS."""
        target_qps = PERFORMANCE_CONFIG["streaming_target_qps"]
        duration_ms = PERFORMANCE_CONFIG["test_duration_ms"]
        message_size = PERFORMANCE_CONFIG["default_message_size"]

        # Create test queries
        num_samples = 100
        queries = [
            create_test_query(prompt_size=message_size, stream=True)
            for _ in range(num_samples)
        ]

        # Create dataloader
        dataloader = QueryDataLoader(queries)
        dataloader.load()

        # Create runtime settings
        rt_settings = RuntimeSettings(
            metric_target=metrics.Throughput(target_qps),
            reported_metrics=[metrics.Throughput(target_qps)],
            min_duration_ms=duration_ms,
            max_duration_ms=duration_ms * 2,
            n_samples_from_dataset=num_samples,
            n_samples_to_issue=None,  # Run until duration
            rng_sched=random.Random(1234),
            rng_sample_index=random.Random(1234),
        )

        # Create scheduler with metrics-enabled factory
        scheduler = MaxThroughputScheduler(
            rt_settings,
            dataloader,
            MetricsSampleFactory,
            WithoutReplacementSampleOrder,
        )

        # Use the scheduler's factory instance for the issuer
        sample_factory = scheduler.sample_factory
        sample_issuer = HttpClientSampleIssuer(http_client)
        sample_issuer.start()

        # Create load generator
        load_gen = LoadGenerator(scheduler, sample_issuer)

        # Start test
        try:
            sess = load_gen.start_test()
            sess.wait_for_test_end()

            # Wait for all pending responses to complete
            sample_issuer.wait_for_all_complete()
        finally:
            sample_factory.metrics.stop()
            sample_issuer.shutdown()

        # Get summary from factory's metrics
        summary = sample_factory.metrics.get_summary()

        # Verify performance
        achieved_qps = summary["qps"]
        min_achievement = PERFORMANCE_CONFIG["target_qps_tolerance"]

        logger.info(
            f"Streaming baseline: Target={target_qps} QPS, Achieved={achieved_qps:.2f} QPS, "
            f"Success Rate={summary['success_rate']:.2f}%, P99={summary['latencies']['p99']:.3f}s"
        )

        assert (
            achieved_qps >= target_qps * min_achievement
        ), f"Failed to achieve {min_achievement*100:.0f}% of target {target_qps} QPS (got {achieved_qps:.2f})"

        required_success_rate = PERFORMANCE_CONFIG["required_success_rate_percent"]
        assert (
            summary["success_rate"] >= required_success_rate
        ), f"Success rate must be >= {required_success_rate}%, got {summary['success_rate']}%"

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    def test_offline_baseline_performance(
        self,
        http_client,
        create_test_query,
    ):
        """Test baseline offline (non-streaming) performance at target QPS."""
        target_qps = PERFORMANCE_CONFIG["offline_target_qps"]
        duration_ms = PERFORMANCE_CONFIG["test_duration_ms"]
        message_size = PERFORMANCE_CONFIG["default_message_size"]

        # Create test queries
        num_samples = 100
        queries = [
            create_test_query(prompt_size=message_size, stream=False)
            for _ in range(num_samples)
        ]

        # Create dataloader
        dataloader = QueryDataLoader(queries)
        dataloader.load()

        # Create runtime settings
        rt_settings = RuntimeSettings(
            metric_target=metrics.Throughput(target_qps),
            reported_metrics=[metrics.Throughput(target_qps)],
            min_duration_ms=duration_ms,
            max_duration_ms=duration_ms * 2,
            n_samples_from_dataset=num_samples,
            n_samples_to_issue=None,
            rng_sched=random.Random(1234),
            rng_sample_index=random.Random(1234),
        )

        # Create scheduler with metrics-enabled factory
        scheduler = MaxThroughputScheduler(
            rt_settings,
            dataloader,
            MetricsSampleFactory,
            WithoutReplacementSampleOrder,
        )

        # Use the scheduler's factory instance for the issuer
        sample_factory = scheduler.sample_factory
        sample_issuer = HttpClientSampleIssuer(http_client)
        sample_issuer.start()

        # Create load generator
        load_gen = LoadGenerator(scheduler, sample_issuer)

        # Start test
        try:
            sess = load_gen.start_test()
            sess.wait_for_test_end()

            # Wait for all pending responses to complete
            sample_issuer.wait_for_all_complete()
        finally:
            sample_factory.metrics.stop()
            sample_issuer.shutdown()

        # Get summary
        summary = sample_factory.metrics.get_summary()

        # Verify performance
        achieved_qps = summary["qps"]
        min_achievement = PERFORMANCE_CONFIG["target_qps_tolerance"]

        logger.info(
            f"Offline baseline: Target={target_qps} QPS, Achieved={achieved_qps:.2f} QPS, "
            f"Success Rate={summary['success_rate']:.2f}%"
        )

        assert (
            achieved_qps >= target_qps * min_achievement
        ), f"Failed to achieve {min_achievement*100:.0f}% of target {target_qps} QPS (got {achieved_qps:.2f})"

        required_success_rate = PERFORMANCE_CONFIG["required_success_rate_percent"]
        assert (
            summary["success_rate"] >= required_success_rate
        ), f"Success rate must be >= {required_success_rate}%, got {summary['success_rate']}%"

    # =========================================================================
    # 2. THROUGHPUT UNDER VARIOUS MESSAGE SIZES
    # =========================================================================

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    @pytest.mark.parametrize("message_size", PERFORMANCE_CONFIG["message_sizes"])
    def test_streaming_throughput_various_message_sizes(
        self,
        http_client,
        create_test_query,
        message_size,
    ):
        """Test that streaming maintains target QPS across different message sizes."""
        target_qps = PERFORMANCE_CONFIG["streaming_target_qps"]
        duration_ms = PERFORMANCE_CONFIG["test_duration_ms"]

        # Create test queries
        num_samples = 100
        queries = [
            create_test_query(prompt_size=message_size, stream=True)
            for _ in range(num_samples)
        ]

        # Create dataloader
        dataloader = QueryDataLoader(queries)
        dataloader.load()

        # Create runtime settings
        rt_settings = RuntimeSettings(
            metric_target=metrics.Throughput(target_qps),
            reported_metrics=[metrics.Throughput(target_qps)],
            min_duration_ms=duration_ms,
            max_duration_ms=duration_ms * 2,
            n_samples_from_dataset=num_samples,
            n_samples_to_issue=None,
            rng_sched=random.Random(1234),
            rng_sample_index=random.Random(1234),
        )

        # Create scheduler with metrics-enabled factory
        scheduler = MaxThroughputScheduler(
            rt_settings,
            dataloader,
            MetricsSampleFactory,
            WithoutReplacementSampleOrder,
        )

        # Use the scheduler's factory instance for the issuer
        sample_factory = scheduler.sample_factory
        sample_issuer = HttpClientSampleIssuer(http_client)
        sample_issuer.start()

        # Create load generator
        load_gen = LoadGenerator(scheduler, sample_issuer)

        # Start test
        try:
            sess = load_gen.start_test()
            sess.wait_for_test_end()

            # Wait for all pending responses to complete
            sample_issuer.wait_for_all_complete()
        finally:
            sample_factory.metrics.stop()
            sample_issuer.shutdown()

        # Get summary
        summary = sample_factory.metrics.get_summary()
        achieved_qps = summary["qps"]
        min_achievement = PERFORMANCE_CONFIG["target_qps_tolerance"]

        logger.info(
            f"Streaming (size={message_size} characters): Target={target_qps} QPS, "
            f"Achieved={achieved_qps:.2f} QPS, Success Rate={summary['success_rate']:.2f}%, "
            f"P99={summary['latencies']['p99']:.3f}s"
        )

        assert (
            achieved_qps >= target_qps * min_achievement
        ), f"Failed to achieve {min_achievement*100:.0f}% of target {target_qps} QPS at message size {message_size} characters (got {achieved_qps:.2f})"

        required_success_rate = PERFORMANCE_CONFIG["required_success_rate_percent"]
        assert (
            summary["success_rate"] >= required_success_rate
        ), f"Success rate must be >= {required_success_rate}%, got {summary['success_rate']}%"

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    @pytest.mark.parametrize("message_size", PERFORMANCE_CONFIG["message_sizes"])
    def test_offline_throughput_various_message_sizes(
        self,
        http_client,
        create_test_query,
        message_size,
    ):
        """Test that offline mode maintains target QPS across different message sizes."""
        target_qps = PERFORMANCE_CONFIG["offline_target_qps"]
        duration_ms = PERFORMANCE_CONFIG["test_duration_ms"]

        # Create test queries
        num_samples = 100
        queries = [
            create_test_query(prompt_size=message_size, stream=False)
            for _ in range(num_samples)
        ]

        # Create dataloader
        dataloader = QueryDataLoader(queries)
        dataloader.load()

        # Create runtime settings
        rt_settings = RuntimeSettings(
            metric_target=metrics.Throughput(target_qps),
            reported_metrics=[metrics.Throughput(target_qps)],
            min_duration_ms=duration_ms,
            max_duration_ms=duration_ms * 2,
            n_samples_from_dataset=num_samples,
            n_samples_to_issue=None,
            rng_sched=random.Random(1234),
            rng_sample_index=random.Random(1234),
        )

        # Create scheduler with metrics-enabled factory
        scheduler = MaxThroughputScheduler(
            rt_settings,
            dataloader,
            MetricsSampleFactory,
            WithoutReplacementSampleOrder,
        )

        # Use the scheduler's factory instance for the issuer
        sample_factory = scheduler.sample_factory
        sample_issuer = HttpClientSampleIssuer(http_client)
        sample_issuer.start()

        # Create load generator
        load_gen = LoadGenerator(scheduler, sample_issuer)

        # Start test
        try:
            sess = load_gen.start_test()
            sess.wait_for_test_end()

            # Wait for all pending responses to complete
            sample_issuer.wait_for_all_complete()
        finally:
            sample_factory.metrics.stop()
            sample_issuer.shutdown()

        # Get summary
        summary = sample_factory.metrics.get_summary()
        achieved_qps = summary["qps"]
        min_achievement = PERFORMANCE_CONFIG["target_qps_tolerance"]

        logger.info(
            f"Offline (size={message_size} characters): Target={target_qps} QPS, "
            f"Achieved={achieved_qps:.2f} QPS, Success Rate={summary['success_rate']:.2f}%"
        )

        assert (
            achieved_qps >= target_qps * min_achievement
        ), f"Failed to achieve {min_achievement*100:.0f}% of target {target_qps} QPS at message size {message_size} characters (got {achieved_qps:.2f})"

        required_success_rate = PERFORMANCE_CONFIG["required_success_rate_percent"]
        assert (
            summary["success_rate"] >= required_success_rate
        ), f"Success rate must be >= {required_success_rate}%, got {summary['success_rate']}%"

    # =========================================================================
    # 3. LATENCY TESTS - P99 DEGRADATION ACROSS MESSAGE SIZES
    # =========================================================================

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    def test_streaming_p99_latency_no_degradation(
        self,
        http_client,
        create_test_query,
    ):
        """Test that streaming P99 latency does not degrade across message sizes."""
        target_qps = PERFORMANCE_CONFIG["streaming_target_qps"]
        duration_ms = PERFORMANCE_CONFIG["test_duration_ms"]
        message_sizes = PERFORMANCE_CONFIG["message_sizes"]
        tolerance = PERFORMANCE_CONFIG["p99_degradation_tolerance"]

        baseline_p99 = None
        p99_results = {}

        for message_size in message_sizes:
            # Create test queries
            num_samples = 100
            queries = [
                create_test_query(prompt_size=message_size, stream=True)
                for _ in range(num_samples)
            ]

            # Create dataloader
            dataloader = QueryDataLoader(queries)
            dataloader.load()

            # Create runtime settings
            rt_settings = RuntimeSettings(
                metric_target=metrics.Throughput(target_qps),
                reported_metrics=[metrics.Throughput(target_qps)],
                min_duration_ms=duration_ms,
                max_duration_ms=duration_ms * 2,
                n_samples_from_dataset=num_samples,
                n_samples_to_issue=None,
                rng_sched=random.Random(1234),
                rng_sample_index=random.Random(1234),
            )

            # Create scheduler with metrics-enabled factory
            scheduler = MaxThroughputScheduler(
                rt_settings,
                dataloader,
                MetricsSampleFactory,
                WithoutReplacementSampleOrder,
            )

            # Use the scheduler's factory instance for the issuer
            sample_factory = scheduler.sample_factory
            sample_issuer = HttpClientSampleIssuer(http_client)
            sample_issuer.start()

            # Create load generator
            load_gen = LoadGenerator(scheduler, sample_issuer)

            # Start test
            try:
                sess = load_gen.start_test()
                sess.wait_for_test_end()

                # Wait for all pending responses to complete
                sample_issuer.wait_for_all_complete()
            finally:
                sample_factory.metrics.stop()
                sample_issuer.shutdown()

            # Get summary
            summary = sample_factory.metrics.get_summary()
            p99 = summary["latencies"]["p99"]
            p99_results[message_size] = p99

            # Set baseline from first (smallest) message size
            if baseline_p99 is None:
                baseline_p99 = p99
                logger.info(
                    f"Streaming baseline (size={message_size} characters): P99={p99:.3f}s"
                )
            else:
                max_allowed = baseline_p99 * tolerance
                logger.info(
                    f"Streaming (size={message_size} characters): P99={p99:.3f}s "
                    f"(baseline={baseline_p99:.3f}s, max_allowed={max_allowed:.3f}s)"
                )

                assert p99 <= max_allowed, (
                    f"Streaming P99 latency degraded at message size {message_size} characters: "
                    f"{p99:.3f}s exceeds {tolerance}x baseline ({baseline_p99:.3f}s) = {max_allowed:.3f}s"
                )

            # Ensure we have sufficient samples
            assert (
                summary["total_requests"] >= 50
            ), f"Too few samples to validate latency: {summary['total_requests']}"

        # Log summary of all results
        logger.info("Streaming P99 latency summary:")
        for size, p99 in p99_results.items():
            degradation = (p99 / baseline_p99) if baseline_p99 > 0 else 1.0
            logger.info(
                f"  {size:8d} characters: {p99:.3f}s ({degradation:.2f}x baseline)"
            )

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    def test_offline_p99_latency_no_degradation(
        self,
        http_client,
        create_test_query,
    ):
        """Test that offline P99 latency does not degrade across message sizes."""
        target_qps = PERFORMANCE_CONFIG["offline_target_qps"]
        duration_ms = PERFORMANCE_CONFIG["test_duration_ms"]
        message_sizes = PERFORMANCE_CONFIG["message_sizes"]
        tolerance = PERFORMANCE_CONFIG["p99_degradation_tolerance"]

        baseline_p99 = None
        p99_results = {}

        for message_size in message_sizes:
            # Create test queries
            num_samples = 100
            queries = [
                create_test_query(prompt_size=message_size, stream=False)
                for _ in range(num_samples)
            ]

            # Create dataloader
            dataloader = QueryDataLoader(queries)
            dataloader.load()

            # Create runtime settings
            rt_settings = RuntimeSettings(
                metric_target=metrics.Throughput(target_qps),
                reported_metrics=[metrics.Throughput(target_qps)],
                min_duration_ms=duration_ms,
                max_duration_ms=duration_ms * 2,
                n_samples_from_dataset=num_samples,
                n_samples_to_issue=None,
                rng_sched=random.Random(1234),
                rng_sample_index=random.Random(1234),
            )

            # Create scheduler with metrics-enabled factory
            scheduler = MaxThroughputScheduler(
                rt_settings,
                dataloader,
                MetricsSampleFactory,
                WithoutReplacementSampleOrder,
            )

            # Use the scheduler's factory instance for the issuer
            sample_factory = scheduler.sample_factory
            sample_issuer = HttpClientSampleIssuer(http_client)
            sample_issuer.start()

            # Create load generator
            load_gen = LoadGenerator(scheduler, sample_issuer)

            # Start test
            try:
                sess = load_gen.start_test()
                sess.wait_for_test_end()

                # Wait for all pending responses to complete
                sample_issuer.wait_for_all_complete()
            finally:
                sample_factory.metrics.stop()
                sample_issuer.shutdown()

            # Get summary
            summary = sample_factory.metrics.get_summary()
            p99 = summary["latencies"]["p99"]
            p99_results[message_size] = p99

            # Set baseline from first (smallest) message size
            if baseline_p99 is None:
                baseline_p99 = p99
                logger.info(
                    f"Offline baseline (size={message_size} characters): P99={p99:.3f}s"
                )
            else:
                max_allowed = baseline_p99 * tolerance
                logger.info(
                    f"Offline (size={message_size} characters): P99={p99:.3f}s "
                    f"(baseline={baseline_p99:.3f}s, max_allowed={max_allowed:.3f}s)"
                )

                assert p99 <= max_allowed, (
                    f"Offline P99 latency degraded at message size {message_size} characters: "
                    f"{p99:.3f}s exceeds {tolerance}x baseline ({baseline_p99:.3f}s) = {max_allowed:.3f}s"
                )

            # Ensure we have sufficient samples
            assert (
                summary["total_requests"] >= 50
            ), f"Too few samples to validate latency: {summary['total_requests']}"

        # Log summary of all results
        logger.info("Offline P99 latency summary:")
        for size, p99 in p99_results.items():
            degradation = (p99 / baseline_p99) if baseline_p99 > 0 else 1.0
            logger.info(
                f"  {size:8d} characters: {p99:.3f}s ({degradation:.2f}x baseline)"
            )
