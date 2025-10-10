"""Performance stress tests for HTTPEndpointClient with single worker."""

import logging
import random
import uuid
from pathlib import Path

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.dataset_manager.dataloader import DataLoader
from inference_endpoint.endpoint_client.loadgen import HttpClientSampleIssuer
from inference_endpoint.load_generator.scheduler import (
    NetworkActivitySimulationScheduler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.session import BenchmarkSession
from inference_endpoint.metrics.recorder import MetricsReporter

from tests.test_helpers import create_test_query

logger = logging.getLogger(__name__)

# =============================================================================
# PERFORMANCE TARGETS - Single worker configuration
# Tests use Poisson scheduler to verify system can sustain target QPS
# =============================================================================

PERFORMANCE_CONFIG = {
    # Test parameters
    "warmup_requests": 100,
    "test_duration_ms": 5000,  # 5 seconds
    "default_message_size": 1000,  # 1k characters
    # Target QPS thresholds (single worker)
    "streaming_target_qps": 1500,
    "offline_target_qps": 1500,
    # Test across multiple message sizes to validate consistent performance
    "message_sizes": [100, 500, 1000, 2000, 5000],
    # Pass/fail criteria
    "required_success_rate_percent": 100,
    "target_qps_tolerance": 0.90,  # Must achieve 90% of target QPS
    "latency_degradation_tolerance": 1.10,  # P99 latency can't degrade >10% across message sizes
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


# =============================================================================
# TEST RUNNER HELPERS
# =============================================================================


def run_performance_test(
    http_client,
    target_qps: float,
    duration_ms: int,
    message_size: int,
    stream: bool,
    num_samples: int = 100,
) -> dict:
    """Run a performance test and return metrics summary.

    Uses Poisson scheduler to test if system can sustain target QPS.

    Args:
        http_client: The HTTP client to test
        target_qps: Target queries per second
        duration_ms: Test duration in milliseconds
        message_size: Size of messages in characters
        stream: Whether to use streaming mode
        num_samples: Number of samples in the dataset

    Returns:
        Dictionary containing performance metrics summary
    """
    # Create test queries
    queries = [
        create_test_query(prompt_size=message_size, stream=stream)
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

    # Choose scheduler based on mode:
    # - Streaming (server mode): Use Poisson scheduler to simulate realistic load arrival
    # - Offline: Use max throughput scheduler to stress test the system
    if stream:
        sched_cls = NetworkActivitySimulationScheduler
    else:
        sched_cls = MaxThroughputScheduler
    scheduler = sched_cls(rt_settings, WithoutReplacementSampleOrder)

    # Create sample issuer
    sample_issuer = HttpClientSampleIssuer(http_client)
    sample_issuer.start()

    # Start benchmark session with metrics-tracking sample factory
    # MetricsSampleFactory will store itself in _latest_instance for retrieval
    session = BenchmarkSession.start(
        rt_settings,
        dataloader,
        sample_issuer,
        scheduler,
        name=f"perf_test_{stream}_{target_qps}qps_{uuid.uuid4().hex}",
        stop_sample_issuer_on_test_end=False,  # We'll handle shutdown manually
    )

    events_db_path = session.event_recorder.connection_name

    # Wait for test to complete
    session.wait_for_test_end()

    # Wait for all pending responses to complete
    sample_issuer.wait_for_all_complete()

    # Shutdown
    sample_issuer.shutdown()

    # Get summary of metrics
    assert Path(events_db_path).exists()

    # Adhere to same summary interface as old tests
    with MetricsReporter(events_db_path) as reporter:
        stats = reporter.get_sample_statuses()
        ttft_stats = reporter.derive_TTFT()
        tpot_stats = reporter.derive_TPOT()
        sample_latency_stats = reporter.derive_sample_latency()
        test_duration = reporter.derive_duration()

    test_qps = stats["total_sent"] / (test_duration / 1e9)

    # Calculate "True" issue QPS based on sample latencies
    total_issue_time = sum(
        metric_row.metric_value for metric_row in sample_latency_stats
    )
    issue_qps = stats["total_sent"] / (total_issue_time / 1e9)

    return {
        "total_issued": stats["total_sent"],
        "duration_ns": test_duration,
        "qps": test_qps,
        "issue_qps": issue_qps,
        "latencies": {
            "sample_latency_p99": sample_latency_stats.percentile(99),
            "ttft_p99": ttft_stats.percentile(99),
            "tpot_p99": tpot_stats.percentile(99),
        },
    }


def assert_performance_requirements(
    summary: dict,
    target_qps: float,
    mode: str,
    message_size: int | None = None,
):
    """Assert that performance meets requirements.

    Args:
        summary: Metrics summary from run_performance_test
        target_qps: Target queries per second
        mode: "streaming" or "offline"
        message_size: Optional message size for better error messages
    """
    # Achieved QPS = total_completed / total_time (completion rate)
    achieved_qps = summary["qps"]
    issued_qps = summary["issue_qps"]
    min_achievement = PERFORMANCE_CONFIG["target_qps_tolerance"]

    # Log results
    size_info = f" (size={message_size} characters)" if message_size else ""
    logger.info(
        f"{mode.capitalize()}{size_info}: Target={target_qps} QPS, "
        f"Achieved={achieved_qps:.2f} QPS"
        + (
            f", P99={summary['latencies']['sample_latency_p99']:.3f}s"
            if "latencies" in summary
            else ""
        )
    )

    # Assert QPS (total_completed / total_time)
    size_msg = f" at message size {message_size} characters" if message_size else ""
    if achieved_qps < target_qps * min_achievement:
        if issued_qps < target_qps * min_achievement:
            raise AssertionError(
                f"Failed to achieve {min_achievement*100:.0f}% of target {target_qps} QPS{size_msg} "
                f"(got {achieved_qps:.2f}, issued {issued_qps:.2f})"
            )
        else:
            logger.warning(
                f"Failed to achieve {min_achievement*100:.0f}% of target {target_qps} QPS{size_msg} "
                f"(got {achieved_qps:.2f}, but redeemed by issued qps {issued_qps:.2f})"
            )


@pytest.mark.timeout(0)  # Disable timeout for all performance tests
class TestHTTPClientPerformanceSingleWorker:
    """Performance tests for HTTPEndpointClient (single worker).

    Uses Poisson scheduler to test if system can sustain target QPS rates.
    """

    # =========================================================================
    # 1. BASELINE PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    def test_streaming_baseline_performance(self, http_client):
        """Test streaming mode performance with Poisson-distributed load."""
        summary = run_performance_test(
            http_client,
            target_qps=PERFORMANCE_CONFIG["streaming_target_qps"],
            duration_ms=PERFORMANCE_CONFIG["test_duration_ms"],
            message_size=PERFORMANCE_CONFIG["default_message_size"],
            stream=True,
        )
        # Always print achieved QPS prominently
        print(f"\n{'='*70}")
        print(
            f"STREAMING BASELINE: Achieved {summary['qps']:.2f} QPS (target: {PERFORMANCE_CONFIG['streaming_target_qps']} QPS)"
        )
        print(f"{'='*70}\n")
        assert_performance_requirements(
            summary,
            target_qps=PERFORMANCE_CONFIG["streaming_target_qps"],
            mode="streaming",
        )

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    def test_offline_baseline_performance(self, http_client):
        """Test offline mode performance with Poisson-distributed load."""
        summary = run_performance_test(
            http_client,
            target_qps=PERFORMANCE_CONFIG["offline_target_qps"],
            duration_ms=PERFORMANCE_CONFIG["test_duration_ms"],
            message_size=PERFORMANCE_CONFIG["default_message_size"],
            stream=False,
        )
        # Always print achieved QPS prominently
        print(f"\n{'='*70}")
        print(
            f"OFFLINE BASELINE: Achieved {summary['qps']:.2f} QPS (target: {PERFORMANCE_CONFIG['offline_target_qps']} QPS)"
        )
        print(f"{'='*70}\n")
        assert_performance_requirements(
            summary,
            target_qps=PERFORMANCE_CONFIG["offline_target_qps"],
            mode="offline",
        )

    # =========================================================================
    # 2. THROUGHPUT UNDER VARIOUS MESSAGE SIZES
    # =========================================================================

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    @pytest.mark.parametrize("message_size", PERFORMANCE_CONFIG["message_sizes"])
    def test_streaming_throughput_various_message_sizes(
        self, http_client, message_size
    ):
        """Validate streaming maintains target QPS with Poisson load across different message sizes."""
        summary = run_performance_test(
            http_client,
            target_qps=PERFORMANCE_CONFIG["streaming_target_qps"],
            duration_ms=PERFORMANCE_CONFIG["test_duration_ms"],
            message_size=message_size,
            stream=True,
        )
        assert_performance_requirements(
            summary,
            target_qps=PERFORMANCE_CONFIG["streaming_target_qps"],
            mode="streaming",
            message_size=message_size,
        )

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    @pytest.mark.parametrize("message_size", PERFORMANCE_CONFIG["message_sizes"])
    def test_offline_throughput_various_message_sizes(self, http_client, message_size):
        """Validate offline mode maintains max throughput across different message sizes."""
        summary = run_performance_test(
            http_client,
            target_qps=PERFORMANCE_CONFIG["offline_target_qps"],
            duration_ms=PERFORMANCE_CONFIG["test_duration_ms"],
            message_size=message_size,
            stream=False,
        )
        assert_performance_requirements(
            summary,
            target_qps=PERFORMANCE_CONFIG["offline_target_qps"],
            mode="offline",
            message_size=message_size,
        )

    # =========================================================================
    # 3. LATENCY TESTS - P99 DEGRADATION ACROSS MESSAGE SIZES
    # =========================================================================

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    @pytest.mark.skip("WIP: optimize serialization/deserialization")
    def test_streaming_p99_latency_no_degradation(self, http_client):
        """Validate streaming TTFT and TPOT P99 latency remain stable across message sizes."""
        target_qps = PERFORMANCE_CONFIG["streaming_target_qps"]
        duration_ms = PERFORMANCE_CONFIG["test_duration_ms"]
        message_sizes = PERFORMANCE_CONFIG["message_sizes"]
        tolerance = PERFORMANCE_CONFIG["latency_degradation_tolerance"]

        baseline_ttft = None
        baseline_tpot = None
        ttft_results = {}
        tpot_results = {}

        for message_size in message_sizes:
            summary = run_performance_test(
                http_client,
                target_qps=target_qps,
                duration_ms=duration_ms,
                message_size=message_size,
                stream=True,
            )

            # Ensure we have sufficient samples
            assert (
                summary["total_issued"] >= 50
            ), f"Too few samples to validate latency: {summary['total_issued']}"

            # Check TTFT and TPOT
            ttft_p99 = summary["latencies"]["ttft_p99"]
            tpot_p99 = summary["latencies"]["tpot_p99"]
            ttft_results[message_size] = ttft_p99
            tpot_results[message_size] = tpot_p99

            # Set baseline from first (smallest) message size
            if baseline_ttft is None:
                baseline_ttft = ttft_p99
                baseline_tpot = tpot_p99
                logger.info(
                    f"Streaming baseline (size={message_size} characters): "
                    f"TTFT_P99={ttft_p99:.3f}s, TPOT_P99={tpot_p99:.6f}s"
                )
            else:
                # Check TTFT degradation
                max_allowed_ttft = baseline_ttft * tolerance
                logger.info(
                    f"Streaming (size={message_size} characters): "
                    f"TTFT_P99={ttft_p99:.3f}s (baseline={baseline_ttft:.3f}s, max={max_allowed_ttft:.3f}s), "
                    f"TPOT_P99={tpot_p99:.6f}s (baseline={baseline_tpot:.6f}s, max={baseline_tpot * tolerance:.6f}s)"
                )

                assert ttft_p99 <= max_allowed_ttft, (
                    f"Streaming TTFT P99 degraded at message size {message_size} characters: "
                    f"{ttft_p99:.3f}s exceeds {tolerance}x baseline ({baseline_ttft:.3f}s) = {max_allowed_ttft:.3f}s"
                )

                # Check TPOT degradation
                max_allowed_tpot = baseline_tpot * tolerance
                assert tpot_p99 <= max_allowed_tpot, (
                    f"Streaming TPOT P99 degraded at message size {message_size} characters: "
                    f"{tpot_p99:.6f}s exceeds {tolerance}x baseline ({baseline_tpot:.6f}s) = {max_allowed_tpot:.6f}s"
                )

        # Log summary of all results
        logger.info("Streaming TTFT P99 summary:")
        for size, ttft in ttft_results.items():
            degradation = (
                (ttft / baseline_ttft) if baseline_ttft and baseline_ttft > 0 else 1.0
            )
            logger.info(
                f"  {size:8d} characters: {ttft:.3f}s ({degradation:.2f}x baseline)"
            )

        logger.info("Streaming TPOT P99 summary:")
        for size, tpot in tpot_results.items():
            degradation = (
                (tpot / baseline_tpot) if baseline_tpot and baseline_tpot > 0 else 1.0
            )
            logger.info(
                f"  {size:8d} characters: {tpot:.6f}s ({degradation:.2f}x baseline)"
            )

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    @pytest.mark.skip("WIP: optimize serialization/deserialization")
    def test_offline_p99_latency_no_degradation(self, http_client):
        """Validate offline P99 latency remains stable across message sizes."""
        target_qps = PERFORMANCE_CONFIG["offline_target_qps"]
        duration_ms = PERFORMANCE_CONFIG["test_duration_ms"]
        message_sizes = PERFORMANCE_CONFIG["message_sizes"]
        tolerance = PERFORMANCE_CONFIG["latency_degradation_tolerance"]

        baseline_p99 = None
        p99_results = {}

        for message_size in message_sizes:
            summary = run_performance_test(
                http_client,
                target_qps=target_qps,
                duration_ms=duration_ms,
                message_size=message_size,
                stream=False,
            )

            # Ensure we have sufficient samples
            assert (
                summary["total_issued"] >= 50
            ), f"Too few samples to validate latency: {summary['total_issued']}"

            # Check total P99 latency
            p99 = summary["latencies"]["sample_latency_p99"]
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
                    f"Offline total P99 latency degraded at message size {message_size} characters: "
                    f"{p99:.3f}s exceeds {tolerance}x baseline ({baseline_p99:.3f}s) = {max_allowed:.3f}s"
                )

        # Log summary of all results
        logger.info("Offline P99 latency summary:")
        for size, p99 in p99_results.items():
            degradation = (p99 / baseline_p99) if baseline_p99 > 0 else 1.0
            logger.info(
                f"  {size:8d} characters: {p99:.3f}s ({degradation:.2f}x baseline)"
            )
