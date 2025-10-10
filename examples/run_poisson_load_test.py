#!/usr/bin/env python3
"""
Example: Poisson Load Test for HTTP Endpoint

This example demonstrates how to use the LoadGenerator with a Poisson scheduler
to simulate realistic server load patterns. It creates an echo server and runs
a load test with configurable parameters.

Usage:
    # Basic usage
    python examples/run_poisson_load_test.py
"""

import logging
import random
import string
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

# Add the src directory to the Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference_endpoint import metrics
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.dataset_manager.dataloader import DataLoader
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.loadgen import HttpClientSampleIssuer
from inference_endpoint.load_generator import LoadGenerator
from inference_endpoint.load_generator.scheduler import (
    NetworkActivitySimulationScheduler,
    SampleEvent,
    SampleFactory,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.testing.echo_server import EchoServer
from inference_endpoint.utils.logging import setup_logging

# Configure logging
setup_logging(level="WARNING")
logger = logging.getLogger(__name__)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_CONFIG = {
    "server_workers": 4,
    "client_workers": 4,
    "test_duration_ms": 5000,  # 5 seconds
    "message_size": 1000,  # 1k characters
    "num_samples": 100,
    "target_qps": 4000,  # Target QPS for Poisson scheduling
    "stream": True,  # Enable streaming mode
}


# =============================================================================
# UTILITY FUNCTIONS (included for self-contained example)
# =============================================================================


def _generate_random_word(
    rng: random.Random, mean_length: float = 5.0, std_dev: float = 2.0
) -> str:
    """Generate a random word with length following a normal distribution."""
    length = int(rng.gauss(mean_length, std_dev))
    length = max(1, min(15, length))  # Clamp to reasonable bounds
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def create_test_query(
    prompt_size: int = 100,
    stream: bool = False,
    query_id: str | None = None,
    seed: int | None = None,
) -> Query:
    """Create a test query with specified parameters.

    Args:
        prompt_size: Target size of the prompt in characters
        stream: Whether to enable streaming mode
        query_id: Custom query ID, or None to generate a UUID
        seed: Random seed for reproducible prompt generation

    Returns:
        Query object with the specified parameters
    """
    # Use a local random instance for reproducibility if seed is provided
    rng = random.Random(seed) if seed is not None else random

    # Generate prompt from random words until we reach approximately the target size
    words = []
    current_length = 0

    while current_length < prompt_size:
        word = _generate_random_word(rng)
        words.append(word)
        # Add 1 for the space character (except for the first word)
        current_length += len(word) + (1 if words else 0)

    prompt = " ".join(words)

    # Trim to exact size if we overshot
    if len(prompt) > prompt_size:
        prompt = prompt[:prompt_size].rstrip()

    return Query(
        id=query_id or str(uuid.uuid4()),
        data={
            "model": "test-model",
            "prompt": prompt,
            "stream": stream,
        },
    )


class PerformanceMetrics:
    """Collect and analyze performance metrics for load testing."""

    def __init__(self):
        self.latencies: list[float] = []
        self.ttft_latencies: list[float] = []
        self.tpot_latencies: list[float] = []  # Time per output token
        self.errors: list[Exception] = []
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.request_times: dict[str | int, float] = {}
        self.first_chunk_times: dict[str | int, float] = {}
        self.output_tokens: dict[str | int, int] = {}
        self.issue_times: list[float] = []
        self.first_issue_time: float | None = None
        self.last_issue_time: float | None = None

    def start(self):
        """Start measurement period."""
        self.start_time = time.time()

    def stop(self):
        """Stop measurement period."""
        self.end_time = time.time()

    def record_request_start(
        self, sample_id: str | int, start_time: float | None = None
    ):
        """Record request start time."""
        current_time = start_time if start_time is not None else time.time()
        self.request_times[sample_id] = current_time

        # Track issue times for issue rate calculation
        self.issue_times.append(current_time)
        if self.first_issue_time is None:
            self.first_issue_time = current_time
        self.last_issue_time = current_time

    def record_first_chunk(self, sample_id: str | int):
        """Record first chunk arrival."""
        if sample_id in self.request_times:
            ttft = time.time() - self.request_times[sample_id]
            self.ttft_latencies.append(ttft)
            self.first_chunk_times[sample_id] = time.time()

    def record_request_complete(
        self, sample_id: str | int, output_tokens: int | None = None
    ):
        """Record request completion and calculate latency and TPOT."""
        if sample_id in self.request_times:
            completion_time = time.time()
            latency = completion_time - self.request_times[sample_id]
            self.latencies.append(latency)

            # Calculate TPOT: (total_duration - ttft) / (output_tokens - 1)
            if (
                sample_id in self.first_chunk_times
                and output_tokens
                and output_tokens > 1
            ):
                ttft = self.first_chunk_times[sample_id] - self.request_times[sample_id]
                generation_time = latency - ttft
                tpot = generation_time / (output_tokens - 1)
                self.tpot_latencies.append(tpot)

            del self.request_times[sample_id]
            self.first_chunk_times.pop(sample_id, None)
            self.output_tokens.pop(sample_id, None)
        else:
            logger.warning(
                f"record_request_complete called for unknown sample_id: {sample_id}"
            )

    def record_error(self, error: Exception):
        """Record an error."""
        self.errors.append(error)

    def calculate_percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    @property
    def duration(self) -> float:
        """Total measurement duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def total_requests(self) -> int:
        """Total number of completed requests."""
        return len(self.latencies)

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return len(self.errors)

    @property
    def qps(self) -> float:
        """Queries per second (completion rate)."""
        if self.duration > 0:
            return self.total_requests / self.duration
        return 0.0

    @property
    def issue_qps(self) -> float:
        """Issue rate - queries issued per second."""
        if self.first_issue_time and self.last_issue_time:
            issue_duration = self.last_issue_time - self.first_issue_time
            if issue_duration > 0:
                return len(self.issue_times) / issue_duration
        return 0.0

    @property
    def total_issued(self) -> int:
        """Total number of requests issued."""
        return len(self.issue_times)

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        total = self.total_requests + self.error_count
        if total > 0:
            return (self.error_count / total) * 100
        return 0.0

    def get_summary(self) -> dict:
        """Get summary statistics including TTFT and TPOT metrics."""
        summary = {
            "total_requests": self.total_requests,
            "total_issued": self.total_issued,
            "duration": self.duration,
            "qps": self.qps,
            "issue_qps": self.issue_qps,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "success_rate": 100.0 - self.error_rate,
            "latencies": {
                "p99": self.calculate_percentile(self.latencies, 99),
                "ttft_p99": self.calculate_percentile(self.ttft_latencies, 99)
                if self.ttft_latencies
                else 0.0,
                "tpot_p99": self.calculate_percentile(self.tpot_latencies, 99)
                if self.tpot_latencies
                else 0.0,
            },
        }

        return summary


def format_performance_report(metrics: dict) -> str:
    """Format performance metrics into a readable report."""
    report = []
    report.append("=" * 60)
    report.append("Performance Test Results")
    report.append("=" * 60)

    report.append(f"\nTotal Issued: {metrics.get('total_issued', 0):,}")
    report.append(f"Total Completed: {metrics.get('total_requests', 0):,}")
    report.append(f"Duration: {metrics.get('duration', 0):.2f}s")
    report.append(f"Issue Rate: {metrics.get('issue_qps', 0):.2f} QPS")
    report.append(f"Completion Rate: {metrics.get('qps', 0):.2f} QPS")
    report.append(f"Success Rate: {metrics.get('success_rate', 0):.2f}%")
    report.append(f"Error Rate: {metrics.get('error_rate', 0):.2f}%")

    if "latencies" in metrics:
        latencies = metrics["latencies"]
        if "p99" in latencies:
            report.append(f"P99 Latency: {latencies['p99']:.3f}s")
        if "ttft_p99" in latencies:
            report.append(f"TTFT P99: {latencies['ttft_p99']:.3f}s")
        if "tpot_p99" in latencies:
            report.append(f"TPOT P99: {latencies['tpot_p99']:.6f}s")

    report.append("=" * 60)
    return "\n".join(report)


class MetricsSampleFactory(SampleFactory):
    """SampleFactory with built-in performance metrics tracking.

    Manages its own PerformanceMetrics collector internally.
    Access metrics via the .metrics attribute.
    """

    def __init__(self, dataloader):
        super().__init__(dataloader)
        self.metrics = PerformanceMetrics()
        self.metrics.start()

    def get_sample_callbacks(self, sample_index: int) -> dict[SampleEvent, Callable]:
        """Return callbacks that record metrics for each event."""

        def on_request_sent(query: Query):
            """Record when request is sent."""
            self.metrics.record_request_start(query.id)

        def on_first_chunk(chunk: StreamChunk):
            """Record first chunk arrival (TTFT)."""
            self.metrics.record_first_chunk(chunk.id)

        def on_non_first_chunk(chunk: StreamChunk):
            """Subsequent chunks - no-op for now."""
            pass

        def on_complete(result: QueryResult | StreamChunk):
            """Record completion and calculate latency and TPOT."""
            if isinstance(result, QueryResult) and result.error:
                self.metrics.record_error(Exception(result.error))
            else:
                response_id = result.id
                # Extract output tokens from the response (count of words as a proxy for tokens)
                output_tokens = None
                if hasattr(result, "response_output") and result.response_output:
                    output_tokens = len(result.response_output.split())
                self.metrics.record_request_complete(response_id, output_tokens)

        return {
            SampleEvent.REQUEST_SENT: on_request_sent,
            SampleEvent.FIRST_CHUNK: on_first_chunk,
            SampleEvent.NON_FIRST_CHUNK: on_non_first_chunk,
            SampleEvent.COMPLETE: on_complete,
        }


# =============================================================================
# DATALOADER
# =============================================================================


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
# TEST RUNNER
# =============================================================================


def run_poisson_load_test():
    """Run a Poisson load test against an echo server and print results."""

    print("\n" + "=" * 70)
    print("POISSON LOAD TEST - ECHO SERVER")
    print("=" * 70)
    print(f"Server Workers: {TEST_CONFIG['server_workers']}")
    print(f"Client Workers: {TEST_CONFIG['client_workers']}")
    print(f"Test Duration: {TEST_CONFIG['test_duration_ms']/1000:.1f} seconds")
    print(f"Message Size: {TEST_CONFIG['message_size']} characters")
    print(f"Target QPS: {TEST_CONFIG['target_qps']}")
    print("Mode: Poisson Scheduling (realistic arrival simulation)")
    print("=" * 70 + "\n")

    # Start echo server
    print(f"Starting echo server with {TEST_CONFIG['server_workers']} workers...")
    server = EchoServer(port=0, workers=TEST_CONFIG["server_workers"])
    server.start()
    print(f"Echo server started on {server.url}")

    try:
        # Create HTTP client
        print(f"Creating HTTP client with {TEST_CONFIG['client_workers']} worker...")
        http_config = HTTPClientConfig(
            endpoint_url=f"{server.url}/v1/chat/completions",
            num_workers=TEST_CONFIG["client_workers"],
        )

        zmq_config = ZMQConfig(
            zmq_io_threads=TEST_CONFIG["client_workers"] * 4,
            zmq_high_water_mark=100_000,
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )
        client.start()
        print("HTTP client started")

        try:
            # Create test queries
            print(f"Creating {TEST_CONFIG['num_samples']} test queries...")
            queries = [
                create_test_query(
                    prompt_size=TEST_CONFIG["message_size"],
                    stream=TEST_CONFIG["stream"],
                )
                for _ in range(TEST_CONFIG["num_samples"])
            ]

            # Create dataloader
            dataloader = QueryDataLoader(queries)
            dataloader.load()

            # Create runtime settings
            rt_settings = RuntimeSettings(
                metric_target=metrics.Throughput(TEST_CONFIG["target_qps"]),
                reported_metrics=[metrics.Throughput(TEST_CONFIG["target_qps"])],
                min_duration_ms=TEST_CONFIG["test_duration_ms"],
                max_duration_ms=TEST_CONFIG["test_duration_ms"] * 2,
                n_samples_from_dataset=TEST_CONFIG["num_samples"],
                n_samples_to_issue=None,  # Run until duration
                rng_sched=random.Random(1234),
                rng_sample_index=random.Random(1234),
            )

            # Create Poisson scheduler for realistic server load simulation
            print("Creating Poisson scheduler...")
            scheduler = NetworkActivitySimulationScheduler(
                rt_settings,
                dataloader,
                MetricsSampleFactory,
                WithoutReplacementSampleOrder,
            )

            # Create sample issuer
            sample_factory = scheduler.sample_factory
            sample_issuer = HttpClientSampleIssuer(client)
            sample_issuer.start()

            # Create load generator
            load_gen = LoadGenerator(scheduler, sample_issuer)

            # Run test
            print(
                f"\nRunning Poisson load test for {TEST_CONFIG['test_duration_ms']/1000:.1f} seconds..."
            )
            print("Please wait...\n")

            try:
                sess = load_gen.start_test()
                sess.wait_for_test_end()

                # Wait for all pending responses to complete
                sample_issuer.wait_for_all_complete()
            finally:
                sample_factory.metrics.stop()
                sample_issuer.shutdown()

            # Get and display results
            summary = sample_factory.metrics.get_summary()

            print("\n" + format_performance_report(summary))

            # Print highlighted results
            print("\n" + "=" * 70)
            print("KEY RESULTS")
            print("=" * 70)
            print(f"  Target QPS: {TEST_CONFIG['target_qps']}")
            print(f"  Achieved QPS: {summary['qps']:.2f}")
            print(f"  Total Completed: {summary['total_requests']:,}")
            print(f"  Duration: {summary['duration']:.2f}s")
            print(f"  Success Rate: {summary['success_rate']:.2f}%")
            print(f"  P99 Latency: {summary['latencies']['p99']:.3f}s")
            print("=" * 70 + "\n")

            return summary

        finally:
            print("Shutting down HTTP client...")
            client.shutdown()

    finally:
        print("Shutting down echo server...")
        server.stop()

    print("Test complete!\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        summary = run_poisson_load_test()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
