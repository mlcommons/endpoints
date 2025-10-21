"""Performance testing utilities with LoadGenerator integration."""

import asyncio
import logging
import time
from collections.abc import Callable

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.load_generator.events import SampleEvent

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    uvloop = None


logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Collect and analyze performance metrics for load testing."""

    def __init__(self):
        self.latencies: list[float] = []
        self.ttft_latencies: list[float] = []
        self.tpot_latencies: list[float] = []  # Time per output token
        self.errors: list[Exception] = []
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.request_times: dict[
            str | int, float
        ] = {}  # Support both UUID strings and ints
        self.first_chunk_times: dict[
            str | int, float
        ] = {}  # Support both UUID strings and ints
        self.output_tokens: dict[str | int, int] = {}  # Track output token counts
        self.issue_times: list[float] = []  # Track when requests are issued
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
        """Record request completion and calculate latency and TPOT.

        Args:
            sample_id: Unique identifier for the request
            output_tokens: Number of output tokens (for TPOT calculation)
        """
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
                f"record_request_complete called for unknown sample_id: {sample_id}, tracked IDs: {list(self.request_times.keys())[:5]}"
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
    """Format performance metrics into a readable report - simplified."""
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


class MetricsSampleFactory:
    """SampleFactory with built-in performance metrics tracking.

    Manages its own PerformanceMetrics collector internally.
    Access metrics via the .metrics attribute.
    """

    def __init__(self, dataloader):
        super().__init__(dataloader, None)
        self.metrics = PerformanceMetrics()
        self.metrics.start()

    def get_sample_callbacks(
        self, sample_index: int, sample_uuid: str
    ) -> dict[SampleEvent, Callable]:
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
