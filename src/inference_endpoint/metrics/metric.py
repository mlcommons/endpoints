"""This module contains the Metric class and its subclasses.

Metrics are used to evaluate the performance of a benchmark, or set baseline expectations of a Server-Under-Test (SUT). The
target values are set by the benchmark submitter and are used to calculate the size / frequency of the load sent by loadgen to the SUT.
Empirical measurements are compared against the target values to determine if the submission is valid.
"""
import math


class Metric:
    """Metrics are used to evaluate the performance of a benchmark. The target values are set by the benchmark submitter
    and are used to calculate the size / frequency of the load sent by loadgen to the SUT. Empirical measurements are compared
    against the target values to determine if the submission is valid.
    """
    def __init__(self, target: float):
        self.target = target

    def is_valid(self, measurement: float) -> bool:
        raise NotImplementedError("Subclasses must implement this method")


class Throughput(Metric):
    REL_TOL = 0.1  # Relative tolerance for throughput

    def __init__(self, target: float):
        super().__init__(target)

    def is_valid(self, measurement: float) -> bool:
        return math.isclose(measurement, self.target, rel_tol=self.REL_TOL)


class QueryLatency(Metric):
    REL_TOL = 0.1  # Relative tolerance for query latency

    def __init__(self, target_latency_ms: float = None, target_qps: float = None):
        """
        Args:
            target_latency_ms: The target latency in milliseconds
            target_qps: The target queries per second. If set, target_latency will be ignored, and the inverse of target_qps will be used as the target.
        """
        if target_qps:
            target_latency_ms = 1000 / target_qps
        else:
            assert target_latency_ms is not None, "Either target_latency_ms or target_qps must be set"
        super().__init__(target_latency_ms)

    def is_valid(self, measurement: float) -> bool:
        return math.isclose(measurement, self.target, rel_tol=self.REL_TOL)


class TTFT(Metric):
    def __init__(self, max_ttft_latency_ms: float):
        super().__init__(max_ttft_latency_ms)

    def is_valid(self, measurement: float) -> bool:
        return measurement <= self.target


class TPOT(Metric):
    def __init__(self, max_tpot_latency_ms: float):
        super().__init__(max_tpot_latency_ms)

    def is_valid(self, measurement: float) -> bool:
        return measurement <= self.target
