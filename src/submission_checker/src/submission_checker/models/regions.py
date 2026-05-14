"""Region boundary computation — §5 Pareto Collection Methodology.

Implements the reference algorithm from §5.5 verbatim, using Python's built-in
``round()`` (banker's / round-half-to-even) for all boundary calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RegionBounds:
    """Inclusive ``[start, end]`` concurrency range for one region.

    Attributes:
        start: Lowest valid concurrency level (inclusive).
        end: Highest valid concurrency level (inclusive).
    """

    start: int
    end: int

    def contains(self, concurrency: int) -> bool:
        """Return True if *concurrency* falls within this region."""
        return self.start <= concurrency <= self.end

    def __str__(self) -> str:
        return f"{self.start}–{self.end}"


@dataclass(frozen=True)
class Regions:
    """All region boundaries for a given Maximum Supported Concurrency.

    Attributes:
        low_latency: Fixed 1–32 range.
        low_throughput: First logarithmic throughput region.
        med_throughput: Second logarithmic throughput region.
        high_throughput: Third logarithmic throughput region, extended by the
            §5.4 10 % margin — spans from ``med_throughput.end + 1`` to
            ``ceil(M * 1.10)``.
    """

    low_latency: RegionBounds
    low_throughput: RegionBounds
    med_throughput: RegionBounds
    high_throughput: RegionBounds


# Minimum steady-state duration per region (§6.2, illustrative — WIP).
MIN_DURATION_MS: dict[str, int] = {
    "low_latency": 600_000,
    "low_throughput": 1_200_000,
    "med_throughput": 1_200_000,
    "high_throughput": 1_200_000,
}


def compute_regions(M: int) -> Regions:
    """Compute region boundaries for a declared Maximum Supported Concurrency *M*.

    This is the reference algorithm from §5.5, using banker's rounding
    (Python's built-in ``round()``).

    Args:
        M: Maximum Supported Concurrency declared in the system description.
           Must be greater than 32.

    Returns:
        A :class:`Regions` instance with inclusive ``[start, end]`` boundaries.

    Raises:
        ValueError: If *M* is not greater than 32.

    Example::

        regions = compute_regions(1024)
        print(regions.low_throughput)  # 33–42
    """
    if M <= 32:
        raise ValueError(f"Maximum Supported Concurrency must be > 32, got {M}")

    interval = math.log2(M - 32) / 3

    low_tput_end = round(32 + 2**interval)
    med_tput_end = round(32 + 2 ** (2 * interval))
    ht_end = math.ceil(M * 1.10)  # includes §5.4 10% margin

    return Regions(
        low_latency=RegionBounds(1, 32),
        low_throughput=RegionBounds(33, low_tput_end),
        med_throughput=RegionBounds(low_tput_end + 1, med_tput_end),
        high_throughput=RegionBounds(med_tput_end + 1, ht_end),
    )


def classify_concurrency(concurrency: int, regions: Regions) -> str | None:
    """Return the canonical region name for *concurrency*, or ``None`` if out of range.

    The High Throughput region already includes the §5.4 10 % margin, so
    concurrencies up to ``ceil(M * 1.10)`` classify as ``"high_throughput"``.

    Args:
        concurrency: The concurrency level to classify.
        regions: Pre-computed region boundaries from :func:`compute_regions`.

    Returns:
        One of ``"low_latency"``, ``"low_throughput"``, ``"med_throughput"``,
        ``"high_throughput"``, or ``None`` if the concurrency exceeds the
        high throughput region (including the 10% margin).  A ``None`` return
        means the concurrency is out of range for submission purposes and the
        corresponding run should be flagged as invalid by the caller.
    """
    if regions.low_latency.contains(concurrency):
        return "low_latency"
    if regions.low_throughput.contains(concurrency):
        return "low_throughput"
    if regions.med_throughput.contains(concurrency):
        return "med_throughput"
    if regions.high_throughput.contains(concurrency):
        return "high_throughput"
    return None
