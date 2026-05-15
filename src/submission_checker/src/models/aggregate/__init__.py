"""Aggregate validation models — each model validates across multiple submission artifacts."""

from .context import ModelContext
from .point_result import MIN_QUERY_COUNT, PointResult

__all__ = ["MIN_QUERY_COUNT", "ModelContext", "PointResult"]
