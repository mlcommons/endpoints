"""Accuracy result model — §4.3, §6.6 accuracy_result.json schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AccuracyResult(BaseModel):
    """Parsed contents of ``accuracy/accuracy_result.json`` (§4.3, §6.6).

    Attributes:
        metric: Name of the accuracy metric (e.g., ``rouge1``).
        score: Achieved score on the accuracy dataset.
        quality_target: Minimum acceptable score defined by the benchmark.
        passed: Whether the score meets the quality target.
    """

    model_config = ConfigDict(extra="allow")

    metric: str
    score: float
    quality_target: float
    passed: bool
