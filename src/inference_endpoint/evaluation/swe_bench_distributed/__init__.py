# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed SWE-bench execution across a fleet of SWE-bench services.

The single-service :class:`~inference_endpoint.evaluation.swe_bench_scorer.SWEBenchScorer`
issues one run covering every instance. This package shards the instance list
into units, dispatches units across several services concurrently, classifies
infrastructure damage separately from genuine model failures, and refuses to
emit an accuracy number unless every planned instance id is accounted for
exactly once.
"""

from .merge import MergeRefusal, MergeResult, merge_run, verify_inventory
from .queue import (
    ClaimError,
    UnitOutcome,
    UnitResult,
    WorkQueue,
)
from .units import Unit, UnitPlan, plan_units

__all__ = [
    "ClaimError",
    "MergeRefusal",
    "MergeResult",
    "Unit",
    "UnitOutcome",
    "UnitPlan",
    "UnitResult",
    "WorkQueue",
    "merge_run",
    "plan_units",
    "verify_inventory",
]
