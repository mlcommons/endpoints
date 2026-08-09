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

from .guards import HealthTerm, HealthVerdict, MemoryGuard, combine_terms, kill_by_pid
from .merge import (
    CompletenessReport,
    MergeRefusal,
    MergeResult,
    assess_run,
    merge_run,
    verify_inventory,
)
from .queue import (
    ClaimError,
    UnitOutcome,
    UnitResult,
    WorkQueue,
)
from .reaper import LocalProcessLiveness, OwnerLiveness, SlurmStepLiveness, reap
from .units import Unit, UnitPlan, plan_units

__all__ = [
    "ClaimError",
    "CompletenessReport",
    "HealthTerm",
    "HealthVerdict",
    "LocalProcessLiveness",
    "MemoryGuard",
    "MergeRefusal",
    "MergeResult",
    "OwnerLiveness",
    "SlurmStepLiveness",
    "Unit",
    "UnitOutcome",
    "UnitPlan",
    "UnitResult",
    "WorkQueue",
    "assess_run",
    "combine_terms",
    "kill_by_pid",
    "merge_run",
    "plan_units",
    "reap",
    "verify_inventory",
]
