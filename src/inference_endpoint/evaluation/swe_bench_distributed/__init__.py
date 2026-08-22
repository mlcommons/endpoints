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

from .classify import (
    GENUINE_KINDS,
    INFRA_KINDS,
    ErrorKind,
    UnitClassification,
    classify_eval_log,
    classify_unit,
)
from .gates import (
    CheckpointIdentityGate,
    EndpointFingerprintGate,
    Gate,
    GateFailure,
    GateReport,
    GateScaleError,
    ToolCallGate,
    run_gates,
)
from .guards import HealthTerm, HealthVerdict, MemoryGuard, combine_terms, kill_by_pid
from .merge import MergeRefusal, MergeResult, merge_run, verify_inventory
from .queue import (
    ClaimError,
    UnitOutcome,
    UnitResult,
    WorkQueue,
)
from .reaper import LocalProcessLiveness, OwnerLiveness, SlurmStepLiveness, reap
from .units import Unit, UnitPlan, plan_units

__all__ = [
    "GENUINE_KINDS",
    "INFRA_KINDS",
    "CheckpointIdentityGate",
    "ClaimError",
    "EndpointFingerprintGate",
    "ErrorKind",
    "Gate",
    "GateFailure",
    "GateReport",
    "GateScaleError",
    "HealthTerm",
    "HealthVerdict",
    "LocalProcessLiveness",
    "MemoryGuard",
    "MergeRefusal",
    "MergeResult",
    "OwnerLiveness",
    "SlurmStepLiveness",
    "ToolCallGate",
    "Unit",
    "UnitClassification",
    "UnitOutcome",
    "UnitPlan",
    "UnitResult",
    "WorkQueue",
    "classify_eval_log",
    "classify_unit",
    "combine_terms",
    "kill_by_pid",
    "merge_run",
    "plan_units",
    "reap",
    "run_gates",
    "verify_inventory",
]
