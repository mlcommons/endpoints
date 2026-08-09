# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split a unit's error instances into infrastructure damage and genuine failures.

A SWE-bench run reports three per-instance outcomes: resolved, unresolved, and
*error*. The agent phase has an infrastructure sentinel (the Pyxis
``infrastructure_failure_path``), but the eval phase has none: an instance whose
evaluation container wedged is booked as ``error``, which counts as "accounted
for", so the unit is published successful and is never retried. Those instances
silently poison a run that can then never reach a full result.

Classification exists to catch exactly that. It reads each error instance's
``run_instance.log`` and assigns one kind.

BIAS RULE -- this is the whole design and it is deliberately asymmetric. If an
error cannot be classified confidently it is treated as GENUINE, never as
infrastructure. A false bad-run costs one redo; a false retry silently biases
the measurement toward optimism, and an optimistic accuracy number is worse than
no number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorKind(StrEnum):
    """One classification of an error instance."""

    # Infrastructure: a defect in the runtime we provided, safe to retry.
    CONTAINER_EXEC_REFUSED = "container_exec_refused"
    CONTAINER_FORK_EAGAIN = "container_fork_eagain"
    RUNTIME_READ_TIMEOUT = "runtime_read_timeout"
    IMAGE_BUILD_TIMEOUT = "image_build_timeout"
    IMAGE_BUILD_ERROR = "image_build_error"
    STEP_INFRASTRUCTURE_FAILURE = "step_infrastructure_failure"
    ENDPOINT_CHANGED = "endpoint_changed"

    # Genuine: a real outcome of the model's patch, or unreadable. Never retried.
    TEST_TIMEOUT = "test_timeout"
    TEST_MEMORY_EXCEEDED = "test_memory_exceeded"
    PATCH_APPLY_FAILED = "patch_apply_failed"
    UNKNOWN = "unknown"


#: Retryable. Every member is a defect in infrastructure we control.
INFRA_KINDS: frozenset[ErrorKind] = frozenset(
    {
        ErrorKind.CONTAINER_EXEC_REFUSED,
        ErrorKind.CONTAINER_FORK_EAGAIN,
        ErrorKind.RUNTIME_READ_TIMEOUT,
        ErrorKind.IMAGE_BUILD_TIMEOUT,
        ErrorKind.IMAGE_BUILD_ERROR,
        ErrorKind.STEP_INFRASTRUCTURE_FAILURE,
        ErrorKind.ENDPOINT_CHANGED,
    }
)

#: Never retried.
#:
#: ``TEST_TIMEOUT`` is a plausible model outcome: a patch that makes the suite
#: loop is a failing patch. ``TEST_MEMORY_EXCEEDED`` is its exact parallel -- a
#: patch that makes a graded test allocate without bound is a failing patch, and
#: the alternative to killing it was never "the test passes", it was "the host
#: OOMs and the instance still never completes". ``PATCH_APPLY_FAILED`` is the
#: model emitting a diff that does not apply; SWE-bench books it as ``error``
#: rather than ``unresolved``, but it is model behaviour. ``UNKNOWN`` is the bias
#: rule.
GENUINE_KINDS: frozenset[ErrorKind] = frozenset(
    {
        ErrorKind.TEST_TIMEOUT,
        ErrorKind.TEST_MEMORY_EXCEEDED,
        ErrorKind.PATCH_APPLY_FAILED,
        ErrorKind.UNKNOWN,
    }
)

# ORDERED. First match wins, and the order is load-bearing.
#
# CONTAINER_FORK_EAGAIN and TEST_TIMEOUT come BEFORE CONTAINER_EXEC_REFUSED: a
# timed-out or fork-failed evaluation frequently *also* emits "container state
# improper" while the harness tears the container down, and reading that as a
# wedge would retry a genuine model outcome.
#
# PATCH_APPLY_FAILED is checked LAST: if a container also wedged, the wedge
# wins, because a wedged container's verdict is unreliable either way. Only a
# log with no infrastructure signature at all reaches this rule.
_RULES: tuple[tuple[ErrorKind, tuple[str, ...]], ...] = (
    (
        ErrorKind.CONTAINER_FORK_EAGAIN,
        ("fork/exec /usr/bin/conmon: resource temporarily unavailable",),
    ),
    (ErrorKind.TEST_TIMEOUT, ("Test timed out after",)),
    (
        ErrorKind.CONTAINER_EXEC_REFUSED,
        (
            "can only create exec sessions on running containers",
            "container state improper",
        ),
    ),
    (ErrorKind.RUNTIME_READ_TIMEOUT, ("Read timed out. (read timeout=",)),
    (
        ErrorKind.PATCH_APPLY_FAILED,
        (
            "Reversed (or previously applied) patch detected",
            ">>>>> Patch Apply Failed",
            "hunk FAILED",
            "hunk failed",
        ),
    ),
)


def classify_eval_log(text: str) -> ErrorKind:
    """Classify one instance's evaluation log.

    ``BuildImageError`` is checked before the ordered rules because its message
    embeds the same "Read timed out" / "500" strings the other rules look for,
    so any other order misattributes a build failure.
    """
    if "BuildImageError" in text:
        if "Read timed out" in text:
            return ErrorKind.IMAGE_BUILD_TIMEOUT
        return ErrorKind.IMAGE_BUILD_ERROR
    for kind, needles in _RULES:
        if any(needle in text for needle in needles):
            return kind
    return ErrorKind.UNKNOWN


@dataclass(slots=True)
class UnitClassification:
    """Per-kind counts for one unit's error instances."""

    kinds: dict[ErrorKind, int] = field(default_factory=dict)
    error_instance_ids: tuple[str, ...] = ()
    #: False when the run's report could not be read at all. "Not measured" and
    #: "measured zero" are different, and conflating them once let a damaged
    #: unit into a clean set.
    measured: bool = False

    @property
    def infra_count(self) -> int:
        return sum(count for kind, count in self.kinds.items() if kind in INFRA_KINDS)

    @property
    def genuine_count(self) -> int:
        return sum(count for kind, count in self.kinds.items() if kind in GENUINE_KINDS)

    @property
    def should_retry(self) -> bool:
        return self.infra_count > 0

    def as_counts(self) -> dict[str, int]:
        return {kind.value: count for kind, count in sorted(self.kinds.items())}


def _find_instance_log(output_dir: Path, instance_id: str) -> Path | None:
    patterns = (
        f"logs/run_evaluation/*/*/{instance_id}/run_instance.log",
        f"logs/run_evaluation/*/*/*/{instance_id}/run_instance.log",
    )
    for pattern in patterns:
        for match in sorted(output_dir.glob(pattern)):
            return match
    return None


def memory_kill_markers(killed_dir: Path, instance_id: str) -> bool:
    """True only for an *eval*-phase memory kill.

    Phase is load-bearing and the two cases must never be collapsed. An eval
    kill destroyed a graded result, so the instance's error is a genuine
    failure. An agent kill merely makes one tool call return an error
    observation and the agent carries on, so the instance still reaches a real
    outcome; that marker exists for audit and must not influence classification.

    A marker beats any log heuristic: a SIGKILLed test leaves an ambiguous log,
    but the kill itself is a fact recorded before acting.
    """
    return any(killed_dir.glob(f"eval.{instance_id}.*.json"))


def classify_unit(
    output_dir: Path,
    error_instance_ids: list[str] | tuple[str, ...] | None,
    *,
    killed_dir: Path | None = None,
    infrastructure_failure: bool = False,
    endpoint_changed: bool = False,
) -> UnitClassification:
    """Classify every error instance of one unit.

    ``infrastructure_failure`` carries the Pyxis agent-phase sentinel, and
    ``endpoint_changed`` carries a mismatch between the inference endpoint
    fingerprint recorded at claim time and at publish time -- an engine
    restarted under a live client produces a plausible-looking run that must not
    be scored.
    """
    kinds: dict[ErrorKind, int] = {}

    if infrastructure_failure:
        kinds[ErrorKind.STEP_INFRASTRUCTURE_FAILURE] = (
            kinds.get(ErrorKind.STEP_INFRASTRUCTURE_FAILURE, 0) + 1
        )
    if endpoint_changed:
        kinds[ErrorKind.ENDPOINT_CHANGED] = kinds.get(ErrorKind.ENDPOINT_CHANGED, 0) + 1

    if error_instance_ids is None:
        return UnitClassification(kinds=kinds, error_instance_ids=(), measured=False)

    ids = tuple(str(x) for x in error_instance_ids)
    for instance_id in ids:
        if killed_dir is not None and memory_kill_markers(killed_dir, instance_id):
            kinds[ErrorKind.TEST_MEMORY_EXCEEDED] = (
                kinds.get(ErrorKind.TEST_MEMORY_EXCEEDED, 0) + 1
            )
            continue
        log_path = _find_instance_log(output_dir, instance_id)
        kind = ErrorKind.UNKNOWN
        if log_path is not None:
            try:
                kind = classify_eval_log(log_path.read_text(errors="replace"))
            except OSError:
                logger.debug("could not read %s", log_path, exc_info=True)
        kinds[kind] = kinds.get(kind, 0) + 1

    return UnitClassification(kinds=kinds, error_instance_ids=ids, measured=True)
