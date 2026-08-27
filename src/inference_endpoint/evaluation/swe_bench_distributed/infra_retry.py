# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry infrastructure faults, but only where non-execution is *provable*.

A retry is a correctness decision, not a convenience. Re-running a command that
may already have run can apply an edit twice, delete something twice, or double
a test run, and none of those announce themselves. So the gate here is not "an
error happened" -- it is "the work provably did not happen".

The evidence comes from the failure itself. An exception may expose
``provable_non_execution``: for a Pyxis step that is the status file still
reading ``pending`` **and** no in-band sentinel, meaning the step script did not
run even its first line. Anything that does not make that claim is not retried,
which is the safe default for every exception type this module has never heard
of.

Retries are bounded and, more importantly, **counted**. The banked campaign this
is ported from retried environment faults without limit and without counting
them (``wq_worker.sh:41`` ``WQ_MAX_ATTEMPTS=5``, with ``:256`` "ENVIRONMENT
FAULTS DO NOT CONSUME THE UNIT'S ATTEMPT BUDGET"), which is precisely why nobody
knew how many there had been. A retry loop that quietly absorbs the defect it
compensates for turns a broken cluster into an invisible one: the measured
effect of adding this loop was ``RunnerError`` 59 -> 7 and resolve 47.0% ->
70.0% against a banked 70.67% on the identical 200 instances, and a run that
needs that much rescuing is not a clean run even when it finishes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
#: Above this share of operations needing a retry, a run is DEGRADED even if
#: every unit eventually succeeded. Rescuing one operation in fifty is not a
#: healthy fleet, it is a fleet that happened to be caught.
DEGRADED_RETRY_FRACTION = 0.02

_T = TypeVar("_T")


class RetryOutcome(StrEnum):
    #: Provably never executed; another attempt follows.
    RETRYING = "retrying"
    #: A later attempt succeeded.
    RECOVERED = "recovered"
    #: Provably never executed, but the attempt budget ran out.
    EXHAUSTED = "exhausted"
    #: The work may have executed. Retrying could double-apply it, so this is a
    #: hard failure by construction.
    NOT_RETRYABLE = "not_retryable"


class RunQuality(StrEnum):
    CLEAN = "CLEAN"
    OK_WITH_RETRIES = "OK_WITH_RETRIES"
    DEGRADED = "DEGRADED"


def is_provable_non_execution(error: BaseException) -> bool:
    """Whether ``error`` proves its operation never ran.

    Read as an attribute rather than an isinstance check so the producer of the
    evidence (the Pyxis step runner) and this consumer stay decoupled. An
    exception that does not claim the property is never retried.
    """
    return getattr(error, "provable_non_execution", False) is True


@dataclass(frozen=True, slots=True)
class RetryRecord:
    target: str
    attempt: int
    outcome: RetryOutcome
    detail: str | None = None
    at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "attempt": self.attempt,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "at": self.at,
        }


class InfraRetryLedger:
    """Every provable non-execution and what became of it.

    Appends one JSON line per event when given a path, so a run that dies still
    leaves its retry history behind, and holds the same records in memory for
    :meth:`summary`. Accounting must never be able to take a run down, so a
    write failure is logged and swallowed -- but the in-memory counters are
    updated first, so the summary is correct even then.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._records: list[RetryRecord] = []
        self._operations = 0

    @property
    def records(self) -> list[RetryRecord]:
        with self._lock:
            return list(self._records)

    @property
    def operations(self) -> int:
        """Operations submitted to the retry wrapper: the denominator."""
        with self._lock:
            return self._operations

    def note_operation(self) -> None:
        with self._lock:
            self._operations += 1

    def record(
        self,
        *,
        target: str,
        attempt: int,
        outcome: RetryOutcome,
        detail: str | None = None,
    ) -> None:
        entry = RetryRecord(
            target=target,
            attempt=attempt,
            outcome=outcome,
            detail=detail,
            at=time.time(),
        )
        with self._lock:
            self._records.append(entry)
            path = self.path
            if path is not None:
                try:
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry.to_dict()) + "\n")
                except OSError:
                    logger.warning(
                        "could not append to the infra retry ledger", exc_info=True
                    )

    @classmethod
    def from_jsonl(cls, path: Path) -> InfraRetryLedger:
        """Load a ledger written by another process.

        The SWE-bench service is an isolated subproject that must not import the
        benchmark client, so its Pyxis step runner writes this same record shape
        directly. Sharing a file format rather than a module is the only way the
        two halves can agree, and reading it here is what turns per-step retries
        into a run-level `run_quality`.

        Unparseable lines are skipped: a truncated final line from a run that
        died is expected, and losing the whole history to it would be worse.
        """
        ledger = cls(path)
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            return ledger
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                record = RetryRecord(
                    target=str(raw["target"]),
                    attempt=int(raw["attempt"]),
                    outcome=RetryOutcome(raw["outcome"]),
                    detail=raw.get("detail"),
                    at=float(raw.get("at") or 0.0),
                )
            except (ValueError, KeyError, TypeError):
                logger.warning("skipping unreadable infra retry record")
                continue
            ledger._records.append(record)
        # Every first attempt that needed a retry represents one operation; a
        # writer that only logs failures cannot report the clean denominator, so
        # it is reported as unknown rather than guessed.
        ledger._operations = sum(1 for r in ledger._records if r.attempt == 1)
        return ledger

    def summary(self) -> dict[str, Any]:
        """Counters a report can publish without re-deriving anything."""
        records = self.records
        operations = self.operations
        recovered = {r.target for r in records if r.outcome is RetryOutcome.RECOVERED}
        exhausted = {r.target for r in records if r.outcome is RetryOutcome.EXHAUSTED}
        return {
            "infra_retries_total": len(records),
            "infra_retry_operations": operations,
            "infra_retry_outcomes": dict(
                Counter(r.outcome.value for r in records).most_common()
            ),
            "infra_retry_succeeded_on_attempt": dict(
                Counter(
                    str(r.attempt)
                    for r in records
                    if r.outcome is RetryOutcome.RECOVERED
                ).most_common()
            ),
            # A target that recovered and later exhausted was not saved.
            "instances_saved_by_retry": len(recovered - exhausted),
            "infra_retries_exhausted": sum(
                1 for r in records if r.outcome is RetryOutcome.EXHAUSTED
            ),
            "run_quality": self.run_quality().value,
        }

    def run_quality(self) -> RunQuality:
        records = self.records
        if any(
            r.outcome in (RetryOutcome.EXHAUSTED, RetryOutcome.NOT_RETRYABLE)
            for r in records
        ):
            return RunQuality.DEGRADED
        if not records:
            return RunQuality.CLEAN
        operations = max(1, self.operations)
        if len(records) > DEGRADED_RETRY_FRACTION * operations:
            return RunQuality.DEGRADED
        return RunQuality.OK_WITH_RETRIES


def retry_on_provable_non_execution(
    operation: Callable[[], _T],
    *,
    target: str,
    ledger: InfraRetryLedger | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_s: float = 2.0,
    max_backoff_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Call ``operation``, retrying only failures that prove it never ran.

    Raises the last failure when the budget is exhausted, and re-raises
    immediately -- without consuming the budget -- for anything that does not
    prove non-execution.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if ledger is not None:
        ledger.note_operation()
    for attempt in range(1, max_attempts + 1):
        try:
            result = operation()
        except Exception as exc:
            if not is_provable_non_execution(exc):
                # The work may have run. Retrying could double-apply it.
                if ledger is not None:
                    ledger.record(
                        target=target,
                        attempt=attempt,
                        outcome=RetryOutcome.NOT_RETRYABLE,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                raise
            if attempt == max_attempts:
                if ledger is not None:
                    ledger.record(
                        target=target,
                        attempt=attempt,
                        outcome=RetryOutcome.EXHAUSTED,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                raise
            if ledger is not None:
                ledger.record(
                    target=target,
                    attempt=attempt,
                    outcome=RetryOutcome.RETRYING,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            logger.warning(
                "%s provably never executed (attempt %d/%d): %s -- retrying",
                target,
                attempt,
                max_attempts,
                exc,
            )
            sleep(min(max_backoff_s, backoff_s * attempt))
            continue
        if attempt > 1 and ledger is not None:
            ledger.record(
                target=target, attempt=attempt, outcome=RetryOutcome.RECOVERED
            )
        return result
    raise AssertionError("unreachable")  # pragma: no cover
