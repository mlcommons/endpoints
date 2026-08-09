# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable work queue for distributed SWE-bench units.

The queue is a directory tree so that a client crash costs nothing but the
in-flight units, and so that the merge gate reads durable records rather than
process memory.

Layout under ``root``::

    units.json                     immutable plan (see units.py)
    claims/<unit_id>/owner         json owner record, written temp+rename
    claims/<unit_id>/hb            heartbeat, mtime only
    results/<unit_id>.json         terminal record (succeeded OR abandoned)
    failed/<unit_id>.<n>.json      one record per *counted* attempt
    failed/env/<unit_id>.*.json    environment faults, NOT counted
    failed/artifacts/<unit_id>...  evidence snapshot taken before a retry

Two invariants are load-bearing and are enforced here rather than by
convention:

1. ``claim()`` is ``os.mkdir`` and nothing else. ``mkdir`` on an existing
   directory fails atomically with ``EEXIST`` on every filesystem we run on,
   including Lustre, so exactly one of N racing callers wins. ``makedirs(...,
   exist_ok=True)`` would hand the unit to every caller.
2. ``requeue()`` is the only way to make a terminal unit runnable again, and it
   removes the result, the claim tombstone *and* the counted attempt records
   together. Deleting a result file by hand does not requeue a unit -- the
   claim tombstone still hides it -- and that misunderstanding has cost real
   campaign time.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import socket
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import msgspec

from .units import PLAN_FILENAME, UnitPlan, read_plan

logger = logging.getLogger(__name__)

_OWNER = "owner"
_HEARTBEAT = "hb"
_CLAIM_CONTENTS = frozenset({_OWNER, _HEARTBEAT})

# Small files that explain a failure. Snapshotted before a retry reuses the
# unit's run directory: a unit that fails and then succeeds otherwise leaves
# only the success's artifacts, and a post-mortem then reads the wrong run.
EVIDENCE_FILES = (
    "status.json",
    "swe_bench_results.json",
    "preds.json",
    "swe_bench_service_status.json",
)
EVIDENCE_LOG_TAIL_BYTES = 200_000
EVIDENCE_LOGS = ("swe_bench_agent.log", "swe_bench_eval.log")


class ClaimError(RuntimeError):
    """A claim operation could not be performed."""


class UnitOutcome(StrEnum):
    """How an attempt at a unit ended.

    ``ENV_FAULT`` is deliberately separate from ``FAILED``: a broken service, an
    unreachable endpoint or a refused gate is a property of the *worker*, not of
    the unit. Charging it to the unit's attempt budget abandons perfectly good
    units because they happened to land on a sick host.
    """

    SUCCEEDED = "succeeded"
    INFRA = "infra"
    FAILED = "failed"
    ENV_FAULT = "env_fault"


#: Outcomes that consume one of the unit's ``max_attempts``.
COUNTED_OUTCOMES = frozenset({UnitOutcome.INFRA, UnitOutcome.FAILED})


@dataclass(slots=True)
class UnitResult:
    """A terminal or attempt record for one unit."""

    unit_id: str
    run_id: str
    plan_digest: str
    outcome: UnitOutcome
    accounted_instance_ids: tuple[str, ...] = ()
    resolved_instance_ids: tuple[str, ...] = ()
    infra_error_count: int = 0
    genuine_error_count: int = 0
    error_kinds: dict[str, int] = field(default_factory=dict)
    service_url: str | None = None
    endpoint_fingerprint: str | None = None
    service_run_id: str | None = None
    attempt: int = 0
    abandoned: bool = False
    duration_s: float = 0.0
    detail: str | None = None
    finished_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "run_id": self.run_id,
            "plan_digest": self.plan_digest,
            "outcome": self.outcome.value,
            "accounted_instance_ids": list(self.accounted_instance_ids),
            "resolved_instance_ids": list(self.resolved_instance_ids),
            "infra_error_count": self.infra_error_count,
            "genuine_error_count": self.genuine_error_count,
            "error_kinds": dict(self.error_kinds),
            "service_url": self.service_url,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "service_run_id": self.service_run_id,
            "attempt": self.attempt,
            "abandoned": self.abandoned,
            "duration_s": self.duration_s,
            "detail": self.detail,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> UnitResult:
        return cls(
            unit_id=str(raw["unit_id"]),
            run_id=str(raw["run_id"]),
            plan_digest=str(raw["plan_digest"]),
            outcome=UnitOutcome(str(raw["outcome"])),
            accounted_instance_ids=tuple(
                str(x) for x in raw.get("accounted_instance_ids") or ()
            ),
            resolved_instance_ids=tuple(
                str(x) for x in raw.get("resolved_instance_ids") or ()
            ),
            infra_error_count=int(raw.get("infra_error_count") or 0),
            genuine_error_count=int(raw.get("genuine_error_count") or 0),
            error_kinds=dict(raw.get("error_kinds") or {}),
            service_url=raw.get("service_url"),
            endpoint_fingerprint=raw.get("endpoint_fingerprint"),
            service_run_id=raw.get("service_run_id"),
            attempt=int(raw.get("attempt") or 0),
            abandoned=bool(raw.get("abandoned")),
            duration_s=float(raw.get("duration_s") or 0.0),
            detail=raw.get("detail"),
            finished_at=float(raw.get("finished_at") or 0.0),
        )


@dataclass(slots=True)
class OwnerRecord:
    unit_id: str
    host: str
    pid: int
    boot_id: str
    plan_digest: str
    claimed_at: float
    endpoint_fingerprint: str | None = None
    slurm_job_id: str | None = None
    slurm_step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "host": self.host,
            "pid": self.pid,
            "boot_id": self.boot_id,
            "plan_digest": self.plan_digest,
            "claimed_at": self.claimed_at,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "slurm_job_id": self.slurm_job_id,
            "slurm_step_id": self.slurm_step_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> OwnerRecord:
        return cls(
            unit_id=str(raw["unit_id"]),
            host=str(raw.get("host") or ""),
            pid=int(raw.get("pid") or 0),
            boot_id=str(raw.get("boot_id") or ""),
            plan_digest=str(raw.get("plan_digest") or ""),
            claimed_at=float(raw.get("claimed_at") or 0.0),
            endpoint_fingerprint=raw.get("endpoint_fingerprint"),
            slurm_job_id=raw.get("slurm_job_id"),
            slurm_step_id=raw.get("slurm_step_id"),
        )


def boot_id() -> str:
    """Identify this boot of this host.

    A pid alone is not proof of liveness: after a reboot the same pid can belong
    to something else entirely, and the reaper would then conclude a dead owner
    is alive and leave its unit blocked forever.
    """
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        try:
            return str(int(time.time() - time.monotonic()))
        except (OSError, ValueError):  # pragma: no cover - defensive
            return "unknown"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(msgspec.json.encode(payload))
    tmp.replace(path)


class WorkQueue:
    """Filesystem-backed queue over an immutable :class:`UnitPlan`."""

    def __init__(self, root: os.PathLike[str] | str, plan: UnitPlan) -> None:
        self.root = Path(root)
        self.plan = plan
        self.claims_dir = self.root / "claims"
        self.results_dir = self.root / "results"
        self.failed_dir = self.root / "failed"
        self.env_failed_dir = self.failed_dir / "env"
        self.artifacts_dir = self.failed_dir / "artifacts"
        for directory in (
            self.root,
            self.claims_dir,
            self.results_dir,
            self.failed_dir,
            self.env_failed_dir,
            self.artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.plan.write(self.root)
        self._boot_id = boot_id()

    # ------------------------------------------------------------------ open --

    @classmethod
    def open(cls, root: os.PathLike[str] | str) -> WorkQueue:
        """Reopen an existing queue, reading its plan from disk."""
        root_path = Path(root)
        return cls(root_path, read_plan(root_path / PLAN_FILENAME))

    # ----------------------------------------------------------- inspection --

    def claimed_unit_ids(self) -> set[str]:
        try:
            return {entry.name for entry in self.claims_dir.iterdir() if entry.is_dir()}
        except FileNotFoundError:  # pragma: no cover - created in __init__
            return set()

    def completed_unit_ids(self) -> set[str]:
        return {path.stem for path in self.results_dir.glob("*.json")}

    def available_unit_ids(self) -> list[str]:
        """Units that are neither claimed nor terminal, in plan order.

        Subtracting *both* claims and results is what makes a hand-deleted
        result file a no-op: the claim tombstone still hides the unit. Use
        :meth:`requeue`.
        """
        taken = self.claimed_unit_ids() | self.completed_unit_ids()
        return [unit_id for unit_id in self.plan.unit_ids if unit_id not in taken]

    def attempts(self, unit_id: str) -> int:
        """Number of *counted* attempts recorded for a unit."""
        return len(list(self.failed_dir.glob(f"{unit_id}.*.json")))

    def owner(self, unit_id: str) -> OwnerRecord | None:
        path = self.claims_dir / unit_id / _OWNER
        try:
            raw = msgspec.json.decode(path.read_bytes(), type=dict)
        except (OSError, msgspec.DecodeError):
            return None
        try:
            return OwnerRecord.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            return None

    def heartbeat_age(self, unit_id: str, *, now: float | None = None) -> float | None:
        path = self.claims_dir / unit_id / _HEARTBEAT
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        return (time.time() if now is None else now) - mtime

    def result(self, unit_id: str) -> UnitResult | None:
        path = self.results_dir / f"{unit_id}.json"
        try:
            raw = msgspec.json.decode(path.read_bytes(), type=dict)
        except (OSError, msgspec.DecodeError):
            return None
        try:
            return UnitResult.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            return None

    def results(self) -> dict[str, UnitResult]:
        found: dict[str, UnitResult] = {}
        for path in sorted(self.results_dir.glob("*.json")):
            result = self.result(path.stem)
            if result is not None:
                found[path.stem] = result
        return found

    # ---------------------------------------------------------------- claim --

    def claim(
        self,
        unit_id: str,
        *,
        endpoint_fingerprint: str | None = None,
    ) -> OwnerRecord | None:
        """Take exclusive ownership of ``unit_id``; ``None`` if someone else has it."""
        if unit_id not in self.plan.unit_ids:
            raise ClaimError(
                f"{unit_id!r} is not in the plan for run {self.plan.run_id}"
            )
        claim_dir = self.claims_dir / unit_id
        try:
            # THE RACE IS DECIDED HERE. mkdir, never makedirs/exist_ok.
            os.mkdir(claim_dir)
        except FileExistsError:
            return None
        except OSError as exc:
            if exc.errno == errno.EEXIST:  # pragma: no cover - platform variance
                return None
            raise ClaimError(f"could not claim {unit_id}: {exc}") from exc

        record = OwnerRecord(
            unit_id=unit_id,
            host=socket.gethostname(),
            pid=os.getpid(),
            boot_id=self._boot_id,
            plan_digest=self.plan.digest,
            claimed_at=time.time(),
            endpoint_fingerprint=endpoint_fingerprint,
            slurm_job_id=os.environ.get("SLURM_JOB_ID") or None,
            slurm_step_id=os.environ.get("SLURM_STEP_ID") or None,
        )
        # Sole owner from here, but still temp+rename so the reaper never reads
        # a half-written owner record and calls it malformed.
        _atomic_write_json(claim_dir / _OWNER, record.to_dict())
        (claim_dir / _HEARTBEAT).touch()
        return record

    def beat(self, unit_id: str) -> None:
        path = self.claims_dir / unit_id / _HEARTBEAT
        try:
            path.touch()
        except OSError:
            logger.debug("could not refresh heartbeat for %s", unit_id, exc_info=True)

    def release(self, unit_id: str) -> bool:
        """Hand a claimed unit back to the queue.

        Removes the claim *directory*. Removing only the ``owner`` file leaves an
        ownerless directory, which still hides the unit and merely relabels the
        problem.
        """
        claim_dir = self.claims_dir / unit_id
        if not claim_dir.exists():
            return False
        shutil.rmtree(claim_dir, ignore_errors=True)
        return True

    def is_pure_bookkeeping(self, unit_id: str) -> bool:
        """True when a claim directory holds only ``owner``/``hb``."""
        claim_dir = self.claims_dir / unit_id
        try:
            contents = {entry.name for entry in claim_dir.iterdir()}
        except OSError:
            return False
        return not (contents - _CLAIM_CONTENTS)

    # -------------------------------------------------------------- publish --

    def publish(self, result: UnitResult) -> None:
        """Record a terminal result and release the claim.

        Releasing here is not optional. The abandon path once published a result
        but kept the claim directory, so ``claims/`` and ``results/`` disagreed
        for the rest of the campaign and every reaper pass had a phantom to
        reason about. Releasing is safe because :meth:`available_unit_ids`
        subtracts results as well as claims.
        """
        self._check_digest(result)
        _atomic_write_json(
            self.results_dir / f"{result.unit_id}.json", result.to_dict()
        )
        self.release(result.unit_id)

    def record_attempt(self, result: UnitResult) -> int:
        """Record a non-terminal attempt. Returns the counted-attempt total.

        ``ENV_FAULT`` attempts are written to ``failed/env/`` and do not
        increment the counter.
        """
        self._check_digest(result)
        if result.outcome is UnitOutcome.ENV_FAULT:
            path = self.env_failed_dir / f"{result.unit_id}.{time.time_ns()}.json"
            _atomic_write_json(path, result.to_dict())
            return self.attempts(result.unit_id)
        count = self.attempts(result.unit_id) + 1
        result.attempt = count
        _atomic_write_json(
            self.failed_dir / f"{result.unit_id}.{count}.json", result.to_dict()
        )
        return count

    def snapshot_evidence(self, unit_id: str, source_dir: Path, attempt: int) -> Path:
        """Copy the small files that explain a failure before a retry overwrites them."""
        target = self.artifacts_dir / f"{unit_id}.attempt{attempt}"
        target.mkdir(parents=True, exist_ok=True)
        for name in EVIDENCE_FILES:
            candidate = source_dir / name
            if candidate.is_file():
                try:
                    shutil.copy2(candidate, target / name)
                except OSError:
                    logger.debug("could not snapshot %s", candidate, exc_info=True)
        for name in EVIDENCE_LOGS:
            candidate = source_dir / name
            if not candidate.is_file():
                continue
            try:
                with candidate.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - EVIDENCE_LOG_TAIL_BYTES))
                    (target / f"{name}.tail").write_bytes(handle.read())
            except OSError:
                logger.debug("could not snapshot %s", candidate, exc_info=True)
        return target

    def abandon(self, result: UnitResult) -> None:
        """Publish a terminal, explicitly-abandoned result."""
        result.abandoned = True
        self.publish(result)

    # -------------------------------------------------------------- requeue --

    def requeue(self, unit_id: str) -> dict[str, list[str]]:
        """Make a unit runnable again. The *only* supported way.

        Removes, together: the terminal result, the claim tombstone, and every
        counted attempt record. Removing any subset leaves the unit invisible or
        already out of attempts, which is how "I deleted the result, why is it
        not rerunning?" happens.
        """
        if unit_id not in self.plan.unit_ids:
            raise ClaimError(
                f"{unit_id!r} is not in the plan for run {self.plan.run_id}"
            )
        removed: dict[str, list[str]] = {"results": [], "claims": [], "attempts": []}
        result_path = self.results_dir / f"{unit_id}.json"
        if result_path.exists():
            result_path.unlink()
            removed["results"].append(str(result_path))
        claim_dir = self.claims_dir / unit_id
        if claim_dir.exists():
            shutil.rmtree(claim_dir, ignore_errors=True)
            removed["claims"].append(str(claim_dir))
        for path in sorted(self.failed_dir.glob(f"{unit_id}.*.json")):
            path.unlink()
            removed["attempts"].append(str(path))
        return removed

    # ---------------------------------------------------------------- utils --

    def _check_digest(self, result: UnitResult) -> None:
        if result.plan_digest != self.plan.digest:
            raise ClaimError(
                f"refusing to record {result.unit_id}: plan digest "
                f"{result.plan_digest[:12]} does not match this queue's "
                f"{self.plan.digest[:12]}"
            )
        if result.run_id != self.plan.run_id:
            raise ClaimError(
                f"refusing to record {result.unit_id}: run id {result.run_id!r} "
                f"does not match this queue's {self.plan.run_id!r}"
            )
