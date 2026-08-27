# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Return orphaned claims to the queue. Nothing else.

A false reap is the worst thing this system can do. Releasing a claim whose
owner is still running puts the unit back in the queue while it is executing, a
second worker takes it, both write results, and the run has duplicate work, a
wrong denominator, and no error anywhere -- the exact silent corruption the
atomic claim exists to prevent, reintroduced by the janitor.

Therefore the reaper is conservative in one specific direction: **uncertainty
never escalates.** If liveness cannot be determined, nothing is released.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .queue import OwnerRecord, WorkQueue

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 60


class Liveness(StrEnum):
    ALIVE = "alive"
    DEAD = "dead"
    #: Could not tell. Treated as ALIVE for the purpose of reaping.
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class LivenessVerdict:
    state: Liveness
    #: Which layer decided. ``"step"`` gets a shorter staleness threshold: a
    #: step that died inside a live job took its tasks with it immediately, so
    #: there is no reason to wait an hour to believe it.
    scope: str = "process"
    detail: str = ""


class OwnerLiveness(Protocol):
    """Decides whether the process that claimed a unit still exists."""

    def probe(self, owner: OwnerRecord) -> LivenessVerdict:
        pass


class LocalProcessLiveness:
    """Liveness by pid, scoped to one host and one boot.

    A pid on its own is not evidence: after a reboot the same number can belong
    to something unrelated, so an owner from a different boot of this host is
    dead, and an owner from a different host is indeterminate (we cannot see it).
    """

    def __init__(self, *, host: str | None = None, boot: str | None = None) -> None:
        import socket

        from .queue import boot_id

        self.host = host if host is not None else socket.gethostname()
        self.boot = boot if boot is not None else boot_id()

    def probe(self, owner: OwnerRecord) -> LivenessVerdict:
        if owner.host != self.host:
            return LivenessVerdict(
                Liveness.INDETERMINATE, "process", f"owner is on {owner.host}"
            )
        if owner.boot_id and owner.boot_id != self.boot:
            return LivenessVerdict(
                Liveness.DEAD, "process", "host rebooted since claim"
            )
        if owner.pid <= 0:
            return LivenessVerdict(Liveness.INDETERMINATE, "process", "no pid recorded")
        try:
            os.kill(owner.pid, 0)
        except ProcessLookupError:
            return LivenessVerdict(Liveness.DEAD, "process", "pid gone")
        except PermissionError:
            # Exists, owned by someone else.
            return LivenessVerdict(Liveness.ALIVE, "process", "pid exists")
        except OSError:
            return LivenessVerdict(Liveness.INDETERMINATE, "process", "kill(0) failed")
        return LivenessVerdict(Liveness.ALIVE, "process", "pid exists")


class SlurmStepLiveness:
    """Liveness by SLURM job *and step*.

    An owner is dead when its job is absent from ``squeue``, **or** when the job
    is alive but its step is gone. The second clause is not optional: a step can
    die inside a live job (a killed srun, an OOM-terminated step) and SLURM
    kills that step's tasks, but the job never leaves ``squeue``, so a
    job-level-only rule blocks those units for the entire life of the
    allocation.

    Step liveness comes from ``scontrol show step``, never ``squeue -s``: on the
    clusters this was built for ``squeue -s`` reports only ``.extern`` and never
    the worker step, so using it would mark every live step dead and falsely
    reap every claim.

    Every failure to read SLURM yields ``INDETERMINATE``. An unavailable
    ``squeue`` must never be read as "no jobs are running".
    """

    def __init__(self, *, timeout_s: int = _PROBE_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s

    def _run(self, argv: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.timeout_s
            )
        except (OSError, subprocess.SubprocessError):
            logger.warning("reaper: %s unavailable; releasing nothing", argv[0])
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    def live_job_ids(self) -> set[str] | None:
        out = self._run(["squeue", "-h", "-o", "%i"])
        if out is None:
            return None
        ids: set[str] = set()
        for token in out.split():
            token = token.strip()
            if not token:
                continue
            ids.add(token)
            ids.add(token.split("_")[0].split(".")[0])
        if not ids and os.environ.get("SLURM_JOB_ID"):
            # An empty queue is legitimate in general, but not while we are
            # ourselves inside a job. That is what a broken squeue looks like.
            logger.warning(
                "reaper: squeue returned empty while inside a job; releasing nothing"
            )
            return None
        return ids

    def live_step_ids(self, job_id: str) -> set[str] | None:
        out = self._run(["scontrol", "show", "step", str(job_id)])
        if out is None:
            return None
        steps = {
            token.split("=", 1)[1].split(".", 1)[1]
            for token in out.split()
            if token.startswith("StepId=") and "." in token.split("=", 1)[1]
        }
        # A successful scontrol listing no step at all is implausible while the
        # job exists (there is always .extern): indeterminate, not empty.
        return steps or None

    def probe(self, owner: OwnerRecord) -> LivenessVerdict:
        if not owner.slurm_job_id:
            return LivenessVerdict(Liveness.INDETERMINATE, "job", "no job id recorded")
        jobs = self.live_job_ids()
        if jobs is None:
            return LivenessVerdict(Liveness.INDETERMINATE, "job", "squeue unreadable")
        if owner.slurm_job_id not in jobs:
            return LivenessVerdict(Liveness.DEAD, "job", "job absent from squeue")
        if not owner.slurm_step_id:
            return LivenessVerdict(
                Liveness.ALIVE, "job", "job present, no step recorded"
            )
        steps = self.live_step_ids(owner.slurm_job_id)
        if steps is None:
            # Indeterminate step liveness must not become MORE aggressive than
            # the job-level answer, which is "alive".
            return LivenessVerdict(Liveness.ALIVE, "job", "step list unreadable")
        if owner.slurm_step_id in steps:
            return LivenessVerdict(Liveness.ALIVE, "step", "step present")
        return LivenessVerdict(Liveness.DEAD, "step", "step gone inside a live job")


@dataclass(slots=True)
class ReapReport:
    released: list[str] = field(default_factory=list)
    kept: dict[str, str] = field(default_factory=dict)
    dry_run: bool = True

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self.released)


def reap(
    queue: WorkQueue,
    liveness: OwnerLiveness,
    *,
    stale_after_s: float = 3600.0,
    step_stale_after_s: float = 900.0,
    apply: bool = False,
    now: float | None = None,
) -> ReapReport:
    """Release claims whose owner is provably gone and which produced no result.

    All three conditions must hold: no result, a stale-enough heartbeat, and a
    ``DEAD`` liveness verdict. A verdict scoped to ``"step"`` uses the shorter
    ``step_stale_after_s``: when a step dies inside a job that stays in the
    queue, the job-level rule alone never fires and those units stay blocked for
    the entire life of the allocation.
    """
    report = ReapReport(dry_run=not apply)
    completed = queue.completed_unit_ids()
    for unit_id in sorted(queue.claimed_unit_ids()):
        if unit_id in completed:
            # Claims for completed units are harmless bookkeeping.
            report.kept[unit_id] = "has result"
            continue
        age = queue.heartbeat_age(unit_id, now=now)
        if age is None:
            report.kept[unit_id] = "no heartbeat to age"
            continue
        owner = queue.owner(unit_id)
        if owner is None:
            # We cannot prove anything about an unreadable owner, so age alone
            # decides. A claim holding anything but pure bookkeeping is not ours
            # to reason about at all.
            if not queue.is_pure_bookkeeping(unit_id):
                report.kept[unit_id] = "claim holds non-bookkeeping contents"
                continue
            verdict = LivenessVerdict(Liveness.DEAD, "process", "owner unreadable")
        else:
            verdict = liveness.probe(owner)
        threshold = step_stale_after_s if verdict.scope == "step" else stale_after_s
        if age < threshold:
            report.kept[unit_id] = f"heartbeat {age:.0f}s < {threshold:.0f}s"
            continue
        if verdict.state is not Liveness.DEAD:
            report.kept[unit_id] = f"owner {verdict.state.value}: {verdict.detail}"
            continue
        report.released.append(unit_id)
        if apply:
            queue.release(unit_id)
    return report
