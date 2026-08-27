# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource guards for graded SWE-bench evaluation.

A graded test runs inside an evaluation container with no memory limit. A model
patch that makes a test allocate without bound will take the host down, and when
a whole client fleet shares one scheduler step, one host's OOM destroys every
peer's work along with it -- ``--kill-on-bad-exit=0`` does **not** prevent that,
because the scheduler escalates OOM separately from task exit codes.

Killing such a process is correct, not a distortion. A patch that makes a graded
test allocate without bound is a failing patch, exactly as a patch that makes it
loop forever is; the alternative to killing was never "the test passes", it was
"the host dies and the instance still never completes". The kill is recorded as
a marker file and the classifier books it as a genuine failure.

TWO RULES THAT ARE ENFORCED BY CONSTRUCTION HERE:

1. **Kill by pid, never by pattern.** A pattern such as ``runtests.py`` can
   appear in the guard's own command line, and a long-lived daemon can carry a
   dead process's argv for days. This module contains no ``pkill``/``pgrep``
   path at all, and :func:`kill_by_pid` refuses self and its own ancestors.
2. **A conjunctive guard must not degenerate.** When one honest term of an
   AND-guard permanently loses its data source, the conjunction collapses into
   its remaining, weaker clauses and starts firing on healthy targets -- that is
   how an idle watchdog killed a live bring-up. :func:`combine_terms` therefore
   returns ``INDETERMINATE``, never ``UNHEALTHY``, if any term has no evidence.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KILL_BYTES = 150 * 1024**3
DEFAULT_WARN_BYTES = 100 * 1024**3
#: Ancestors that prove a process is inside a container supervisor.
CONTAINER_SUPERVISORS = ("conmon", "containerd-shim", "runc", "enroot", "crun")
_ANCESTOR_DEPTH = 6


class HealthVerdict(StrEnum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    INDETERMINATE = "indeterminate"


@dataclass(slots=True)
class HealthTerm:
    """One clause of a conjunctive guard, with its evidence count.

    ``evidence`` is the number of observations the term actually made. A term
    that made none cannot vote, and must not be silently read as ``HEALTHY``
    (which would let the conjunction fire on the strength of the other clauses
    alone) nor as ``UNHEALTHY``.
    """

    name: str
    verdict: HealthVerdict
    evidence: int
    detail: str = ""


def combine_terms(terms: list[HealthTerm]) -> tuple[HealthVerdict, str]:
    """AND the terms, refusing to act on an unevidenced conjunction."""
    if not terms:
        return HealthVerdict.INDETERMINATE, "no terms"
    blind = [term.name for term in terms if term.evidence <= 0]
    if blind:
        return (
            HealthVerdict.INDETERMINATE,
            "no evidence for term(s): "
            + ", ".join(blind)
            + " -- a conjunction with a blind term cannot be trusted to be true",
        )
    indeterminate = [
        term.name for term in terms if term.verdict is HealthVerdict.INDETERMINATE
    ]
    if indeterminate:
        return (
            HealthVerdict.INDETERMINATE,
            "indeterminate term(s): " + ", ".join(indeterminate),
        )
    healthy = [term.name for term in terms if term.verdict is HealthVerdict.HEALTHY]
    if healthy:
        return HealthVerdict.HEALTHY, "healthy term(s): " + ", ".join(healthy)
    return HealthVerdict.UNHEALTHY, "; ".join(
        f"{term.name}: {term.detail}" for term in terms
    )


class SelfKillRefused(RuntimeError):
    """Refused to signal this process or one of its ancestors."""


def ancestors(
    pid: int, *, depth: int = _ANCESTOR_DEPTH, proc: Path | None = None
) -> list[int]:
    """Parent pids of ``pid``, nearest first."""
    root = proc if proc is not None else Path("/proc")
    found: list[int] = []
    current = pid
    for _ in range(depth):
        try:
            stat = (root / str(current) / "status").read_text()
        except OSError:
            break
        parent = None
        for line in stat.splitlines():
            if line.startswith("PPid:"):
                try:
                    parent = int(line.split()[1])
                except (IndexError, ValueError):
                    parent = None
                break
        if parent is None or parent <= 0 or parent in found:
            break
        found.append(parent)
        current = parent
    return found


def kill_by_pid(
    pid: int, *, sig: int = signal.SIGKILL, proc: Path | None = None
) -> bool:
    """Signal exactly one pid.

    Refuses this process and any of its ancestors. There is deliberately no
    pattern-matching variant of this function: matching by command line is how a
    guard kills itself, or kills whatever inherited a stale argv.
    """
    if pid <= 0:
        raise SelfKillRefused(f"refusing to signal pid {pid}")
    if pid == os.getpid():
        raise SelfKillRefused("refusing to signal self")
    if pid in ancestors(os.getpid(), proc=proc):
        raise SelfKillRefused(f"refusing to signal ancestor pid {pid}")
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except OSError:
        logger.warning("could not signal pid %d", pid, exc_info=True)
        return False
    return True


@dataclass(slots=True)
class ProcessSample:
    pid: int
    rss_bytes: int
    #: Name of the container this process belongs to, if it could be resolved.
    #: This is what determines the phase, so an unresolvable name is not "not a
    #: test" -- see :meth:`MemoryGuard.phase_for`.
    container_name: str | None = None
    ancestor_names: tuple[str, ...] = ()
    #: Advisory only. Deliberately NOT a predicate: see MemoryGuard's docstring.
    cwd: str = ""


@dataclass(slots=True)
class GuardAction:
    pid: int
    rss_bytes: int
    verdict: HealthVerdict
    reason: str
    killed: bool = False
    terms: list[HealthTerm] = field(default_factory=list)


class MemoryGuard:
    """Kill a runaway graded test, and only a runaway graded test.

    A process is a candidate only when **both** terms hold:

    * resident memory at or above ``kill_bytes`` (default 150 GiB; a healthy
      graded test uses single-digit GiB, so the headroom is roughly thirty-fold)
    * it has a container-supervisor ancestor -- it is inside a container

    THERE IS DELIBERATELY NO WORKING-DIRECTORY TERM. An earlier version required
    the process's cwd to be inside the testbed, on the reasoning that a graded
    test runs there. It does not always: a runaway that had grown to 667 GiB was
    skipped for 105 minutes because its cwd was ``/tmp``. Every additional
    conjunct is another way for the guard to miss what it exists to catch, so
    the predicate set is the smallest one that cannot match a benchmark client,
    an engine, a login shell or the guard itself -- all of which fail the
    container term. ``cwd`` is still sampled, as advisory detail only.
    """

    def __init__(
        self,
        *,
        kill_bytes: int = DEFAULT_KILL_BYTES,
        warn_bytes: int = DEFAULT_WARN_BYTES,
        killed_dir: Path | None = None,
        supervisors: tuple[str, ...] = CONTAINER_SUPERVISORS,
        eval_container_prefixes: tuple[str, ...] = ("sweb.eval",),
        agent_container_prefixes: tuple[str, ...] = ("minisweagent",),
    ) -> None:
        self.kill_bytes = kill_bytes
        self.warn_bytes = warn_bytes
        self.killed_dir = killed_dir
        self.supervisors = supervisors
        self.eval_container_prefixes = eval_container_prefixes
        self.agent_container_prefixes = agent_container_prefixes

    def phase_for(self, sample: ProcessSample) -> str:
        """Which phase a runaway belongs to, from its container name.

        Fails closed to ``"unknown"``. An unresolvable container name must not
        stop the kill -- the process is still a confirmed runaway inside a
        container -- but it must also not be booked as an eval kill, because
        only an eval kill turns an instance's error into a genuine failure.
        """
        name = sample.container_name or ""
        if any(name.startswith(prefix) for prefix in self.eval_container_prefixes):
            return "eval"
        if any(name.startswith(prefix) for prefix in self.agent_container_prefixes):
            return "agent"
        return "unknown"

    def evaluate(self, sample: ProcessSample) -> GuardAction:
        terms = [
            HealthTerm(
                name="rss",
                verdict=(
                    HealthVerdict.UNHEALTHY
                    if sample.rss_bytes >= self.kill_bytes
                    else HealthVerdict.HEALTHY
                ),
                evidence=1 if sample.rss_bytes >= 0 else 0,
                detail=f"{sample.rss_bytes / 1024**3:.1f} GiB",
            ),
            HealthTerm(
                name="in_container",
                verdict=(
                    HealthVerdict.UNHEALTHY
                    if any(name in self.supervisors for name in sample.ancestor_names)
                    else HealthVerdict.HEALTHY
                ),
                evidence=len(sample.ancestor_names),
                detail=f"ancestors={list(sample.ancestor_names)}",
            ),
        ]
        verdict, reason = combine_terms(terms)
        return GuardAction(
            pid=sample.pid,
            rss_bytes=sample.rss_bytes,
            verdict=verdict,
            reason=reason,
            terms=terms,
        )

    def act(
        self,
        sample: ProcessSample,
        *,
        instance_id: str | None = None,
        phase: str | None = None,
        apply: bool = False,
    ) -> GuardAction:
        """Evaluate and, when ``apply``, kill by pid and record a marker.

        The marker is written *before* the kill: a SIGKILLed test leaves an
        ambiguous log, so the record of having killed it is the only reliable
        evidence, and it has to exist even if the process dies first.
        """
        action = self.evaluate(sample)
        if action.verdict is not HealthVerdict.UNHEALTHY or not apply:
            return action
        resolved_phase = phase if phase is not None else self.phase_for(sample)
        if self.killed_dir is not None and instance_id:
            self.record_kill(instance_id, sample, phase=resolved_phase)
        action.killed = kill_by_pid(sample.pid)
        return action

    def record_kill(
        self, instance_id: str, sample: ProcessSample, *, phase: str = "eval"
    ) -> Path:
        """Write the ``<phase>.<instance>.<host>.<pid>.json`` marker.

        Phase is load-bearing: only ``eval`` markers make an instance's error a
        genuine failure. An ``agent`` kill merely makes one tool call return an
        error observation and the agent carries on, so it must never influence
        classification.
        """
        import socket

        assert self.killed_dir is not None
        self.killed_dir.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        path = self.killed_dir / f"{phase}.{instance_id}.{host}.{sample.pid}.json"
        path.write_text(
            json.dumps(
                {
                    "phase": phase,
                    "instance_id": instance_id,
                    "host": host,
                    "pid": sample.pid,
                    "rss_bytes": sample.rss_bytes,
                    "killed_at": time.time(),
                }
            )
        )
        return path
