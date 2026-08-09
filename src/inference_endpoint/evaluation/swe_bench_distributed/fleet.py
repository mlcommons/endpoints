# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fan a SWE-bench accuracy run out across a fleet of SWE-bench services."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import msgspec
import yaml

from ...exceptions import SetupError
from .classify import classify_unit
from .gates import (
    CheckpointIdentityGate,
    EndpointFingerprintGate,
    Gate,
    GateFailure,
    ToolCallGate,
    run_gates,
)
from .merge import MergeRefusal, merge_run
from .queue import UnitOutcome, UnitResult, WorkQueue
from .reaper import LocalProcessLiveness, reap
from .units import Unit, plan_units

logger = logging.getLogger(__name__)

QUEUE_DIRNAME = "swe_bench_wq"
UNITS_DIRNAME = "units"

#: Buckets a SWE-bench run report uses for instances that reached an outcome.
#: ``incomplete_ids`` is deliberately absent: an incomplete instance is exactly
#: what "not accounted for" means, and it must fail the merge gate.
ACCOUNTED_ID_KEYS = (
    "resolved_ids",
    "unresolved_ids",
    "empty_patch_ids",
    "error_ids",
)


class ServiceQuarantined(RuntimeError):
    """A service was withdrawn from the fleet."""


@dataclass(slots=True)
class ServiceState:
    """Per-service bookkeeping for the dispatcher."""

    url: str
    completed_units: int = 0
    consecutive_env_faults: int = 0
    last_progress_at: float = field(default_factory=time.monotonic)
    quarantined_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.quarantined_reason is None


@dataclass(slots=True)
class DispatchOutcome:
    """What one attempt at one unit produced."""

    result: UnitResult
    terminal: bool


class FleetDispatcher:
    """Claim units, run them on services, classify, retry, and merge.

    Concurrency is one in-flight service run per service. The service itself
    parallelises within a run (``workers`` / ``max_eval_workers``), so a second
    concurrent run per service would only contend for the same host.
    """

    def __init__(
        self,
        *,
        queue: WorkQueue,
        service_urls: list[str],
        submit: Any,
        poll: Any,
        collect: Any,
        fingerprint: Any = None,
        max_attempts: int = 3,
        stall_timeout_s: float = 3 * 60 * 60,
        max_consecutive_env_faults: int = 3,
        idle_poll_s: float = 1.0,
        unit_root: Path | None = None,
        killed_dir: Path | None = None,
    ) -> None:
        if not service_urls:
            raise SetupError("the SWE-bench fleet needs at least one service URL")
        self.queue = queue
        self.services = {url: ServiceState(url=url) for url in service_urls}
        self.submit = submit
        self.poll = poll
        self.collect = collect
        self.fingerprint = fingerprint
        self.max_attempts = max_attempts
        self.stall_timeout_s = stall_timeout_s
        self.max_consecutive_env_faults = max_consecutive_env_faults
        self.idle_poll_s = idle_poll_s
        self.unit_root = unit_root
        self.killed_dir = killed_dir
        self._lock = threading.Lock()
        self._in_flight = 0

    # ------------------------------------------------------------ dispatch --

    def run(self) -> None:
        """Drive every planned unit to a terminal result."""
        with ThreadPoolExecutor(max_workers=len(self.services)) as pool:
            futures = [
                pool.submit(self._service_loop, url) for url in list(self.services)
            ]
            for future in futures:
                future.result()

    def _service_loop(self, url: str) -> None:
        state = self.services[url]
        while state.available:
            fingerprint = self._fingerprint(url)
            unit = self._take_unit(fingerprint)
            if unit is None:
                # An empty queue does not mean the run is finished. A unit in
                # flight on another service can be released back at any moment
                # -- a failed attempt, a quarantined peer -- and a worker that
                # exits on the first empty poll leaves that unit for nobody.
                # Only "nothing available AND nothing in flight" ends the run.
                if self._idle_is_terminal():
                    return
                time.sleep(self.idle_poll_s)
                continue
            try:
                outcome = self._attempt(unit, state, fingerprint)
                self._settle(unit, state, outcome)
            finally:
                with self._lock:
                    self._in_flight -= 1
            self._check_stall(state)

    def _take_unit(self, fingerprint: str | None) -> Unit | None:
        """Claim the next available unit and mark it in flight, atomically.

        Claiming and counting must happen under one lock. With the increment
        after the claim there is a window in which a peer sees "nothing
        available" (this unit is claimed) and "nothing in flight" (not yet
        counted), concludes the run is over, and exits -- leaving the unit with
        nobody to retry it if this attempt fails.
        """
        with self._lock:
            for unit_id in self.queue.available_unit_ids():
                if self.queue.claim(unit_id, endpoint_fingerprint=fingerprint) is None:
                    continue  # another process won the filesystem claim
                self._in_flight += 1
                return self.queue.plan.unit(unit_id)
        return None

    def _idle_is_terminal(self) -> bool:
        with self._lock:
            return self._in_flight == 0 and not self.queue.available_unit_ids()

    def _fingerprint(self, url: str) -> str | None:
        if self.fingerprint is None:
            return None
        try:
            return self.fingerprint()
        except Exception:  # noqa: BLE001 - a fingerprint is advisory at claim time
            logger.debug("could not fingerprint endpoints for %s", url, exc_info=True)
            return None

    def _attempt(
        self, unit: Unit, state: ServiceState, claim_fingerprint: str | None
    ) -> DispatchOutcome:
        started = time.monotonic()
        base = UnitResult(
            unit_id=unit.unit_id,
            run_id=unit.run_id,
            plan_digest=self.queue.plan.digest,
            outcome=UnitOutcome.FAILED,
            service_url=state.url,
            endpoint_fingerprint=claim_fingerprint,
        )
        try:
            service_run_id = self.submit(state.url, unit)
        except Exception as exc:  # noqa: BLE001 - any submit failure is the host's
            base.outcome = UnitOutcome.ENV_FAULT
            base.detail = f"submit failed: {type(exc).__name__}: {exc}"
            base.duration_s = time.monotonic() - started
            return DispatchOutcome(result=base, terminal=False)

        base.service_run_id = service_run_id
        try:
            status = self.poll(state.url, service_run_id)
        except Exception as exc:  # noqa: BLE001
            base.outcome = UnitOutcome.ENV_FAULT
            base.detail = f"poll failed: {type(exc).__name__}: {exc}"
            base.duration_s = time.monotonic() - started
            return DispatchOutcome(result=base, terminal=False)

        base.duration_s = time.monotonic() - started
        if status.get("status") != "succeeded":
            base.outcome = UnitOutcome.FAILED
            base.detail = (
                f"service run ended {status.get('status')}: {status.get('error')}"
            )
            return DispatchOutcome(result=base, terminal=False)

        report, output_dir = self.collect(state.url, service_run_id, unit, status)
        # An engine restarted mid-unit yields a plausible run that scores near
        # zero and exits successfully. Comparing the fingerprint is the only
        # thing that distinguishes it from a genuinely bad model.
        publish_fingerprint = self._fingerprint(state.url)
        endpoint_changed = (
            claim_fingerprint is not None
            and publish_fingerprint is not None
            and claim_fingerprint != publish_fingerprint
        )

        accounted, resolved = accounted_and_resolved(report)
        classification = classify_unit(
            output_dir,
            report.get("error_ids"),
            killed_dir=self.killed_dir,
            infrastructure_failure=bool(report.get("infrastructure_failure")),
            endpoint_changed=endpoint_changed,
        )
        base.accounted_instance_ids = accounted
        base.resolved_instance_ids = resolved
        base.infra_error_count = classification.infra_count
        base.genuine_error_count = classification.genuine_count
        base.error_kinds = classification.as_counts()

        if classification.should_retry:
            # The agent phase succeeded and the service said so, but instances
            # were lost to infrastructure. Publishing this as a success is how a
            # run silently becomes unable to ever reach a full result.
            base.outcome = UnitOutcome.INFRA
            base.detail = (
                f"{classification.infra_count} instance(s) lost to infrastructure: "
                + ", ".join(f"{k}={v}" for k, v in base.error_kinds.items())
            )
            return DispatchOutcome(result=base, terminal=False)

        missing = set(unit.instance_ids) - set(accounted)
        if missing:
            base.outcome = UnitOutcome.FAILED
            base.detail = f"{len(missing)} instance(s) unaccounted for"
            return DispatchOutcome(result=base, terminal=False)

        base.outcome = UnitOutcome.SUCCEEDED
        return DispatchOutcome(result=base, terminal=True)

    def _settle(
        self, unit: Unit, state: ServiceState, outcome: DispatchOutcome
    ) -> None:
        result = outcome.result
        if outcome.terminal:
            self.queue.publish(result)
            state.completed_units += 1
            state.consecutive_env_faults = 0
            state.last_progress_at = time.monotonic()
            return

        if self.unit_root is not None:
            attempt = self.queue.attempts(unit.unit_id) + 1
            self.queue.snapshot_evidence(
                unit.unit_id, self.unit_root / unit.unit_id, attempt
            )

        attempts = self.queue.record_attempt(result)

        if result.outcome is UnitOutcome.ENV_FAULT:
            # The unit is fine; the host is not. Do not charge the unit, and
            # withdraw the service if it keeps doing this.
            state.consecutive_env_faults += 1
            if state.consecutive_env_faults >= self.max_consecutive_env_faults:
                state.quarantined_reason = (
                    f"{state.consecutive_env_faults} consecutive environment faults"
                )
                logger.error(
                    "quarantining SWE-bench service %s: %s",
                    state.url,
                    state.quarantined_reason,
                )
            self.queue.release(unit.unit_id)
            return

        state.consecutive_env_faults = 0
        if attempts >= self.max_attempts:
            # Stop burning slots. An abandoned unit is a loud, terminal record
            # that the merge gate refuses, not a unit that spins forever.
            self.queue.abandon(result)
            state.last_progress_at = time.monotonic()
            return
        self.queue.release(unit.unit_id)

    def _check_stall(self, state: ServiceState) -> None:
        """Withdraw a service that is healthy but not producing.

        Health is not progress. A service that answers ``/health`` while
        completing nothing is the silent failure mode: verify the effect, never
        the status.
        """
        if not state.available:
            return
        idle = time.monotonic() - state.last_progress_at
        if idle > self.stall_timeout_s:
            state.quarantined_reason = (
                f"no unit completed in {idle:.0f}s despite a healthy service"
            )
            logger.error(
                "quarantining SWE-bench service %s: %s",
                state.url,
                state.quarantined_reason,
            )

    @property
    def quarantined(self) -> dict[str, str]:
        return {
            url: state.quarantined_reason
            for url, state in self.services.items()
            if state.quarantined_reason is not None
        }


def accounted_and_resolved(
    report: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract the accounted and resolved instance ids from a SWE-bench report.

    Ids, not counts. A shard with one duplicate and one missing id has the right
    count and the wrong content, and only an id comparison catches it.
    """
    accounted: list[str] = []
    seen: set[str] = set()
    for key in ACCOUNTED_ID_KEYS:
        for instance_id in report.get(key) or ():
            text = str(instance_id)
            if text in seen:
                # Preserve the duplicate so the merge gate can refuse it rather
                # than silently deduplicating a real accounting bug.
                accounted.append(text)
                continue
            seen.add(text)
            accounted.append(text)
    resolved = tuple(str(x) for x in report.get("resolved_ids") or ())
    return tuple(accounted), resolved


def build_gates(
    *,
    expected_model: str | None,
    tool_call_model: str | None,
    min_prompt_tokens: int,
    api_key: str | None = None,
) -> tuple[list[Gate], EndpointFingerprintGate]:
    """Assemble the pre-dispatch gates.

    ``expected_model`` is optional only because not every deployment pins a
    checkpoint path; when it is set the identity gate is mandatory.
    """
    fingerprint_gate = EndpointFingerprintGate(api_key=api_key)
    gates: list[Gate] = []
    if expected_model:
        gates.append(CheckpointIdentityGate(expected_model, api_key=api_key))
    if tool_call_model:
        gates.append(
            ToolCallGate(
                tool_call_model,
                min_prompt_tokens=min_prompt_tokens,
                api_key=api_key,
            )
        )
    gates.append(fingerprint_gate)
    return gates, fingerprint_gate


def load_benchmark_config(report_dir: Path) -> dict[str, Any]:
    config_path = report_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. The fleet scorer must run "
            "inside a benchmark that has already written its config."
        )
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"benchmark config at {config_path} must be a YAML mapping")
    return config


def write_merge_artifacts(
    report_dir: Path, payload: dict[str, Any], name: str = "swe_bench_merge.json"
) -> Path:
    path = report_dir / name
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(msgspec.json.encode(payload))
    tmp.replace(path)
    return path


__all__ = [
    "ACCOUNTED_ID_KEYS",
    "QUEUE_DIRNAME",
    "UNITS_DIRNAME",
    "DispatchOutcome",
    "FleetDispatcher",
    "GateFailure",
    "LocalProcessLiveness",
    "MergeRefusal",
    "ServiceQuarantined",
    "ServiceState",
    "accounted_and_resolved",
    "build_gates",
    "load_benchmark_config",
    "merge_run",
    "plan_units",
    "reap",
    "run_gates",
    "urljoin",
    "write_merge_artifacts",
]
