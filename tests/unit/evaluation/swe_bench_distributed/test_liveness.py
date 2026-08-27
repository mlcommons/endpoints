# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner-liveness probes. Uncertainty must never escalate to DEAD."""

from __future__ import annotations

import os
import socket
import subprocess
from typing import Any

import pytest
from inference_endpoint.evaluation.swe_bench_distributed.queue import OwnerRecord
from inference_endpoint.evaluation.swe_bench_distributed.reaper import (
    Liveness,
    LocalProcessLiveness,
    SlurmStepLiveness,
)

pytestmark = pytest.mark.unit


def owner(**overrides) -> OwnerRecord:
    payload: dict[str, Any] = {
        "unit_id": "run-a.s00",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "boot_id": "boot-1",
        "plan_digest": "d" * 64,
        "claimed_at": 0.0,
    }
    payload.update(overrides)
    return OwnerRecord(**payload)


class TestLocalProcessLiveness:
    def test_a_live_pid_on_this_boot_is_alive(self):
        probe = LocalProcessLiveness(boot="boot-1")
        assert probe.probe(owner()).state is Liveness.ALIVE

    def test_a_missing_pid_is_dead(self):
        probe = LocalProcessLiveness(boot="boot-1")
        # 2**22 is above the default pid_max on Linux, so it cannot exist.
        assert probe.probe(owner(pid=2**22)).state is Liveness.DEAD

    def test_a_different_boot_is_dead(self):
        # After a reboot the same pid number can belong to something unrelated,
        # so a live-looking pid is not evidence that the owner survived.
        probe = LocalProcessLiveness(boot="boot-2")
        assert probe.probe(owner()).state is Liveness.DEAD

    def test_another_host_is_indeterminate_not_dead(self):
        probe = LocalProcessLiveness(boot="boot-1")
        assert probe.probe(owner(host="elsewhere")).state is Liveness.INDETERMINATE

    def test_a_missing_pid_record_is_indeterminate(self):
        probe = LocalProcessLiveness(boot="boot-1")
        assert probe.probe(owner(pid=0)).state is Liveness.INDETERMINATE


class FakeSlurm(SlurmStepLiveness):
    def __init__(self, responses):
        super().__init__()
        self.responses = responses
        self.calls: list[list[str]] = []

    def _run(self, argv):
        self.calls.append(argv)
        response = self.responses.get(argv[0])
        if isinstance(response, Exception):
            raise response
        return response


class TestSlurmStepLiveness:
    def slurm_owner(self, **overrides):
        return owner(slurm_job_id="1000", slurm_step_id="3", **overrides)

    def test_a_job_absent_from_squeue_is_dead(self):
        probe = FakeSlurm({"squeue": "2000\n"})
        verdict = probe.probe(self.slurm_owner())
        assert verdict.state is Liveness.DEAD
        assert verdict.scope == "job"

    def test_a_live_job_and_step_is_alive(self):
        probe = FakeSlurm(
            {
                "squeue": "1000\n",
                "scontrol": "StepId=1000.3 State=RUNNING StepId=1000.extern",
            }
        )
        assert probe.probe(self.slurm_owner()).state is Liveness.ALIVE

    def test_a_dead_step_inside_a_live_job_is_dead(self):
        # A step can die inside a live job -- a killed srun, an OOM-terminated
        # step -- and the job never leaves the queue, so a job-level-only rule
        # blocks those units for the entire allocation.
        probe = FakeSlurm(
            {"squeue": "1000\n", "scontrol": "StepId=1000.extern State=RUNNING"}
        )
        verdict = probe.probe(self.slurm_owner())
        assert verdict.state is Liveness.DEAD
        assert verdict.scope == "step"

    def test_step_liveness_uses_scontrol_not_squeue_s(self):
        probe = FakeSlurm(
            {"squeue": "1000\n", "scontrol": "StepId=1000.3 State=RUNNING"}
        )
        probe.probe(self.slurm_owner())
        # `squeue -s` reports only `.extern` on the clusters this targets, so it
        # would mark every live step dead and falsely reap every claim.
        assert ["scontrol", "show", "step", "1000"] in probe.calls
        assert not any("-s" in argv for argv in probe.calls if argv[0] == "squeue")

    def test_an_unreadable_squeue_is_indeterminate(self):
        probe = FakeSlurm({"squeue": None})
        assert probe.probe(self.slurm_owner()).state is Liveness.INDETERMINATE

    def test_an_unreadable_step_list_falls_back_to_the_job_answer(self):
        # Indeterminate step liveness must never be more aggressive than the
        # job-level answer, which is "alive".
        probe = FakeSlurm({"squeue": "1000\n", "scontrol": None})
        assert probe.probe(self.slurm_owner()).state is Liveness.ALIVE

    def test_an_empty_scontrol_listing_is_treated_as_unreadable(self):
        probe = FakeSlurm({"squeue": "1000\n", "scontrol": "no steps here"})
        assert probe.probe(self.slurm_owner()).state is Liveness.ALIVE

    def test_an_owner_without_a_job_id_is_indeterminate(self):
        probe = FakeSlurm({"squeue": "1000\n"})
        assert probe.probe(owner()).state is Liveness.INDETERMINATE

    def test_an_empty_queue_inside_a_job_is_implausible(self, monkeypatch):
        # An empty successful squeue is what a broken squeue looks like. It must
        # never be read as "no jobs are running" while we are inside a job.
        monkeypatch.setenv("SLURM_JOB_ID", "1000")
        probe = FakeSlurm({"squeue": ""})
        assert probe.live_job_ids() is None

    def test_an_empty_queue_outside_a_job_is_trusted(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        probe = FakeSlurm({"squeue": ""})
        assert probe.live_job_ids() == set()

    def test_array_job_ids_are_matched_by_base_id(self):
        probe = FakeSlurm(
            {"squeue": "1000_4\n", "scontrol": "StepId=1000.3 State=RUNNING"}
        )
        assert probe.probe(self.slurm_owner()).state is Liveness.ALIVE

    def test_a_failing_command_is_indeterminate_not_dead(self):
        probe = SlurmStepLiveness(timeout_s=1)

        def boom(argv, **kwargs):
            raise subprocess.SubprocessError("no slurm here")

        original = subprocess.run
        subprocess.run = boom
        try:
            assert probe.live_job_ids() is None
        finally:
            subprocess.run = original
