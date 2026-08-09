# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure-versus-genuine classification of error instances."""

from __future__ import annotations

import pytest

from inference_endpoint.evaluation.swe_bench_distributed.classify import (
    GENUINE_KINDS,
    INFRA_KINDS,
    ErrorKind,
    classify_eval_log,
    classify_unit,
)

pytestmark = pytest.mark.unit


def write_log(output_dir, instance_id: str, text: str) -> None:
    log_dir = output_dir / "logs" / "run_evaluation" / "run-1" / "model" / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run_instance.log").write_text(text)


class TestBiasRule:
    def test_unknown_is_genuine_never_infra(self):
        # A false bad-run costs one redo; a false retry biases the measurement
        # toward optimism. Unclassifiable therefore means "keep the result".
        assert ErrorKind.UNKNOWN in GENUINE_KINDS
        assert ErrorKind.UNKNOWN not in INFRA_KINDS

    def test_kinds_are_partitioned(self):
        assert not (INFRA_KINDS & GENUINE_KINDS)
        assert INFRA_KINDS | GENUINE_KINDS == set(ErrorKind)

    def test_model_outcomes_are_genuine(self):
        for kind in (
            ErrorKind.TEST_TIMEOUT,
            ErrorKind.TEST_MEMORY_EXCEEDED,
            ErrorKind.PATCH_APPLY_FAILED,
        ):
            assert kind in GENUINE_KINDS


class TestLogRules:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "fork/exec /usr/bin/conmon: resource temporarily unavailable",
                ErrorKind.CONTAINER_FORK_EAGAIN,
            ),
            ("Test timed out after 1800s", ErrorKind.TEST_TIMEOUT),
            (
                "can only create exec sessions on running containers",
                ErrorKind.CONTAINER_EXEC_REFUSED,
            ),
            ("container state improper", ErrorKind.CONTAINER_EXEC_REFUSED),
            ("Read timed out. (read timeout=60)", ErrorKind.RUNTIME_READ_TIMEOUT),
            (
                "Reversed (or previously applied) patch detected",
                ErrorKind.PATCH_APPLY_FAILED,
            ),
            ("1 out of 3 hunk FAILED", ErrorKind.PATCH_APPLY_FAILED),
            ("nothing recognisable here", ErrorKind.UNKNOWN),
        ],
    )
    def test_each_rule(self, text, expected):
        assert classify_eval_log(text) is expected

    def test_timeout_wins_over_teardown_noise(self):
        # A timed-out evaluation also emits "container state improper" while the
        # harness tears the container down. Reading that as a wedge would retry
        # a genuine model outcome, so rule order is load-bearing.
        text = "Test timed out after 1800s\ncontainer state improper\n"
        assert classify_eval_log(text) is ErrorKind.TEST_TIMEOUT

    def test_fork_failure_wins_over_teardown_noise(self):
        text = (
            "fork/exec /usr/bin/conmon: resource temporarily unavailable\n"
            "can only create exec sessions on running containers\n"
        )
        assert classify_eval_log(text) is ErrorKind.CONTAINER_FORK_EAGAIN

    def test_a_wedge_wins_over_patch_apply(self):
        # A wedged container's verdict is unreliable either way, so the wedge
        # decides and the unit is retried.
        text = "container state improper\nhunk FAILED\n"
        assert classify_eval_log(text) is ErrorKind.CONTAINER_EXEC_REFUSED

    def test_build_error_is_checked_before_every_other_rule(self):
        # BuildImageError's message embeds the same needles the other rules look
        # for, so any other ordering misattributes a build failure.
        assert (
            classify_eval_log("BuildImageError: Read timed out. (read timeout=60)")
            is ErrorKind.IMAGE_BUILD_TIMEOUT
        )
        assert (
            classify_eval_log("BuildImageError: 500 Server Error")
            is ErrorKind.IMAGE_BUILD_ERROR
        )


class TestClassifyUnit:
    def test_infra_and_genuine_are_counted_separately(self, tmp_path):
        write_log(tmp_path, "a-1", "container state improper")
        write_log(tmp_path, "a-2", "Test timed out after 1800s")

        classification = classify_unit(tmp_path, ["a-1", "a-2"])

        assert classification.infra_count == 1
        assert classification.genuine_count == 1
        assert classification.should_retry

    def test_only_genuine_errors_do_not_trigger_a_retry(self, tmp_path):
        write_log(tmp_path, "a-1", "Test timed out after 1800s")
        assert not classify_unit(tmp_path, ["a-1"]).should_retry

    def test_a_missing_log_is_unknown_and_therefore_genuine(self, tmp_path):
        classification = classify_unit(tmp_path, ["absent"])
        assert classification.kinds == {ErrorKind.UNKNOWN: 1}
        assert not classification.should_retry

    def test_none_error_ids_means_not_measured(self, tmp_path):
        # "We did not measure" and "we measured zero" are different; conflating
        # them once let a damaged unit into a clean set.
        classification = classify_unit(tmp_path, None)
        assert classification.measured is False

    def test_empty_error_ids_means_measured_zero(self, tmp_path):
        classification = classify_unit(tmp_path, [])
        assert classification.measured is True
        assert classification.infra_count == 0

    def test_eval_memory_kill_marker_is_genuine(self, tmp_path):
        killed = tmp_path / "killed"
        killed.mkdir()
        (killed / "eval.a-1.host.999.json").write_text("{}")
        write_log(tmp_path, "a-1", "container state improper")

        classification = classify_unit(tmp_path, ["a-1"], killed_dir=killed)

        # The marker beats the log: a SIGKILLed test leaves an ambiguous log,
        # but the kill is a fact recorded before acting.
        assert classification.kinds == {ErrorKind.TEST_MEMORY_EXCEEDED: 1}
        assert not classification.should_retry

    def test_agent_phase_kill_marker_does_not_classify(self, tmp_path):
        killed = tmp_path / "killed"
        killed.mkdir()
        (killed / "agent.a-1.host.999.json").write_text("{}")
        write_log(tmp_path, "a-1", "Test timed out after 1800s")

        # An agent kill only makes one tool call return an error observation;
        # the instance still reaches a real outcome, so the marker is audit-only.
        classification = classify_unit(tmp_path, ["a-1"], killed_dir=killed)
        assert classification.kinds == {ErrorKind.TEST_TIMEOUT: 1}

    def test_step_infrastructure_failure_forces_a_retry(self, tmp_path):
        classification = classify_unit(tmp_path, [], infrastructure_failure=True)
        assert classification.should_retry
        assert classification.kinds == {ErrorKind.STEP_INFRASTRUCTURE_FAILURE: 1}

    def test_a_changed_endpoint_forces_a_retry(self, tmp_path):
        # An engine restarted under a live client produces a plausible run that
        # scores near zero and exits successfully.
        classification = classify_unit(tmp_path, [], endpoint_changed=True)
        assert classification.should_retry
        assert classification.kinds == {ErrorKind.ENDPOINT_CHANGED: 1}
