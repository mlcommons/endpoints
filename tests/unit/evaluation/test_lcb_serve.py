# SPDX-FileCopyrightText: Copyright (c) 2026 CoreWeave, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for lcb_serve.py's grading-child error attribution.

run_code_subprocess forks a grading child per sample and must classify a
child that reports no result into exactly one of:
  -1 timeout            (submission ran past the deadline)
  -6 GradingChildDied   (judge-side: died before or while doing its own
                         setup -- reliability_guard, suite parse -- i.e.
                         before the submission's code ever ran)
  -8 SubmissionKilledChild (submission-side: died while its own code, or
                         run_test's dispatch into it, was executing)

These tests call run_code_subprocess directly (not through _LCBWorker's
pool) so that monkeypatches applied in the test process are inherited by
the forked grading child; the batch-level guard tests below go through the
real pool instead, and rely on the interpreter's default start method
being "fork" (true on 3.11/3.12) for the same reason.
"""

import os

import pytest
from inference_endpoint.evaluation.livecodebench import lcb_serve, run_lcb_tests

pytestmark = pytest.mark.unit

CALL_BASED_SUITE = '{"fn_name": "solve", "inputs": ["1"], "outputs": ["1"]}'
PASSING_CALL_BASED = "def solve(x):\n    return x\n"
CALL_SYSEXIT = "def solve(x):\n    import sys\n    sys.exit(3)\n"
CALL_OS_EXIT = "import os\nos._exit(3)\ndef solve(x):\n    return x\n"

# Keep timeouts short; the grading itself is instant, this only bounds how
# long a genuinely hung test would take to fail.
TIMEOUT_SEC = 5


def test_passing_submission_scores_true():
    res, metadata = lcb_serve.run_code_subprocess(
        CALL_BASED_SUITE, PASSING_CALL_BASED, timeout_sec=TIMEOUT_SEC
    )
    assert res == [True]
    assert metadata.get("error_code") is None


def test_sys_exit_is_submission_exit_not_infra():
    res, metadata = lcb_serve.run_code_subprocess(
        CALL_BASED_SUITE, CALL_SYSEXIT, timeout_sec=TIMEOUT_SEC
    )
    assert metadata["error_code"] == -7
    assert metadata["error_code"] not in lcb_serve._LCB_INFRA_ERROR_CODES


def test_os_exit_in_submission_is_submission_killed_child_not_infra():
    res, metadata = lcb_serve.run_code_subprocess(
        CALL_BASED_SUITE, CALL_OS_EXIT, timeout_sec=TIMEOUT_SEC
    )
    assert metadata["error_code"] == -8
    assert metadata["error_code"] not in lcb_serve._LCB_INFRA_ERROR_CODES


def test_death_before_run_test_is_infra_error():
    """A judge process that dies before run_test is even reached (e.g. a
    bad start method, or a crash while importing) is -6, not -8."""
    orig = lcb_serve.execute_code_single_suppressed_errors

    def _die_immediately(*args, **kwargs):
        os._exit(1)

    lcb_serve.execute_code_single_suppressed_errors = _die_immediately
    try:
        res, metadata = lcb_serve.run_code_subprocess(
            CALL_BASED_SUITE, PASSING_CALL_BASED, timeout_sec=TIMEOUT_SEC
        )
    finally:
        lcb_serve.execute_code_single_suppressed_errors = orig

    assert metadata["error_code"] == -6
    assert metadata["error_code"] in lcb_serve._LCB_INFRA_ERROR_CODES


def test_death_during_judge_setup_is_infra_error_not_submission():
    """A death inside run_test's own setup (reliability_guard, suite parse)
    -- after run_test has started but before started_flag is set -- must
    still be -6. This is the exact boundary the started_flag fix moved: it
    used to flip True on wrapper entry, before this setup ran, which would
    have misattributed this case as -8."""
    orig_guard = run_lcb_tests.reliability_guard

    def _guard_then_die(*args, **kwargs):
        orig_guard(*args, **kwargs)
        os._exit(1)

    run_lcb_tests.reliability_guard = _guard_then_die
    try:
        res, metadata = lcb_serve.run_code_subprocess(
            CALL_BASED_SUITE, PASSING_CALL_BASED, timeout_sec=TIMEOUT_SEC
        )
    finally:
        run_lcb_tests.reliability_guard = orig_guard

    assert metadata["error_code"] == -6
    assert metadata["error_code"] in lcb_serve._LCB_INFRA_ERROR_CODES


def test_compile_error_in_submission_is_not_infra_but_is_logged():
    """A submission that doesn't even compile hits the outer except in
    grade_call_based's caller, gets -4, and is the submission's fault, not
    the judge's -- but must still carry an "error" key so it's visible."""
    bad_syntax = "def solve(x\n    return x\n"  # missing closing paren
    res, metadata = lcb_serve.run_code_subprocess(
        CALL_BASED_SUITE, bad_syntax, timeout_sec=TIMEOUT_SEC
    )
    assert metadata["error_code"] == -4
    assert "error" in metadata
    assert metadata["error_code"] not in lcb_serve._LCB_INFRA_ERROR_CODES


def test_timeout_is_not_infra_error():
    # grade_call_based enforces its own per-test-case timeout via
    # signal.alarm and returns a normal -3 result well within it, so
    # exercising the *outer* process-level timeout (run_code_subprocess's
    # p.join(timeout=...)) needs a submission that defeats the alarm too.
    ignore_alarm_and_hang = (
        "import signal\n"
        "signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
        "def solve(x):\n"
        "    while True: pass\n"
    )
    res, metadata = lcb_serve.run_code_subprocess(
        CALL_BASED_SUITE, ignore_alarm_and_hang, timeout_sec=1
    )
    assert metadata["error_code"] == -1
    assert metadata["error_code"] not in lcb_serve._LCB_INFRA_ERROR_CODES


class _DictTestLoader(dict):
    """Minimal stand-in for LCBTestLoader: _LCBWorker only does
    self.test_loader[qid], which dict already supports."""


def test_all_os_exit_batch_scores_zero_without_raising():
    """A batch where every submission kills its own interpreter is a
    legitimate 0, not an infra failure -- the guard must not fire."""
    worker = lcb_serve._LCBWorker(
        _DictTestLoader(q1=CALL_BASED_SUITE, q2=CALL_BASED_SUITE),
        n_lcb_workers=2,
        worker_timeout_sec=TIMEOUT_SEC,
    )
    results = worker(["q1", "q2"], [[CALL_OS_EXIT], [CALL_OS_EXIT]])
    assert results == {"q1": [False], "q2": [False]}


def test_submission_error_logs_at_warning_not_error(caplog):
    """Routine submission-attributed outcomes (-8 here) must not log at
    ERROR -- that level is reserved for judge-side infra failures, so ops
    can alert on it without being flooded by ordinary bad submissions."""
    worker = lcb_serve._LCBWorker(
        _DictTestLoader(q1=CALL_BASED_SUITE),
        n_lcb_workers=1,
        worker_timeout_sec=TIMEOUT_SEC,
    )
    with caplog.at_level("WARNING", logger=lcb_serve.logger.name):
        results = worker(["q1"], [[CALL_OS_EXIT]])

    assert results == {"q1": [False]}
    levels = [r.levelname for r in caplog.records]
    assert "WARNING" in levels
    assert "ERROR" not in levels


def test_infra_error_logs_at_error_level(monkeypatch, caplog):
    """A judge-side death (-6) must log at ERROR so it's distinguishable
    from the routine submission-attributed WARNINGs above.

    Relies on the pool using the default "fork" start method (true on
    3.11/3.12) so the monkeypatch below reaches the pool workers.
    """

    def _die_immediately(*args, **kwargs):
        os._exit(1)

    monkeypatch.setattr(
        lcb_serve, "execute_code_single_suppressed_errors", _die_immediately
    )

    worker = lcb_serve._LCBWorker(
        _DictTestLoader(q1=CALL_BASED_SUITE),
        n_lcb_workers=1,
        worker_timeout_sec=TIMEOUT_SEC,
    )
    with caplog.at_level("WARNING", logger=lcb_serve.logger.name):
        # Single-sample batch is also all-infra, so the guard raises; the
        # log call happens before that, inside the executor's with-block.
        with pytest.raises(RuntimeError, match="infrastructure errors"):
            worker(["q1"], [[PASSING_CALL_BASED]])

    levels = [r.levelname for r in caplog.records]
    assert "ERROR" in levels
    assert "WARNING" not in levels


def test_all_judge_startup_death_batch_raises(monkeypatch):
    """A batch where every grading child dies before run_test is reached
    means the judge itself is broken, not that every sample failed -- the
    all-infra guard must raise instead of silently reporting a 0.

    Relies on the pool using the default "fork" start method (true on
    3.11/3.12) so the monkeypatch below (applied in this process) reaches
    the pool workers; a spawn/forkserver worker would re-import the module
    fresh and not see it. The guard logic under test lives in
    _LCBWorker.__call__ and is identical regardless of pool context.
    """

    def _die_immediately(*args, **kwargs):
        os._exit(1)

    monkeypatch.setattr(
        lcb_serve, "execute_code_single_suppressed_errors", _die_immediately
    )

    worker = lcb_serve._LCBWorker(
        _DictTestLoader(q1=CALL_BASED_SUITE, q2=CALL_BASED_SUITE),
        n_lcb_workers=2,
        worker_timeout_sec=TIMEOUT_SEC,
    )
    with pytest.raises(RuntimeError, match="infrastructure errors"):
        worker(["q1", "q2"], [[PASSING_CALL_BASED], [PASSING_CALL_BASED]])
