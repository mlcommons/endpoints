# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry only where non-execution is provable, and always count the retries."""

from __future__ import annotations

import json

import pytest
from inference_endpoint.evaluation.swe_bench_distributed.infra_retry import (
    InfraRetryLedger,
    RetryOutcome,
    RunQuality,
    is_provable_non_execution,
    retry_on_provable_non_execution,
)

pytestmark = pytest.mark.unit


class NeverLaunched(RuntimeError):
    """Stands in for a step whose status file still read ``pending``."""

    provable_non_execution = True


class MayHaveRun(RuntimeError):
    """Stands in for a step that reached ``started`` before it failed."""

    provable_non_execution = False


def _never_sleep(_seconds: float) -> None:
    return None


def _run(operation, **kwargs):
    kwargs.setdefault("target", "run-a.s00")
    kwargs.setdefault("sleep", _never_sleep)
    return retry_on_provable_non_execution(operation, **kwargs)


class TestTheSafetyGate:
    def test_a_provable_non_execution_is_retried(self):
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise NeverLaunched("status=pending")
            return "ok"

        assert _run(operation, max_attempts=3) == "ok"
        assert len(calls) == 3

    def test_a_failure_that_may_have_run_is_never_retried(self):
        """The whole safety argument. Re-running could double-apply the work."""
        calls = []

        def operation():
            calls.append(1)
            raise MayHaveRun("status=started")

        with pytest.raises(MayHaveRun):
            _run(operation, max_attempts=5)

        assert len(calls) == 1

    def test_an_unfamiliar_exception_is_not_retried(self):
        """Absence of the claim is not evidence for it."""
        calls = []

        def operation():
            calls.append(1)
            raise ValueError("something else entirely")

        with pytest.raises(ValueError):
            _run(operation, max_attempts=5)

        assert len(calls) == 1

    def test_the_attempt_budget_is_bounded(self):
        calls = []

        def operation():
            calls.append(1)
            raise NeverLaunched("status=pending")

        with pytest.raises(NeverLaunched):
            _run(operation, max_attempts=4)

        assert len(calls) == 4

    def test_a_successful_first_attempt_costs_nothing(self):
        ledger = InfraRetryLedger()

        assert _run(lambda: "ok", ledger=ledger) == "ok"

        assert ledger.records == []
        assert ledger.run_quality() is RunQuality.CLEAN

    @pytest.mark.parametrize(
        ("error", "provable"),
        [(NeverLaunched(""), True), (MayHaveRun(""), False), (ValueError(""), False)],
    )
    def test_provability_is_read_from_the_failure(self, error, provable):
        assert is_provable_non_execution(error) is provable


class TestAccounting:
    def test_a_recovered_operation_is_counted_and_attributed(self):
        ledger = InfraRetryLedger()
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 2:
                raise NeverLaunched("status=pending")
            return "ok"

        _run(operation, ledger=ledger, max_attempts=3)
        summary = ledger.summary()

        assert summary["infra_retries_total"] == 2
        assert summary["instances_saved_by_retry"] == 1
        assert summary["infra_retries_exhausted"] == 0
        assert summary["infra_retry_succeeded_on_attempt"] == {"2": 1}
        assert summary["infra_retry_outcomes"] == {"retrying": 1, "recovered": 1}

    def test_an_exhausted_operation_is_counted_as_exhausted(self):
        ledger = InfraRetryLedger()

        with pytest.raises(NeverLaunched):
            _run(
                lambda: (_ for _ in ()).throw(NeverLaunched("status=pending")),
                ledger=ledger,
                max_attempts=2,
            )
        summary = ledger.summary()

        assert summary["infra_retries_exhausted"] == 1
        assert summary["instances_saved_by_retry"] == 0
        assert summary["run_quality"] == RunQuality.DEGRADED.value

    def test_a_not_retryable_failure_is_recorded(self):
        ledger = InfraRetryLedger()

        with pytest.raises(MayHaveRun):
            _run(
                lambda: (_ for _ in ()).throw(MayHaveRun("status=started")),
                ledger=ledger,
            )

        assert ledger.records[0].outcome is RetryOutcome.NOT_RETRYABLE

    def test_the_ledger_is_durable(self, tmp_path):
        """A run that dies must still leave its retry history behind."""
        path = tmp_path / "infra_retries.jsonl"
        ledger = InfraRetryLedger(path)
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 2:
                raise NeverLaunched("status=pending")
            return "ok"

        _run(operation, ledger=ledger, max_attempts=3)

        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["outcome"] for row in rows] == ["retrying", "recovered"]
        assert all(row["target"] == "run-a.s00" for row in rows)

    def test_accounting_never_takes_the_run_down(self, tmp_path):
        """A ledger that cannot be written must not fail the operation."""
        ledger = InfraRetryLedger(tmp_path / "no-such-dir" / "retries.jsonl")
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 2:
                raise NeverLaunched("status=pending")
            return "ok"

        assert _run(operation, ledger=ledger, max_attempts=3) == "ok"
        # In-memory counters are still correct.
        assert ledger.summary()["instances_saved_by_retry"] == 1


class TestRunQuality:
    def test_no_retries_is_clean(self):
        ledger = InfraRetryLedger()
        for _ in range(100):
            _run(lambda: "ok", ledger=ledger)

        assert ledger.summary()["run_quality"] == RunQuality.CLEAN.value

    def test_a_few_retries_is_ok_with_retries(self):
        ledger = InfraRetryLedger()
        for _ in range(200):
            _run(lambda: "ok", ledger=ledger)
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 2:
                raise NeverLaunched("status=pending")
            return "ok"

        _run(operation, ledger=ledger, max_attempts=3)

        assert ledger.summary()["run_quality"] == RunQuality.OK_WITH_RETRIES.value

    def test_many_retries_is_degraded_even_when_everything_succeeded(self):
        """A run that leaned on the retry loop is not a clean run."""
        ledger = InfraRetryLedger()
        for index in range(10):
            calls = []

            def operation(calls=calls):
                calls.append(1)
                if len(calls) < 2:
                    raise NeverLaunched("status=pending")
                return "ok"

            _run(operation, ledger=ledger, target=f"unit-{index}", max_attempts=3)

        assert ledger.summary()["run_quality"] == RunQuality.DEGRADED.value

    def test_an_exhaustion_is_degraded_regardless_of_volume(self):
        ledger = InfraRetryLedger()
        for _ in range(1000):
            _run(lambda: "ok", ledger=ledger)
        with pytest.raises(NeverLaunched):
            _run(
                lambda: (_ for _ in ()).throw(NeverLaunched("status=pending")),
                ledger=ledger,
                max_attempts=2,
            )

        assert ledger.run_quality() is RunQuality.DEGRADED


class TestReadingBackAWrittenLedger:
    """The SWE-bench service writes this shape from another process.

    It is an isolated subproject and cannot import this package, so the two
    halves share a file format. If that agreement breaks, per-step retries stop
    reaching the run-level `run_quality` and the run looks clean.
    """

    def test_a_written_ledger_round_trips(self, tmp_path):
        path = tmp_path / "infra_retries.jsonl"
        source = InfraRetryLedger(path)
        source.record(target="unit-1", attempt=1, outcome=RetryOutcome.RETRYING)
        source.record(target="unit-1", attempt=2, outcome=RetryOutcome.RECOVERED)

        loaded = InfraRetryLedger.from_jsonl(path)

        assert [r.outcome for r in loaded.records] == [
            RetryOutcome.RETRYING,
            RetryOutcome.RECOVERED,
        ]
        assert loaded.summary()["instances_saved_by_retry"] == 1

    def test_a_truncated_final_line_does_not_lose_the_history(self, tmp_path):
        path = tmp_path / "infra_retries.jsonl"
        path.write_text(
            '{"target": "u", "attempt": 1, "outcome": "retrying", "at": 1.0}\n'
            '{"target": "u", "attempt": 2, "outcome": "recov'
        )

        loaded = InfraRetryLedger.from_jsonl(path)

        assert len(loaded.records) == 1

    def test_a_missing_ledger_is_an_empty_clean_one(self, tmp_path):
        loaded = InfraRetryLedger.from_jsonl(tmp_path / "never-written.jsonl")

        assert loaded.records == []
        assert loaded.run_quality() is RunQuality.CLEAN

    def test_an_exhaustion_written_elsewhere_still_degrades_the_run(self, tmp_path):
        path = tmp_path / "infra_retries.jsonl"
        path.write_text(
            '{"target": "u", "attempt": 1, "outcome": "retrying", "at": 1.0}\n'
            '{"target": "u", "attempt": 2, "outcome": "exhausted", "at": 2.0}\n'
        )

        assert InfraRetryLedger.from_jsonl(path).run_quality() is RunQuality.DEGRADED
