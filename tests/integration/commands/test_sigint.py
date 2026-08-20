# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Whole-process Ctrl-C integration tests.

The interruption paths no unit test can compose: a real
``inference-endpoint`` subprocess in its own process group receives SIGINT
(exactly what a terminal ^C delivers to the foreground group — parent and
service children alike). The contract, at every delivery point:

- exit code 130 (user abort, distinct from failure exit codes 1-4);
- artifacts are honest: a run whose measurement was cut short lands
  ``result_summary.json`` ``state=interrupted``, ``complete: false`` — or no
  artifacts at all, never a COMPLETE-looking outcome for an aborted
  measurement. (A ^C during post-measurement finalization of an
  already-completed run keeps the completed perf artifacts; see
  TestFinalizeBenchmark.) ``final_snapshot.json`` usually reads
  ``state=interrupted`` too (the session's INTERRUPTED marker drives the
  aggregator's ENDED finalize), except in the post-ENDED drain window,
  where the aggregator legitimately records the normally-ended run it
  observed — ``result_summary.json`` + the exit code are the run-level
  truth;
- ``events.jsonl`` survives — the event logger ignores the group SIGINT and
  flushes on the session's terminal ENDED;
- no service child outlives the run;
- teardown is prompt, not a hang on an unbounded drain.
"""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import pytest

_TESTS_DIR = Path(__file__).resolve().parents[2]
_CHAR_TOKENIZER_DIR = _TESTS_DIR / "assets/tokenizers/char"
_DS_DATASET = _TESTS_DIR / "assets/datasets/ds_samples.jsonl"


def _write_config(report_dir: Path, endpoint_url: str, config_path: Path) -> None:
    """~120 s workload (600 samples @ 5 QPS): only the ^C can end the run."""
    config_path.write_text(
        f"""
type: online
endpoint_config:
  endpoints: ["{endpoint_url}"]
model_params:
  name: "{_CHAR_TOKENIZER_DIR}"
  streaming: "off"
datasets:
  - path: "{_DS_DATASET}"
    type: performance
report_dir: {report_dir}
settings:
  load_pattern:
    type: poisson
    target_qps: 5
  client:
    num_workers: 1
    warmup_connections: 0
    max_connections: 10
  runtime:
    n_samples_to_issue: 600
  warmup:
    enabled: false
"""
    )


def _cli() -> str:
    cli = shutil.which("inference-endpoint")
    assert cli is not None, "console script must be installed in the test venv"
    return cli


@contextlib.contextmanager
def _benchmark_proc(
    argv: list[str], log_file: Path | None = None
) -> Iterator[subprocess.Popen]:
    """A benchmark subprocess in its own group, SIGKILLed on exit if alive.

    ``log_file`` captures stdout+stderr (logging goes to stdout) for tests
    that key on run-lifecycle log lines.
    """
    with contextlib.ExitStack() as stack:
        out: IO[bytes] | int = (
            stack.enter_context(log_file.open("wb"))
            if log_file is not None
            else subprocess.DEVNULL
        )
        proc = subprocess.Popen(
            argv,
            stdout=out,
            stderr=subprocess.STDOUT if log_file is not None else subprocess.DEVNULL,
            start_new_session=True,  # own process group, like a foreground job
        )
        try:
            yield proc
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def _wait_services_ready(
    proc: subprocess.Popen, report_dir: Path, timeout_s: float = 60.0
) -> None:
    """Block until the aggregator touches metrics/.ready (handlers installed);
    the session starts issuing right after service readiness."""
    ready = report_dir / "metrics" / ".ready"
    deadline = time.monotonic() + timeout_s
    while not ready.exists():
        assert proc.poll() is None, "benchmark died before services came up"
        assert time.monotonic() < deadline, "services never became ready"
        time.sleep(0.1)


def _iter_proc_cmdlines() -> Iterator[tuple[int, bytes]]:
    """(pid, cmdline) for every live process (Linux); racing exits skipped."""
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            yield (
                int(pid_dir.name),
                (pid_dir / "cmdline").read_bytes().replace(b"\0", b" "),
            )
        except OSError:
            continue  # process exited mid-scan


def _procs_referencing(needle: str) -> list[str]:
    """Cmdlines of live processes whose argv mentions ``needle``."""
    return [
        cmdline.decode(errors="replace")
        for _, cmdline in _iter_proc_cmdlines()
        if needle.encode() in cmdline
    ]


def _assert_no_leftover_children(report_dir: Path, what: str) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        leftovers = _procs_referencing(str(report_dir))
        if not leftovers:
            return
        time.sleep(0.2)
    raise AssertionError(f"service children outlived the {what}: {leftovers}")


def _assert_interrupted_artifacts(report_dir: Path) -> None:
    snapshot = json.loads((report_dir / "metrics" / "final_snapshot.json").read_text())
    assert snapshot["state"] == "interrupted"
    summary = json.loads(
        (report_dir / "performance" / "result_summary.json").read_text()
    )
    assert summary["complete"] is False


def _pid_of_child(needle: str, extra: str) -> int | None:
    """PID of a live process whose argv mentions both needles."""
    for pid, cmdline in _iter_proc_cmdlines():
        if needle.encode() in cmdline and extra.encode() in cmdline:
            return pid
    return None


@pytest.mark.integration
def test_sigint_mid_run_exits_130_with_interrupted_artifacts(
    mock_http_echo_server, tmp_path
):
    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    _write_config(report_dir, mock_http_echo_server.url, config_path)

    with _benchmark_proc(
        [_cli(), "benchmark", "from-config", "-c", str(config_path)]
    ) as proc:
        _wait_services_ready(proc, report_dir)
        time.sleep(3.0)  # comfortably inside the ~120 s performance phase
        os.killpg(proc.pid, signal.SIGINT)
        rc = proc.wait(timeout=60.0)

    assert rc == 130, f"user abort must exit 130, got {rc}"
    _assert_interrupted_artifacts(report_dir)
    # The event logger must survive the group SIGINT and flush on ENDED.
    assert (
        report_dir / "events.jsonl"
    ).exists(), "events.jsonl missing — event logger died on ^C instead of flushing"
    _assert_no_leftover_children(report_dir, "run")


@pytest.mark.integration
def test_sigint_grace_expiry_abandons_wedged_drain(mock_http_echo_server, tmp_path):
    """A single ^C against a wedged metrics drain exits within the grace.

    The aggregator child is SIGSTOPped to simulate a wedged drain — the exact
    hang the teardown grace exists for. One SIGINT to the MAIN process only
    (``os.kill``, not the group): the graceful stop parks on the wedged
    drain; grace expiry must SIGTERM→SIGKILL the children so the drain's
    wait-for-exit unblocks and the run exits 130 without a second keystroke.
    The grace is shrunk to 3s via the class constant (fixed 30s in
    production) so the test stays fast.
    """
    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    _write_config(report_dir, mock_http_echo_server.url, config_path)

    wrapper = (
        "from inference_endpoint.commands.benchmark.watchdog import SigintGovernor; "
        "SigintGovernor.TEARDOWN_GRACE_S = 3.0; "
        "from inference_endpoint.main import run; run()"
    )
    agg_pid: int | None = None
    try:
        with _benchmark_proc(
            [
                shutil.which("python") or "python",
                "-c",
                wrapper,
                "benchmark",
                "from-config",
                "-c",
                str(config_path),
            ]
        ) as proc:
            _wait_services_ready(proc, report_dir)
            time.sleep(3.0)  # comfortably inside the ~120 s performance phase

            agg_pid = _pid_of_child("metrics_aggregator", str(report_dir))
            assert agg_pid is not None, "aggregator child not found"
            os.kill(agg_pid, signal.SIGSTOP)  # wedge the drain

            os.kill(proc.pid, signal.SIGINT)
            start = time.monotonic()
            rc = proc.wait(timeout=30.0)
            abort_latency = time.monotonic() - start
    finally:
        if agg_pid is not None:
            try:
                os.kill(agg_pid, signal.SIGKILL)  # SIGKILL reaps stopped procs
            except ProcessLookupError:
                pass  # already gone — grace escalation killed it

    assert rc == 130, f"^C must exit 130, got {rc}"
    assert abort_latency > 2.0, "exited before the grace — drain was not wedged"
    assert (
        abort_latency < 20.0
    ), f"abort took {abort_latency:.1f}s — grace escalation did not fire"
    # The SIGKILLed aggregator never wrote a terminal snapshot, so the report
    # was built from the last live pub/sub frame — the split-brain guard must
    # still land the summary as interrupted, never state:"live".
    summary = json.loads(
        (report_dir / "performance" / "result_summary.json").read_text()
    )
    assert summary["state"] == "interrupted"
    assert summary["complete"] is False
    _assert_no_leftover_children(report_dir, "grace-expired abort")


@pytest.mark.integration
def test_sigint_before_session_exits_130(mock_http_echo_server, tmp_path):
    """A ^C before the session exists (setup/service launch) exits 130.

    No session is bound yet, so the governor falls back to an immediate
    KeyboardInterrupt — the run must not hang or exit 0.
    """
    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    _write_config(report_dir, mock_http_echo_server.url, config_path)

    with _benchmark_proc(
        [_cli(), "benchmark", "from-config", "-c", str(config_path)]
    ) as proc:
        time.sleep(1.5)  # interpreter up, setup underway; services not ready
        os.killpg(proc.pid, signal.SIGINT)
        rc = proc.wait(timeout=30.0)

    assert rc == 130, f"pre-session ^C must exit 130, got {rc}"
    _assert_no_leftover_children(report_dir, "aborted run")


@pytest.mark.integration
def test_single_group_sigint_under_uv_run_is_graceful(mock_http_echo_server, tmp_path):
    """One keystroke under `uv run` stays graceful.

    `uv run` forwards the terminal's group SIGINT to its child, so a single
    ^C is delivered twice (~200 ms apart, past kernel coalescing). The
    duplicate is a harmless no-op (the stop is already in flight): the run
    takes the graceful path — report written, exit 130.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not available")

    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    _write_config(report_dir, mock_http_echo_server.url, config_path)

    with _benchmark_proc(
        [
            uv,
            "run",
            "inference-endpoint",
            "benchmark",
            "from-config",
            "-c",
            str(config_path),
        ]
    ) as proc:
        _wait_services_ready(proc, report_dir, timeout_s=90.0)
        time.sleep(3.0)
        os.killpg(proc.pid, signal.SIGINT)  # one keystroke: group + uv forward
        rc = proc.wait(timeout=60.0)

    assert rc == 130, f"user abort must exit 130, got {rc}"
    # The graceful path writes the report before exiting.
    _assert_interrupted_artifacts(report_dir)


@pytest.mark.integration
def test_sigint_in_drain_window_keeps_summary_authoritative(
    mock_http_echo_server, tmp_path
):
    """^C after the session's terminal ENDED: result_summary is the truth.

    The session ends normally and the aggregator drains a deferred
    tokenization backlog when the ^C reaches the main process. The session
    never publishes an INTERRUPTED marker (it already ENDED) and the
    aggregator is not signaled, so final_snapshot.json legitimately reads
    state=complete — that is what the aggregator observed. The run-level
    artifacts must still be honest: exit 130 and result_summary.json
    state=interrupted, complete=false. Pins the artifact-precedence contract
    (docs/CLI_QUICK_REFERENCE.md "Ctrl-C (SIGINT)").
    """
    dataset_path = tmp_path / "big_prompts.jsonl"
    prompt = "lorem ipsum " * 21_000  # ~250 KB per sample
    with dataset_path.open("w") as f:
        for i in range(30):
            f.write(json.dumps({"prompt": f"{i} {prompt}"}) + "\n")

    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    config_path.write_text(
        f"""
type: offline
endpoint_config:
  endpoints: ["{mock_http_echo_server.url}"]
model_params:
  name: "{_CHAR_TOKENIZER_DIR}"
  streaming: "off"
datasets:
  - path: "{dataset_path}"
    type: performance
report_dir: {report_dir}
settings:
  load_pattern:
    type: max_throughput
  client:
    num_workers: 1
    warmup_connections: 0
    max_connections: 10
  metrics_tokenizer_workers: 0  # defer all tokenization to the drain
  warmup:
    enabled: false
"""
    )

    log_file = tmp_path / "run.log"
    with _benchmark_proc(
        [_cli(), "benchmark", "from-config", "-c", str(config_path)],
        log_file=log_file,
    ) as proc:
        # The pipeline logs this line once the session has ENDED and the
        # aggregator drain wait begins — the divergence window.
        deadline = time.monotonic() + 120.0
        while "Waiting for services to finish processing" not in log_file.read_text(
            errors="replace"
        ):
            assert proc.poll() is None, "benchmark exited before the drain"
            assert time.monotonic() < deadline, "drain window never reached"
            time.sleep(0.05)

        os.kill(proc.pid, signal.SIGINT)  # main process only: aggregator unsignaled
        rc = proc.wait(timeout=120.0)

    assert rc == 130, f"drain-window ^C must exit 130, got {rc}"

    summary = json.loads(
        (report_dir / "performance" / "result_summary.json").read_text()
    )
    assert summary["state"] == "interrupted"
    assert summary["complete"] is False

    # The aggregator observed a normally-ended run: its own artifact says so.
    snapshot = json.loads((report_dir / "metrics" / "final_snapshot.json").read_text())
    assert snapshot["state"] == "complete"
    _assert_no_leftover_children(report_dir, "drain-window run")
