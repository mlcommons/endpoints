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

"""Whole-process Ctrl-C integration test.

The one interruption path no unit test can compose: a real
``inference-endpoint`` subprocess in its own process group receives SIGINT
(exactly what a terminal ^C delivers to the foreground group — parent and
service children alike) mid-run. The contract:

- exit code 130 (user abort, distinct from failure exit codes 1-4);
- artifacts are honest: ``final_snapshot.json`` ``state=interrupted`` (the
  session's INTERRUPTED marker drives the aggregator's ENDED finalize) and
  ``result_summary.json`` ``complete: false``;
- ``events.jsonl`` survives — the event logger ignores the group SIGINT and
  flushes on the session's terminal ENDED;
- no service child outlives the run;
- teardown is prompt, not a hang on an unbounded drain.
"""

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

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


def _procs_referencing(needle: str) -> list[str]:
    """Cmdlines of live processes whose argv mentions ``needle`` (Linux)."""
    hits = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue  # process exited mid-scan
        if needle.encode() in cmdline:
            hits.append(cmdline.decode(errors="replace"))
    return hits


@pytest.mark.integration
def test_sigint_mid_run_exits_130_with_interrupted_artifacts(
    mock_http_echo_server, tmp_path
):
    cli = shutil.which("inference-endpoint")
    assert cli is not None, "console script must be installed in the test venv"

    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    _write_config(report_dir, mock_http_echo_server.url, config_path)

    proc = subprocess.Popen(
        [cli, "benchmark", "from-config", "-c", str(config_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # own process group, like a foreground job
    )
    try:
        # The aggregator touches metrics/.ready once its signal handlers are
        # registered; the session starts issuing right after service readiness.
        ready = report_dir / "metrics" / ".ready"
        deadline = time.monotonic() + 60.0
        while not ready.exists():
            assert proc.poll() is None, "benchmark died before services came up"
            assert time.monotonic() < deadline, "services never became ready"
            time.sleep(0.1)
        time.sleep(3.0)  # comfortably inside the ~120 s performance phase

        os.killpg(proc.pid, signal.SIGINT)
        rc = proc.wait(timeout=60.0)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    assert rc == 130, f"user abort must exit 130, got {rc}"

    snapshot = json.loads((report_dir / "metrics" / "final_snapshot.json").read_text())
    assert snapshot["state"] == "interrupted"

    summary = json.loads(
        (report_dir / "performance" / "result_summary.json").read_text()
    )
    assert summary["complete"] is False

    # The event logger must survive the group SIGINT and flush on ENDED.
    assert (
        report_dir / "events.jsonl"
    ).exists(), "events.jsonl missing — event logger died on ^C instead of flushing"

    # No aggregator/event-logger child may outlive the run.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        leftovers = _procs_referencing(str(report_dir))
        if not leftovers:
            break
        time.sleep(0.2)
    assert not leftovers, f"service children outlived the run: {leftovers}"


def _pid_of_child(needle: str, extra: str) -> int | None:
    """PID of a live process whose argv mentions both needles (Linux)."""
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue  # process exited mid-scan
        if needle.encode() in cmdline and extra.encode() in cmdline:
            return int(pid_dir.name)
    return None


@pytest.mark.integration
def test_second_sigint_force_quits_immediately(mock_http_echo_server, tmp_path):
    """Second ^C abandons a wedged metrics drain and exits 130 promptly.

    The aggregator child is SIGSTOPped to simulate a wedged drain — the exact
    hang the force-quit path exists for. SIGINT goes to the MAIN process only
    (``os.kill``, not the group), as the governor's contract is per-process:
    the first ^C stops the session gracefully and then parks forever waiting
    for the stopped aggregator; the second ^C must SIGKILL the children and
    exit 130 within seconds.
    """
    cli = shutil.which("inference-endpoint")
    assert cli is not None, "console script must be installed in the test venv"

    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    _write_config(report_dir, mock_http_echo_server.url, config_path)

    proc = subprocess.Popen(
        [cli, "benchmark", "from-config", "-c", str(config_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    agg_pid: int | None = None
    try:
        ready = report_dir / "metrics" / ".ready"
        deadline = time.monotonic() + 60.0
        while not ready.exists():
            assert proc.poll() is None, "benchmark died before services came up"
            assert time.monotonic() < deadline, "services never became ready"
            time.sleep(0.1)
        time.sleep(3.0)  # comfortably inside the ~120 s performance phase

        agg_pid = _pid_of_child("metrics_aggregator", str(report_dir))
        assert agg_pid is not None, "aggregator child not found"
        os.kill(agg_pid, signal.SIGSTOP)  # wedge the drain

        os.kill(proc.pid, signal.SIGINT)
        time.sleep(2.0)  # graceful path engaged; drain parked on the wedge
        assert proc.poll() is None, "first ^C must keep waiting on the drain"

        os.kill(proc.pid, signal.SIGINT)
        start = time.monotonic()
        rc = proc.wait(timeout=15.0)
        force_quit_latency = time.monotonic() - start
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if agg_pid is not None:
            try:
                os.kill(agg_pid, signal.SIGKILL)  # SIGKILL reaps stopped procs
            except ProcessLookupError:
                pass  # already gone — the force-quit killed it

    assert rc == 130, f"force quit must exit 130, got {rc}"
    assert (
        force_quit_latency < 10.0
    ), f"force quit took {force_quit_latency:.1f}s — the drain was not abandoned"

    # SIGKILLed children must not outlive the run.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        leftovers = _procs_referencing(str(report_dir))
        if not leftovers:
            break
        time.sleep(0.2)
    assert not leftovers, f"service children outlived the force quit: {leftovers}"


@pytest.mark.integration
def test_sigint_before_session_exits_130(mock_http_echo_server, tmp_path):
    """A ^C before the session exists (setup/service launch) exits 130.

    No session is bound yet, so the governor falls back to an immediate
    KeyboardInterrupt — the run must not hang or exit 0.
    """
    cli = shutil.which("inference-endpoint")
    assert cli is not None, "console script must be installed in the test venv"

    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    _write_config(report_dir, mock_http_echo_server.url, config_path)

    proc = subprocess.Popen(
        [cli, "benchmark", "from-config", "-c", str(config_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        time.sleep(1.5)  # interpreter up, setup underway; services not ready
        os.killpg(proc.pid, signal.SIGINT)
        rc = proc.wait(timeout=30.0)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    assert rc == 130, f"pre-session ^C must exit 130, got {rc}"

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        leftovers = _procs_referencing(str(report_dir))
        if not leftovers:
            break
        time.sleep(0.2)
    assert not leftovers, f"children outlived the aborted run: {leftovers}"


@pytest.mark.integration
def test_single_group_sigint_under_uv_run_is_graceful(mock_http_echo_server, tmp_path):
    """One keystroke under `uv run` counts once.

    `uv run` forwards the terminal's group SIGINT to its child, so a single
    ^C is delivered twice (~200 ms apart, past kernel coalescing). The
    duplicate must be suppressed: the run takes the graceful path — report
    written, exit 130 — instead of force-quitting and losing the metrics.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not available")

    report_dir = tmp_path / "report"
    config_path = tmp_path / "bench.yaml"
    _write_config(report_dir, mock_http_echo_server.url, config_path)

    proc = subprocess.Popen(
        [
            uv,
            "run",
            "inference-endpoint",
            "benchmark",
            "from-config",
            "-c",
            str(config_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        ready = report_dir / "metrics" / ".ready"
        deadline = time.monotonic() + 90.0
        while not ready.exists():
            assert proc.poll() is None, "benchmark died before services came up"
            assert time.monotonic() < deadline, "services never became ready"
            time.sleep(0.1)
        time.sleep(3.0)

        os.killpg(proc.pid, signal.SIGINT)  # one keystroke: group + uv forward
        rc = proc.wait(timeout=60.0)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    assert rc == 130, f"user abort must exit 130, got {rc}"

    # The graceful path writes the report; a force quit would have skipped it.
    summary = json.loads(
        (report_dir / "performance" / "result_summary.json").read_text()
    )
    assert summary["complete"] is False
    snapshot = json.loads((report_dir / "metrics" / "final_snapshot.json").read_text())
    assert snapshot["state"] == "interrupted"
