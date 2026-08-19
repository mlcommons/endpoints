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
