# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from inference_endpoint.evaluation.swebench_service.swebench_service import (
    pyxis_environment as environment_mod,
)
from inference_endpoint.evaluation.swebench_service.swebench_service import (
    pyxis_persistent as transport,
)
from inference_endpoint.evaluation.swebench_service.swebench_service.runner import (
    RunnerError,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def portable_unshare(tmp_path, monkeypatch):
    """Only bypass namespace creation; run the actual shell worker."""
    if not all(shutil.which(tool) for tool in ("timeout", "bash")):
        pytest.skip("requires GNU timeout and bash")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    unshare = bin_dir / "unshare"
    unshare.write_text(
        '#!/bin/sh\nwhile [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"\n'
    )
    unshare.chmod(0o700)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


@pytest.fixture
def worker(tmp_path, monkeypatch, portable_unshare):
    root = tmp_path / "protocol"
    commands = []

    command = [
        "bash",
        str(Path(transport.__file__).with_name("pyxis_command_worker.sh")),
        str(root),
        "bash",
        "-c",
    ]
    popen = subprocess.Popen

    def launch(argv, **kwargs):
        commands.append(argv)
        return popen(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)

    channel = transport.PersistentExecChannel(
        root,
        command,
        environment_mod.safe_srun_env(),
        failure_path=tmp_path / "failed",
        launch_timeout_s=2,
        driver_grace_s=1,
        shutdown_grace_s=0.2,
    )
    channel.start()
    yield channel, commands
    channel.close()


def execute(channel, command, cwd, timeout=2):
    return channel.execute(command=command, cwd=str(cwd), timeout_s=timeout)


@pytest.mark.parametrize("code", [0, 7, 124, 137])
def test_preserves_exit_code_and_merged_output(worker, tmp_path, code):
    channel, launches = worker
    result = execute(
        channel, f"printf stdout; printf stderr >&2; exit {code}", tmp_path
    )
    assert result.output == "stdoutstderr"
    assert result.returncode == code
    assert result.timed_out is False
    assert len(launches) == 1


def test_files_persist_but_shell_state_does_not(worker, tmp_path):
    channel, launches = worker
    execute(channel, "printf persisted > state; export PRIVATE=value; cd /", tmp_path)
    result = execute(
        channel, 'cat state; printf "|%s|%s" "$PWD" "${PRIVATE-unset}"', tmp_path
    )
    assert result.output == f"persisted|{tmp_path}|unset"
    assert result.returncode == 0
    assert len(launches) == 1


def test_preserves_multiline_command_and_binary_output(worker, tmp_path):
    channel, _ = worker
    result = execute(
        channel, "printf 'a\\000b\\377\\n'\n# trailing comment\n", tmp_path
    )
    assert result.output == "a\x00b\ufffd\n"
    assert result.returncode == 0


def test_timeout_is_reported_and_worker_remains_usable(worker, tmp_path):
    channel, launches = worker
    result = execute(channel, "printf before; exec sleep 5", tmp_path, timeout=1)
    assert result.output == "before"
    assert result.timed_out is True
    assert result.returncode == 124
    assert execute(channel, "printf recovered", tmp_path).output == "recovered"
    assert len(launches) == 1


def test_missing_cwd_is_a_command_error_and_worker_remains_usable(worker, tmp_path):
    channel, launches = worker
    result = execute(channel, "echo must-not-run", tmp_path / "missing")
    assert result.returncode == 125
    assert not result.timed_out
    assert "No such file or directory" in result.output
    assert execute(channel, "printf recovered", tmp_path).output == "recovered"
    assert len(launches) == 1


def test_environment_survives_tool_tmp_cleanup(monkeypatch, tmp_path, portable_unshare):
    """Use production mount sources and worker, substituting local paths for mounts."""
    scratch = None

    def initialize(**kwargs):
        nonlocal scratch
        scratch = next(source for source, dest in kwargs["mounts"] if dest == "/tmp")

    def local_command(*, argv, mounts, **kwargs):
        result = []
        for arg in argv:
            for source, dest in mounts:
                if arg == dest or arg.startswith(dest + "/"):
                    arg = str(source) + arg[len(dest) :]
                    break
            result.append(arg)
        return result

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(environment_mod, "run_srun_step", initialize)
    monkeypatch.setattr(environment_mod, "build_srun_command", local_command)
    env = environment_mod.PyxisEnvironment(
        image="image", run_id="cleanup", cwd=str(tmp_path)
    )
    try:
        command = (
            "import os, shutil, tempfile; "
            f"location = tempfile.mkdtemp(dir={str(scratch)!r}); "
            "shutil.rmtree(os.path.dirname(location)); print('cleaned')"
        )
        result = env.execute(
            {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(command)}"}
        )
        assert result["returncode"] == 0
        assert result["output"] == "cleaned\n"
        assert env.execute({"command": "printf recovered"})["output"] == "recovered"
    finally:
        env.cleanup()


@pytest.mark.parametrize(
    "message",
    [
        "curl: (56) CONNECT tunnel failed, response 403",
        "srun: error: Unable to confirm allocation for job 123",
    ],
)
def test_startup_retries_confirmed_prelaunch_failure(
    monkeypatch, tmp_path, portable_unshare, message
):
    root = tmp_path / "protocol"
    launches = []
    delays = []
    popen = subprocess.Popen

    def launch(argv, **kwargs):
        launches.append(argv)
        assert not (root / "request").exists()
        assert not (tmp_path / "failed").exists()
        if len(launches) < 3:
            return popen(
                ["bash", "-c", 'printf "%s\\n" "$1" >&2; exit 1', "prelaunch", message],
                **kwargs,
            )
        return popen(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(transport, "wait_for_prelaunch_retry", delays.append)
    channel = transport.PersistentExecChannel(
        root,
        ["bash", str(environment_mod._COMMAND_WORKER), str(root), "bash", "-c"],
        environment_mod.safe_srun_env(),
        failure_path=tmp_path / "failed",
        launch_timeout_s=2,
    )
    try:
        channel.start()
        assert (
            execute(channel, "printf once >> count; cat count", tmp_path).output
            == "once"
        )
        assert len(launches) == 3
        assert delays == [1, 2]
        assert not (tmp_path / "failed").exists()
    finally:
        channel.close()


@pytest.mark.parametrize(
    "finish, attempts",
    [
        ("exit 1", 5),
        ("exit 0", 1),
        ("exec sleep 10", 1),
    ],
)
def test_startup_retry_boundaries(monkeypatch, tmp_path, finish, attempts):
    launches = []
    popen = subprocess.Popen

    def launch(argv, **kwargs):
        launches.append(argv)
        return popen(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(transport, "wait_for_prelaunch_retry", lambda attempt: None)
    channel = transport.PersistentExecChannel(
        tmp_path / "protocol",
        ["bash", "-c", "echo 'unable to confirm allocation for job' >&2; " + finish],
        environment_mod.safe_srun_env(),
        failure_path=tmp_path / "failed",
        launch_timeout_s=0.2,
        shutdown_grace_s=0.1,
    )
    try:
        with pytest.raises(RunnerError, match="did not become ready"):
            channel.start()
        assert len(launches) == attempts
        assert channel._process.poll() is not None
        assert (tmp_path / "failed").exists()
    finally:
        channel.close()


def test_ready_worker_is_never_relaunched(monkeypatch, tmp_path):
    delays = []
    monkeypatch.setattr(transport, "wait_for_prelaunch_retry", delays.append)
    root = tmp_path / "protocol"
    channel = transport.PersistentExecChannel(
        root,
        [
            "bash",
            "-c",
            'touch "$1/ready"; echo "unable to confirm allocation for job"; exit 1',
            "worker",
            str(root),
        ],
        environment_mod.safe_srun_env(),
        failure_path=tmp_path / "failed",
    )
    try:
        channel.start()
        channel._process.wait(timeout=2)
        with pytest.raises(RunnerError, match="not running"):
            execute(channel, "touch unexpected", tmp_path)
        assert delays == []
        assert not (tmp_path / "unexpected").exists()
        assert (tmp_path / "failed").exists()
    finally:
        channel.close()


def test_process_cleanup_is_a_command_failure(tmp_path):
    """Run destructive process matching only inside real, private PID namespaces."""
    if sys.platform != "linux" or not all(
        shutil.which(tool) for tool in ("unshare", "timeout", "pkill")
    ):
        pytest.skip("requires Linux PID namespaces and procps")
    isolation = ["unshare", "--pid", "--fork", "--mount-proc", "--kill-child"]
    probe = subprocess.run([*isolation, "true"], capture_output=True, timeout=5)
    if probe.returncode:
        pytest.skip("PID namespaces are not permitted")
    root = tmp_path / "protocol"
    channel = transport.PersistentExecChannel(
        root,
        [
            *isolation,
            "bash",
            str(environment_mod._COMMAND_WORKER),
            str(root),
            "bash",
            "-c",
        ],
        environment_mod.safe_srun_env(),
        launch_timeout_s=2,
    )
    try:
        channel.start()
        result = execute(channel, "sleep 20 & pkill -f 'sleep 20'", tmp_path)
        assert result.returncode == 143
        assert not result.timed_out
        assert execute(channel, "printf recovered", tmp_path).output == "recovered"
    finally:
        channel.close()


@pytest.mark.parametrize("timeout", [0, -1])
def test_invalid_timeout_does_not_execute_command(worker, tmp_path, timeout):
    channel, _ = worker
    with pytest.raises(ValueError, match="positive"):
        execute(channel, "touch unexpected", tmp_path, timeout=timeout)
    assert not (tmp_path / "unexpected").exists()
    assert execute(channel, "true", tmp_path).returncode == 0


def test_concurrent_callers_are_serialized(worker, tmp_path):
    channel, launches = worker

    def increment(_):
        return execute(
            channel,
            "n=$(cat count 2>/dev/null || echo 0); sleep 0.05; echo $((n+1)) > count",
            tmp_path,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(increment, range(4)))
    assert all(result.returncode == 0 for result in results)
    assert (tmp_path / "count").read_text() == "4\n"
    assert len(launches) == 1


def test_dead_worker_does_not_restart_or_replay(worker, tmp_path):
    channel, launches = worker
    channel._process.kill()
    channel._process.wait(timeout=2)
    with pytest.raises(RunnerError, match="not running"):
        execute(channel, "echo duplicated >> count", tmp_path)
    with pytest.raises(RunnerError, match="closed"):
        execute(channel, "echo duplicated >> count", tmp_path)
    assert (tmp_path / "failed").exists()
    assert not (tmp_path / "count").exists()
    assert len(launches) == 1


def test_driver_deadline_never_replays_an_active_command(worker, tmp_path, monkeypatch):
    channel, launches = worker
    channel._driver_grace_s = 0.2
    wait_for = channel._wait_for
    monkeypatch.setattr(
        channel,
        "_wait_for",
        lambda path, timeout: wait_for(path.with_name("missing"), timeout),
    )
    with pytest.raises(RunnerError, match="driver deadline"):
        execute(channel, "echo once >> count; sleep 0.4", tmp_path, timeout=1)
    with pytest.raises(RunnerError, match="closed"):
        execute(channel, "echo twice >> count", tmp_path)
    assert (tmp_path / "count").read_text() == "once\n"
    assert (tmp_path / "failed").exists()
    assert len(launches) == 1


@pytest.mark.parametrize("fault", ["metadata", "truncated", "missing"])
def test_completion_corruption_fails_the_run(worker, tmp_path, monkeypatch, fault):
    channel, launches = worker
    read_completion = channel._read_completion

    def corrupt(request):
        if fault == "metadata":
            (request / "complete").write_text("garbled")
        elif fault == "truncated":
            (request / "output").write_bytes(b"")
        else:
            (request / "output").unlink()
        return read_completion(request)

    monkeypatch.setattr(channel, "_read_completion", corrupt)
    with pytest.raises(RunnerError, match="infrastructure failure"):
        execute(channel, "echo once >> count; printf result", tmp_path)
    with pytest.raises(RunnerError, match="closed"):
        execute(channel, "echo twice >> count", tmp_path)
    assert (tmp_path / "count").read_text() == "once\n"
    assert (tmp_path / "failed").exists()
    assert len(launches) == 1


def test_close_reaps_worker_and_is_idempotent(worker):
    channel, _ = worker
    process = channel._process
    channel.close()
    channel.close()
    assert process.poll() == 0


def test_failure_marker_error_does_not_skip_worker_shutdown(
    worker, tmp_path, monkeypatch
):
    channel, _ = worker
    touch = Path.touch

    def fail_marker(path, *args, **kwargs):
        if path == tmp_path / "failed":
            raise OSError("disk full")
        return touch(path, *args, **kwargs)

    def bad_reply(request):
        raise RunnerError("invalid reply")

    monkeypatch.setattr(Path, "touch", fail_marker)
    monkeypatch.setattr(channel, "_read_completion", bad_reply)
    with pytest.raises(RunnerError, match="invalid reply"):
        execute(channel, "true", tmp_path)
    assert channel._closed
    assert channel._process.poll() is not None


def test_startup_failure_is_infrastructure_failure(tmp_path):
    channel = transport.PersistentExecChannel(
        tmp_path / "protocol",
        ["bash", "-c", "echo denied >&2; exit 1"],
        environment_mod.safe_srun_env(),
        failure_path=tmp_path / "failed",
    )
    try:
        with pytest.raises(RunnerError, match="(?s)did not become ready.*denied"):
            channel.start()
        assert (tmp_path / "failed").exists()
    finally:
        channel.close()


def test_environment_routes_commands_to_one_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "node")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-worker")
    steps = []
    channels = []

    def step(**kwargs):
        steps.append(kwargs)
        return subprocess.CompletedProcess([], 0, "", "")

    class Channel:
        def __init__(self, protocol_dir, command, env, **kwargs):
            self.argv = command
            self.env = env
            self.closed = False
            self.commands = 0
            channels.append(self)

        def start(self):
            pass

        def execute(self, **kwargs):
            self.commands += 1
            return transport.CommandResult(7, "failed", False)

        def close(self):
            self.closed = True

    def cleanup(command, **kwargs):
        assert channels[0].closed
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(environment_mod, "run_srun_step", step)
    monkeypatch.setattr(environment_mod, "PersistentExecChannel", Channel)
    monkeypatch.setattr(subprocess, "run", cleanup)
    env = environment_mod.PyxisEnvironment(image="image", run_id="run")
    try:
        for _ in range(3):
            assert env.execute({"command": "exit 7"})["returncode"] == 7
        assert len(steps) == 1
        assert len(channels) == 1
        assert channels[0].commands == 3
        assert "OPENAI_API_KEY" not in channels[0].env
        assert "--kill-child" in channels[0].argv
        assert "--jobid=123" in channels[0].argv
        assert (env._tmp_dir.stat().st_mode & 0o777) == 0o700

        def fail(**kwargs):
            raise RunnerError("worker died; execution is uncertain")

        monkeypatch.setattr(channels[0], "execute", fail)
        with pytest.raises(RunnerError, match="execution is uncertain"):
            env.execute({"command": "touch state"})
        assert len(steps) == 1
        assert len(channels) == 1
    finally:
        env.cleanup()
    assert not env._tmp_dir.exists()


@pytest.mark.parametrize(
    "manifest", ["--1 0 0", "0 0 -1", "0 2 0", "256 0 0", "truncated"]
)
def test_rejects_corrupt_completion_metadata(tmp_path, manifest):
    channel = transport.PersistentExecChannel(tmp_path / "protocol", [], {})
    request = tmp_path / "request"
    request.mkdir()
    (request / "complete").write_text(manifest)
    with pytest.raises(RunnerError, match="completion marker"):
        channel._read_completion(request)


def test_missing_worker_executable_marks_infrastructure_failure(tmp_path):
    channel = transport.PersistentExecChannel(
        tmp_path / "protocol",
        [str(tmp_path / "missing")],
        {},
        failure_path=tmp_path / "failed",
    )
    try:
        with pytest.raises(RunnerError, match="did not become ready"):
            channel.start()
        assert channel._process is None
        assert (tmp_path / "failed").exists()
    finally:
        channel.close()


def test_worker_that_never_becomes_ready_is_reaped(tmp_path):
    channel = transport.PersistentExecChannel(
        tmp_path / "protocol",
        ["sleep", "10"],
        environment_mod.safe_srun_env(),
        launch_timeout_s=0.1,
        shutdown_grace_s=0.1,
        failure_path=tmp_path / "failed",
    )
    try:
        with pytest.raises(RunnerError, match="did not become ready"):
            channel.start()
        assert channel._process.poll() is not None
    finally:
        channel.close()
    assert channel._process.poll() is not None
    assert (tmp_path / "failed").exists()
