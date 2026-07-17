# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import threading
from pathlib import Path

import msgspec.json
import pytest
import yaml

_SERVICE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "inference_endpoint"
    / "evaluation"
    / "swebench_service"
)
sys.path.insert(0, str(_SERVICE_ROOT))

from swebench_service import runner as runner_mod  # noqa: E402
from swebench_service.runner import (  # noqa: E402
    CancellationToken,
    RunCancelled,
    RunnerError,
    SweBenchRunner,
)
from swebench_service.schemas import RunRequest  # noqa: E402

pytestmark = pytest.mark.unit


def _request(endpoints: list[str]) -> RunRequest:
    return RunRequest(
        model_name="test-model",
        endpoint_urls=endpoints,
        endpoint_api_key=None,
        generation_params={"name": "test-model"},
        subset="lite",
        split="test",
        num_instances=1,
        workers=1,
        max_eval_workers=1,
        evaluated_instance_ids=["repo__repo-1"],
    )


def test_run_subprocess_streams_output_to_log(tmp_path):
    log_path = tmp_path / "subprocess.log"

    runner_mod._run_subprocess(
        [sys.executable, "-c", "print('first'); print('second')"],
        log_path,
        cwd=tmp_path,
        timeout_s=5,
    )

    assert log_path.read_text() == "first\nsecond\n"


def test_run_subprocess_reports_bounded_failure_tail(tmp_path):
    log_path = tmp_path / "subprocess.log"
    script = (
        "import sys\n"
        "print('early-marker')\n"
        "for i in range(700): print(f'{i:04d}-' + 'x' * 100)\n"
        "print('final-marker')\n"
        "sys.exit(7)\n"
    )

    with pytest.raises(RunnerError, match="exited with code 7") as exc_info:
        runner_mod._run_subprocess(
            [sys.executable, "-c", script],
            log_path,
            cwd=tmp_path,
            timeout_s=5,
        )

    assert "early-marker" in log_path.read_text()
    failure_tail = str(exc_info.value).partition("\n")[2]
    assert "final-marker" in failure_tail
    assert "early-marker" not in failure_tail


def test_run_subprocess_timeout_preserves_partial_log(tmp_path):
    log_path = tmp_path / "subprocess.log"

    with pytest.raises(RunnerError, match="timed out after 1s"):
        runner_mod._run_subprocess(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            log_path,
            cwd=tmp_path,
            timeout_s=1,
        )

    assert log_path.read_text() == "started\n"


def test_run_subprocess_cancellation_preserves_partial_log(tmp_path):
    log_path = tmp_path / "subprocess.log"
    cancel_token = CancellationToken()
    cancel_timer = threading.Timer(0.2, cancel_token.cancel)
    cancel_timer.start()
    try:
        with pytest.raises(RunCancelled, match="subprocess cancelled"):
            runner_mod._run_subprocess(
                [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                log_path,
                cwd=tmp_path,
                timeout_s=5,
                cancel_token=cancel_token,
            )
    finally:
        cancel_timer.cancel()
        cancel_timer.join()

    assert log_path.read_text() == "started\n"


def test_base_env_keeps_proxies_and_sets_no_proxy_for_loopback(monkeypatch, tmp_path):
    monkeypatch.setenv("http_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("all_proxy", "socks5://proxy.example:1080")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.example:1080")
    monkeypatch.setenv("NO_PROXY", "intel.com")

    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    env = runner._base_env(_request(["http://localhost:30000"]))

    assert env["http_proxy"] == "http://proxy.example:8080"
    assert env["https_proxy"] == "http://proxy.example:8080"
    assert env["HTTP_PROXY"] == "http://proxy.example:8080"
    assert env["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert env["all_proxy"] == "socks5://proxy.example:1080"
    assert env["ALL_PROXY"] == "socks5://proxy.example:1080"
    assert {"127.0.0.1", "localhost", "intel.com"} <= set(env["NO_PROXY"].split(","))
    assert env["NO_PROXY"] == env["no_proxy"]


def test_base_env_keeps_proxies_for_non_loopback_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("https_proxy", "http://proxy.example:8080")

    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    env = runner._base_env(_request(["http://swebench-host:30000"]))

    assert env["https_proxy"] == "http://proxy.example:8080"
    assert "swebench-host" in env["NO_PROXY"].split(",")


def test_patch_config_rewrites_localhost_api_base_to_127_0_0_1(tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    patched = runner._patch_config(
        tmp_path,
        _request(["http://localhost:30000"]),
        run_id="run-123",
    )

    cfg = yaml.safe_load(patched.read_text())
    assert cfg["model"]["model_kwargs"]["api_base"] == "http://127.0.0.1:30000/v1"
    assert "model_class" not in cfg["model"]
    assert "api_key" not in cfg["model"]["model_kwargs"]
    assert cfg["environment"]["run_args"] == [
        "--rm",
        "--label",
        "com.mlcommons.endpoints.swebench-run=run-123",
    ]


def test_patch_config_keeps_api_key_out_of_yaml_and_forwards_generation(tmp_path):
    request = _request(["http://endpoint:30000"])
    request.endpoint_api_key = "real-secret"
    request.generation_params = {
        "temperature": 0.2,
        "seed": 23,
        "max_new_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    patched = runner._patch_config(tmp_path, request, run_id="run-1")

    text = patched.read_text()
    cfg = yaml.safe_load(text)
    model_kwargs = cfg["model"]["model_kwargs"]
    assert "real-secret" not in text
    assert "api_key" not in model_kwargs
    assert model_kwargs["temperature"] == 0.2
    assert model_kwargs["seed"] == 23
    assert model_kwargs["max_tokens"] == 2048
    assert model_kwargs["chat_template_kwargs"] == {"enable_thinking": False}


def test_base_env_supplies_api_key_only_to_agent_subprocess(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    authenticated = _request(["http://endpoint:30000"])
    authenticated.endpoint_api_key = "real-secret"
    loopback = _request(["http://localhost:30000"])

    assert runner._base_env(loopback)["OPENAI_API_KEY"] == "EMPTY"

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    unauthenticated = _request(["http://endpoint:30000"])
    assert "OPENAI_API_KEY" not in runner._base_env(unauthenticated)


def test_run_agent_filters_exact_instance_ids(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def fake_run_subprocess(cmd, *args, **kwargs):
        commands.append(cmd)
        envs.append(kwargs["env"])

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)
    request = _request(["http://endpoint:30000"])
    request.endpoint_api_key = "agent-secret"
    request.evaluated_instance_ids = ["repo__repo-1", "repo.with.regex+chars"]
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    runner._run_agent(request, tmp_path / "config.yaml", tmp_path, tmp_path)

    cmd = commands[0]
    assert "--slice" not in cmd
    assert cmd[cmd.index("--filter") + 1] == (
        "^(?:repo__repo\\-1|repo\\.with\\.regex\\+chars)$"
    )
    assert envs[0]["OPENAI_API_KEY"] == "agent-secret"


def test_patch_config_selects_qwen_model_without_mutating_pythonpath(
    monkeypatch, tmp_path
):
    envs: list[dict[str, str]] = []
    request = _request(["http://endpoint:30000"])
    request.enable_swebench_toolcall_patch = True
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    def fake_run_subprocess(cmd, log_path, *, env, **kwargs):
        envs.append(env)

    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)

    patched = runner._patch_config(tmp_path, request, run_id="run-qwen")
    cfg = yaml.safe_load(patched.read_text())
    runner._run_agent(request, patched, tmp_path, tmp_path)

    assert cfg["model"]["model_class"] == (
        "swebench_service.qwen_tools_model.QwenToolsModel"
    )
    assert envs[0]["PYTHONPATH"] == "/existing/path"


def test_validate_prediction_ids_rejects_unexpected_instances(tmp_path):
    request = _request(["http://endpoint:30000"])
    request.evaluated_instance_ids = ["repo__repo-1"]
    preds = tmp_path / "preds.json"
    preds.write_bytes(
        msgspec.json.encode({"repo__repo-1": "patch", "repo__repo-2": "patch"})
    )
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    with pytest.raises(RunnerError, match="unexpected SWE-bench"):
        runner._validate_prediction_ids(request, preds)


def test_run_eval_persists_harness_run_id(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    request = _request(["http://endpoint:30000"])
    output_dir = tmp_path / "output"
    run_dir = tmp_path / "run"
    output_dir.mkdir()
    run_dir.mkdir()
    preds_path = output_dir / "preds.json"
    preds_path.write_text('{"repo__repo-1":"patch"}')

    def fake_run_subprocess(cmd, log_path, *, cwd, **kwargs):
        assert cmd[:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]
        run_id = cmd[cmd.index("--run_id") + 1]
        assert (run_dir / "swe_bench_eval_run_id.txt").read_text() == run_id
        (cwd / f"test-model.{run_id}.json").write_text(
            '{"resolved_instances":1,"submitted_instances":1}'
        )

    monkeypatch.setattr(runner_mod, "_run_subprocess", fake_run_subprocess)

    result_path = runner._run_eval(request, preds_path, output_dir, run_dir)

    assert result_path.exists()
    assert (run_dir / "swe_bench_eval_run_id.txt").read_text().startswith("endpoints_")


def _stub_successful_run(monkeypatch, runner: SweBenchRunner) -> None:
    def fake_run_agent(request, patched_config, output_dir, run_dir, cancel_token=None):
        (output_dir / "preds.json").write_text('{"repo__repo-1":"patch"}')

    def fake_run_eval(request, preds_path, output_dir, run_dir, cancel_token=None):
        result_path = output_dir / "result.json"
        result_path.write_text('{"resolved_instances":1,"submitted_instances":1}')
        return result_path

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_run_eval", fake_run_eval)


def test_run_cleans_labeled_containers_after_success(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    _stub_successful_run(monkeypatch, runner)
    cleaned: list[str] = []
    monkeypatch.setattr(runner, "_cleanup_containers", cleaned.append)

    result = runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-1")

    assert result == {"resolved_instances": 1, "submitted_instances": 1}
    assert cleaned == ["run-1"]


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (RuntimeError("agent failed"), "agent failed"),
        (RunnerError("subprocess timed out"), "timed out"),
        (RunCancelled("subprocess cancelled"), "cancelled"),
    ],
)
def test_run_cleans_labeled_containers_after_failure(
    monkeypatch, tmp_path, error, match
):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    cleaned: list[str] = []

    def fail_agent(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner, "_run_agent", fail_agent)
    monkeypatch.setattr(runner, "_cleanup_containers", cleaned.append)

    with pytest.raises(type(error), match=match):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-2")

    assert cleaned == ["run-2"]


def test_run_cleans_harness_containers_after_eval_cancellation(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    cleaned: list[tuple[str, dict]] = []

    def fake_run_agent(request, patched_config, output_dir, run_dir, cancel_token=None):
        (output_dir / "preds.json").write_text('{"repo__repo-1":"patch"}')

    def cancel_eval(request, preds_path, output_dir, run_dir, cancel_token=None):
        (run_dir / "swe_bench_eval_run_id.txt").write_text("endpoints_cancelled")
        raise RunCancelled("subprocess cancelled")

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "_run_eval", cancel_eval)
    monkeypatch.setattr(
        runner,
        "_cleanup_containers",
        lambda run_id, **kwargs: cleaned.append((run_id, kwargs)),
    )

    with pytest.raises(RunCancelled, match="cancelled"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-cancelled")

    assert cleaned == [
        (
            "run-cancelled",
            {
                "eval_run_id": "endpoints_cancelled",
                "instance_ids": ["repo__repo-1"],
            },
        )
    ]


def test_cleanup_failure_fails_successful_run(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    _stub_successful_run(monkeypatch, runner)
    monkeypatch.setattr(
        runner,
        "_cleanup_containers",
        lambda run_id: (_ for _ in ()).throw(RunnerError("cleanup failed")),
    )

    with pytest.raises(RunnerError, match="cleanup failed"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-3")


def test_cleanup_failure_does_not_mask_primary_failure(monkeypatch, tmp_path):
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)
    monkeypatch.setattr(
        runner,
        "_run_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RunCancelled("cancelled")),
    )
    monkeypatch.setattr(
        runner,
        "_cleanup_containers",
        lambda run_id: (_ for _ in ()).throw(RunnerError("cleanup failed")),
    )

    with pytest.raises(RunCancelled, match="cancelled"):
        runner.run(_request(["http://endpoint:30000"]), tmp_path / "run-4")


def test_cleanup_uses_exact_run_label_and_leaves_unrelated_containers(
    monkeypatch, tmp_path
):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        stdout = "matched-1\nmatched-2\n" if cmd[1:3] == ["ps", "-aq"] else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    runner._cleanup_containers("run-exact")

    assert calls == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.mlcommons.endpoints.swebench-run=run-exact",
        ],
        ["docker", "rm", "-f", "matched-1", "matched-2"],
    ]


def test_cleanup_exactly_matches_harness_container_names(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["ps", "-aq"]:
            stdout = "agent-container\n"
        elif cmd[1:3] == ["ps", "-a"]:
            stdout = (
                "eval-container\tsweb.eval.repo__repo-1.endpoints_eval\n"
                "other-instance\tsweb.eval.repo__repo-2.endpoints_eval\n"
                "other-run\tsweb.eval.repo__repo-1.endpoints_other\n"
                "unrelated\tunrelated.endpoints_eval\n"
            )
        else:
            stdout = ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
    runner = SweBenchRunner(project_root=tmp_path, subprocess_timeout_s=30)

    runner._cleanup_containers(
        "run-exact",
        eval_run_id="endpoints_eval",
        instance_ids=["Repo__Repo-1"],
    )

    assert calls == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.mlcommons.endpoints.swebench-run=run-exact",
        ],
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=endpoints_eval",
            "--format",
            "{{.ID}}\t{{.Names}}",
        ],
        ["docker", "rm", "-f", "agent-container", "eval-container"],
    ]
