# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_bytes
from .pyxis_environment import build_srun_command, resolve_image, safe_srun_env

_PRINT_LOCK = threading.Lock()
_EVAL_SCRIPT = r"""set -eu

patch_path=$1
eval_path=$2
output_path=$3
timeout_s=$4

cd /testbed
if git apply --verbose "$patch_path" || \
    git apply --verbose --reject "$patch_path" || \
    patch --batch --fuzz=5 -p1 -i "$patch_path"; then
    echo ">>>>> Applied Patch"
else
    echo ">>>>> Patch Apply Failed"
    exit 1
fi

set +e
timeout "$timeout_s" /bin/bash "$eval_path" >"$output_path" 2>&1
status=$?
set -e
cat "$output_path"
if [[ $status -eq 124 ]]; then
    echo "Timeout error: $timeout_s seconds exceeded." >>"$output_path"
    exit 124
fi
exit 0
"""


def _run_agent(args: argparse.Namespace) -> None:
    from minisweagent.environments import get_environment
    from minisweagent.run.benchmarks import swebench

    def get_pyxis_environment(config: dict, instance: dict):
        environment_config = copy.deepcopy(config.get("environment", {}))
        environment_config["image"] = resolve_image(
            args.image_registry, instance["instance_id"]
        )
        return get_environment(environment_config)

    base_agent = swebench.ProgressTrackingAgent

    class LiveTrajectoryAgent(base_agent):  # type: ignore[misc, valid-type]
        def __init__(self, *agent_args: Any, instance_id: str = "", **kwargs: Any):
            kwargs["output_path"] = (
                args.output / instance_id / f"{instance_id}.live.json"
            )
            super().__init__(*agent_args, instance_id=instance_id, **kwargs)

    swebench.get_sb_environment = get_pyxis_environment
    swebench.ProgressTrackingAgent = LiveTrajectoryAgent
    swebench.main(
        subset=args.subset,
        split=args.split,
        slice_spec="",
        filter_spec=args.filter,
        shuffle=False,
        output=str(args.output),
        workers=args.workers,
        model=args.model,
        model_class=None,
        redo_existing=False,
        config_spec=[str(args.config)],
        environment_class="swebench_service.pyxis_environment.PyxisEnvironment",
    )


def _evaluate_instance(
    *,
    test_spec: Any,
    prediction: dict[str, Any],
    image: str | Path,
    output_dir: Path,
    run_id: str,
    timeout_s: int,
) -> None:
    instance_id = test_spec.instance_id
    safe_model = prediction["model_name_or_path"].replace("/", "__")
    log_dir = output_dir / "logs" / "run_evaluation" / run_id / safe_model / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    patch_path = log_dir / "patch.diff"
    eval_path = log_dir / "eval.sh"
    output_path = log_dir / "test_output.txt"
    report_path = log_dir / "report.json"
    patch_path.write_text(prediction["model_patch"])
    eval_path.write_text(test_spec.eval_script)
    output_path.write_text("")
    report_path.unlink(missing_ok=True)

    command = build_srun_command(
        image=image,
        mounts=[
            (patch_path, "/tmp/swebench_patch.diff"),
            (eval_path, "/tmp/swebench_eval.sh"),
            (output_path, "/tmp/swebench_test_output.txt"),
        ],
        workdir="/testbed",
        argv=[
            "bash",
            "-c",
            _EVAL_SCRIPT,
            "pyxis-eval",
            "/tmp/swebench_patch.diff",
            "/tmp/swebench_eval.sh",
            "/tmp/swebench_test_output.txt",
            str(timeout_s),
        ],
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s + 60,
        env=safe_srun_env(),
    )
    with _PRINT_LOCK:
        print(f"[{instance_id}]\n{result.stdout}{result.stderr}", flush=True)
    if result.returncode != 0:
        return

    from swebench.harness.grading import get_eval_report

    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=output_path,
        include_tests_status=True,
    )
    atomic_write_bytes(report_path, (json.dumps(report, indent=4) + "\n").encode())


def _run_eval(args: argparse.Namespace) -> None:
    from swebench.harness.reporting import make_run_report
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.utils import (
        get_predictions_from_file,
        load_swebench_dataset,
    )

    predictions = {
        prediction["instance_id"]: prediction
        for prediction in get_predictions_from_file(
            str(args.predictions_path), args.dataset_name, args.split
        )
        if prediction["instance_id"] in args.instance_ids
    }
    rows = load_swebench_dataset(args.dataset_name, args.split, args.instance_ids)
    images = {
        instance_id: resolve_image(args.image_registry, instance_id)
        for instance_id in args.instance_ids
    }
    payloads = []
    for row in rows:
        instance_id = row["instance_id"]
        prediction = predictions.get(instance_id)
        if prediction is None or prediction.get("model_patch") in {"", None}:
            continue
        payloads.append(
            {
                "test_spec": make_test_spec(row, arch="arm64"),
                "prediction": prediction,
                "image": images[instance_id],
                "output_dir": args.output_dir,
                "run_id": args.run_id,
                "timeout_s": args.timeout,
            }
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [
            executor.submit(_evaluate_instance, **payload) for payload in payloads
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with _PRINT_LOCK:
                    print(f"Pyxis evaluation failed: {exc}", flush=True)

    output_dir = args.output_dir.resolve()
    with contextlib.chdir(output_dir):
        make_run_report(
            predictions,
            [{"instance_id": instance_id} for instance_id in args.instance_ids],
            args.run_id,
            client=None,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    agent_parser = commands.add_parser("agent")
    agent_parser.add_argument("--model", required=True)
    agent_parser.add_argument("--config", type=Path, required=True)
    agent_parser.add_argument("--subset", required=True)
    agent_parser.add_argument("--split", required=True)
    agent_parser.add_argument("--filter", required=True)
    agent_parser.add_argument("--workers", type=int, required=True)
    agent_parser.add_argument("--output", type=Path, required=True)
    agent_parser.add_argument("--image-registry", required=True)

    eval_parser = commands.add_parser("eval")
    eval_parser.add_argument("--dataset-name", required=True)
    eval_parser.add_argument("--split", required=True)
    eval_parser.add_argument("--predictions-path", type=Path, required=True)
    eval_parser.add_argument("--max-workers", type=int, required=True)
    eval_parser.add_argument("--run-id", required=True)
    eval_parser.add_argument("--image-registry", required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--timeout", type=int, default=1800)
    eval_parser.add_argument("--instance-ids", nargs="+", required=True)
    args = parser.parse_args(argv)

    if args.command == "agent":
        _run_agent(args)
    else:
        _run_eval(args)


if __name__ == "__main__":
    main()
