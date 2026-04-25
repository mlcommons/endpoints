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

"""Run a concurrency sweep benchmark using a template YAML config.

The provided config is used as a template: for each concurrency value the
load_pattern is overridden to type=concurrency with the given target, and
the report_dir is set to a per-run subdirectory under the sweep root.

Usage:
    python scripts/concurrency_sweep/run.py --config path/to/config.yaml
    python scripts/concurrency_sweep/run.py \\
        --config examples/08_Qwen2.5-0.5B_Example/online_qwen_benchmark.yaml \\
        --concurrency 1 2 4 8 16 32 64 \\
        --duration-ms 120000 \\
        --output-dir results/my_sweep

After the sweep completes, run the summarize script:
    python scripts/concurrency_sweep/summarize.py <sweep_root>/concurrency_sweep/
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

DEFAULT_CONCURRENCY_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_DURATION_MS = 600_000
DEFAULT_TIMEOUT_S = 12 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a concurrency sweep using a template YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the base benchmark YAML config to use as a template.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCY_VALUES,
        metavar="N",
        help="Concurrency values to sweep over.",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=DEFAULT_DURATION_MS,
        help="Per-run benchmark duration in milliseconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for sweep output. Defaults to the report_dir "
            "defined in the config file."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="Per-run subprocess timeout in seconds (includes setup and teardown).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Stream benchmark output to the terminal in real time in addition "
            "to saving it to the per-run log file. Useful for debugging failures."
        ),
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def render_config(
    base_config: dict, concurrency: int, report_dir: Path, duration_ms: int
) -> dict:
    config = copy.deepcopy(base_config)
    config["name"] = f"{config.get('name', 'benchmark')}-c{concurrency}"
    config["report_dir"] = str(report_dir)

    runtime = config.setdefault("settings", {}).setdefault("runtime", {})
    runtime["min_duration_ms"] = duration_ms
    runtime["max_duration_ms"] = duration_ms

    load_pattern = config["settings"].setdefault("load_pattern", {})
    load_pattern["type"] = "concurrency"
    load_pattern["target_concurrency"] = concurrency
    load_pattern.pop("target_qps", None)

    return config


def run_single_benchmark(
    config: dict, timeout_seconds: int, log_path: Path, verbose: bool = False
) -> tuple[str, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="."
    ) as tmp:
        yaml.safe_dump(config, tmp, sort_keys=False)
        temp_config_path = Path(tmp.name)

    cmd = [
        "inference-endpoint",
        "benchmark",
        "from-config",
        "-c",
        str(temp_config_path),
    ]

    try:
        if verbose:
            with log_path.open("w") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                try:
                    proc.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise
            returncode = proc.returncode
        else:
            with log_path.open("w") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            returncode = result.returncode

        if returncode == 0:
            return "success", ""
        return "failed", f"exit code {returncode}, see {log_path}"
    except subprocess.TimeoutExpired:
        return "timeout", f"exceeded {timeout_seconds} seconds"
    finally:
        temp_config_path.unlink(missing_ok=True)


def write_summary(sweep_root: Path, rows: list[dict]) -> None:
    summary_path = sweep_root / "summary.json"
    with summary_path.open("w") as f:
        json.dump(rows, f, indent=2)

    csv_path = summary_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["concurrency", "status", "report_dir", "log_file", "detail"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote run summary to {summary_path}")


def main() -> int:
    args = parse_args()

    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1

    base_config = load_config(args.config)

    if args.output_dir is not None:
        base_report_dir = args.output_dir
    elif "report_dir" in base_config:
        base_report_dir = Path(base_config["report_dir"])
    else:
        print(
            "Error: no --output-dir given and config has no report_dir field.",
            file=sys.stderr,
        )
        return 1

    concurrency_values = sorted(set(args.concurrency))
    sweep_root = base_report_dir / "concurrency_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    print(f"Config       : {args.config}")
    print(f"Concurrency  : {concurrency_values}")
    print(f"Duration/run : {args.duration_ms / 60_000:.1f} minutes")
    print(f"Sweep root   : {sweep_root}")

    summary_rows: list[dict] = []

    for concurrency in concurrency_values:
        run_dir = sweep_root / f"concurrency_{concurrency}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "benchmark.log"
        config = render_config(base_config, concurrency, run_dir, args.duration_ms)

        print(f"\nRunning concurrency={concurrency} ...")
        status, detail = run_single_benchmark(
            config=config,
            timeout_seconds=args.timeout_seconds,
            log_path=log_path,
            verbose=args.verbose,
        )
        print(f"  status: {status}" + (f"  ({detail})" if detail else ""))
        if status != "success" and not args.verbose:
            print(f"  Re-run with --verbose to stream output, or inspect: {log_path}")

        summary_rows.append(
            {
                "concurrency": concurrency,
                "status": status,
                "report_dir": str(run_dir),
                "log_file": str(log_path),
                "detail": detail,
            }
        )

    write_summary(sweep_root, summary_rows)

    n_ok = sum(1 for r in summary_rows if r["status"] == "success")
    print(f"\nCompleted {n_ok}/{len(summary_rows)} runs successfully.")
    print("To summarize results run:")
    print(f"  python scripts/concurrency_sweep/summarize.py {sweep_root}")

    return 0 if n_ok == len(summary_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
