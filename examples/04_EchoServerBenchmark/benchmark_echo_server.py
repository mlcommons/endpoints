#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Benchmark comparison between inference-endpoint and vLLM using the echo server.

This script starts a local echo server and runs both benchmarking tools against it
to compare their performance characteristics. Since the echo server responds instantly,
this isolates the benchmarking tool overhead from actual inference latency.

Usage:
    python benchmark_echo_server.py --num-prompts 30000 --workers 4
    python benchmark_echo_server.py --num-prompts 30000 --workers 4 --verbose
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_NUM_PROMPTS = 30000
ECHO_SERVER_WORKERS = 4
IE_WORKERS = 1  # Match vLLM's single async client
SCRIPT_DIR = Path(__file__).parent.resolve()
VLLM_VENV_DIR = SCRIPT_DIR.parent / "03_BenchmarkComparison" / "vllm_venv"

# Sample prompts for benchmarking
SAMPLE_PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "What is machine learning and how does it work?",
    "Describe the process of photosynthesis.",
    "How does blockchain technology work?",
    "What are the health benefits of regular exercise?",
    "Explain the theory of relativity in simple terms.",
    "How do neural networks learn from data?",
    "What causes climate change and its effects?",
    "Describe how the internet works at a high level.",
    "What is the difference between AI and machine learning?",
]


def create_dataset_files(temp_dir: Path) -> tuple[Path, Path]:
    """Create dataset files for both benchmarks.

    Args:
        temp_dir: Temporary directory to store files

    Returns:
        Tuple of (vllm_dataset_path, ie_dataset_path)
    """
    vllm_path = temp_dir / "vllm_prompts.jsonl"
    ie_path = temp_dir / "ie_prompts.jsonl"

    with open(vllm_path, "w") as f:
        for prompt in SAMPLE_PROMPTS:
            f.write(json.dumps({"prompt": prompt}) + "\n")

    with open(ie_path, "w") as f:
        for prompt in SAMPLE_PROMPTS:
            f.write(json.dumps({"text_input": prompt}) + "\n")

    return vllm_path, ie_path


def run_vllm_benchmark(
    endpoint_url: str,
    dataset_path: Path,
    num_prompts: int,
    streaming: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run vLLM benchmark.

    Args:
        endpoint_url: Echo server URL
        dataset_path: Path to vLLM format dataset
        num_prompts: Number of prompts to send
        streaming: Whether to use streaming mode
        verbose: Print command output

    Returns:
        Dictionary of parsed metrics
    """
    vllm_bin = VLLM_VENV_DIR / "bin" / "vllm"
    if not vllm_bin.exists():
        raise FileNotFoundError(
            f"vLLM not found at {vllm_bin}. "
            "Run examples/03_BenchmarkComparison/setup_vllm_venv.sh first."
        )

    cmd = [
        str(vllm_bin),
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        endpoint_url,
        "--model",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--num-prompts",
        str(num_prompts),
        "--endpoint",
        "/v1/chat/completions",
        "--skip-chat-template",
    ]

    if not streaming:
        cmd.append("--no-stream")

    if verbose:
        logger.info(f"Running: {' '.join(cmd)}")

    captured_output = []
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    for line in process.stdout:
        if verbose:
            print(line, end="")
        captured_output.append(line)

    process.wait()
    if process.returncode != 0:
        logger.error(f"vLLM benchmark failed with code {process.returncode}")
        return {}

    return parse_vllm_output("".join(captured_output))


def parse_vllm_output(output: str) -> dict[str, float]:
    """Parse vLLM benchmark output."""
    patterns = {
        "throughput": r"Request throughput \(req/s\):\s+([\d\.]+)",
        "output_throughput": r"Output token throughput \(tok/s\):\s+([\d\.]+)",
        "total_output_tokens": r"Total generated tokens:\s+([\d]+)",
        "total_time": r"Benchmark duration \(s\):\s+([\d\.]+)",
        "ttft_mean": r"Mean TTFT \(ms\):\s+([\d\.]+)",
        "ttft_median": r"Median TTFT \(ms\):\s+([\d\.]+)",
        "ttft_p99": r"P99 TTFT \(ms\):\s+([\d\.]+)",
        "tpot_mean": r"Mean TPOT \(ms\):\s+([\d\.]+)",
        "tpot_median": r"Median TPOT \(ms\):\s+([\d\.]+)",
        "tpot_p99": r"P99 TPOT \(ms\):\s+([\d\.]+)",
        "itl_mean": r"Mean ITL \(ms\):\s+([\d\.]+)",
        "itl_median": r"Median ITL \(ms\):\s+([\d\.]+)",
        "itl_p99": r"P99 ITL \(ms\):\s+([\d\.]+)",
    }

    return {
        key: float(match.group(1))
        for key, pattern in patterns.items()
        if (match := re.search(pattern, output))
    }


def run_inference_endpoint_benchmark(
    endpoint_url: str,
    dataset_path: Path,
    num_prompts: int,
    workers: int,
    streaming: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run inference-endpoint benchmark.

    Args:
        endpoint_url: Echo server URL
        dataset_path: Path to IE format dataset
        num_prompts: Number of prompts to send
        workers: Number of parallel workers
        streaming: Whether to use streaming mode
        verbose: Print command output

    Returns:
        Dictionary of parsed metrics
    """
    cmd = [
        "inference-endpoint",
        "benchmark",
        "offline",
        "--endpoint",
        endpoint_url,
        "--model",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset",
        str(dataset_path),
        "--num-samples",
        str(num_prompts),
        "--workers",
        str(workers),
        "--streaming",
        "on" if streaming else "off",
    ]

    if verbose:
        logger.info(f"Running: {' '.join(cmd)}")

    captured_output = []
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    for line in process.stdout:
        if verbose:
            print(line, end="")
        captured_output.append(line)

    process.wait()
    if process.returncode != 0:
        logger.error(
            f"inference-endpoint benchmark failed with code {process.returncode}"
        )
        return {}

    return parse_ie_output("".join(captured_output))


def parse_ie_output(output: str) -> dict[str, float]:
    """Parse inference-endpoint benchmark output."""
    patterns = {
        "qps": r"QPS:\s+([\d\.]+)",
        "tps": r"TPS:\s+([\d\.]+)",
        "total_time": r"Duration:\s+([\d\.]+)\s*seconds",
        "ttft_mean": r"TTFT:.*?Avg\.?:\s+([\d\.]+)\s*ms",
        "ttft_median": r"TTFT:.*?Median:\s+([\d\.]+)\s*ms",
        "tpot_mean": r"TPOT.*?:.*?Avg\.?:\s+([\d\.]+)\s*ms",
        "tpot_median": r"TPOT.*?:.*?Median:\s+([\d\.]+)\s*ms",
    }

    results = {}
    for key, pattern in patterns.items():
        if match := re.search(pattern, output, re.DOTALL | re.IGNORECASE):
            results[key] = float(match.group(1))

    # Parse TTFT percentiles
    ttft_section = re.search(r"TTFT:(.*?)(?:TPOT|Latency:)", output, re.DOTALL)
    if ttft_section:
        if p99_match := re.search(r"99:\s+([\d\.]+)\s*ms", ttft_section.group(1)):
            results["ttft_p99"] = float(p99_match.group(1))

    # Parse TPOT percentiles
    tpot_section = re.search(
        r"TPOT.*?:(.*?)(?:Latency:|Output sequence)", output, re.DOTALL
    )
    if tpot_section:
        if p99_match := re.search(r"99:\s+([\d\.]+)\s*ms", tpot_section.group(1)):
            results["tpot_p99"] = float(p99_match.group(1))

    return results


def print_comparison(ie_results: dict, vllm_results: dict) -> None:
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("ECHO SERVER BENCHMARK COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<35} | {'Inference-Endpoint':<22} | {'vLLM Benchmark':<22}")
    print("-" * 90)

    def fmt(v: Any) -> str:
        if isinstance(v, int | float):
            return f"{v:.2f}"
        return str(v) if v else "N/A"

    comparisons = [
        ("Duration (s)", ie_results.get("total_time"), vllm_results.get("total_time")),
        (
            "Request Throughput (req/s)",
            ie_results.get("qps"),
            vllm_results.get("throughput"),
        ),
        (
            "Output Token Throughput (tok/s)",
            ie_results.get("tps"),
            vllm_results.get("output_throughput"),
        ),
        ("", "", ""),
        ("Mean TTFT (ms)", ie_results.get("ttft_mean"), vllm_results.get("ttft_mean")),
        (
            "Median TTFT (ms)",
            ie_results.get("ttft_median"),
            vllm_results.get("ttft_median"),
        ),
        ("P99 TTFT (ms)", ie_results.get("ttft_p99"), vllm_results.get("ttft_p99")),
        ("", "", ""),
        ("Mean TPOT (ms)", ie_results.get("tpot_mean"), vllm_results.get("tpot_mean")),
        (
            "Median TPOT (ms)",
            ie_results.get("tpot_median"),
            vllm_results.get("tpot_median"),
        ),
        ("P99 TPOT (ms)", ie_results.get("tpot_p99"), vllm_results.get("tpot_p99")),
    ]

    for label, ie_val, vllm_val in comparisons:
        if label == "":
            print("-" * 90)
        else:
            print(f"{label:<35} | {fmt(ie_val):<22} | {fmt(vllm_val):<22}")

    print("=" * 90)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare inference-endpoint and vLLM benchmarks using echo server"
    )
    parser.add_argument(
        "--num-prompts",
        "-n",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help=f"Number of prompts to benchmark (default: {DEFAULT_NUM_PROMPTS})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Echo server port (default: auto-assign)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use non-streaming mode",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    # Import echo server
    try:
        from inference_endpoint.testing.echo_server import EchoServer
    except ImportError:
        logger.error(
            "Cannot import EchoServer. Make sure inference-endpoint is installed."
        )
        sys.exit(1)

    # Start echo server
    print(f"Starting echo server with {ECHO_SERVER_WORKERS} workers...")
    server = EchoServer(port=args.port, workers=ECHO_SERVER_WORKERS)
    server.start()
    endpoint_url = server.url
    print(f"Echo server running at {endpoint_url}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            vllm_dataset, ie_dataset = create_dataset_files(temp_path)

            streaming = not args.no_stream
            mode = "streaming" if streaming else "non-streaming"
            print(f"\nRunning benchmarks ({mode} mode, {args.num_prompts} prompts)...")

            # Run vLLM benchmark
            print("\n[1/2] Running vLLM benchmark...")
            vllm_results = run_vllm_benchmark(
                endpoint_url,
                vllm_dataset,
                args.num_prompts,
                streaming=streaming,
                verbose=args.verbose,
            )

            # Run inference-endpoint benchmark
            print("\n[2/2] Running inference-endpoint benchmark...")
            ie_results = run_inference_endpoint_benchmark(
                endpoint_url,
                ie_dataset,
                args.num_prompts,
                IE_WORKERS,
                streaming=streaming,
                verbose=args.verbose,
            )

            # Print comparison
            print_comparison(ie_results, vllm_results)

    finally:
        print("Stopping echo server...")
        server.stop()


if __name__ == "__main__":
    main()
