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

"""Compare inference-endpoint and vLLM benchmarking tools.

Generates dummy prompts on the fly from a base set of 10 phrases.
Prompts are cycled if num-prompts exceeds the base set size.

Usage:
    python compare_with_vllm.py --model "Qwen/Qwen2.5-0.5B-Instruct" --num-prompts 100
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml

# Base prompts for generating benchmark datasets (cycled as needed)
BASE_PROMPTS = [
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

# Constants
DEFAULT_ENDPOINT = "http://localhost:8000"
DEFAULT_NUM_PROMPTS = 100
DEFAULT_MAX_OUTPUT_TOKENS = 2000
DEFAULT_WARMUP_TIMEOUT = 300
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_VLLM_VENV_DIR = SCRIPT_DIR / "vllm_venv"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_vllm_executable(venv_dir: Path) -> Path:
    """Get the vllm executable path from the virtualenv.

    Args:
        venv_dir: Path to the vLLM virtualenv directory

    Returns:
        Path to the vllm executable

    Raises:
        FileNotFoundError: If the venv or vllm executable doesn't exist
    """
    vllm_bin = venv_dir / "bin" / "vllm"
    if not vllm_bin.exists():
        raise FileNotFoundError(
            f"vLLM executable not found at {vllm_bin}. "
            f"Please run setup_vllm_venv.sh to create the vLLM virtualenv."
        )
    return vllm_bin


@contextmanager
def temp_dataset_files(num_prompts: int) -> Generator[tuple[Path, Path], None, None]:
    """Context manager that creates temporary dataset files for benchmarking.

    Generates prompts by cycling through BASE_PROMPTS and creates two JSONL files:
    - vLLM format: {"prompt": "..."}
    - inference-endpoint format: {"text_input": "..."}

    Args:
        num_prompts: Number of prompts to generate

    Yields:
        Tuple of (vllm_dataset_path, ie_dataset_path)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        prompts = [BASE_PROMPTS[i % len(BASE_PROMPTS)] for i in range(num_prompts)]

        vllm_path = temp_path / "prompts_vllm.jsonl"
        ie_path = temp_path / "prompts_ie.jsonl"

        with open(vllm_path, "w") as f:
            for prompt in prompts:
                f.write(json.dumps({"prompt": prompt}) + "\n")

        with open(ie_path, "w") as f:
            for prompt in prompts:
                f.write(json.dumps({"text_input": prompt}) + "\n")

        yield vllm_path, ie_path


def generate_ie_config(
    model: str,
    dataset_path: Path,
    endpoint_url: str,
    num_requests: int,
    max_output_tokens: int,
    workers: int,
    timeout: int,
    report_dir: Path,
    config_path: Path,
) -> None:
    """Generate a YAML config file for inference-endpoint benchmark.

    Creates a config with sampling params matching vLLM defaults for fair comparison:
    - temperature: 0 (greedy decoding)
    - top_p: 1.0 (no nucleus sampling)
    - top_k: null (no top-k filtering, vLLM uses -1)
    - repetition_penalty: 1.0 (no penalty)

    Args:
        model: Model name
        dataset_path: Path to JSONL dataset with {"text_input": "..."} format
        endpoint_url: Server endpoint URL
        num_requests: Number of requests to send
        max_output_tokens: Maximum output tokens per request
        workers: Number of parallel http-client workers
        timeout: Timeout in seconds
        report_dir: Directory to save reports
        config_path: Path to write the config file
    """
    config = {
        "name": "vllm-comparison-benchmark",
        "version": "1.0",
        "type": "offline",
        "model_params": {
            "name": model,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": None,
            "repetition_penalty": 1.0,
            "max_new_tokens": max_output_tokens,
            "streaming": "on",
        },
        "datasets": [
            {
                "name": "comparison-dataset",
                "type": "performance",
                "path": str(dataset_path),
                "format": "jsonl",
                "samples": num_requests,
                "parser": {"prompt": "text_input"},
            }
        ],
        "settings": {
            "runtime": {
                "min_duration_ms": 0,
                "max_duration_ms": timeout * 1000,
                "n_samples_to_issue": num_requests,
            },
            "load_pattern": {"type": "max_throughput"},
            "client": {"workers": workers},
        },
        "endpoint_config": {"endpoint": endpoint_url},
        "report_dir": str(report_dir),
        "timeout": timeout,
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare inference-endpoint benchmark with vLLM benchmark using JSONL datasets.",
        epilog="Example: python compare_with_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct --num-prompts 100",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name (e.g., Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--num-prompts",
        "-n",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help=f"Number of prompts to send (default: {DEFAULT_NUM_PROMPTS})",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Server endpoint URL (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Maximum output tokens per request (default: {DEFAULT_MAX_OUTPUT_TOKENS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds for inference-endpoint (default: 900)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers for inference-endpoint (default: 1)",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print commands without executing them (dry run mode)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output from benchmark commands (default: only show final results)",
    )
    parser.add_argument(
        "--vllm-venv-dir",
        type=Path,
        default=DEFAULT_VLLM_VENV_DIR,
        help=f"Path to vLLM virtualenv directory (default: {DEFAULT_VLLM_VENV_DIR})",
    )
    return parser.parse_args()


def run_inference_endpoint(
    model: str,
    dataset_path: Path,
    endpoint_url: str,
    num_requests: int,
    max_output_tokens: int,
    timeout: int,
    workers: int,
    temp_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run inference-endpoint benchmark and parse output.

    Uses a YAML config file to specify sampling params for fair comparison with vLLM:
    - temperature=0, top_p=1.0, repetition_penalty=1.0 (matching vLLM defaults)

    Args:
        model: Model name
        dataset_path: Path to JSONL dataset with {"text_input": "..."} format
        endpoint_url: Server endpoint URL
        num_requests: Number of requests to send
        max_output_tokens: Maximum output tokens per request
        timeout: Timeout in seconds
        workers: Number of parallel http-client workers
        temp_dir: Temporary directory to save report and config
        dry_run: If True, print command without executing it
        verbose: If True, print command output to stdout

    Returns:
        Dictionary of parsed benchmark metrics (from stdout and report JSON)
    """
    if verbose:
        logger.info("Running inference-endpoint benchmark...")

    report_dir = temp_dir / "ie_report"
    config_path = temp_dir / "ie_config.yaml"

    # Generate config with sampling params matching vLLM for fair comparison
    generate_ie_config(
        model=model,
        dataset_path=dataset_path,
        endpoint_url=endpoint_url,
        num_requests=num_requests,
        max_output_tokens=max_output_tokens,
        workers=workers,
        timeout=timeout,
        report_dir=report_dir,
        config_path=config_path,
    )

    cmd = [
        "inference-endpoint",
        "benchmark",
        "from-config",
        "--config",
        str(config_path),
        "--timeout",
        str(timeout),
    ]

    if verbose:
        with open(config_path) as f:
            config_contents = yaml.safe_load(f)
        logger.info(
            f"Generated config file contents:\n{yaml.dump(config_contents, default_flow_style=False, sort_keys=False)}"
        )
        logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        print(f"\n[DRY RUN] Would execute: {' '.join(cmd)}\n")
        return {}

    # Stream output live while capturing it for parsing
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
        raise subprocess.CalledProcessError(process.returncode, cmd)

    # Parse metrics from captured output
    full_output = "".join(captured_output)
    results = parse_inference_endpoint_output(full_output)

    # Load report JSON for total generated tokens
    report_json_path = report_dir / "result_summary.json"
    if report_json_path.exists():
        try:
            with open(report_json_path) as f:
                report_data = json.load(f)
                if report_data.get("output_sequence_lengths"):
                    results["total_output_tokens"] = report_data[
                        "output_sequence_lengths"
                    ]["total"]
                    if verbose:
                        logger.info(
                            f"Loaded total output tokens from report: {results['total_output_tokens']}"
                        )
        except Exception as e:
            logger.warning(f"Failed to parse report JSON from {report_json_path}: {e}")
    else:
        logger.warning(f"Report JSON not found: {report_json_path}")

    return results


def parse_inference_endpoint_output(output: str) -> dict[str, float]:
    """Parse inference-endpoint benchmark output to extract metrics.

    Args:
        output: inference-endpoint benchmark stdout

    Returns:
        Dictionary of parsed metrics
    """
    patterns = {
        "qps": r"QPS:\s+([\d\.]+)",
        "tps": r"TPS:\s+([\d\.]+)",
        "ttft_min": r"TTFT:.*?Min:\s+([\d\.]+)\s*ms",
        "ttft_max": r"TTFT:.*?Max:\s+([\d\.]+)\s*ms",
        "ttft_median": r"TTFT:.*?Median:\s+([\d\.]+)\s*ms",
        "ttft_mean": r"TTFT:.*?Avg\.?:\s+([\d\.]+)\s*ms",
        "tpot_min": r"TPOT.*?:.*?Min:\s+([\d\.]+)\s*ms",
        "tpot_max": r"TPOT.*?:.*?Max:\s+([\d\.]+)\s*ms",
        "tpot_median": r"TPOT.*?:.*?Median:\s+([\d\.]+)\s*ms",
        "tpot_mean": r"TPOT.*?:.*?Avg\.?:\s+([\d\.]+)\s*ms",
        "total_time": r"Duration:\s+([\d\.]+)\s*seconds",
        "output_len_min": r"Output sequence lengths:.*?Min:\s+([\d\.]+)\s*tokens",
        "output_len_max": r"Output sequence lengths:.*?Max:\s+([\d\.]+)\s*tokens",
        "output_len_median": r"Output sequence lengths:.*?Median:\s+([\d\.]+)\s*tokens",
        "output_len_mean": r"Output sequence lengths:.*?Avg\.?:\s+([\d\.]+)\s*tokens",
    }

    results = {}
    for key, pattern in patterns.items():
        if match := re.search(pattern, output, re.DOTALL | re.IGNORECASE):
            results[key] = float(match.group(1))

    # Parse percentiles separately (they appear in Percentiles: block)
    # TTFT percentiles
    ttft_section = re.search(r"TTFT:(.*?)(?:TPOT|Latency:)", output, re.DOTALL)
    if ttft_section:
        if p99_match := re.search(r"99:\s+([\d\.]+)\s*ms", ttft_section.group(1)):
            results["ttft_p99"] = float(p99_match.group(1))

    # TPOT percentiles
    tpot_section = re.search(
        r"TPOT.*?:(.*?)(?:Latency:|Output sequence)", output, re.DOTALL
    )
    if tpot_section:
        if p99_match := re.search(r"99:\s+([\d\.]+)\s*ms", tpot_section.group(1)):
            results["tpot_p99"] = float(p99_match.group(1))

    # Output sequence length percentiles
    output_len_section = re.search(
        r"Output sequence lengths:(.*?)(?:^-+\s*Summary|^={3,}|\Z)",
        output,
        re.DOTALL | re.MULTILINE,
    )
    if output_len_section:
        # Look for "99:" in the percentiles section (not "99.9:")
        if p99_match := re.search(
            r"^\s*99:\s+([\d\.]+)\s*tokens", output_len_section.group(1), re.MULTILINE
        ):
            results["output_len_p99"] = float(p99_match.group(1))

    return results


def run_vllm_benchmark(
    model: str,
    dataset_path: Path,
    endpoint_url: str,
    num_requests: int,
    max_output_tokens: int,
    temp_dir: Path,
    venv_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run vLLM benchmark with openai-chat backend and custom dataset.

    Args:
        model: Model name
        dataset_path: Path to JSONL dataset with {"prompt": "..."} format
        endpoint_url: Server endpoint URL
        num_requests: Number of requests to send
        max_output_tokens: Maximum output tokens per request
        temp_dir: Temporary directory to save detailed results
        venv_dir: Path to the vLLM virtualenv directory
        dry_run: If True, print command without executing it
        verbose: If True, print command output to stdout

    Returns:
        Dictionary of parsed benchmark metrics (from both stdout and detailed JSON)
    """
    if verbose:
        logger.info("Running vLLM benchmark...")

    # Get vllm executable from the virtualenv
    vllm_bin = get_vllm_executable(venv_dir)

    # Generate result filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_filename = f"vllm_benchmark_{timestamp}.json"
    result_path = temp_dir / result_filename

    cmd = [
        str(vllm_bin),
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        endpoint_url,
        "--model",
        model,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--custom-output-len",
        str(max_output_tokens),
        "--num-prompts",
        str(num_requests),
        "--temperature",
        "0",
        "--top-p",
        "1.0",
        "--top-k",
        "-1",
        "--repetition-penalty",
        "1.0",
        "--endpoint",
        "/v1/chat/completions",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(temp_dir),
        "--result-filename",
        result_filename,
    ]

    if verbose:
        logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        print(f"\n[DRY RUN] Would execute: {' '.join(cmd)}\n")
        return {}

    # Stream output live while capturing it for parsing
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
        raise subprocess.CalledProcessError(process.returncode, cmd)

    # Parse metrics from captured output
    full_output = "".join(captured_output)
    results = parse_vllm_output(full_output)

    # Load detailed JSON for output_lens statistics
    if result_path.exists():
        try:
            with open(result_path) as f:
                detailed_data = json.load(f)
                if verbose:
                    logger.info(
                        f"vLLM JSON keys available: {list(detailed_data.keys())[:10]}..."
                    )  # Show first 10 keys
                results.update(parse_vllm_detailed_output_lens(detailed_data))
                if verbose:
                    logger.info(f"Parsed detailed results from {result_path}")
        except Exception as e:
            logger.warning(f"Failed to parse detailed results from {result_path}: {e}")
    else:
        logger.warning(f"vLLM result file not found: {result_path}")

    return results


def parse_vllm_output(output: str) -> dict[str, float]:
    """Parse vLLM benchmark output to extract metrics.

    Args:
        output: vLLM benchmark stdout

    Returns:
        Dictionary of parsed metrics
    """
    patterns = {
        "throughput": r"Request throughput \(req/s\):\s+([\d\.]+)",
        "output_throughput": r"Output token throughput \(tok/s\):\s+([\d\.]+)",
        "total_output_tokens": r"Total generated tokens:\s+([\d]+)",
        "ttft_mean": r"Mean TTFT \(ms\):\s+([\d\.]+)",
        "ttft_median": r"Median TTFT \(ms\):\s+([\d\.]+)",
        "ttft_p99": r"P99 TTFT \(ms\):\s+([\d\.]+)",
        "tpot_mean": r"Mean TPOT \(ms\):\s+([\d\.]+)",
        "tpot_median": r"Median TPOT \(ms\):\s+([\d\.]+)",
        "tpot_p99": r"P99 TPOT \(ms\):\s+([\d\.]+)",
        "itl_mean": r"Mean ITL \(ms\):\s+([\d\.]+)",
        "itl_median": r"Median ITL \(ms\):\s+([\d\.]+)",
        "itl_p99": r"P99 ITL \(ms\):\s+([\d\.]+)",
        "total_time": r"Benchmark duration \(s\):\s+([\d\.]+)",
    }

    return {
        key: float(match.group(1))
        for key, pattern in patterns.items()
        if (match := re.search(pattern, output))
    }


def parse_vllm_detailed_output_lens(detailed_data: dict) -> dict[str, float]:
    """Parse vLLM detailed JSON output to extract output length statistics.

    Args:
        detailed_data: Parsed JSON from vLLM's --save-result output

    Returns:
        Dictionary of output length statistics
    """
    results = {}

    # Check multiple possible key names
    output_lens_key = None
    for key in ["output_lens", "output_len", "output_lengths", "output_tokens"]:
        if key in detailed_data:
            output_lens_key = key
            break

    if output_lens_key is None:
        logger.warning(
            f"No output length field found in vLLM detailed output. Available keys: {list(detailed_data.keys())}"
        )
        return results

    output_lens = detailed_data[output_lens_key]
    if not output_lens:
        logger.warning("Output length array is empty")
        return results

    output_lens_array = np.array(output_lens)

    results["output_len_mean"] = float(np.mean(output_lens_array))
    results["output_len_median"] = float(np.median(output_lens_array))
    results["output_len_min"] = float(np.min(output_lens_array))
    results["output_len_max"] = float(np.max(output_lens_array))
    results["output_len_p99"] = float(np.percentile(output_lens_array, 99))

    logger.info(
        f"Successfully parsed {len(output_lens_array)} output lengths from vLLM results"
    )

    return results


def print_comparison(ie_results: dict, vllm_results: dict) -> None:
    """Print comparison table of benchmark results.

    Args:
        ie_results: inference-endpoint results (parsed from stdout)
        vllm_results: vLLM benchmark results (parsed from stdout and detailed JSON)
    """
    print("\n" + "=" * 100)
    print(f"{'Metric':<35} | {'Inference Endpoint':<25} | {'vLLM Benchmark':<25}")
    print("-" * 100)

    def fmt(v: Any, is_integer: bool = False) -> str:
        """Format value for display."""
        if is_integer and isinstance(v, int | float):
            return f"{int(v)}"
        return f"{v:.2f}" if isinstance(v, int | float) else str(v)

    # ITL is essentially the same as TPOT in streaming mode
    ie_itl_mean = ie_results.get("tpot_mean", "N/A")
    ie_itl_median = ie_results.get("tpot_median", "N/A")
    ie_itl_p99 = ie_results.get("tpot_p99", "N/A")

    comparisons = [
        ("", "", ""),  # Separator
        (
            "Test Duration (s)",
            ie_results.get("total_time", "N/A"),
            vllm_results.get("total_time", "N/A"),
        ),
        ("", "", ""),  # Separator
        (
            "Throughput (req/s)",
            ie_results.get("qps", "N/A"),
            vllm_results.get("throughput", "N/A"),
        ),
        (
            "Total Generated Tokens",
            ie_results.get("total_output_tokens", "N/A"),
            vllm_results.get("total_output_tokens", "N/A"),
        ),
        (
            "Output Token Throughput (tok/s)",
            ie_results.get("tps", "N/A"),
            vllm_results.get("output_throughput", "N/A"),
        ),
        ("", "", ""),  # Separator
        (
            "Mean TTFT (ms)",
            ie_results.get("ttft_mean", "N/A"),
            vllm_results.get("ttft_mean", "N/A"),
        ),
        (
            "Median TTFT (ms)",
            ie_results.get("ttft_median", "N/A"),
            vllm_results.get("ttft_median", "N/A"),
        ),
        (
            "P99 TTFT (ms)",
            ie_results.get("ttft_p99", "N/A"),
            vllm_results.get("ttft_p99", "N/A"),
        ),
        ("", "", ""),  # Separator
        (
            "Mean TPOT (ms)",
            ie_results.get("tpot_mean", "N/A"),
            vllm_results.get("tpot_mean", "N/A"),
        ),
        (
            "Median TPOT (ms)",
            ie_results.get("tpot_median", "N/A"),
            vllm_results.get("tpot_median", "N/A"),
        ),
        (
            "P99 TPOT (ms)",
            ie_results.get("tpot_p99", "N/A"),
            vllm_results.get("tpot_p99", "N/A"),
        ),
        ("", "", ""),  # Separator
        ("Mean ITL (ms)", ie_itl_mean, vllm_results.get("itl_mean", "N/A")),
        ("Median ITL (ms)", ie_itl_median, vllm_results.get("itl_median", "N/A")),
        ("P99 ITL (ms)", ie_itl_p99, vllm_results.get("itl_p99", "N/A")),
        ("", "", ""),  # Separator
        (
            "Mean Output Length (tokens)",
            ie_results.get("output_len_mean", "N/A"),
            vllm_results.get("output_len_mean", "N/A"),
        ),
        (
            "Median Output Length (tokens)",
            ie_results.get("output_len_median", "N/A"),
            vllm_results.get("output_len_median", "N/A"),
        ),
        (
            "P99 Output Length (tokens)",
            ie_results.get("output_len_p99", "N/A"),
            vllm_results.get("output_len_p99", "N/A"),
        ),
    ]

    for label, ie_val, vllm_val in comparisons:
        if label == "":  # Separator row
            print("-" * 100)
        else:
            # Use integer formatting for token counts
            is_int = "Total Generated Tokens" in label or "tokens)" in label
            print(
                f"{label:<35} | {fmt(ie_val, is_int):<25} | {fmt(vllm_val, is_int):<25}"
            )
    print("=" * 100 + "\n")


def warmup_server(
    url: str, model: str, timeout: int = DEFAULT_WARMUP_TIMEOUT, verbose: bool = False
):
    """Send a warmup request to ensure the server is fully ready for inference.

    Args:
        url: Server endpoint URL
        model: Model name
        timeout: Timeout in seconds for warmup
        verbose: If True, print detailed warmup progress

    Returns True if successful, False otherwise.
    """
    if verbose:
        logger.info(f"Warming up server at {url}...")
    warmup_url = f"{url}/v1/chat/completions"
    warmup_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    }

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if verbose:
                logger.info("Sending warmup request...")
            resp = requests.post(
                warmup_url,
                json=warmup_payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if resp.status_code == 200:
                elapsed = time.time() - start_time
                if verbose:
                    logger.info(f"Warmup successful! (took {elapsed:.1f}s)")
                return True
            else:
                if verbose:
                    logger.warning(
                        f"Warmup returned {resp.status_code}: {resp.text[:200]}"
                    )
                time.sleep(5)
        except requests.Timeout:
            if verbose:
                logger.warning("Warmup timed out, retrying...")
            time.sleep(5)
        except requests.RequestException as e:
            if verbose:
                logger.warning(f"Warmup failed: {e}, retrying...")
            time.sleep(5)

    logger.error(f"Server warmup failed after {timeout}s")
    return False


def main():
    args = parse_args()

    # Set logging level based on verbose flag
    # Use ERROR level when not verbose to suppress INFO and WARNING messages
    if not args.verbose:
        logging.getLogger().setLevel(logging.ERROR)

    if args.verbose:
        logger.info(f"Model: {args.model}")
        logger.info(f"Endpoint: {args.endpoint}")
        logger.info(f"Number of prompts: {args.num_prompts}")
        logger.info(f"Max output tokens: {args.max_output_tokens}")
        logger.info(f"vLLM venv: {args.vllm_venv_dir}")
    if args.dry:
        print("DRY RUN MODE: Commands will be printed but not executed")

    # Generate temp dataset files and run benchmarks
    with temp_dataset_files(args.num_prompts) as (vllm_dataset, ie_dataset):
        if args.verbose:
            logger.info(f"Generated {args.num_prompts} prompts in temp directory")

        # Use the same temp directory for vLLM results
        temp_dir = vllm_dataset.parent

        # Warmup before vLLM benchmark
        if not args.dry:
            print("Running vLLM benchmark...")
            if not warmup_server(args.endpoint, args.model, verbose=args.verbose):
                logger.error("Warmup before vLLM benchmark failed.")
                sys.exit(1)

        vllm_res = run_vllm_benchmark(
            args.model,
            vllm_dataset,
            args.endpoint,
            args.num_prompts,
            args.max_output_tokens,
            temp_dir,
            args.vllm_venv_dir,
            dry_run=args.dry,
            verbose=args.verbose,
        )

        # Warmup before inference-endpoint benchmark
        if not args.dry:
            print("Running inference-endpoint benchmark...")
            if not warmup_server(args.endpoint, args.model, verbose=args.verbose):
                logger.error("Warmup before inference-endpoint benchmark failed.")
                sys.exit(1)

        ie_res = run_inference_endpoint(
            args.model,
            ie_dataset,
            args.endpoint,
            args.num_prompts,
            args.max_output_tokens,
            args.timeout,
            args.workers,
            temp_dir,
            dry_run=args.dry,
            verbose=args.verbose,
        )

    if not args.dry:
        print_comparison(ie_res, vllm_res)
    else:
        logger.info("Skipping comparison in dry run mode")


if __name__ == "__main__":
    main()
