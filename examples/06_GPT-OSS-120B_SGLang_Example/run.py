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

"""Manual example for GPQA dataset with SGLang endpoint.

This example demonstrates:
1. Generating the GPQA dataset to a file
2. Applying transforms (UserPromptFormatter, Harmonize, DropColumns)
3. Launching an SGLang API session using the transformed dataset

Prerequisites:
- SGLang server running at localhost:30000
- Run: python -m sglang.launch_server --model-path openai/gpt-oss-120b --port 30000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import LoadPattern, LoadPatternType
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined.gpqa import GPQA
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    DropColumns,
    Harmonize,
    UserPromptFormatter,
)
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator import (
    BenchmarkSession,
    MaxThroughputScheduler,
    SampleEvent,
    SampleEventHandler,
    WithoutReplacementSampleOrder,
)
from tqdm import tqdm

# Configuration for SGLang server
SGLANG_SERVER_HOST = "localhost"
SGLANG_SERVER_PORT = 30000
SGLANG_ENDPOINT = f"http://{SGLANG_SERVER_HOST}:{SGLANG_SERVER_PORT}/generate"


class ProgressBarHook:
    """Hook to update progress bar on sample completion."""

    def __init__(self, pbar: tqdm | None = None):
        self.pbar = pbar

    def __call__(self, _):
        if isinstance(self.pbar, tqdm):
            self.pbar.update(1)

    def set_pbar(self, pbar: tqdm):
        self.pbar = pbar


def generate_gpqa_dataset(
    datasets_dir: Path,
    variant: str = "diamond",
    max_samples: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Generate the GPQA dataset to a file.

    Args:
        datasets_dir: Directory where datasets are stored
        variant: GPQA variant to use (default: "diamond")
        max_samples: Maximum number of samples to include (default: None = all)
        force: Force regeneration of dataset even if it exists

    Returns:
        DataFrame containing the GPQA dataset
    """
    df = GPQA.generate(
        datasets_dir=Path(datasets_dir),
        variant=variant,
        max_samples=max_samples,
        force=force,
    )
    return df


def create_transforms() -> list:
    """Create the list of transforms to apply to the GPQA dataset.

    Returns:
        List of transforms to apply
    """
    prompt_format = (
        "{question}\n\n"
        "(A) {choice1}\n"
        "(B) {choice2}\n"
        "(C) {choice3}\n"
        "(D) {choice4}\n\n"
        "Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'."
    )

    return [
        # Step 1: Format the prompt from question and choices
        UserPromptFormatter(
            user_prompt_format=prompt_format,
            output_column="user_prompt",
        ),
        # Step 2: Harmonize the prompt for SGLang/GPT-OSS
        Harmonize(
            prompt_column="user_prompt",
        ),
        # Step 3: Drop columns we don't need for inference
        DropColumns(
            columns=[
                "question",
                "choice1",
                "choice2",
                "choice3",
                "choice4",
                "domain",
                "subdomain",
                "user_prompt",
            ],
            errors="ignore",
        ),
        # Step 4: Add metadata columns since we don't want to do a dict update every iteration
        AddStaticColumns(
            {
                "stream": True,
                "max_new_tokens": 32768,
                "temperature": 1.0,
                "top_p": 1.0,
                "tok_k": -1,
            }
        ),
    ]


def create_sglang_client(tmp_dir: Path) -> HTTPEndpointClient:
    """Create an SGLang HTTP client for issuing queries.

    Args:
        tmp_dir: Temporary directory for ZMQ sockets

    Returns:
        Configured HTTPEndpointClient for SGLang
    """
    http_config = HTTPClientConfig(
        endpoint_url=SGLANG_ENDPOINT,
        num_workers=4,
        api_type="sglang",
    )

    zmq_config = ZMQConfig(
        zmq_request_queue_prefix=f"ipc://{tmp_dir}/req",
        zmq_response_queue_addr=f"ipc://{tmp_dir}/resp",
        zmq_readiness_queue_addr=f"ipc://{tmp_dir}/ready",
    )

    aiohttp_config = AioHttpConfig()

    client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
    return client


class EmptyDataset(Dataset):
    """Empty dataset for performance run."""

    def __init__(self):
        super().__init__(None)

    def load_sample(self, index: int):
        return None

    def num_samples(self):
        return 0


def run_benchmark_session(dataset: Dataset, issuer: HttpClientSampleIssuer, args):
    """Run a benchmark session with the SGLang endpoint.

    Args:
        dataset: Dataset to use for benchmarking
        issuer: SampleIssuer for issuing queries
        args: Command line arguments
    """
    # Set up progress bar hook
    pbar_hook = ProgressBarHook()
    SampleEventHandler.register_hook(SampleEvent.COMPLETE, pbar_hook)

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(6),
        reported_metrics=[],
        min_duration_ms=args.min_duration * 1000,
        max_duration_ms=args.max_duration * 1000,
        n_samples_from_dataset=0,
        n_samples_to_issue=0,
        min_sample_count=0,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    # Use max throughput scheduler for offline benchmarking
    scheduler = MaxThroughputScheduler(rt_settings, WithoutReplacementSampleOrder)

    # Run the benchmark session
    n_total = dataset.num_samples() * dataset.repeats

    with tqdm(desc="GPQA Benchmark", total=n_total, unit="samples") as pbar:
        pbar_hook.set_pbar(pbar)
        sess = BenchmarkSession.start(
            rt_settings,
            EmptyDataset(),
            issuer,
            scheduler,
            accuracy_datasets=[dataset],
            name="gpqa_sglang_benchmark",
            report_dir=args.report_dir,
            dump_events_log=True,
            max_shutdown_timeout_s=600.0,
        )
        sess.wait_for_test_end()


def run_main(args):
    """Main function to run the example."""
    # Setup paths
    tmp_dir = Path("/tmp/sglang_manual_example")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Generate GPQA dataset
        print("Generating GPQA diamond dataset...")
        df = generate_gpqa_dataset(
            datasets_dir="datasets",
            force=args.force_regenerate,
        )
        print(f"Loaded {len(df)} samples from GPQA diamond")

        # Step 2: Create transforms
        print("Creating transforms...")
        transforms = create_transforms()

        # Step 3: Create Dataset with transforms (transforms will be applied during load())
        print("Creating dataset with transforms...")
        print(df.columns)
        df.to_parquet("datasets/gqpa_diamond_pre-transformed_gpt-oss.parquet")
        dataset = GPQA(df, transforms=transforms)
        dataset.load()
        print(f"Dataset loaded with {dataset.num_samples()} samples")

        # Step 4: Create SGLang client
        print(f"Creating SGLang client for endpoint: {SGLANG_ENDPOINT}")
        client = create_sglang_client(tmp_dir)
        sample_issuer = HttpClientSampleIssuer(client)

        # Step 5: Run benchmark session
        print("Starting benchmark session...")
        run_benchmark_session(dataset, sample_issuer, args)

        print(f"\nBenchmark complete! Results saved to {args.report_dir}/")

    finally:
        # Cleanup
        if "client" in locals():
            client.shutdown()


def main():
    """Main entry point for the manual example."""
    parser = argparse.ArgumentParser(
        description="GPQA dataset example with SGLang endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset generation arguments
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of dataset even if it exists",
    )

    # Benchmark configuration arguments
    parser.add_argument(
        "--min-duration",
        type=int,
        default=10,
        help="Minimum duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=600,
        help="Maximum duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="gpqa_sglang_report",
        help="Directory to save benchmark reports (default: gpqa_sglang_report)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GPQA Dataset Example with SGLang")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  SGLang endpoint: {SGLANG_ENDPOINT}")
    print(f"  Report directory: {args.report_dir}\n")

    # Run the main function
    run_main(args)


if __name__ == "__main__":
    main()
