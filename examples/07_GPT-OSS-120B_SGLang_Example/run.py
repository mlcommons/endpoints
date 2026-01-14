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
import logging
import random
import shutil
import tempfile
from pathlib import Path

from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import LoadPattern, LoadPatternType
from inference_endpoint.dataset_manager import Dataset, EmptyDataset
from inference_endpoint.dataset_manager.predefined.aime25 import AIME25
from inference_endpoint.dataset_manager.predefined.gpqa import GPQA
from inference_endpoint.dataset_manager.predefined.livecodebench import LiveCodeBench
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.evaluation.extractor import ABCDExtractor, BoxedMathExtractor
from inference_endpoint.evaluation.scoring import LiveCodeBenchScorer, PassAt1Scorer
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


def run_benchmark_session(
    accuracy_datasets: list[Dataset], issuer: HttpClientSampleIssuer, args
):
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
    n_total = sum(
        [dataset.num_samples() * dataset.repeats for dataset in accuracy_datasets]
    )

    with tqdm(
        desc="SGLang Accuracy Suite Benchmark", total=n_total, unit="samples"
    ) as pbar:
        pbar_hook.set_pbar(pbar)
        sess = BenchmarkSession.start(
            rt_settings,
            EmptyDataset(),
            issuer,
            scheduler,
            accuracy_datasets=accuracy_datasets,
            name="sglang_accuracy_suite_benchmark",
            report_dir=args.report_dir,
            dump_events_log=True,
            max_shutdown_timeout_s=None,
        )
        sess.wait_for_test_end()

    # Create the scorer
    scorer = PassAt1Scorer(
        GPQA.DATASET_ID,
        accuracy_datasets[0],
        args.report_dir,
        extractor=ABCDExtractor,
    )

    # Score the dataset
    score, n_repeats = scorer.score()
    logging.info(f"GPQA Pass@1 Score ({n_repeats} repeats): {score}")

    scorer = PassAt1Scorer(
        AIME25.DATASET_ID,
        accuracy_datasets[1],
        args.report_dir,
        extractor=BoxedMathExtractor,
        ground_truth_column="answer",
    )

    # Score the dataset
    score, n_repeats = scorer.score()
    logging.info(f"AIME25 Pass@1 Score ({n_repeats} repeats): {score}")

    # Score the LCB dataset
    scorer = LiveCodeBenchScorer(
        LiveCodeBench.DATASET_ID,
        accuracy_datasets[2],
        args.report_dir,
    )

    # Score the dataset
    score, n_repeats = scorer.score()
    logging.info(f"LCB Pass@1 Score ({n_repeats} repeats): {score}")


def run_main(args):
    """Main function to run the example."""
    # Setup paths
    tmp_dir = Path(tempfile.mkdtemp(prefix="sglang_manual_example_"))
    num_repeats = args.num_repeats

    client = None
    try:
        # Always generate GPQA diamond dataset
        logging.info("Generating GPQA diamond dataset...")
        gpqa_dataset = GPQA.get_dataloader(
            num_repeats=num_repeats, transforms=GPQA.PRESETS.gptoss_sglang()
        )
        gpqa_dataset.load()
        # Always generate AIME25 dataset
        logging.info("Generating AIME25 dataset...")
        aime25_dataset = AIME25.get_dataloader(
            num_repeats=num_repeats, transforms=AIME25.PRESETS.gptoss_sglang()
        )
        aime25_dataset.load()
        logging.info(f"Dataset loaded with {aime25_dataset.num_samples()} samples")
        # Generate LCB Dataset
        logging.info("Generating LCB dataset...")
        lcb_dataset = LiveCodeBench.get_dataloader(
            num_repeats=num_repeats,
            transforms=LiveCodeBench.PRESETS.gptoss_sglang(),
        )
        lcb_dataset.load()
        logging.info(f"Dataset loaded with {lcb_dataset.num_samples()} samples")

        # Step 4: Create SGLang client
        logging.info(f"Creating SGLang client for endpoint: {SGLANG_ENDPOINT}")
        client = create_sglang_client(tmp_dir)
        sample_issuer = HttpClientSampleIssuer(client)

        # Step 5: Run benchmark session
        logging.info("Starting benchmark session...")
        run_benchmark_session(
            [gpqa_dataset, aime25_dataset, lcb_dataset], sample_issuer, args
        )

        logging.info(f"\nBenchmark complete! Results saved to {args.report_dir}/")

    finally:
        # Cleanup
        if client is not None:
            client.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    """Main entry point for the manual example."""
    parser = argparse.ArgumentParser(
        description="SGLang Accuracy Suite Benchmark",
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
        default="sglang_accuracy_report",
        help="Directory to save benchmark reports (default: sglang_accuracy_report)",
    )

    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Number of repeats to run (default: 1)",
    )

    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("SGLang Accuracy Suite Benchmark")
    logging.info("=" * 60)
    logging.info("\nConfiguration:")
    logging.info(f"  SGLang endpoint: {SGLANG_ENDPOINT}")
    logging.info(f"  Report directory: {args.report_dir}\n")

    # Run the main function
    run_main(args)


if __name__ == "__main__":
    main()
