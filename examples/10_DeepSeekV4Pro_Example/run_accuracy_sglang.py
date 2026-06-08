# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-only benchmark for DeepSeek-V4-Pro via SGLang /v1/chat/completions.

Mirrors examples/04_GPTOSS120B_Example/run.py but uses the DeepSeek-V4 chat path
(api_type=openai) instead of pre-tokenized /generate.
"""

from __future__ import annotations

import argparse
import logging
import os
import random

from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    APIType,
    LoadPattern,
    LoadPatternType,
    ModelParams,
    StreamingMode,
)
from inference_endpoint.dataset_manager import Dataset, EmptyDataset
from inference_endpoint.dataset_manager.predefined.aime25 import AIME25
from inference_endpoint.dataset_manager.predefined.gpqa import GPQA
from inference_endpoint.dataset_manager.predefined.livecodebench import LiveCodeBench
from inference_endpoint.endpoint_client.config import HTTPClientConfig
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

DEFAULT_MAX_NEW_TOKENS = 50_000


class ProgressBarHook:
    def __init__(self, pbar: tqdm | None = None):
        self.pbar = pbar

    def __call__(self, _):
        if isinstance(self.pbar, tqdm):
            self.pbar.update(1)

    def set_pbar(self, pbar: tqdm):
        self.pbar = pbar


def create_sglang_client(endpoint_url: str, num_workers: int) -> HTTPEndpointClient:
    http_config = HTTPClientConfig(
        endpoint_urls=[endpoint_url.rstrip("/")],
        num_workers=num_workers,
        api_type=APIType.OPENAI,
    )
    return HTTPEndpointClient(http_config)


def run_benchmark_session(
    accuracy_datasets: list[Dataset], issuer: HttpClientSampleIssuer, args
) -> None:
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

    scheduler = MaxThroughputScheduler(rt_settings, WithoutReplacementSampleOrder)
    n_total = sum(
        dataset.num_samples() * dataset.repeats for dataset in accuracy_datasets
    )

    with tqdm(
        desc="DeepSeek-V4-Pro SGLang Accuracy Benchmark",
        total=n_total,
        unit="samples",
    ) as pbar:
        pbar_hook.set_pbar(pbar)
        sess = BenchmarkSession.start(
            rt_settings,
            EmptyDataset(),
            issuer,
            scheduler,
            accuracy_datasets=accuracy_datasets,
            name="deepseek_v4_pro_sglang_accuracy_benchmark",
            report_dir=args.report_dir,
            dump_events_log=True,
            max_shutdown_timeout_s=None,
        )
        sess.wait_for_test_end()

    scorer = PassAt1Scorer(
        GPQA.DATASET_ID,
        accuracy_datasets[0],
        args.report_dir,
        extractor=ABCDExtractor,
    )
    score, n_repeats = scorer.score()
    logging.info("GPQA Pass@1 Score (%d repeats): %s", n_repeats, score)

    scorer = PassAt1Scorer(
        AIME25.DATASET_ID,
        accuracy_datasets[1],
        args.report_dir,
        extractor=BoxedMathExtractor,
        ground_truth_column="answer",
    )
    score, n_repeats = scorer.score()
    logging.info("AIME25 Pass@1 Score (%d repeats): %s", n_repeats, score)

    scorer = LiveCodeBenchScorer(
        LiveCodeBench.DATASET_ID,
        accuracy_datasets[2],
        args.report_dir,
    )
    score, n_repeats = scorer.score()
    logging.info("LCB Pass@1 Score (%d repeats): %s", n_repeats, score)


def run_main(args) -> None:
    model_params = ModelParams(
        name=args.model_name,
        temperature=1.0,
        max_new_tokens=args.max_new_tokens,
        top_k=-1,
        top_p=1.0,
        streaming=StreamingMode.ON,
    )

    client = None
    try:
        logging.info("Generating GPQA diamond dataset...")
        gpqa_dataset = GPQA.get_dataloader(
            num_repeats=args.num_repeats,
            transforms=GPQA.PRESETS.deepseek_v4(),
            force_regenerate=args.force_regenerate,
        )
        gpqa_dataset.load(api_type=APIType.OPENAI, model_params=model_params)

        logging.info("Generating AIME25 dataset...")
        aime25_dataset = AIME25.get_dataloader(
            num_repeats=args.num_repeats,
            transforms=AIME25.PRESETS.deepseek_v4(),
            force_regenerate=args.force_regenerate,
        )
        aime25_dataset.load(api_type=APIType.OPENAI, model_params=model_params)

        logging.info("Generating LiveCodeBench dataset...")
        lcb_dataset = LiveCodeBench.get_dataloader(
            num_repeats=args.num_repeats,
            transforms=LiveCodeBench.PRESETS.deepseek_v4(),
            force_regenerate=args.force_regenerate,
        )
        lcb_dataset.load(api_type=APIType.OPENAI, model_params=model_params)

        logging.info("Creating SGLang client for endpoint: %s", args.endpoint_url)
        client = create_sglang_client(args.endpoint_url, args.num_workers)
        sample_issuer = HttpClientSampleIssuer(client)

        logging.info("Starting accuracy benchmark session...")
        run_benchmark_session(
            [gpqa_dataset, aime25_dataset, lcb_dataset], sample_issuer, args
        )
        logging.info("Benchmark complete! Results saved to %s/", args.report_dir)
    finally:
        if client is not None:
            client.shutdown()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="DeepSeek-V4-Pro SGLang accuracy benchmark (GPQA + AIME25 + LCB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("SGLANG_ENDPOINT_URL", "http://localhost:30000"),
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4-Pro"),
    )
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--min-duration", type=int, default=10)
    parser.add_argument("--max-duration", type=int, default=3600)
    parser.add_argument(
        "--report-dir",
        type=str,
        default="results/sglang_deepseek_v4_pro_accuracy",
    )
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)),
    )
    args = parser.parse_args()
    run_main(args)


if __name__ == "__main__":
    main()
