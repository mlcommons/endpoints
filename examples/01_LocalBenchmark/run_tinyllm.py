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

from __future__ import annotations

import argparse
import threading

import inference_endpoint.config.rulesets.mlcommons.models as mlcommons_models
from inference_endpoint.config.rulesets.mlcommons.rules import CURRENT
from inference_endpoint.config.user_config import UserConfig
from inference_endpoint.core.types import QueryResult, StreamChunk
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.load_generator import (
    BenchmarkSession,
    MaxThroughputScheduler,
    SampleEvent,
    SampleEventHandler,
    SampleIssuer,
    WithoutReplacementSampleOrder,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.utils import logging

logging.set_verbosity_error()  # Suppress HuggingFace warnings


# Example: TinyLlama for local benchmarking
# This demonstrates using a small model from HuggingFace for testing
# TinyLlama-1.1B-Chat is a popular small model (~4M downloads/month)
# Note: This is a minimal example - for production use, consider larger models
class TinyLLM:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

    def generate_text_streamed(
        self, prompt, max_length=512, temperature=1, top_k=50, top_p=0.95
    ):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        streamer = TextIteratorStreamer(self.tokenizer)
        thread = threading.Thread(
            target=self.model.generate,
            args=(inputs,),
            kwargs={
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": True,
                "streamer": streamer,
            },
        )
        thread.start()

        yield from streamer

        thread.join()  # No timeout here, if the streamer loop finished, the thread is definitely done

    def generate_text_non_streamed(
        self, prompt, max_length=512, temperature=1, top_k=50, top_p=0.95
    ):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


class TinyDataset(Dataset):
    """Example Dataset with hardcoded prompts for testing.

    In production, you would load prompts from a dataset file.
    """

    def __init__(self):
        super().__init__(None)

        # Sample prompts for testing - replace with your dataset
        self.samples = [
            "Can you describe what a proof of concept is?",
            "What is the capital of France?",
            "A Python generator is",
            "Python was created by who?",
            "What was Python named after?",
            "147.93 + 42.12 = ?",
            "Write a quine in Python that includes this prompt but backwards.",
            "What is the 17th root of the 87th Fibonacci number?",
        ]

    def load_sample(self, index: int):
        return self.samples[index]

    def num_samples(self):
        return len(self.samples)


class ProgressBarHook:
    """Hook to update progress bar on sample completion.

    This demonstrates how to use event hooks to monitor benchmark progress.
    """

    def __init__(self, pbar: tqdm | None = None):
        self.pbar = pbar

    def __call__(self, _):
        if isinstance(self.pbar, tqdm):
            self.pbar.update(1)

    def set_pbar(self, pbar: tqdm):
        self.pbar = pbar


class SerialSampleIssuer(SampleIssuer):
    """Example SampleIssuer for local model benchmarking.

    This is a synchronous, blocking issuer suitable for local testing.
    For production endpoints, use the HTTPClient from endpoint_client module.

    If computation is streamed, the compute function should be a generator yielding
    response chunks. Otherwise, it should return a single string response.
    """

    def __init__(self, compute_func=None, streaming: bool = True):
        if compute_func is None:
            self.compute_func = lambda x: x  # If streaming, assumes x is Iterable[str]
        else:
            self.compute_func = compute_func

        self.streaming = streaming

    def issue(self, sample):
        if self.streaming:
            first = True
            chunks = []
            for chunk in self.compute_func(sample.data):
                chunks.append(str(chunk))
                stream_chunk = StreamChunk(
                    id=sample.uuid,
                    metadata={"first_chunk": first},
                    response_chunk=chunk,
                )
                SampleEventHandler.stream_chunk_complete(stream_chunk)
                first = False
            query_result = QueryResult(id=sample.uuid, response_output=chunks)
        else:
            response = self.compute_func(sample.data)
            query_result = QueryResult(id=sample.uuid, response_output=response)
        SampleEventHandler.query_result_complete(query_result)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run local benchmark with TinyLLM model"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming mode for TTFT metrics",
    )
    parser.add_argument(
        "--dump-events-log",
        action="store_true",
        help="Dump the events to a log file",
    )
    parser.add_argument(
        "--total-sample-count",
        type=int,
        help="Total number of samples to issue (Debug only)",
    )
    args = parser.parse_args()

    # Set up progress bar hook to monitor sample completion
    pbar_hook = ProgressBarHook()
    SampleEventHandler.register_hook(SampleEvent.COMPLETE, pbar_hook)

    # Initialize dataset with sample prompts
    dataset = TinyDataset()

    # Configure benchmark runtime settings
    # Note: Using MLCommons ruleset for configuration
    user_config = UserConfig(
        user_metric_target=2,  # Target QPS baseline
        min_sample_count=100,  # Minimum samples to issue
        min_duration_ms=10 * 1000,  # 10 seconds minimum
        max_duration_ms=5 * 60 * 1000,  # 5 minutes maximum
        total_sample_count=args.total_sample_count if args.total_sample_count else None,
        ds_subset_size=dataset.num_samples(),  # Use all available samples
    )

    # Apply user config with model specifications
    rt_settings = CURRENT.apply_user_config(
        model=mlcommons_models.Llama3_1_8b,  # Model specification for ruleset
        user_config=user_config,
    )

    # Initialize the model
    model_runner = TinyLLM()

    # Create sample issuer based on streaming preference
    if args.streaming:
        issuer = SerialSampleIssuer(
            compute_func=model_runner.generate_text_streamed, streaming=True
        )
    else:
        issuer = SerialSampleIssuer(
            compute_func=model_runner.generate_text_non_streamed, streaming=False
        )

    # Use max throughput scheduler for offline benchmarking
    scheduler = MaxThroughputScheduler(rt_settings, WithoutReplacementSampleOrder)

    # Run the benchmark session
    n_total = rt_settings.total_samples_to_issue()
    with tqdm(desc="Local Benchmark", total=n_total, unit="samples") as pbar:
        pbar_hook.set_pbar(pbar)
        sess = BenchmarkSession.start(
            rt_settings,
            dataset,
            issuer,
            scheduler,
            name="tinyllm_benchmark",
            report_dir="tinyllm_benchmark_report",
            tokenizer_override=model_runner.tokenizer,
            dump_events_log=args.dump_events_log,
        )
        sess.wait_for_test_end()

    print("\nBenchmark complete! Results saved to tinyllm_benchmark_report/")
