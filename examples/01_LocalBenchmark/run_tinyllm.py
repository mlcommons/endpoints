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

from __future__ import annotations

import argparse
import threading

import inference_endpoint.rulesets.mlcommons.models as mlcommons_models
from inference_endpoint.config.user_config import UserConfig
from inference_endpoint.core.types import QueryResult, StreamChunk
from inference_endpoint.dataset_manager.dataloader import DataLoader
from inference_endpoint.load_generator import (
    BenchmarkSession,
    MaxThroughputScheduler,
    SampleEvent,
    SampleEventHandler,
    SampleIssuer,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.rulesets.mlcommons.rules import CURRENT
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.utils import logging

logging.set_verbosity_error()  # Suppress HuggingFace warnings


# Purely for testing, I just googled "Smallest LLM on huggingface" and found this one.
class TinyLLM:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("arnir0/Tiny-LLM")
        self.tokenizer = AutoTokenizer.from_pretrained("arnir0/Tiny-LLM")

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


class TinyDataLoader(DataLoader):
    def __init__(self):
        super().__init__(None)

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
    def __init__(self, pbar: tqdm | None = None):
        self.pbar = pbar

    def __call__(self, _):
        if isinstance(self.pbar, tqdm):
            self.pbar.update(1)

    def set_pbar(self, pbar: tqdm):
        self.pbar = pbar


class SerialSampleIssuer(SampleIssuer):
    """SampleIssuer for testing. No threading, and is blocking. Whenever issue is called,
    it performs the provided compute function, calling callbacks when necessary.

    If computation is streamed, then the compute function should be a generator, yielding
    the 'chunks' of the supposed response.

    Otherwise, the compute function should return a singular string value.
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
            query_result = QueryResult(id=sample.uuid, response_output="".join(chunks))
        else:
            response = self.compute_func(sample.data)
            query_result = QueryResult(id=sample.uuid, response_output=response)
        SampleEventHandler.query_result_complete(query_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    pbar_hook = ProgressBarHook()
    SampleEventHandler.register_hook(SampleEvent.COMPLETE, pbar_hook)

    dataloader = TinyDataLoader()
    user_config = UserConfig(
        user_metric_target=2,
        min_sample_count=100,
        min_duration_ms=10 * 1000,
        max_duration_ms=5 * 60 * 1000,
        ds_subset_size=dataloader.num_samples(),  # Otherwise it will use the size of Llama3.1's dataset
    )
    rt_settings = CURRENT.apply_user_config(
        model=mlcommons_models.Llama3_1_8b,
        user_config=user_config,
    )
    model_runner = TinyLLM()

    if args.streaming:
        issuer = SerialSampleIssuer(
            compute_func=model_runner.generate_text_streamed, streaming=True
        )
    else:
        issuer = SerialSampleIssuer(
            compute_func=model_runner.generate_text_non_streamed, streaming=False
        )
    scheduler = MaxThroughputScheduler(rt_settings, WithoutReplacementSampleOrder)

    n_total = rt_settings.total_samples_to_issue()
    with tqdm(desc="poc_full_run", total=n_total, unit="samples") as pbar:
        pbar_hook.set_pbar(pbar)
        sess = BenchmarkSession.start(
            rt_settings,
            dataloader,
            issuer,
            scheduler,
            name="poc_full_run",
            stop_sample_issuer_on_test_end=False,
            report_path="poc_full_run_report",
            tokenizer_override=model_runner.tokenizer,
        )
        sess.wait_for_test_end()
