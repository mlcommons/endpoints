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

"""Dataset replay server — replays a dataset's reference answers with simulated pacing.

Unlike :mod:`echo_server`, which returns the prompt back, this server looks the
prompt up in the benchmark dataset and returns that row's *reference answer*.
A benchmark run against it therefore scores 100% accuracy under any
response-comparing scorer, which is what makes it useful for exercising the
performance + accuracy pipeline end to end without a GPU or a real model.

Pacing is simulated so the emitted metrics are shaped like a real serving stack
rather than being uniformly zero:

* ``--ttft-ms`` — base time before the first token of a response.
* ``--tpot-ms`` — base time between subsequent tokens.
* ``--slots``   — batch capacity. Up to this many concurrent requests are served
  at the base rate; beyond it, per-token time degrades in proportion to the
  overflow, the way continuous batching degrades under load.

The degradation is what produces a non-trivial pareto curve: ``system_tps``
saturates near ``slots * 1000 / tpot_ms`` while ``tps_per_user`` falls off as
concurrency climbs, so the measurement points at different concurrency levels
are actually distinguishable from one another.

Usage::

    python -m inference_endpoint.testing.dataset_replay_server \\
        --port 8765 \\
        --dataset /tmp/pareto_ci.jsonl \\
        --prompt-key text_input --response-key ref_output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from aiohttp import web

from inference_endpoint.dataset_manager import Dataset
from inference_endpoint.dataset_manager.transforms import ColumnRemap
from inference_endpoint.openai.openai_types_gen import CreateChatCompletionRequest
from inference_endpoint.testing.echo_server import EchoServer
from inference_endpoint.utils.logging import setup_logging

logger = logging.getLogger(__name__)

#: Returned when a prompt is not found in the dataset. Kept distinctive so a
#: lookup miss shows up as an accuracy failure rather than silently scoring.
_MISS_RESPONSE = "REPLAY_LOOKUP_MISS"


class DatasetReplayServer(EchoServer):
    """Serves each prompt's reference answer, with optional simulated pacing.

    Pacing defaults to zero, which makes the server a pure lookup table that
    answers as fast as it can. :meth:`main` supplies non-zero defaults for the
    standalone server, where realistic timing is the point.
    """

    def __init__(
        self,
        *,
        answers: dict[str, str],
        ttft_ms: float = 0.0,
        tpot_ms: float = 0.0,
        slots: int = 64,
        **kwargs: object,
    ):
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._answers = answers
        self._ttft_s = ttft_ms / 1000.0
        self._tpot_s = tpot_ms / 1000.0
        self._slots = max(1, slots)
        self._inflight = 0
        self._misses = 0

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        *,
        prompt_key: str = "text_input",
        response_key: str = "ref_output",
        **kwargs: object,
    ) -> DatasetReplayServer:
        """Build a server whose answers come from a benchmark dataset file.

        Loads through :meth:`Dataset.load_from_file` rather than parsing the
        file directly, so the server accepts every format the benchmark itself
        accepts and applies the same column-remap machinery.

        Raises:
            ValueError: If no row yields both a prompt and an answer — otherwise
                every request would miss, and the run would score 0% with no
                indication of the cause.
        """
        loader = Dataset.load_from_file(
            Path(dataset_path),
            transforms=[ColumnRemap({prompt_key: "prompt", response_key: "output"})],
        )
        loader.load()

        answers: dict[str, str] = {}
        for i in range(loader.num_samples()):
            sample = loader.load_sample(i)
            prompt, answer = sample.get("prompt"), sample.get("output")
            if prompt is None or answer is None:
                continue
            answers[str(prompt)] = str(answer)

        if not answers:
            raise ValueError(
                f"{dataset_path}: no rows with both {prompt_key!r} and "
                f"{response_key!r}; every request would score as a lookup miss"
            )
        return cls(answers=answers, **kwargs)  # type: ignore[arg-type]

    @property
    def answer_count(self) -> int:
        """How many prompt -> answer pairs the server can serve."""
        return len(self._answers)

    @property
    def miss_count(self) -> int:
        """How many requests failed to match a known prompt."""
        return self._misses

    def get_response(self, request: str) -> str:
        """Return the reference answer for *request*, or the miss sentinel.

        A miss means the client sent something other than the raw dataset
        column — every response would then be the sentinel and accuracy would
        read 0% with no indication of why, so the first one is logged loudly
        with both sides of the comparison.
        """
        answer = self._answers.get(request)
        if answer is not None:
            return answer

        self._misses += 1
        if self._misses == 1:
            sample_key = next(iter(self._answers))
            logger.warning(
                "Prompt lookup MISS - the client is not sending the raw dataset column.\n"
                "  received: %r\n"
                "  a known key: %r",
                request[:400],
                sample_key[:400],
            )
        return _MISS_RESPONSE

    def _load_factor(self) -> float:
        """Per-token slowdown for the current in-flight count.

        Returns 1.0 while the batch is within capacity, then grows linearly —
        the shape that makes ``system_tps`` saturate instead of scaling forever.
        """
        return max(1.0, self._inflight / self._slots)

    async def _handle_streaming_response(
        self,
        id: str,
        request: web.Request,
        completion_request: CreateChatCompletionRequest,
        content: str,
    ) -> web.StreamResponse:
        """Stream the reference answer one whitespace token per SSE chunk.

        ``content`` arrives already resolved: the caller in :class:`EchoServer`
        has run it through :meth:`get_response`. Looking it up again here would
        search for an *answer* in a map keyed by *prompts*, miss every time, and
        silently drive accuracy to zero.

        One chunk per token is what lets the client measure TPOT; batching
        tokens into fewer chunks would collapse the inter-token timing that the
        pareto curve's interactivity axis is built from.
        """
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        model = str(completion_request.model.root)
        tokens = content.split()

        self._inflight += 1
        try:
            await asyncio.sleep(self._ttft_s * self._load_factor())

            for i, token in enumerate(tokens):
                if i > 0:
                    await asyncio.sleep(self._tpot_s * self._load_factor())
                chunk = {
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token if i == 0 else f" {token}"},
                            "finish_reason": None,
                        }
                    ],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())

            final = {
                "id": id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            await response.write(f"data: {json.dumps(final)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
        finally:
            self._inflight -= 1


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dataset replay server — replays dataset reference answers with simulated pacing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="hostname/address to bind to"
    )
    parser.add_argument("--port", type=int, default=8765, help="port to bind to")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="dataset file to replay answers from (any format Dataset accepts)",
    )
    parser.add_argument(
        "--prompt-key", default="text_input", help="dataset column holding the prompt"
    )
    parser.add_argument(
        "--response-key", default="ref_output", help="dataset column holding the answer"
    )
    parser.add_argument(
        "--ttft-ms", type=float, default=40.0, help="base time to first token, ms"
    )
    parser.add_argument(
        "--tpot-ms", type=float, default=8.0, help="base time per output token, ms"
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=64,
        help="batch capacity; per-token time degrades linearly beyond it",
    )
    return parser


def main() -> None:
    setup_logging()
    args = create_parser().parse_args()

    server = DatasetReplayServer.from_dataset(
        args.dataset,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        ttft_ms=args.ttft_ms,
        tpot_ms=args.tpot_ms,
        slots=args.slots,
        host=args.host,
        port=args.port,
    )
    logger.info(
        "Dataset replay server: %d reference answers from %s (ttft=%.1fms tpot=%.1fms slots=%d)",
        server.answer_count,
        args.dataset,
        args.ttft_ms,
        args.tpot_ms,
        args.slots,
    )
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
