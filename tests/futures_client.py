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

"""Futures-based wrapper for testing HTTPEndpointClient."""

import asyncio
import concurrent.futures
import logging

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient

logger = logging.getLogger(__name__)


class FuturesHttpClient(HTTPEndpointClient):
    """
    HTTPEndpointClient with futures-based API for testing.
    Returns thread-safe futures from issue_query() that can be awaited from any context.
    """

    def __init__(
        self,
        config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
    ):
        # Auto-starts with own event loop thread (loop=None)
        super().__init__(config, aiohttp_config, zmq_config)

        # Start response handler on client's loop
        self._pending: dict[str | int, concurrent.futures.Future] = {}
        self._handler_future = asyncio.run_coroutine_threadsafe(
            self._handle_responses(), self.loop
        )
        self._is_shutting_down = False

    def issue_query(self, query: Query) -> concurrent.futures.Future[QueryResult]:
        """Issue query and return a future for the result."""
        if self._is_shutting_down:
            raise RuntimeError("Cannot issue query: client is shutting down")

        future: concurrent.futures.Future[QueryResult] = concurrent.futures.Future()
        self._pending[query.id] = future
        super().issue_query(query)
        return future

    async def _handle_responses(self):
        """Route responses to their corresponding futures."""
        while True:
            try:
                response = await self.try_receive()
                if response is None:
                    continue

                future = self._pending[response.id]
                match response:
                    case StreamChunk(is_complete=False):
                        pass  # Ignore intermediate stream chunks
                    case QueryResult(error=err) if err:
                        future.set_exception(Exception(err))
                        del self._pending[response.id]
                    case QueryResult():
                        future.set_result(response)
                        del self._pending[response.id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in response handler: {e}")

    def shutdown(self):
        """Shutdown handler and HTTP client."""
        self._is_shutting_down = True
        self._handler_future.cancel()

        while self._pending:
            _, future = self._pending.popitem()
            if not future.done():
                future.cancel()

        super().shutdown()
