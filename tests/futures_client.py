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

"""Futures-based wrapper for testing HTTPEndpointClient."""

import asyncio
import concurrent.futures
import logging

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient

logger = logging.getLogger(__name__)


class FuturesHttpClient(HTTPEndpointClient):
    """
    HTTPEndpointClient with futures-based API for testing.
    Returns thread-safe futures from issue() that can be awaited from any context.
    """

    def __init__(
        self,
        config: HTTPClientConfig,
    ):
        # Auto-starts with own event loop thread (loop=None)
        super().__init__(config)

        # Start response handler on client's loop
        self._pending: dict[str | int, concurrent.futures.Future] = {}
        assert (
            self.loop is not None
        ), "Client loop should be initialized by parent __init__"
        self._handler_future = asyncio.run_coroutine_threadsafe(
            self._handle_responses(), self.loop
        )
        self._is_shutting_down = False

    # TODO (vir): fix this type ignore since the base class doesn't have a return value
    def issue(self, query: Query) -> concurrent.futures.Future[QueryResult]:  # type: ignore[override]
        """Issue query and return a future for the result."""
        if self._is_shutting_down:
            raise RuntimeError("Cannot issue query: client is shutting down")

        future: concurrent.futures.Future[QueryResult] = concurrent.futures.Future()
        self._pending[query.id] = future
        super().issue(query)
        return future

    async def _handle_responses(self):
        """Route responses to their corresponding futures."""
        while True:
            try:
                response = await self.recv()
                if response is None:
                    break  # None signals transport closed - exit handler

                match response:
                    case StreamChunk(is_complete=False):
                        # Intermediate stream chunk - future stays pending
                        pass

                    case QueryResult(error=err) if err:
                        # Error response - pop and reject future
                        if future := self._pending.pop(response.id, None):
                            future.set_exception(Exception(err))
                        else:
                            logger.debug(f"Error for unknown request ID: {response.id}")

                    case QueryResult():
                        # Success response - pop and resolve future
                        if future := self._pending.pop(response.id, None):
                            future.set_result(response)
                        else:
                            logger.debug(
                                f"Response for unknown request ID: {response.id}"
                            )

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
