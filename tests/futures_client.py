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

"""Futures-based wrapper for HTTPEndpointClient - Test Infrastructure."""

import asyncio
import logging

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient

logger = logging.getLogger(__name__)


class StreamingFuture(asyncio.Future):
    """Future that also exposes first chunk for streaming responses."""

    def __init__(self):
        super().__init__()
        self.first = asyncio.Future()


class FuturesHttpClient(HTTPEndpointClient):
    """
    HTTP client with futures-based API for async contexts.
    FuturesHttpClient will run on the current event loop.

    This is a test utility that wraps HTTPEndpointClient to provide
    a simpler futures-based interface for testing purposes.

    Final result is available via the future object.
    """

    def __init__(self, *args, **kwargs):
        """Initialize FuturesHttpClient.

        Args:
            *args: Passed to HTTPEndpointClient.
            **kwargs: Passed to HTTPEndpointClient.
        """
        super().__init__(*args, **kwargs)
        self._pending_futures: dict[str | int, asyncio.Future] = {}
        self._response_handler_task: asyncio.Task | None = None

    async def async_start(self):
        """Start HTTP client and response handler."""
        try:
            # Set loop to current running loop
            self.loop = asyncio.get_running_loop()
            await super().async_start()

            # Schedule response handler on current loop
            self._response_handler_task = asyncio.create_task(self._handle_responses())
        except Exception as e:
            logger.exception(f"Failed to start FuturesHttpClient: {e}")

            # Cleanup on failure
            await self.async_shutdown()
            raise e

    async def issue_query(self, query: Query) -> asyncio.Future:
        """Issue query and return future for response."""
        future = (
            StreamingFuture() if query.data.get("stream", False) else asyncio.Future()
        )
        self._pending_futures[query.id] = future

        try:
            await super().issue_query_async(query)
        except Exception as e:
            # Set exception on future and clean up on failure
            logger.exception(f"Failed to send query {query.id}: {e}")
            self._set_future_exception(future, e)
            self._pending_futures.pop(query.id, None)
            raise

        return future

    def _set_future_exception(self, future: asyncio.Future, exception: Exception):
        """Set exception on future and streaming first chunk if applicable."""
        if not future.done():
            future.set_exception(exception)
        if isinstance(future, StreamingFuture) and not future.first.done():
            future.first.set_exception(exception)

    async def _handle_responses(self):
        """Handle responses and complete futures."""
        while True:
            try:
                response = await self.get_ready_responses_async()

                # Handle timeout (no response available)
                if response is None:
                    continue

                future = self._pending_futures.get(response.id)
                if not future:
                    logger.warning(
                        f"Received response for unknown query: {response.id}"
                    )
                    continue

                if future.done():
                    logger.warning(
                        f"Received duplicate response for query: {response.id}"
                    )
                    continue

                # Handle different response types
                match response:
                    case StreamChunk(response_chunk=chunk, is_complete=False):
                        future.first.set_result(chunk)

                    case StreamChunk(is_complete=True):
                        raise NotImplementedError(
                            "StreamChunk(is_complete=True) should not be received, QueryResult is expected instead"
                        )

                    case QueryResult(error=err) if err is not None:
                        self._set_future_exception(future, Exception(err))
                        self._pending_futures.pop(response.id)

                    case QueryResult():
                        future.set_result(response)
                        if (
                            isinstance(future, StreamingFuture)
                            and not future.first.done()
                        ):
                            # For streaming futures with no first chunk, set first chunk to empty string
                            future.first.set_result("")

                        self._pending_futures.pop(response.id)

                    case _:
                        logger.error(f"Unexpected response type: {type(response)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in response handler: {e}")

    async def async_shutdown(self):
        """Async shutdown for external loop usage."""
        # Cancel response handler task first
        if self._response_handler_task:
            self._response_handler_task.cancel()
            try:
                await asyncio.wait_for(self._response_handler_task, timeout=0.2)
            except (TimeoutError, asyncio.CancelledError):
                pass

        # Cancel any pending futures
        for future in self._pending_futures.values():
            if not future.done():
                future.cancel()
        self._pending_futures.clear()

        await super().async_shutdown()

    def start(self):
        """Synchronous start is not supported for FuturesHttpClient.

        Raises:
            RuntimeError: Always raised to prevent improper usage.

        Use async_start() instead:
            await client.async_start()
        """
        raise RuntimeError(
            "FuturesHttpClient does not support synchronous start(). "
            "Use 'await client.async_start()' instead."
        )

    def shutdown(self):
        """Synchronous shutdown is not supported for FuturesHttpClient.

        Raises:
            RuntimeError: Always raised to prevent improper usage.

        Use async_shutdown() instead:
            await client.async_shutdown()
        """
        raise RuntimeError(
            "FuturesHttpClient does not support synchronous shutdown(). "
            "Use 'await client.async_shutdown()' instead."
        )
