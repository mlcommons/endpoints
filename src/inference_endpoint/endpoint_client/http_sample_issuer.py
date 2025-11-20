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

"""LoadGenerator integration for HTTPEndpointClient."""

import asyncio
import logging
import threading

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.load_generator import SampleIssuer
from inference_endpoint.load_generator.sample import Sample, SampleEventHandler
from inference_endpoint.profiling import profile

logger = logging.getLogger(__name__)


class HttpClientSampleIssuer(SampleIssuer):
    """SampleIssuer using HTTPEndpointClient with async response handling.

    Responsibilities:
    - Send queries via HTTP client
    - Route responses to appropriate sample callbacks
    """

    def __init__(
        self,
        http_client: HTTPEndpointClient,
    ):
        super().__init__()
        self.http_client = http_client

        # Task for handling responses
        self.response_task: asyncio.Task | None = None

        # Signals when all pending queries complete
        self._client_idle_event = threading.Event()

        # Shutdown flag for response handler
        self._shutdown = False

        self.n_inflight = 0

    def start(self):
        """Start response handler on the HTTP client's event loop."""
        self.response_task = asyncio.run_coroutine_threadsafe(
            self.handle_responses(), self.http_client.loop
        )

    @profile
    async def handle_responses(self):
        """Handle all responses in async loop. Routes responses to sample callbacks."""
        while not self._shutdown:
            try:
                response = await self.http_client.get_ready_responses_async()
                if response is None:  # timed out without a response
                    continue

                # Route to appropriate callback based on response type
                match response:
                    case StreamChunk(is_complete=False):
                        # NOTE(vir): is_complete=True should not be received, QueryResult is expected instead
                        SampleEventHandler.stream_chunk_complete(response)

                    case QueryResult(error=err):
                        SampleEventHandler.query_result_complete(response)
                        if err is not None:
                            logger.error(f"Error in request {response.id}: {err}")

                        self.n_inflight -= 1
                        if self.n_inflight == 0:
                            self._client_idle_event.set()

                    case _:
                        raise ValueError(f"Unexpected response type: {type(response)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in response handler: {e}", exc_info=True)
                continue

    @profile
    def issue(self, sample: Sample):
        """Issue sample to HTTP endpoint."""
        # TODO(vir):
        # we were paying idle-event cost on every issue
        # Q&D fix for now, move idle-check into HttpClient itself
        if self.n_inflight == 0:
            self._client_idle_event.clear()
        self.n_inflight += 1
        self.http_client.issue_query(
            Query(
                id=sample.uuid,
                data=sample.data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                    if sample.data.get("stream", False)
                    else "application/json",
                },
            )
        )

    def wait_for_all_complete(self, timeout: float | None = None):
        """Wait (blocking) for all pending queries to complete.

        Args:
            timeout: Maximum time to wait in seconds. None = wait forever.

        Returns:
            True if all complete, False if timeout
        """
        result = self._client_idle_event.wait(timeout=timeout)
        return result

    def shutdown(self):
        """Shutdown issuer response handler."""
        self._shutdown = True

        if self.response_task:
            self.response_task.cancel()
