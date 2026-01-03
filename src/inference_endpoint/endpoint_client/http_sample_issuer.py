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

"""LoadGenerator integration for HTTPEndpointClient."""

import asyncio
import logging

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.load_generator import SampleIssuer
from inference_endpoint.load_generator.sample import Sample, SampleEventHandler
from inference_endpoint.profiling import profile

logger = logging.getLogger(__name__)


class HttpClientSampleIssuer(SampleIssuer):
    """
    SampleIssuer interface for HTTPEndpointClient.
    Routes completed responses to SampleEventHandler.

    Usage:
        # Create HTTP client and sample issuer - auto-initializes
        client = HTTPEndpointClient(config)
        issuer = HttpClientSampleIssuer(client)

        # Issue samples
        issuer.issue(sample)

        # shutdown() is optional - only needed for early exit
    """

    def __init__(
        self,
        http_client: HTTPEndpointClient,
    ):
        super().__init__()
        self.http_client = http_client

        # Start response handler task to route completed responses back to SampleEventHandler
        assert self.http_client.loop is not None
        self._response_task = asyncio.run_coroutine_threadsafe(
            self._handle_responses(), self.http_client.loop
        )

    @profile
    async def _handle_responses(self):
        """Route completed responses to SampleEventHandler."""
        while True:
            try:
                # TODO(vir): consider using recv() + drain
                match response := await self.http_client.recv():
                    case StreamChunk(is_complete=False):
                        # NOTE(vir): is_complete=True should not be received, QueryResult is expected instead
                        SampleEventHandler.stream_chunk_complete(response)

                    case QueryResult(error=err):
                        SampleEventHandler.query_result_complete(response)
                        if err is not None:
                            logger.error(f"Error in request {response.id}: {err}")

                    case None:
                        # Transport closed or shutdown
                        break

                    case _:
                        raise ValueError(f"Unexpected response type: {type(response)}")

            except asyncio.CancelledError:
                # Handle shutdown signal
                break
            except Exception as e:
                logger.error(f"Error in response handler: {e}", exc_info=True)
                continue

    @profile
    def issue(self, sample: Sample):
        """Issue sample to HTTP endpoint."""
        # NOTE(vir):
        # If using extra headers (e.g., Authorization), pre-cache them in
        # worker.py request-template via HttpRequestTemplate.cache_headers()
        # to avoid per-request encoding overhead at runtime.
        self.http_client.issue(Query(id=sample.uuid, data=sample.data))

    def shutdown(self):
        """
        Gracefully shutdown sample issuer.
        Will cancel the response-handler task.
        """
        self._response_task.cancel()
