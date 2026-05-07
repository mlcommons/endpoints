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

"""SampleIssuer implementation wrapping HTTPEndpointClient.

Thin adapter: delegates issue/recv/shutdown to the underlying HTTP client.
The BenchmarkSession owns the response receive loop — this class does NOT
run its own _handle_responses coroutine.
"""

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient


class HttpClientSampleIssuer:
    """SampleIssuer wrapping an HTTPEndpointClient.

    Satisfies the SampleIssuer protocol from load_generator.session.

    Usage:
        client = await HTTPEndpointClient.create(config, loop)
        issuer = HttpClientSampleIssuer(client)

        issuer.issue(query)                    # sync ZMQ push
        response = await issuer.recv()         # async ZMQ recv
        issuer.shutdown()                      # no-op (client shutdown called separately)
    """

    def __init__(self, http_client: HTTPEndpointClient):
        self.http_client = http_client

    def issue(self, query: Query) -> None:
        """Issue query to HTTP endpoint. Non-blocking (ZMQ push)."""
        self.http_client.issue(query)

    async def recv(self) -> QueryResult | StreamChunk | None:
        """Wait for next response. Returns None when transport is closed."""
        return await self.http_client.recv()

    def shutdown(self) -> None:
        """No-op. HTTPEndpointClient.shutdown() is called separately by the caller."""
        pass
