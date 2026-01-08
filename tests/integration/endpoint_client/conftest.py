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

"""Shared fixtures for endpoint client integration tests."""

from pathlib import Path

import pytest
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)

from tests.futures_client import FuturesHttpClient
from tests.test_helpers import get_test_socket_path


def create_futures_client(
    url: str,
    tmp_path: Path,
    prefix: str,
    num_workers: int = 1,
) -> FuturesHttpClient:
    """Helper to create a FuturesHttpClient with specific config.

    Args:
        url: The endpoint URL to connect to
        tmp_path: pytest tmp_path fixture for creating unique socket paths
        prefix: Unique prefix for socket paths (typically test name)
        num_workers: Number of worker processes (default: 1)

    Returns:
        FuturesHttpClient: Configured client ready to use
    """
    http_config = HTTPClientConfig(
        endpoint_url=url,
        num_workers=num_workers,
    )

    zmq_kwargs = {
        "zmq_request_queue_prefix": get_test_socket_path(tmp_path, prefix, "_req"),
        "zmq_response_queue_addr": get_test_socket_path(tmp_path, prefix, "_resp"),
        "zmq_readiness_queue_addr": get_test_socket_path(tmp_path, prefix, "_ready"),
    }

    zmq_config = ZMQConfig(**zmq_kwargs)
    aiohttp_config = AioHttpConfig()

    return FuturesHttpClient(http_config, aiohttp_config, zmq_config)


@pytest.fixture
def futures_http_client(mock_http_echo_server, tmp_path):
    """Fixture that creates and manages a FuturesHttpClient instance.

    Uses mock_http_echo_server with default configuration.
    Automatically handles client shutdown after test completes.

    Usage:
        async def test_something(self, futures_http_client):
            future = futures_http_client.issue_query(query)
            result = await asyncio.wrap_future(future)
    """
    client = create_futures_client(
        url=f"{mock_http_echo_server.url}/v1/chat/completions",
        tmp_path=tmp_path,
        prefix="futures_client",
    )
    yield client
    client.shutdown()
