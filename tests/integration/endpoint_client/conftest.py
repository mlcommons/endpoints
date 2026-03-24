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

import pytest
from inference_endpoint.endpoint_client.config import HTTPClientConfig

from tests.futures_client import FuturesHttpClient


def create_futures_client(
    url: str,
    num_workers: int = 1,
    max_connections: int = 10,
    warmup_connections: int = 0,
) -> FuturesHttpClient:
    """Helper to create a FuturesHttpClient with specific config."""
    http_config = HTTPClientConfig(
        endpoint_urls=[url],
        num_workers=num_workers,
        max_connections=max_connections,
        warmup_connections=warmup_connections,
    )
    return FuturesHttpClient(http_config)


@pytest.fixture
def futures_http_client(mock_http_echo_server):
    """Fixture that creates and manages a FuturesHttpClient instance.

    Uses mock_http_echo_server with default configuration.
    Transport context is managed internally by WorkerManager.
    Automatically handles client shutdown after test completes.
    """
    client = create_futures_client(
        url=f"{mock_http_echo_server.url}/v1/chat/completions",
    )
    yield client
    client.shutdown()
