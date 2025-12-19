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

"""Shared fixtures for endpoint client integration tests."""

import pytest

from tests.futures_client import FuturesHttpClient


@pytest.fixture
def futures_http_client(request):
    """Fixture that creates and manages a FuturesHttpClient instance.

    This fixture expects the test to provide configs via a `client_config` fixture
    that returns (http_config, aiohttp_config, zmq_config).

    Usage in test class:
        @pytest.fixture
        def client_config(self, mock_http_echo_server, tmp_path):
            http_config = HTTPClientConfig(...)
            aiohttp_config = AioHttpConfig()
            zmq_config = ZMQConfig(...)
            return http_config, aiohttp_config, zmq_config

        async def test_something(self, futures_http_client):
            future = futures_http_client.issue_query(query)
            result = future.result(timeout=5)
    """
    http_config, aiohttp_config, zmq_config = request.getfixturevalue("client_config")

    client = FuturesHttpClient(http_config, aiohttp_config, zmq_config)
    yield client
    client.shutdown()
