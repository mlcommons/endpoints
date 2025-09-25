"""Shared fixtures for endpoint client integration tests."""

import pytest_asyncio
from inference_endpoint.endpoint_client.futures_client import FuturesHttpClient


@pytest_asyncio.fixture
async def futures_http_client(request):
    """Fixture that creates, starts, and manages a FuturesHttpClient instance.

    This fixture expects the test to provide configs via a `client_config` fixture
    that returns (http_config, aiohttp_config, zmq_config).

    The fixture will:
    - Create the client
    - Start it
    - Yield the started client
    - Properly shut it down after the test

    Usage in test class:
        @pytest.fixture
        def client_config(self, mock_http_echo_server, tmp_path):
            http_config = HTTPClientConfig(...)
            aiohttp_config = AioHttpConfig()
            zmq_config = ZMQConfig(...)
            return http_config, aiohttp_config, zmq_config

        async def test_something(self, futures_http_client):
            # Client is already started and ready to use
            future = await futures_http_client.issue_query(query)
    """
    # Get the client_config fixture from the test
    http_config, aiohttp_config, zmq_config = request.getfixturevalue("client_config")

    # Create client with running event loop
    client = FuturesHttpClient(http_config, aiohttp_config, zmq_config)

    try:
        # Start the client
        await client.async_start()
        # Yield to test
        yield client
    finally:
        await client.async_shutdown()
