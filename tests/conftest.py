"""
Pytest configuration and common fixtures for the MLPerf Inference Endpoint
Benchmarking System.
This file provides shared fixtures and configuration for all tests.
"""

import asyncio

# Add src to path for imports
import sys
from pathlib import Path
from typing import Any

import pytest
from inference_endpoint.testing.echo_server import EchoServer
from inference_endpoint.dataset_manager.dataloader import DataLoader

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "environment": "test",
        "logging": {"level": "INFO", "output": "console"},
        "performance": {
            "max_concurrent_requests": 1000,
            "buffer_size": 10000,
            "memory_limit": "4GB",
        },
    }


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    from inference_endpoint.core.types import Query

    return Query(
        prompt="Hello, how are you?",
        model="gpt-3.5-turbo",
        max_tokens=50,
        temperature=0.7,
    )


@pytest.fixture
def event_loop() -> asyncio.AbstractEventLoop:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory containing test data."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("test_artifacts")


@pytest.fixture(scope="class")
def mock_http_echo_server():
    """
    Mock HTTP server that echoes back the request payload in the appropriate format.

    This fixture creates a real HTTP server running on localhost that captures
    any HTTP request and returns the request payload as the response. Useful for
    testing HTTP clients with real network calls but controlled responses.

    Returns:
        A server instance with URL.

    Example:
        def test_my_http_client(mock_http_echo_server):
            server = mock_http_echo_server
            # Make real HTTP requests to server.url
            # The response will contain the exact payload you sent
    """

    # Create and start the server with dynamic port allocation (port=0)
    server = EchoServer(port=0)
    server.start()

    try:
        yield server
    finally:
        server.stop()

@pytest.fixture
def dummy_dataloader():
    class DummyDataLoader(DataLoader):
        def __init__(self, n_samples: int = 100):
            super().__init__(None)
            self.n_samples = n_samples

        def load_sample(self, sample_index: int) -> int:
            assert sample_index >= 0 and sample_index < self.n_samples
            return sample_index
    return DummyDataLoader()
