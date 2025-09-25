"""
Pytest configuration and common fixtures for the MLPerf Inference Endpoint
Benchmarking System.
This file provides shared fixtures and configuration for all tests.
"""

import hashlib

# Add src to path for imports
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from inference_endpoint.dataset_manager.dataloader import (
    DataLoader,
    HFDataLoader,
    PickleReader,
)
from inference_endpoint.testing.echo_server import EchoServer

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Register the profiling plugin
pytest_plugins = ["src.inference_endpoint.profiling.pytest_profiling_plugin"]


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
def create_test_query():
    """Factory for creating test queries with various configurations.

    This is a flexible factory that can create queries with different sizes,
    streaming modes, and custom IDs for testing purposes.

    Examples:
        # Create a simple query
        query = create_test_query()

        # Create a large query for performance testing
        large_query = create_test_query(prompt_size=1000, stream=False)

        # Create a streaming query with custom ID
        streaming_query = create_test_query(stream=True, query_id="test-123")
    """
    from inference_endpoint.core.types import Query

    def _create_query(
        prompt_size: int = 100,
        stream: bool = False,
        query_id: str | None = None,
    ):
        """Create a test query with specified parameters."""
        prompt = "a" * prompt_size  # Simple prompt of specified size
        return Query(
            id=query_id or str(uuid.uuid4()),
            data={
                "model": "test-model",
                "prompt": prompt,
                "stream": stream,
            },
        )

    return _create_query


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
    except Exception as e:
        raise RuntimeError(f"Mock Echo Server error: {e}") from e
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

        def num_samples(self) -> int:
            return self.n_samples

    return DummyDataLoader()


@pytest.fixture
def ds_pickle_dataset_path():
    """
    Returns the path to the ds_samples.pkl file.
    """
    return "tests/datasets/ds_samples.pkl"


@pytest.fixture
def ds_pickle_reader(ds_pickle_dataset_path):
    """
    Returns a PickleReader object for the ds_samples.pkl file.
    """

    def parser(row):
        ret = {}
        for column in row.index.to_list():
            ret[column] = row[column]
        return ret

    return PickleReader(ds_pickle_dataset_path, parser=parser)


@pytest.fixture
def hf_squad_dataset_path():
    """
    Returns the path to the squad dataset.
    """
    return "tests/datasets/squad_pruned"


@pytest.fixture
def hf_squad_dataset(hf_squad_dataset_path):
    """
    Returns a HFDataLoader object for the squad dataset.
    """
    return HFDataLoader(hf_squad_dataset_path, format="arrow")


def get_test_socket_path(tmp_path, test_name, suffix=""):
    """Generate a short socket name using hash to avoid path length limits.

    This avoids Unix domain socket path length limits (typically 108 chars)
    when pytest runs tests in parallel with long temporary directory paths.

    Args:
        tmp_path: The pytest tmp_path fixture
        test_name: A unique identifier for the test
        suffix: Optional suffix to append (e.g., "_req", "_resp")

    Returns:
        A short IPC socket path like "ipc:///tmp/.../a1b2c3d4_req"
    """
    # Create a hash of the test name to ensure uniqueness
    hash_val = hashlib.md5(test_name.encode()).hexdigest()[:8]
    # Combine with a short suffix
    name = f"{hash_val}{suffix}"
    return f"ipc://{tmp_path}/{name}"
