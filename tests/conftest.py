"""
Pytest configuration and common fixtures for the MLPerf Inference Endpoint
Benchmarking System.
This file provides shared fixtures and configuration for all tests.
"""

import random
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any

import inference_endpoint.metrics as metrics
import pytest
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.dataset_manager.dataloader import (
    HFDataLoader,
    PickleReader,
)
from inference_endpoint.testing.echo_server import EchoServer

# Add src to path for imports
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


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory containing test data."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("test_artifacts")


@pytest.fixture(scope="function")
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


@pytest.fixture
def sample_uuids():
    """Generate deterministic UUID strings from integers for testing.

    Returns:
        A function that takes an integer and returns a 32-character hexadecimal UUID string.

    Example:
        uuid1 = sample_uuids(1)  # Returns "00000000000000000000000000000001"
        uuid2 = sample_uuids(2)  # Returns "00000000000000000000000000000002"
    """

    def _generate_uuid(n: int) -> str:
        return uuid.UUID(int=n).hex

    return _generate_uuid


@pytest.fixture
def events_db(tmp_path, sample_uuids):
    """Returns a sample in-memory sqlite database for events.
    This database contains events for 3 sent queries, but only 2 are completed. The 3rd query has no 'received' events.
    """
    test_db = str(tmp_path / f"test_events_{uuid.uuid4().hex}.db")
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER)"
    )

    # Use deterministic UUIDs for testing
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    events = [
        (uuid1, "request_sent", 10000),
        (uuid2, "request_sent", 10003),
        (uuid1, "first_chunk_received", 10010),
        (uuid2, "first_chunk_received", 10190),
        (uuid1, "non_first_chunk_received", 10201),
        (uuid3, "request_sent", 10202),
        (uuid1, "non_first_chunk_received", 10203),
        (uuid2, "non_first_chunk_received", 10210),
        (uuid1, "non_first_chunk_received", 10211),
        (uuid1, "complete", 10211),
        (uuid2, "non_first_chunk_received", 10214),
        (uuid2, "non_first_chunk_received", 10217),
        (uuid2, "non_first_chunk_received", 10219),
        (uuid2, "complete", 10219),
    ]
    cur.executemany(
        "INSERT INTO events (sample_uuid, event_type, timestamp_ns) VALUES (?, ?, ?)",
        events,
    )
    conn.commit()
    yield test_db

    cur.close()
    conn.close()
    Path(test_db).unlink()


@pytest.fixture
def runtime_settings():
    return RuntimeSettings(
        metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=1000,
        n_samples_from_dataset=100,
        n_samples_to_issue=100,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
    )
