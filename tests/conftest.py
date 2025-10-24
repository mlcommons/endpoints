"""
Pytest configuration and common fixtures for the MLPerf Inference Endpoint
Benchmarking System.
This file provides shared fixtures and configuration for all tests.
"""

import logging
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
    DataLoader,
    DeepSeekR1ChatCompletionDataLoader,
    HFDataLoader,
    PickleReader,
)
from inference_endpoint.load_generator.events import SampleEvent, SessionEvent
from inference_endpoint.load_generator.sample import SampleEventHandler
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
def dummy_dataloader():
    """
    Returns a DummyDataLoader object which just returns the sample index.
    """

    class DummyDataLoader(DataLoader):
        def __init__(self, n_samples: int = 100):
            """
            Initialize the DummyDataLoader.

            Args:
                n_samples (int): The number of samples to load.
            """
            super().__init__(None)
            self.n_samples = n_samples

        def load_sample(self, sample_index: int) -> int:
            """
            Load a sample from the dataset.

            Args:
                sample_index (int): The index of the sample to load.
            """
            assert sample_index >= 0 and sample_index < self.n_samples
            return sample_index

        def num_samples(self) -> int:
            """
            Returns the number of samples in the dataset.
            """
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
        ("", SessionEvent.TEST_STARTED.value, 5000),
        (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000),
        (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003),
        (uuid1, SampleEvent.FIRST_CHUNK.value, 10010),
        (uuid2, SampleEvent.FIRST_CHUNK.value, 10190),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10201),
        (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10202),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10203),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10210),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10211),
        (uuid1, SampleEvent.COMPLETE.value, 10211),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10214),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10217),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10219),
        (uuid2, SampleEvent.COMPLETE.value, 10219),
        ("", SessionEvent.TEST_ENDED.value, 10300),
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


class OracleServer(EchoServer):
    def __init__(self, file_path):
        """
        Initialize the Oracle server with a dataset and load predefined prompt-response mappings.

        The server loads chat completion samples from the specified file path using a custom parser.
        Each sample is mapped from its input prompt to its reference output, allowing subsequent
        retrieval of responses based on exact prompt matching.

        Args:
            file_path (str): Path to the dataset file containing chat completion samples
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.file_path = file_path

        def parser(x):
            """
            Extract the prompt and reference output from a dataset sample object.

            Converts a dataset sample into a dictionary with 'prompt' and 'output' keys,
            using the sample's text input as the prompt and reference output as the response.

            Returns:
                dict: A dictionary with 'prompt' and 'output' keys derived from the input sample.
            """
            return {"prompt": x.text_input, "output": x.ref_output}

        self.parser = parser
        data_loader = DeepSeekR1ChatCompletionDataLoader(
            self.file_path, parser=self.parser
        )
        data_loader.load()
        self.data = {}
        for i in range(data_loader.num_samples()):
            sample = data_loader.load_sample(i)
            self.data[sample["prompt"]] = sample["output"]

    def get_response(self, request: str) -> str:
        """
        Retrieve a predefined response for a given request from the loaded dataset.

        Returns the stored output corresponding to the input request. If no matching
        response is found, returns a default "No response found" message.

        Args:
            request (str): The input prompt to look up in the dataset.

        Returns:
            str: The matching output for the request, or a default message if not found.
        """
        return self.data.get(request, "No response found")


@pytest.fixture
def mock_http_oracle_server(ds_pickle_dataset_path):
    """
    Pytest fixture that creates and manages a mock HTTP oracle server for dataset-driven testing.

    Creates an OracleServer instance from a specified dataset pickle file, starts the server
    on a dynamically allocated port, and manages its lifecycle during testing.

    Args:
        ds_pickle_dataset_path (str): Path to the dataset pickle file containing chat completion samples

    Yields:
        OracleServer: A running mock HTTP server serving predefined responses from the dataset

    Raises:
        RuntimeError: If any errors occur during server setup or execution
    """
    # Create and start the server with dynamic port allocation (port=0)
    server = OracleServer(ds_pickle_dataset_path)
    server.start()

    try:
        yield server
    except Exception as e:
        raise RuntimeError(f"Mock Oracle Server error: {e}") from e
    finally:
        server.stop()


@pytest.fixture
def max_throughput_runtime_settings():
    return RuntimeSettings(
        metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=1000,
        n_samples_from_dataset=100,
        n_samples_to_issue=100,
        min_sample_count=100,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
    )


@pytest.fixture
def clean_sample_event_hooks():
    """Fixture to ensure SampleEventHandler hooks are cleared before and after each test."""
    SampleEventHandler.clear_hooks()
    yield SampleEventHandler
    SampleEventHandler.clear_hooks()
