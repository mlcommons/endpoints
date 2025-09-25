"""Performance test fixtures and configuration."""

import sys
from pathlib import Path

import pytest
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.futures_client import FuturesHttpClient
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.testing.echo_server import EchoServer
from inference_endpoint.utils.logging import setup_logging

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Configure logging for tests
setup_logging(level="WARNING")


@pytest.fixture(scope="function")
def perf_http_echo_server():
    """
    Function-scoped HTTP echo server for performance tests.

    Creates a fresh HTTP server instance for each test function to ensure
    isolated performance measurements without interference from previous tests.

    Returns:
        EchoServer: A started echo server instance with dynamically assigned port.
    """
    # Create and start the server with dynamic port allocation (port=0)
    server = EchoServer(port=0)
    server.start()

    try:
        yield server
    except Exception as e:
        raise RuntimeError(f"Performance Echo Server error: {e}") from e
    finally:
        server.stop()


@pytest.fixture(scope="function")
def futures_http_client(perf_http_echo_server, tmp_path):
    """Create single-worker futures-based HTTP client for testing with isolated server."""
    http_config = HTTPClientConfig(
        endpoint_url=f"{perf_http_echo_server.url}/v1/chat/completions",
        num_workers=1,
    )

    aiohttp_config = AioHttpConfig()

    zmq_config = ZMQConfig(
        zmq_io_threads=4,
        zmq_request_queue_prefix=f"ipc://{tmp_path}/perf_test",
        zmq_response_queue_addr=f"ipc://{tmp_path}/perf_test_responses",
        zmq_high_water_mark=100_000,
    )

    raw_client = HTTPEndpointClient(
        config=http_config,
        aiohttp_config=aiohttp_config,
        zmq_config=zmq_config,
    )

    client = FuturesHttpClient(raw_client)
    client.start()
    yield client
    client.shutdown()


@pytest.fixture(scope="function")
def http_client(perf_http_echo_server, tmp_path):
    """Create single-worker HTTP client for LoadGenerator tests."""
    http_config = HTTPClientConfig(
        endpoint_url=f"{perf_http_echo_server.url}/v1/chat/completions",
        num_workers=1,
    )

    aiohttp_config = AioHttpConfig()

    zmq_config = ZMQConfig(
        zmq_io_threads=4,
        zmq_request_queue_prefix=f"ipc://{tmp_path}/perf_test_raw",
        zmq_response_queue_addr=f"ipc://{tmp_path}/perf_test_raw_responses",
        zmq_high_water_mark=100_000,
    )

    client = HTTPEndpointClient(
        config=http_config,
        aiohttp_config=aiohttp_config,
        zmq_config=zmq_config,
    )

    client.start()
    yield client
    client.shutdown()


@pytest.fixture(scope="function")
def futures_http_client_multiworker(perf_http_echo_server, tmp_path):
    """Create multi-worker (12 CPUs) futures-based HTTP client for testing with isolated server."""
    http_config = HTTPClientConfig(
        endpoint_url=f"{perf_http_echo_server.url}/v1/chat/completions",
        num_workers=12,
    )

    aiohttp_config = AioHttpConfig()

    zmq_config = ZMQConfig(
        zmq_io_threads=12,
        zmq_request_queue_prefix=f"ipc://{tmp_path}/perf_test_multi",
        zmq_response_queue_addr=f"ipc://{tmp_path}/perf_test_multi_responses",
        zmq_high_water_mark=100_000,
    )

    raw_client = HTTPEndpointClient(
        config=http_config,
        aiohttp_config=aiohttp_config,
        zmq_config=zmq_config,
    )

    client = FuturesHttpClient(raw_client)
    client.start()
    yield client
    client.shutdown()
