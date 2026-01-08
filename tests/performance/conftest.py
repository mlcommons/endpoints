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

"""Performance test fixtures and configuration."""

import logging
import sys
from pathlib import Path

import pytest
import zmq
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.testing.echo_server import EchoServer
from inference_endpoint.utils.logging import setup_logging

from tests.test_helpers import get_test_socket_path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Configure logging for tests
setup_logging(level="WARNING")


@pytest.fixture(scope="function")
def perf_http_echo_server():
    """
    Function-scoped HTTP echo server for performance tests.

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
def http_client(perf_http_echo_server, tmp_path):
    """Create single-worker HTTP client for perf tests."""
    http_config = HTTPClientConfig(
        endpoint_url=f"{perf_http_echo_server.url}/v1/chat/completions",
        num_workers=1,
    )
    assert (
        len(str(tmp_path)) <= zmq.IPC_PATH_MAX_LEN
    ), "tmp_path is too long for ZMQ - consider setting --basetemp=<short_path> for pytest"

    zmq_config = ZMQConfig(
        zmq_io_threads=4,
        zmq_request_queue_prefix=get_test_socket_path(tmp_path, "perf_test_raw"),
        zmq_response_queue_addr=get_test_socket_path(
            tmp_path, "perf_test_raw_responses"
        ),
        zmq_high_water_mark=100_000,
    )

    client = HTTPEndpointClient(
        config=http_config,
        aiohttp_config=AioHttpConfig(),
        zmq_config=zmq_config,
    )

    try:
        yield client
    except Exception as e:
        raise RuntimeError(f"HttpEndpointClient Error: {e}") from e
    finally:
        client.shutdown()


@pytest.fixture(scope="function")
def cleanup_connections():
    to_cleanup = {
        "close": [],
        "delete": [],
    }
    yield to_cleanup

    for obj in to_cleanup["close"]:
        try:
            obj.close()
        except Exception as e:
            logging.error(f"Error closing object: {e}")
    for obj in to_cleanup["delete"]:
        try:
            Path(obj).unlink(missing_ok=True)
        except Exception as e:
            logging.error(f"Error deleting object: {e}")
