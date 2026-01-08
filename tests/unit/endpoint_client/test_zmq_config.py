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

"""Unit tests for ZMQ configuration components."""

import os
from unittest import mock

from inference_endpoint.endpoint_client.zmq_utils import ZMQConfig


class TestZMQConfig:
    """Tests for ZMQConfig class."""

    def test_default_path_generation_unix(self):
        """Test that default paths are generated correctly on Unix systems."""
        with mock.patch("os.name", "posix"):
            with mock.patch("os.getpid", return_value=12345):
                config = ZMQConfig()

                assert (
                    config.zmq_request_queue_prefix
                    == "ipc:///tmp/mlperf_endpoint_http_worker_requests_12345"
                )
                assert (
                    config.zmq_response_queue_addr
                    == "ipc:///tmp/mlperf_endpoint_http_worker_responses_12345"
                )

    def test_custom_paths_preserved(self):
        """Test that custom paths are preserved when provided."""
        custom_request = "ipc:///custom/request/path"
        custom_response = "ipc:///custom/response/path"

        config = ZMQConfig(
            zmq_request_queue_prefix=custom_request,
            zmq_response_queue_addr=custom_response,
        )

        assert config.zmq_request_queue_prefix == custom_request
        assert config.zmq_response_queue_addr == custom_response

    def test_partial_custom_paths(self):
        """Test that only unspecified paths are auto-generated."""
        custom_request = "ipc:///custom/request/path"

        with mock.patch("os.name", "posix"):
            with mock.patch("os.getpid", return_value=12345):
                config = ZMQConfig(zmq_request_queue_prefix=custom_request)

                assert config.zmq_request_queue_prefix == custom_request
                assert (
                    config.zmq_response_queue_addr
                    == "ipc:///tmp/mlperf_endpoint_http_worker_responses_12345"
                )

    def test_path_generation_includes_pid(self):
        """Test that path generation includes the process ID."""
        with mock.patch("os.name", "posix"):
            config = ZMQConfig()

            # Should include PID in the path
            pid = os.getpid()
            assert (
                f"ipc:///tmp/mlperf_endpoint_http_worker_requests_{pid}"
                == config.zmq_request_queue_prefix
            )
            assert (
                f"ipc:///tmp/mlperf_endpoint_http_worker_responses_{pid}"
                == config.zmq_response_queue_addr
            )

    def test_different_processes_get_different_paths(self):
        """Test that different processes get different socket paths."""
        with mock.patch("os.name", "posix"):
            # Simulate different PIDs
            with mock.patch("os.getpid", return_value=111):
                config1 = ZMQConfig()

            with mock.patch("os.getpid", return_value=222):
                config2 = ZMQConfig()

            # Paths should be different due to PID
            assert config1.zmq_request_queue_prefix != config2.zmq_request_queue_prefix
            assert config1.zmq_response_queue_addr != config2.zmq_response_queue_addr

    def test_zmq_config_other_attributes(self):
        """Test that other ZMQConfig attributes work correctly."""
        config = ZMQConfig(
            zmq_io_threads=8,
            zmq_high_water_mark=20000,
            zmq_linger=100,
            zmq_send_timeout=5000,
            zmq_recv_timeout=10000,
            zmq_recv_buffer_size=20 * 1024 * 1024,
            zmq_send_buffer_size=20 * 1024 * 1024,
        )

        assert config.zmq_io_threads == 8
        assert config.zmq_high_water_mark == 20000
        assert config.zmq_linger == 100
        assert config.zmq_send_timeout == 5000
        assert config.zmq_recv_timeout == 10000
        assert config.zmq_recv_buffer_size == 20 * 1024 * 1024
        assert config.zmq_send_buffer_size == 20 * 1024 * 1024
