"""Integration tests for ZMQ socket communication."""

import asyncio
import time

import msgspec
import pytest
import zmq
import zmq.asyncio
from inference_endpoint.endpoint_client.zmq_utils import (
    ZMQConfig,
    ZMQPullSocket,
    ZMQPushSocket,
)

from ...test_helpers import get_test_socket_path


class SampleData(msgspec.Struct):
    """Simple test data class for serialization tests."""

    id: str
    value: str
    timestamp: float


class TestZMQPushPullIntegration:
    """Integration tests for ZMQ Push/Pull sockets."""

    @pytest.fixture
    def zmq_config(self):
        """Create a ZMQ config for testing."""
        return ZMQConfig()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        [
            # Basic prompt
            {
                "id": "test-simple",
                "prompt": "Hello, world!",
                "description": "simple prompt",
            },
            # Large payload
            {
                "id": "test-large",
                "prompt": "x" * 10000,  # 10KB prompt
                "description": "large payload",
            },
            # Special characters and unicode
            {
                "id": "test-special",
                "prompt": "Test with special chars: 你好 🚀 €£¥ \n\t\r",
                "description": "special characters",
            },
            # Empty prompt
            {
                "id": "test-empty",
                "prompt": "",
                "description": "empty prompt",
            },
            # JSON-like content
            {
                "id": "test-json",
                "prompt": '{"message": "Test JSON", "nested": {"key": "value"}}',
                "description": "JSON content",
            },
            # SQL injection attempt (testing edge cases)
            {
                "id": "test-sql",
                "prompt": "'; DROP TABLE users; --",
                "description": "SQL injection pattern",
            },
            # Multiple lines
            {
                "id": "test-multiline",
                "prompt": "Line 1\nLine 2\nLine 3\n\nLine 5 with gaps",
                "description": "multiline content",
            },
        ],
    )
    async def test_push_pull_communication_various_payloads(
        self, zmq_config, test_case, tmp_path
    ):
        """Test push/pull communication with various payload types."""
        # Create unique address for this test using tmp_path
        address = get_test_socket_path(tmp_path, f"test_payload_{test_case['id']}")

        # Create context
        context = zmq.asyncio.Context()

        try:
            # Create pull socket first (bind)
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)

            # Create push socket (connect)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            # Allow time for connection
            await asyncio.sleep(0.1)

            # Send a query with the test case data
            test_query = {
                "id": test_case["id"],
                "prompt": test_case["prompt"],
                "model": "gpt-3.5-turbo",
                "max_completion_tokens": 50,
                "temperature": 0.7,
                "metadata": {"test_description": test_case["description"]},
            }

            await push_socket.send(test_query)

            # Receive the query
            received = await pull_socket.receive()

            # Verify
            assert isinstance(received, type(test_query))
            assert received["id"] == test_case["id"]
            assert received["prompt"] == test_case["prompt"]
            assert received["model"] == "gpt-3.5-turbo"
            assert received["max_completion_tokens"] == 50
            assert received["temperature"] == 0.7
            assert received["metadata"]["test_description"] == test_case["description"]

        finally:
            push_socket.close()
            pull_socket.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "result_case",
        [
            # Basic response
            {
                "query_id": "test-basic-result",
                "response_output": "This is the generated response",
                "error": None,
                "metadata": {
                    "model": "gpt-3.5-turbo",
                    "tokens_used": 25,
                    "latency": 1.5,
                },
                "description": "basic response",
            },
            # Empty response
            {
                "query_id": "test-empty-result",
                "response_output": "",
                "error": None,
                "metadata": {"model": "gpt-3.5-turbo", "tokens_used": 0},
                "description": "empty response",
            },
            # Error response
            {
                "query_id": "test-error-result",
                "response_output": "",
                "error": "HTTP 500: Internal Server Error",
                "metadata": {"error_code": 500, "retry_after": 60},
                "description": "error response",
            },
            # Large response
            {
                "query_id": "test-large-result",
                "response_output": "Y" * 10000,  # 10KB response
                "error": None,
                "metadata": {
                    "model": "gpt-3.5-turbo",
                    "tokens_used": 1000,
                    "chunks": 1,
                },
                "description": "large response",
            },
            # Response with special characters
            {
                "query_id": "test-special-result",
                "response_output": "Response with special: 你好世界 🌍 €£¥\n\tTabbed line",
                "error": None,
                "metadata": {"model": "gpt-3.5-turbo", "language": "mixed"},
                "description": "special characters response",
            },
            # JSON response
            {
                "query_id": "test-json-result",
                "response_output": '{"status": "success", "data": {"items": [1, 2, 3]}}',
                "error": None,
                "metadata": {"model": "gpt-3.5-turbo", "format": "json"},
                "description": "JSON formatted response",
            },
        ],
    )
    async def test_query_result_various_scenarios(
        self, zmq_config, result_case, tmp_path
    ):
        """Test sending and receiving QueryResult objects with various scenarios."""
        # Use shorter path to avoid Unix socket path length limit (107 chars)
        # Map long query_ids to short names
        id_map = {
            "test-basic-result": "basic",
            "test-empty-result": "empty",
            "test-error-result": "error",
            "test-large-result": "large",
            "test-special-result": "spec",
            "test-json-result": "json",
        }
        short_id = id_map.get(result_case["query_id"], result_case["query_id"][:5])
        address = get_test_socket_path(tmp_path, f"tr_{short_id}")

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Send a QueryResult
            test_result = {
                "query_id": result_case["query_id"],
                "response_output": result_case["response_output"],
                "error": result_case["error"],
                "metadata": result_case["metadata"],
            }

            await push_socket.send(test_result)
            received = await pull_socket.receive()

            # Verify
            assert isinstance(received, type(test_result))
            assert received["query_id"] == result_case["query_id"]
            assert received["response_output"] == result_case["response_output"]
            assert received["error"] == result_case["error"]
            assert received["metadata"] == result_case["metadata"]

        finally:
            push_socket.close()
            pull_socket.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    async def test_streaming_response_communication(self, zmq_config, tmp_path):
        """Test streaming response pattern with first/final chunks."""
        address = get_test_socket_path(tmp_path, "test_streaming")

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Send first chunk
            first_chunk = {
                "query_id": "stream-123",
                "response_output": "Once",
                "metadata": {"first_chunk": True, "final_chunk": False},
            }

            await push_socket.send(first_chunk)
            received_first = await pull_socket.receive()

            assert received_first["query_id"] == "stream-123"
            assert received_first["response_output"] == "Once"
            assert received_first["metadata"]["first_chunk"] is True
            assert received_first["metadata"]["final_chunk"] is False

            # Send final chunk
            final_chunk = {
                "query_id": "stream-123",
                "response_output": "Once upon a time in a land far away...",
                "metadata": {"first_chunk": False, "final_chunk": True},
            }

            await push_socket.send(final_chunk)
            received_final = await pull_socket.receive()

            assert received_final["query_id"] == "stream-123"
            assert (
                received_final["response_output"]
                == "Once upon a time in a land far away..."
            )
            assert received_final["metadata"]["first_chunk"] is False
            assert received_final["metadata"]["final_chunk"] is True

        finally:
            push_socket.close()
            pull_socket.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sequence_config",
        [
            # Multiple push sockets to single pull
            {
                "description": "multiple push sockets",
                "num_push_sockets": 3,
                "num_messages_per_socket": 1,
                "concurrent": False,
            },
            # Single push socket with multiple sequential messages
            {
                "description": "sequential messages",
                "num_push_sockets": 1,
                "num_messages_per_socket": 5,
                "concurrent": False,
            },
            # Single push socket with concurrent messages
            {
                "description": "concurrent messages",
                "num_push_sockets": 1,
                "num_messages_per_socket": 10,
                "concurrent": True,
            },
            # Multiple push sockets with multiple messages each
            {
                "description": "multiple sockets with multiple messages",
                "num_push_sockets": 2,
                "num_messages_per_socket": 3,
                "concurrent": False,
            },
        ],
    )
    async def test_multiple_message_sequences(
        self, zmq_config, sequence_config, tmp_path
    ):
        """Test various patterns of sending multiple messages."""
        address = get_test_socket_path(
            tmp_path,
            f"test_sequence_{sequence_config['description'].replace(' ', '_')}",
        )

        context = zmq.asyncio.Context()

        try:
            # Create pull socket
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)

            # Create push sockets
            push_sockets = []
            for _ in range(sequence_config["num_push_sockets"]):
                push_socket = ZMQPushSocket(context, address, zmq_config)
                push_sockets.append(push_socket)

            await asyncio.sleep(0.1)

            # Send messages
            sent_queries = []
            total_messages = (
                sequence_config["num_push_sockets"]
                * sequence_config["num_messages_per_socket"]
            )

            if sequence_config["concurrent"]:
                # Send all messages concurrently
                tasks = []
                for socket_idx, push_socket in enumerate(push_sockets):
                    for msg_idx in range(sequence_config["num_messages_per_socket"]):
                        query = {
                            "id": f"socket-{socket_idx}-msg-{msg_idx}",
                            "prompt": f"Query from socket {socket_idx}, message {msg_idx}",
                            "model": "gpt-3.5-turbo",
                        }
                        sent_queries.append(query)
                        tasks.append(push_socket.send(query))
                await asyncio.gather(*tasks)
            else:
                # Send messages sequentially
                for socket_idx, push_socket in enumerate(push_sockets):
                    for msg_idx in range(sequence_config["num_messages_per_socket"]):
                        query = {
                            "id": f"socket-{socket_idx}-msg-{msg_idx}",
                            "prompt": f"Query from socket {socket_idx}, message {msg_idx}",
                            "model": "gpt-3.5-turbo",
                        }
                        sent_queries.append(query)
                        await push_socket.send(query)

            # Receive all messages
            received_queries = []
            for _ in range(total_messages):
                received = await pull_socket.receive()
                received_queries.append(received)

            # Verify all messages received (order may vary)
            received_ids = {q["id"] for q in received_queries}
            expected_ids = {q["id"] for q in sent_queries}
            assert received_ids == expected_ids
            assert len(received_queries) == total_messages

        finally:
            for socket in push_sockets:
                socket.close()
            pull_socket.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    async def test_custom_data_serialization(self, zmq_config, tmp_path):
        """Test sending custom data types."""
        address = get_test_socket_path(tmp_path, "test_custom")

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(
                context, address, zmq_config, bind=True, decoder_type=SampleData
            )
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Send custom data
            custom_data = SampleData(
                id="custom-001",
                value="test value with special chars: 你好 🚀",
                timestamp=time.time(),
            )

            await push_socket.send(custom_data)
            received = await pull_socket.receive()

            assert isinstance(received, SampleData)
            assert received.id == "custom-001"
            assert received.value == "test value with special chars: 你好 🚀"
            assert isinstance(received.timestamp, float)

        finally:
            push_socket.close()
            pull_socket.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    async def test_pull_socket_connect_mode(self, zmq_config, tmp_path):
        """Test ZMQPullSocket in connect mode (bind=False)."""
        address = get_test_socket_path(tmp_path, "test_pull_connect")

        context = zmq.asyncio.Context()

        try:
            # First create a push socket that binds
            push_socket = ZMQPushSocket(context, address, zmq_config)
            # Manually bind the push socket's underlying socket
            push_socket.socket.close()  # Close the connected socket
            push_socket.socket = context.socket(zmq.PUSH)
            push_socket.socket.bind(address)
            push_socket.socket.setsockopt(zmq.SNDHWM, zmq_config.zmq_high_water_mark)
            push_socket.socket.setsockopt(zmq.LINGER, zmq_config.zmq_linger)
            push_socket.socket.setsockopt(zmq.SNDBUF, zmq_config.zmq_send_buffer_size)
            push_socket.socket.setsockopt(zmq.SNDTIMEO, zmq_config.zmq_send_timeout)

            # Create pull socket in connect mode (bind=False)
            pull_socket = ZMQPullSocket(
                context, address, zmq_config, bind=False, decoder_type=SampleData
            )

            await asyncio.sleep(0.1)

            # Send a message
            test_data = SampleData(
                id="connect-test",
                value="Testing connect mode",
                timestamp=time.time(),
            )

            await push_socket.send(test_data)

            # Receive the message
            received = await pull_socket.receive()

            # Verify
            assert isinstance(received, SampleData)
            assert received.id == "connect-test"
            assert received.value == "Testing connect mode"

        finally:
            push_socket.close()
            pull_socket.close()
            context.destroy(linger=0)
