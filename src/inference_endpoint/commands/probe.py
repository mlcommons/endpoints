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

"""Probe command implementation for endpoint health checking."""

import argparse
import asyncio
import logging
import tempfile
import time
from urllib.parse import urljoin

from inference_endpoint.config.schema import APIType
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.exceptions import (
    ExecutionError,
    InputValidationError,
    SetupError,
)

logger = logging.getLogger(__name__)


async def run_probe_command(args: argparse.Namespace) -> None:
    """Run endpoint probe to validate connectivity and basic functionality.

    Actions:
    1. Send test requests using HTTP client with futures
    2. Measure basic latency
    3. Report validation status
    """
    # Extract arguments
    endpoint = args.endpoint
    num_requests = args.requests
    test_prompt = args.prompt
    api_type = APIType(getattr(args, "api_type", "openai"))

    # Model: use provided or default to valid OpenAI model name
    model_name = getattr(args, "model", None)
    if not model_name:
        logger.error("Model required: --model or specify in YAML config")
        raise InputValidationError("Model required: --model NAME")
    # Note: API key handling would go in HTTP client config if needed

    logger.info(f"Probing: {endpoint}")

    # Create temp directory for ZMQ
    client = None

    # TODO (Rashid): Add a health check with a separate timeout.
    with tempfile.TemporaryDirectory(prefix="probe_") as tmp_dir:
        try:
            # Setup HTTP client with futures support
            http_config = HTTPClientConfig(
                endpoint_url=urljoin(endpoint, api_type.default_route()),
                api_type=api_type,
                num_workers=1,
            )
            aiohttp_config = AioHttpConfig()
            zmq_config = ZMQConfig(
                zmq_request_queue_prefix=f"ipc://{tmp_dir}/req",
                zmq_response_queue_addr=f"ipc://{tmp_dir}/resp",
                zmq_readiness_queue_addr=f"ipc://{tmp_dir}/ready",
            )

            client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

            logger.info(f"Sending {num_requests} requests...")

            # Send test requests
            start_times: dict[str, float] = {}
            sent_query_ids: list[str] = []
            issue_errors: list[str] = []

            # TODO: this might not work with a real vLLM/SGLang endpoint, fix this.
            for i in range(num_requests):
                query_id = f"probe-{i}"
                query = Query(
                    id=query_id,
                    data={
                        "prompt": test_prompt,
                        "model": model_name,
                        "max_tokens": 50,
                        "stream": False,
                    },
                )

                try:
                    start_times[query_id] = time.time()
                    client.issue_query(query)
                    # Only track successfully issued queries
                    sent_query_ids.append(query_id)
                except Exception as e:
                    issue_errors.append(f"{query_id}: Failed to issue - {str(e)[:50]}")
                    logger.warning(f"Failed to issue request {i}: {str(e)[:50]}")
                    continue

                # Simple progress indicator
                if (i + 1) % max(1, num_requests // 10) == 0 or i == num_requests - 1:
                    logger.info(f"  Issued {i + 1}/{num_requests} requests")

            # Wait for all responses
            latencies: list[float] = []
            errors: list[str] = issue_errors  # Include any issue errors
            responses: list[tuple[str, str]] = []

            # Only count successfully issued queries
            num_expected = len(sent_query_ids)
            if num_expected == 0:
                logger.error("✗ No queries were successfully issued")
                raise ExecutionError("Probe failed: no queries could be issued")

            # Wait for all responses with generous timeout (probe queries can be slow)
            probe_timeout = 60.0  # 60 seconds total
            start_wait = time.time()

            logger.info(f"Waiting for {num_expected} responses...")

            received_ids: set[str] = set()

            while (
                len(received_ids) < num_expected
                and (time.time() - start_wait) < probe_timeout
            ):
                try:
                    result = await client.try_receive()

                    if result is None:
                        await asyncio.sleep(0.01)
                        continue

                    # Skip non-final streaming chunks
                    if not isinstance(result, QueryResult):
                        continue

                    query_id = result.id

                    if query_id in received_ids:
                        logger.warning(f"Received duplicate response for {query_id}")
                        continue

                    received_ids.add(query_id)

                    # Calculate latency - should always be in start_times for issued queries
                    if query_id not in start_times:
                        logger.warning(
                            f"Received response for unknown query_id: {query_id}, skipping"
                        )
                        continue
                    latency_ms = (time.time() - start_times[query_id]) * 1000

                    if result.error:
                        errors.append(f"{query_id}: {result.error}")
                    else:
                        latencies.append(latency_ms)
                        responses.append((query_id, result.response_output))

                    # Simple progress indicator
                    if (
                        len(received_ids) % max(1, num_expected // 10) == 0
                        or len(received_ids) == num_expected
                    ):
                        logger.info(
                            f"  Processed {len(received_ids)}/{num_expected} responses : {query_id} : {result.response_output[:100]}"
                        )

                except Exception as e:
                    logger.warning(f"Error receiving response: {str(e)[:50]}")
                    await asyncio.sleep(0.01)

            # Mark any issued but not received as timeout
            for query_id in sent_query_ids:
                if query_id not in received_ids:
                    errors.append(f"{query_id}: Timeout (>{probe_timeout}s)")

            # Report results
            success_count = len(latencies)
            logger.info(f"✓ Completed: {success_count}/{num_expected} successful")

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                logger.info(f"✓ Avg latency: {avg_latency:.0f}ms")
                logger.info(f"✓ Range: {min(latencies):.0f}ms - {max(latencies):.0f}ms")

            # Show sample responses for sanity check
            if responses:
                logger.info(f"✓ Sample responses ({len(responses)} collected):")
                # Show all responses - can be overwhelming, but useful for debugging
                for query_id, response in responses:
                    # Truncate long responses
                    response_preview = (
                        response[:100] + "..." if len(response) > 100 else response
                    )
                    logger.info(f"  [{query_id}] {response_preview}")

            if errors:
                logger.warning(f"⚠ Errors: {len(errors)}")
                if args.verbose:
                    for error in errors[:3]:
                        logger.warning(f"  {error}")
                    if len(errors) > 3:
                        logger.warning(f"  ... +{len(errors) - 3} more")

            # Check if probe was successful
            if success_count < num_requests * 0.5:
                logger.error("✗ Probe failed: Too many errors")
                raise ExecutionError(
                    f"Probe failed: only {success_count}/{num_requests} requests successful"
                )

            logger.info("✓ Probe successful")

        except ExecutionError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            logger.error("✗ Probe failed")
            raise SetupError(f"Probe setup failed: {e}") from e
        finally:
            # Cleanup
            if client is not None:
                client.shutdown()
