"""Probe command implementation for endpoint health checking."""

import argparse
import asyncio
import logging
import shutil
import tempfile
import time

from inference_endpoint.core.types import Query
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.futures_client import FuturesHttpClient
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

    # Model: use provided or default to valid OpenAI model name
    model_name = getattr(args, "model", None)
    if not model_name:
        logger.error("Model required: --model or specify in YAML config")
        raise InputValidationError("Model required: --model NAME")
    # Note: API key handling would go in HTTP client config if needed

    logger.info(f"Probing: {endpoint}")

    # Create temp directory for ZMQ
    tmp_dir = tempfile.mkdtemp(prefix="probe_")
    client = None

    # TODO (Rashid): Add a health check with a separate timeout.
    try:
        # Setup HTTP client with futures support
        http_config = HTTPClientConfig(
            endpoint_url=f"{endpoint}/v1/chat/completions",
            num_workers=1,
            max_concurrency=num_requests,
        )
        aiohttp_config = AioHttpConfig()
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_dir}/req",
            zmq_response_queue_addr=f"ipc://{tmp_dir}/resp",
            zmq_readiness_queue_addr=f"ipc://{tmp_dir}/ready",
        )

        client = FuturesHttpClient(http_config, aiohttp_config, zmq_config)
        await client.async_start()

        logger.info(f"Sending {num_requests} requests...")

        # Send test requests and collect futures
        futures = []
        start_times = {}

        # TODO: this might not work with a real vLLM/SGLang endpoint, fix this.
        for i in range(num_requests):
            query = Query(
                id=f"probe-{i}",
                data={
                    "prompt": test_prompt,
                    "model": model_name,
                    "max_tokens": 50,
                    "stream": False,
                },
            )
            start_times[f"probe-{i}"] = time.time()

            try:
                future = await client.issue_query(query)
                futures.append((f"probe-{i}", future))
                # Simple progress indicator
                if (i + 1) % max(1, num_requests // 10) == 0 or i == num_requests - 1:
                    logger.info(f"  Issued {i + 1}/{num_requests} requests")
            except Exception as e:
                logger.warning(f"Failed to issue request {i}: {str(e)[:50]}")

        # Wait for all responses
        latencies = []
        errors = []
        responses = []

        # Wait for all responses with generous timeout (probe queries can be slow)
        # Default HTTP client timeout is 30s, give extra buffer for processing
        probe_timeout = 60.0  # 60 seconds per query

        logger.info(f"Waiting for {len(futures)} responses...")

        for idx, (query_id, future) in enumerate(futures):
            try:
                result = await asyncio.wait_for(future, timeout=probe_timeout)
                # Calculate latency - should always be in start_times
                assert (
                    query_id in start_times
                ), f"Query {query_id} not found in start_times"
                latency_ms = (time.time() - start_times[query_id]) * 1000
                latencies.append(latency_ms)

                if result.error:
                    errors.append(f"{query_id}: {result.error}")
                else:
                    # Store successful response for sanity check
                    responses.append((query_id, result.response_output))
            except TimeoutError:
                errors.append(f"{query_id}: Timeout (>{probe_timeout}s)")
            except Exception as e:
                errors.append(f"{query_id}: {str(e)[:50]}")

            # Simple progress indicator
            if (idx + 1) % max(1, len(futures) // 10) == 0 or idx == len(futures) - 1:
                logger.info(f"  Processed {idx + 1}/{len(futures)} responses")

        # Report results
        success_count = len(latencies)
        logger.info(f"✓ Completed: {success_count}/{num_requests} successful")

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            logger.info(f"✓ Avg latency: {avg_latency:.0f}ms")
            logger.info(f"✓ Range: {min(latencies):.0f}ms - {max(latencies):.0f}ms")

        # Show sample responses for sanity check
        if responses:
            logger.info(f"✓ Sample responses ({len(responses)} collected):")
            # Show first 10 responses
            for query_id, response in responses[:10]:
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
            await client.async_shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)
