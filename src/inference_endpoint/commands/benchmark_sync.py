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

"""Threaded (synchronous) benchmark runner.

Extracted from the original ``_run_benchmark()`` in ``benchmark.py``.  This
module contains **only** the execution logic — setup is handled by
``setup_benchmark()`` and post-processing by ``post_benchmark()``, both in
``benchmark.py``.
"""

import logging
import shutil
import signal
import tempfile
import uuid
from typing import Any

from tqdm import tqdm

from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.exceptions import ExecutionError, SetupError
from inference_endpoint.load_generator import (
    BenchmarkSession,
    SampleEvent,
    SampleEventHandler,
    SessionConfig,
)

from .benchmark import ResponseCollector

logger = logging.getLogger(__name__)


def run_benchmark(setup: SessionConfig) -> tuple[Any, ResponseCollector]:
    """Execute a benchmark session using the threaded (sync) runner.

    This is the execution-only counterpart to ``setup_benchmark()`` — it
    assumes all configuration, dataset loading, and scheduler creation have
    already been performed and are available via *setup*.

    Post-benchmark work (accuracy scoring, ``results.json``, error logging)
    is **not** done here; the caller should invoke ``post_benchmark()``
    after this function returns.

    Args:
        setup: A fully-populated :class:`SessionConfig` produced by
            ``setup_benchmark()``.

    Returns:
        A ``(report, response_collector)`` tuple where *report* is the
        :class:`BenchmarkSession` report object (or ``None`` if the session
        failed to produce one) and *response_collector* holds per-query
        results and errors.

    Raises:
        SetupError: If the HTTP endpoint client cannot be created.
        ExecutionError: If the benchmark fails after successful setup.
        KeyboardInterrupt: If the user interrupts with Ctrl+C (re-raised).
    """
    # Progress bar
    pbar = tqdm(
        desc=f"{setup.model_name} (Streaming: {setup.enable_streaming})",
        total=setup.total_samples,
        smoothing=0,  # smoothing=0 shows average instead of EMA
    )

    # Response collector
    response_collector = ResponseCollector(
        collect_responses=setup.collect_responses, pbar=pbar
    )
    SampleEventHandler.register_hook(
        SampleEvent.COMPLETE, response_collector.on_complete_hook
    )

    # Scope ZMQ context so transport and sockets are cleaned up when the block exits.
    with ManagedZMQContext.scoped() as zmq_ctx:
        tmp_dir = tempfile.mkdtemp(prefix="inference_endpoint_")

        try:
            http_client = HTTPEndpointClient(setup.http_config, zmq_context=zmq_ctx)
            sample_issuer = HttpClientSampleIssuer(http_client)
        except Exception as e:
            logger.error("Connection failed")
            raise SetupError(f"Failed to connect to endpoint: {e}") from e

        # Run benchmark
        logger.info("Running...")

        sess = None
        try:
            sess = BenchmarkSession.from_config(
                setup,
                sample_issuer,
                name=f"cli_benchmark_{uuid.uuid4().hex[0:8]}",
                dump_events_log=True,
            )

            # Wait for test end with ability to interrupt
            def signal_handler(signum, frame):
                logger.warning("Interrupt signal received, stopping benchmark...")
                # Raise KeyboardInterrupt to break out of wait_for_test_end()
                raise KeyboardInterrupt()

            # Install our handler, save old one
            old_handler = signal.signal(signal.SIGINT, signal_handler)
            try:
                sess.wait_for_test_end()
            finally:
                # Always restore original handler
                signal.signal(signal.SIGINT, old_handler)

            report = getattr(sess, "report", None)
            return report, response_collector

        except KeyboardInterrupt:
            logger.warning("Benchmark interrupted by user")
            raise
        except ExecutionError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            logger.error("Benchmark failed")
            raise ExecutionError(f"Benchmark execution failed: {e}") from e
        finally:
            # Cleanup — always execute
            logger.info("Cleaning up...")
            try:
                if sess is not None:
                    sess.stop()
                pbar.close()
                sample_issuer.shutdown()
                http_client.shutdown()
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                if setup.config.verbose:
                    logger.warning(f"Cleanup error: {e}")
