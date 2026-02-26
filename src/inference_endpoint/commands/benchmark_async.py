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

"""Async benchmark runner extracted from benchmark.py.

Single-loop uvloop benchmark execution — no threads in the main process.
Uses eager_task_factory so new tasks run synchronously until their first
real suspend point.
"""

import asyncio
import logging
import os
import signal
import time
import uuid
from typing import Any

from tqdm import tqdm

from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.config.schema import LoadPatternType
from inference_endpoint.endpoint_client.http_client import AsyncHttpEndpointClient
from inference_endpoint.exceptions import ExecutionError
from inference_endpoint.load_generator import SessionConfig

from .benchmark import ResponseCollector

logger = logging.getLogger(__name__)


async def run_benchmark(
    setup: SessionConfig,
) -> tuple[Any, ResponseCollector]:
    """Execute benchmark on a single uvloop -- no threads in the main process.

    Architecture:
    - AsyncHttpEndpointClient on the running loop (not sync wrapper)
    - ZmqEventRecordPublisher for event recording (sync ZMQ PUB NOBLOCK)
    - AsyncEventRecorder in background for SQLite writes
    - Online: loop.call_at() callback chain for Poisson scheduling
    - Offline: tight send loop + sleep(0) every 1000
    - Unified receiver: poll() + sleep(0) when idle (benchmark_httpclient.py pattern)

    Args:
        setup: SessionConfig containing all prepared state from setup_benchmark().

    Returns:
        Tuple of (report, response_collector) where report is the MetricsReporter
        output and response_collector holds responses and errors.
    """
    from inference_endpoint.async_utils.transport.record import (
        SampleEventType,
        SessionEventType,
    )
    from inference_endpoint.async_utils.transport.zmq.pubsub import (
        ZmqEventRecordPublisher,
    )
    from inference_endpoint.core.types import Query, QueryResult, StreamChunk
    from inference_endpoint.metrics.async_recorder import AsyncEventRecorder
    from inference_endpoint.metrics.async_reporter import AsyncEventReporter
    from inference_endpoint.metrics.reporter import MetricsReporter

    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]

    # ── Declare resources upfront for finally block ──────────────────────
    # Safe to check even if setup fails partway through (all start as None).
    zmq_ctx: ManagedZMQContext | None = None
    http_client: AsyncHttpEndpointClient | None = None
    publisher: ZmqEventRecordPublisher | None = None
    writer: AsyncEventRecorder | None = None
    recorder: AsyncEventReporter | None = None
    pbar: tqdm | None = None
    response_collector = ResponseCollector(collect_responses=setup.collect_responses)
    session_ended = False
    stop_requested = False
    report = None

    try:
        # ── Resource creation ─────────────────────────────────────────────

        zmq_ctx = ManagedZMQContext(io_threads=4)

        # Client construction blocks on run_coroutine_threadsafe, so use to_thread
        http_client = await asyncio.to_thread(
            AsyncHttpEndpointClient,
            setup.http_config,
            loop=loop,
            zmq_context=zmq_ctx,
        )

        # ── Event recording infrastructure ────────────────────────────────

        session_id = f"cli_benchmark_{uuid.uuid4().hex[:8]}"
        pub_addr = f"ipc://{zmq_ctx.socket_dir}/ev_pub_{session_id[:8]}"
        publisher = ZmqEventRecordPublisher(pub_addr, zmq_ctx, loop=loop)

        writer = AsyncEventRecorder(
            session_id, publisher.bind_address, sub_settle_s=0.5, stop_timeout=5.0
        )
        writer.start()  # blocking: waits for subscriber readiness

        idle_event = asyncio.Event()
        recorder = AsyncEventReporter(publisher, session_id, notify_idle=idle_event)

        # ── Progress bar + response collector ─────────────────────────────

        pbar = tqdm(
            desc=f"{setup.model_name} (Streaming: {setup.enable_streaming})",
            total=setup.total_samples,
            smoothing=0,
        )
        response_collector = ResponseCollector(
            collect_responses=setup.collect_responses, pbar=pbar
        )

        # ── Signal handling ───────────────────────────────────────────────

        send_done = False
        stop_requested = False
        uuid_to_index: dict[str, int] = {}

        def on_sigint():
            nonlocal stop_requested, send_done
            logger.warning("Interrupt received, stopping benchmark...")
            stop_requested = True
            send_done = True  # unblock receiver drain check

        loop.add_signal_handler(signal.SIGINT, on_sigint)

        # ── Send + Receive ────────────────────────────────────────────────

        recorder.record_event(SessionEventType.STARTED, time.monotonic_ns())

        def handle_response(result):
            """Process a received response (QueryResult or StreamChunk)."""
            ts = time.monotonic_ns()
            match result:
                case StreamChunk(is_complete=False):
                    metadata = result.metadata or {}
                    if metadata.get("first_chunk", False):
                        recorder.record_event(
                            SampleEventType.RECV_FIRST,
                            ts,
                            sample_uuid=result.id,
                            data={"response_chunk": result.response_chunk},
                        )
                    else:
                        recorder.record_event(
                            SampleEventType.RECV_NON_FIRST,
                            ts,
                            sample_uuid=result.id,
                        )
                case QueryResult(error=err):
                    if err is not None:
                        logger.error(f"Error in request {result.id}: {err}")
                    recorder.record_event(
                        SampleEventType.COMPLETE,
                        ts,
                        sample_uuid=result.id,
                        data=result.response_output,
                    )
                    response_collector.on_complete_hook(result)
                    setup.scheduler.notify_complete()

        async def receiver():
            """Unified receiver: poll() + sleep(0) when idle."""
            while True:
                result = http_client.poll()
                if result is not None:
                    handle_response(result)
                else:
                    if send_done and (
                        recorder.n_inflight_samples <= 0 or stop_requested
                    ):
                        break
                    await asyncio.sleep(0)

        def issue_sample(s_idx: int) -> str:
            """Issue a single sample -- shared by all send paths."""
            sample_uuid = uuid.uuid4().hex
            sample_data = setup.dataloader.load_sample(s_idx)
            ts = time.monotonic_ns()
            recorder.record_event(
                SampleEventType.ISSUED,
                ts,
                sample_uuid=sample_uuid,
            )
            http_client.issue(Query(id=sample_uuid, data=sample_data))
            uuid_to_index[sample_uuid] = s_idx
            return sample_uuid

        if setup.load_pattern_type == LoadPatternType.MAX_THROUGHPUT:
            # Offline: tight send loop, yield every 1000
            async def sender():
                nonlocal send_done
                sent = 0
                for s_idx, _ in setup.scheduler:
                    if stop_requested:
                        break
                    issue_sample(s_idx)
                    sent += 1
                    if sent % 1000 == 0:
                        await asyncio.sleep(0)
                send_done = True
        else:
            # Online (Poisson / Concurrency): scheduler.__aiter__ handles timing
            async def sender():
                nonlocal send_done
                async for s_idx in setup.scheduler:
                    if stop_requested:
                        break
                    issue_sample(s_idx)
                send_done = True

        await asyncio.gather(sender(), receiver())

        # Restore default SIGINT so Ctrl+C raises KeyboardInterrupt during
        # post-benchmark reporting and cleanup (no more flag-only handler).
        loop.remove_signal_handler(signal.SIGINT)

        # ── Post-benchmark ────────────────────────────────────────────────

        recorder.record_event(
            SessionEventType.STOP_PERFORMANCE_TRACKING, time.monotonic_ns()
        )
        recorder.should_check_idle = True
        recorder.record_event(SessionEventType.STOP_LOADGEN, time.monotonic_ns())

        if recorder.n_inflight_samples > 0 and not stop_requested:
            try:
                await asyncio.wait_for(
                    idle_event.wait(),
                    timeout=setup.config.timeout or 300,
                )
            except TimeoutError:
                logger.warning(
                    f"Timed out waiting for {recorder.n_inflight_samples} inflight samples"
                )

        recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())
        session_ended = True
        writer.stop()

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise ExecutionError(f"Benchmark execution failed: {e}") from e
    finally:
        # Each step guarded individually -- failure in one doesn't skip the rest.
        # Order: signal -> pbar -> session ended -> writer -> client -> report -> zmq

        try:
            loop.remove_signal_handler(signal.SIGINT)
        except Exception:
            pass

        if pbar:
            try:
                pbar.close()
            except Exception:
                pass

        if recorder and not session_ended:
            try:
                recorder.record_event(SessionEventType.ENDED, time.monotonic_ns())
            except Exception:
                pass

        if writer:
            try:
                writer.stop()
            except Exception:
                pass

        if http_client:
            try:
                await http_client.shutdown()
            except Exception:
                pass

        # Reset CPU affinity -- loadgen pinning no longer needed,
        # use all cores for report tokenization
        try:
            os.sched_setaffinity(0, range(os.cpu_count() or 1))
        except (OSError, AttributeError):
            pass
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        if recorder:
            try:
                with MetricsReporter(
                    recorder.connection_name, client_type="sqlite"
                ) as reporter:
                    report = reporter.create_report(setup.tokenizer)
                    report.display(fn=print, summary_only=True)

                    if setup.report_dir:
                        report.to_json(save_to=setup.report_dir / "result_summary.json")
                        with open(setup.report_dir / "report.txt", "w") as f:
                            report.display(
                                fn=f.write,
                                summary_only=False,
                                newline="\n",
                            )
                        reporter.dump_to_json(setup.report_dir / "events.jsonl")
                        logger.info(f"Report saved to: {setup.report_dir}")
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")

        if publisher:
            try:
                publisher.close()
            except Exception:
                pass

        if zmq_ctx:
            try:
                zmq_ctx.cleanup()
            except Exception:
                pass

    return (report, response_collector)
