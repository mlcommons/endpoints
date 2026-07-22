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

"""Metrics/event-log service pipeline for a benchmark run.

``MetricsPipeline`` owns the ZMQ context, the event publisher, the metrics-snapshot
subscriber, and the two service subprocesses (metrics aggregator + event logger).
It exposes explicit lifecycle methods rather than a context manager because the run
has three distinct teardown paths that a single ``__aexit__`` cannot express cleanly:

* ``start()`` — bring the infrastructure up; unwind partially-acquired resources if
  service launch fails (so a launch error can't leak the ZMQ context).
* ``drain_and_build_report()`` — the graceful end-of-run drain (close publisher →
  wait for services → source the final snapshot → build the Report). Runs on both a
  clean finish and a session-run failure, since ``BenchmarkSession.run`` publishes
  ``ENDED`` in its own ``finally`` and the aggregator writes a terminal snapshot.
* ``abort()`` — the connect-failure fast path: kill the services without a graceful
  drain (no ``ENDED`` was ever published).

``close()`` exits the ZMQ scope and is idempotent, so the orchestrator can call it
unconditionally in its ``finally``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_endpoint.async_utils.event_publisher import EventPublisherService
from inference_endpoint.async_utils.services.launcher import (
    ServiceConfig,
    ServiceLauncher,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    snapshot_to_dict,
)
from inference_endpoint.async_utils.services.metrics_aggregator.subscriber import (
    MetricsSnapshotSubscriber,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.config.schema import LoadPatternType
from inference_endpoint.metrics.report import Report

if TYPE_CHECKING:
    from inference_endpoint.config.schema import BenchmarkConfig

logger = logging.getLogger(__name__)

_AGGREGATOR_MODULE = "inference_endpoint.async_utils.services.metrics_aggregator"
_EVENT_LOGGER_MODULE = "inference_endpoint.async_utils.services.event_logger"


def _load_final_snapshot_from_disk(path: Path) -> dict[str, Any] | None:
    """Read the persisted ``final_snapshot.json`` written by the aggregator.

    Returns the snapshot in its dict form — the same shape produced by
    ``snapshot_to_dict`` and consumed by ``Report.from_snapshot``. No
    intermediate Struct decode (see ``Report.from_snapshot`` docstring
    for why the dict shape is the consumer contract).

    Returns ``None`` if the file is missing (the aggregator was killed
    by an uncatchable signal — SIGKILL, OOM-kill — before its handler
    could write) or unreadable.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_bytes())
    except Exception as e:  # noqa: BLE001 — best-effort.
        logger.warning("Failed to read final snapshot %s: %s", path, e)
        return None


def _build_aggregator_args(
    *,
    socket_dir: str,
    pub_socket_name: str,
    metrics_socket_name: str,
    metrics_output_dir: Path,
    enable_streaming: bool,
    tokenizer_name: str | None,
    drain_timeout_s: float,
    tokenizer_workers: int,
) -> list[str]:
    """CLI args for the metrics_aggregator subprocess."""
    args: list[str] = [
        "--socket-dir",
        socket_dir,
        "--socket-name",
        pub_socket_name,
        "--metrics-socket",
        metrics_socket_name,
        "--metrics-output-dir",
        str(metrics_output_dir),
    ]
    if enable_streaming:
        args.append("--streaming")
    if tokenizer_name is not None:
        args.extend(["--tokenizer", tokenizer_name])
    args.extend(["--drain-timeout", str(drain_timeout_s)])
    args.extend(["--tokenizer-workers", str(tokenizer_workers)])
    return args


def _build_event_logger_args(
    *, event_log_dir: Path, socket_dir: str, pub_socket_name: str
) -> list[str]:
    """CLI args for the event_logger subprocess (writes events.jsonl to tmpfs)."""
    return [
        "--log-dir",
        str(event_log_dir),
        "--socket-dir",
        socket_dir,
        "--socket-name",
        pub_socket_name,
        "--writers",
        "jsonl",
    ]


def _build_report_from_snapshot(
    snap_dict: dict[str, Any], config: BenchmarkConfig
) -> Report | None:
    """Build the Report from a metrics snapshot dict; best-effort (None on failure).

    A malformed snapshot must never fail an otherwise-completed benchmark, so any
    build error is logged and swallowed to ``None``.
    """
    try:
        load_pattern = config.settings.load_pattern
        runtime_cfg = config.settings.runtime
        # load_pattern + warmup config and the RNG seeds, so result_summary.json is
        # self-describing and a valid run is identified by its settings. The full,
        # re-runnable config lives in config.yaml alongside. The resolved/effective
        # runtime settings (sample count + ordering, which can differ per audit
        # phase) are deferred to a follow-up. endpoint_config (api_key/URLs) is a
        # sibling of settings and never included, so no secrets.
        run_config = config.settings.model_dump(
            mode="json", include={"load_pattern", "warmup"}
        )
        run_config["scheduler_random_seed"] = runtime_cfg.scheduler_random_seed
        run_config["dataloader_random_seed"] = runtime_cfg.dataloader_random_seed
        report = Report.from_snapshot(
            snap_dict,
            run_config=run_config,
            use_legacy_loadgen_qps_metrics=(
                load_pattern.type == LoadPatternType.POISSON
                and load_pattern.use_legacy_loadgen_qps_metrics
            ),
        )
        if not report.complete:
            logger.warning(
                "Report is incomplete (state=%s, n_pending_tasks=%d)",
                report.state,
                snap_dict.get("n_pending_tasks", 0),
            )
        if report.legacy_loadgen_window_duration_ns is not None:
            logger.warning(
                "Reporting QPS/TPS with the legacy MLPerf LoadGen Server "
                "'completed' definition (deprecated; to be removed once a "
                "formal tail-cutting mechanism lands). Pass "
                "--no-use-legacy-loadgen-qps-metrics for endpoints-native "
                "metrics."
            )
        return report
    except Exception as e:  # noqa: BLE001 — best-effort report build.
        logger.warning(f"Failed to build report from snapshot: {e}")
        return None


class MetricsPipeline:
    """ZMQ + metrics/event-logger subprocess lifecycle for one benchmark run.

    ``event_log_dir`` is on tmpfs (salvaged then removed by the caller);
    ``metrics_output_dir`` is on disk under the report dir (holds
    ``final_snapshot.json`` — the primary Report source) and is never removed here.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        tokenizer_name: str | None,
        enable_streaming: bool,
        event_log_dir: Path,
        metrics_output_dir: Path,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._config = config
        self._tokenizer_name = tokenizer_name
        self._enable_streaming = enable_streaming
        self._event_log_dir = event_log_dir
        self._metrics_output_dir = metrics_output_dir
        self._loop = loop

        self._zmq_cm: Any = None
        self._launcher: ServiceLauncher | None = None
        self.publisher: EventPublisherService | None = None
        self.subscriber: MetricsSnapshotSubscriber | None = None
        self._closed = False

    async def start(self) -> None:
        """Bring up ZMQ + publisher + subscriber + service subprocesses.

        Connects the snapshot subscriber BEFORE launching the aggregator that binds
        the matching PUB socket (ZMQ tolerates connect-before-bind on IPC; starting
        the SUB reader early lets the subscription handshake complete during the
        ~1-2s subprocess-launch window, avoiding the slow-joiner risk of dropping
        early live ticks — or, worst case, missing COMPLETE). On any failure the
        partially-acquired resources are unwound so nothing (notably the ZMQ
        context) leaks.
        """
        self._zmq_cm = ManagedZMQContext.scoped(io_threads=2)
        zmq_ctx = self._zmq_cm.__enter__()
        try:
            self.publisher = EventPublisherService(zmq_ctx)
            pub_socket_name = self.publisher.socket_name
            metrics_socket_name = f"metrics_pub_{uuid.uuid4().hex[:8]}"

            if zmq_ctx.socket_dir is None:
                raise RuntimeError("ZMQ socket_dir must be set after publisher bind")
            self.subscriber = MetricsSnapshotSubscriber(
                metrics_socket_name, zmq_ctx, self._loop
            )
            self.subscriber.start()

            self._launcher = ServiceLauncher(zmq_ctx)
            drain = self._config.settings.drain
            aggregator_args = _build_aggregator_args(
                socket_dir=zmq_ctx.socket_dir,
                pub_socket_name=pub_socket_name,
                metrics_socket_name=metrics_socket_name,
                metrics_output_dir=self._metrics_output_dir,
                enable_streaming=self._enable_streaming,
                tokenizer_name=self._tokenizer_name,
                drain_timeout_s=drain.metrics_drain_timeout_s,
                tokenizer_workers=drain.metrics_tokenizer_workers,
            )
            event_logger_args = _build_event_logger_args(
                event_log_dir=self._event_log_dir,
                socket_dir=zmq_ctx.socket_dir,
                pub_socket_name=pub_socket_name,
            )
            await self._launcher.launch(
                [
                    ServiceConfig(module=_AGGREGATOR_MODULE, args=aggregator_args),
                    ServiceConfig(module=_EVENT_LOGGER_MODULE, args=event_logger_args),
                ],
                timeout=self._config.settings.service_ready_timeout_s,
            )
        except BaseException:
            self._teardown(kill=True)
            raise

    async def drain_and_build_report(self) -> Report | None:
        """Graceful drain: close publisher, wait for services, build the Report.

        Runs on both the clean-finish and session-failure paths. Sources the
        snapshot from the aggregator's on-disk ``final_snapshot.json`` when present,
        else falls back to the last live pub/sub snapshot (only reached when the
        aggregator was killed before it could write). Report construction is
        best-effort — a build failure yields ``None`` rather than aborting.
        """
        assert self.publisher is not None
        assert self._launcher is not None
        logger.info(
            "Closing publisher (buffer=%d, pending=%d)...",
            self.publisher.buffered_count,
            self.publisher.pending_count,
        )
        self.publisher.close()
        logger.info("Waiting for services to finish processing...")
        await asyncio.to_thread(self._launcher.wait_for_exit, None)

        snap_dict = _load_final_snapshot_from_disk(
            self._metrics_output_dir / "final_snapshot.json"
        )
        if snap_dict is not None:
            logger.info("Built report from final_snapshot.json")
        elif self.subscriber is not None and self.subscriber.latest is not None:
            snap_dict = snapshot_to_dict(self.subscriber.latest)
            logger.warning(
                "No final_snapshot.json on disk; falling back to last "
                "pub/sub snapshot (state may or may not be terminal)"
            )
        else:
            logger.error("No metrics snapshot available; cannot build report")

        report = (
            _build_report_from_snapshot(snap_dict, self._config)
            if snap_dict is not None
            else None
        )
        # Null the closed handles so the caller's unconditional close() does not
        # re-close them (close() is idempotent, but this keeps the invariant clean).
        self.publisher = None
        if self.subscriber is not None:
            self.subscriber.close()
            self.subscriber = None
        return report

    async def abort(self) -> None:
        """Connect-failure fast path: kill services without a graceful drain."""
        self._teardown(kill=True)

    def close(self) -> None:
        """Exit the ZMQ scope (idempotent) — safe to call unconditionally."""
        self._teardown(kill=False)

    def _teardown(self, *, kill: bool) -> None:
        if self._closed:
            return
        self._closed = True
        if self.publisher is not None:
            try:
                self.publisher.close()
            except Exception as e:  # noqa: BLE001 — teardown best-effort
                logger.warning("Publisher close error: %s", e)
        if kill and self._launcher is not None:
            self._launcher.kill_all()
        if self.subscriber is not None:
            try:
                self.subscriber.close()
            except Exception as e:  # noqa: BLE001 — teardown best-effort
                logger.warning("Subscriber close error: %s", e)
            self.subscriber = None
        if self._zmq_cm is not None:
            self._zmq_cm.__exit__(None, None, None)
            self._zmq_cm = None
