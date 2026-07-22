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

"""Profiler trigger protocol for benchmark runs (vLLM /start_profile, /stop_profile).

``ProfileController`` owns the whole trigger lifecycle: it pre-derives the per-endpoint
URLs up front (so a misconfigured run fails before any traffic), fires ``/start_profile``
when the performance phase begins, fires ``/stop_profile`` for every start that succeeded,
and produces the ``{engine, starts, stops}`` payload rendered into ``report.txt`` and a
sibling ``profiling.json``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, TextIO
from urllib import error as urllib_error
from urllib import request as urllib_request

from inference_endpoint.config.schema import ProfilerEngine

logger = logging.getLogger(__name__)


# (start_path, stop_path) for each supported inference engine's profiling
# protocol. Add a row when introducing a new ProfilerEngine variant.
_PROFILE_PATHS: dict[ProfilerEngine, tuple[str, str]] = {
    ProfilerEngine.VLLM: ("/start_profile", "/stop_profile"),
}


def _derive_profile_urls(
    endpoints: list[str], engine: ProfilerEngine, action: str
) -> list[str]:
    """One profile URL per endpoint, derived from the engine's HTTP protocol.

    For vLLM: strip a trailing ``/v1`` from each endpoint and append
    ``/{start,stop}_profile``. ``action`` is ``"start"`` or ``"stop"``.
    """
    if not endpoints:
        raise ValueError(
            f"profiling.engine={engine.value} but endpoint_config.endpoints "
            f"is empty; cannot derive {action} URLs"
        )
    start_path, stop_path = _PROFILE_PATHS[engine]
    path = start_path if action == "start" else stop_path
    urls: list[str] = []
    for ep in endpoints:
        base = ep.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        urls.append(f"{base.rstrip('/')}{path}")
    return urls


def _post_profile(url: str) -> dict[str, Any]:
    """POST {url} with empty body; never raises. Returns a record dict suitable
    for report.txt rendering and profiling.json serialization."""
    record: dict[str, Any] = {
        "url": url,
        "sent_at_ns": time.monotonic_ns(),
        "sent_at_iso": datetime.now().isoformat(timespec="milliseconds"),
        "status": None,
        "error": None,
    }
    req = urllib_request.Request(url, method="POST", data=b"")
    try:
        with urllib_request.urlopen(req, timeout=2) as resp:
            record["status"] = resp.status
    except urllib_error.HTTPError as e:
        record["status"] = e.code
        record["error"] = f"{e.code} {e.reason}"
    except Exception as e:  # noqa: BLE001 — profile failures must never abort a run
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def _render_profile_status(rec: dict[str, Any]) -> str:
    status = rec.get("status")
    error = rec.get("error")
    if status == 200:
        return "200 OK"
    if status == 404:
        return (
            "404 (profiling not enabled on server — pass "
            "--profiler-config.profiler=... to server)"
        )
    if error:
        return error
    if status is not None:
        return str(status)
    return "ERROR"


def _write_profiling_section(f: TextIO, profiling: dict[str, Any]) -> None:
    """Append the Profiling section to report.txt (called after report.display)."""
    starts = profiling.get("starts", [])
    stops = profiling.get("stops", [])
    f.write("\n------------------- Profiling -------------------\n")
    f.write(f"Engine: {profiling.get('engine', 'unknown')}\n")
    f.write("Start:\n")
    for rec in starts:
        f.write(
            f"  POST {rec['url']} @ {rec['sent_at_iso']} → "
            f"{_render_profile_status(rec)}\n"
        )
    if stops:
        f.write("Stop:\n")
        for rec in stops:
            suffix = (
                " (from abort handler)" if rec.get("stop_reason") == "abort" else ""
            )
            f.write(
                f"  POST {rec['url']} @ {rec['sent_at_iso']} → "
                f"{_render_profile_status(rec)}{suffix}\n"
            )
    if starts and stops:
        first_start = min(r["sent_at_ns"] for r in starts)
        last_stop = max(r["sent_at_ns"] for r in stops)
        f.write(f"Trigger span: {(last_stop - first_start) / 1e9:.2f} s\n")
    f.write(
        "\nNote: actual trace window is bounded by server-side "
        "--profiler-config.delay_iterations and "
        "--profiler-config.max_iterations.\n"
        "Trace artifact path is in server stdout.\n"
    )


class ProfileController:
    """Owns the profiler start/stop trigger lifecycle for one benchmark run.

    Disabled (a no-op) when ``engine is None``. When enabled it derives the start
    and stop URLs at construction — so an engine set with no endpoints fails before
    the run — and only fires ``/stop_profile`` for the ``/start_profile`` calls that
    returned 200.
    """

    def __init__(
        self,
        engine: ProfilerEngine | None,
        endpoints: list[str],
        urls_override: list[str] | None,
    ) -> None:
        self._engine = engine
        self._start_urls: list[str] = []
        self._stop_urls: list[str] = []
        self._starts: list[dict[str, Any]] = []
        self._stops: list[dict[str, Any]] = []
        if engine is not None:
            # Preserve the truthiness fallback: an explicit empty urls override
            # falls through to endpoint_config.endpoints.
            profile_endpoints = urls_override or endpoints
            self._start_urls = _derive_profile_urls(profile_endpoints, engine, "start")
            self._stop_urls = _derive_profile_urls(profile_endpoints, engine, "stop")

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    def start(self) -> None:
        """Fire /start_profile sequentially before any perf request is issued."""
        for url in self._start_urls:
            rec = _post_profile(url)
            if rec["status"] == 200:
                logger.info("Profile start: %s -> 200 OK", url)
            else:
                logger.warning(
                    "Profile start: %s -> %s", url, rec["error"] or rec["status"]
                )
            self._starts.append(rec)

    def stop(self, completed_normally: bool) -> None:
        """Fire /stop_profile for every start that returned 200.

        Unifies the clean phase-end path and the abort path — both call this.
        """
        if not self._starts:
            return
        stop_reason = "phase_end" if completed_normally else "abort"
        for i, start_rec in enumerate(self._starts):
            if start_rec["status"] != 200 or i >= len(self._stop_urls):
                continue
            rec = _post_profile(self._stop_urls[i])
            rec["stop_reason"] = stop_reason
            if rec["status"] == 200:
                logger.info("Profile stop: %s -> 200 OK", self._stop_urls[i])
            else:
                logger.warning(
                    "Profile stop: %s -> %s",
                    self._stop_urls[i],
                    rec["error"] or rec["status"],
                )
            self._stops.append(rec)

    def payload(self) -> dict[str, Any] | None:
        """The {engine, starts, stops} record, or None when profiling is disabled.

        A payload is emitted whenever an engine is configured, even if no
        performance phase fired (no starts/stops) — matching the pre-refactor
        behavior where the payload's presence tracks the engine, not the triggers.
        """
        if self._engine is None:
            return None
        return {
            "engine": self._engine.value,
            "starts": self._starts,
            "stops": self._stops,
        }
