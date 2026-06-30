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

"""Tiny Prometheus instant-query poller (stdlib only).

Run as a sidecar: every ``--interval`` seconds it issues ``query`` and prints one
canonical JSONL sample per returned series to stdout:

    {"ts": <epoch_s>, "value": <float>, "label": "<series>"}

One transient HTTP failure must not kill the loop — failures are skipped. The
series ``label`` is the metric's full label set so multi-GPU/exporter results
stay distinct (and energy is not double-counted across series).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request


def _series_label(metric: dict[str, str]) -> str:
    """Stable label from a Prometheus metric's labels (gpu/instance preferred)."""
    for key in ("gpu", "device", "instance", "__name__"):
        if key in metric:
            return str(metric[key])
    if not metric:
        return "default"
    return ",".join(f"{k}={v}" for k, v in sorted(metric.items()))


def _poll_once(url: str, query: str, timeout: float) -> None:
    full = f"{url.rstrip('/')}/api/v1/query?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(full, timeout=timeout) as resp:  # noqa: S310 (user URL)
        body = json.loads(resp.read().decode())
    if body.get("status") != "success":
        return
    now = time.time()
    for series in body.get("data", {}).get("result", []):
        value_pair = series.get("value")
        if not value_pair or len(value_pair) != 2:
            continue
        try:
            value = float(value_pair[1])
        except (ValueError, TypeError):
            continue
        sample = {
            "ts": now,
            "value": value,
            "label": _series_label(series.get("metric", {})),
        }
        sys.stdout.write(json.dumps(sample) + "\n")
    sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=None)
    args = p.parse_args()
    timeout = args.timeout if args.timeout is not None else max(args.interval, 1.0)
    while True:
        start = time.monotonic()
        try:
            _poll_once(args.url, args.query, timeout)
        except Exception:  # noqa: BLE001 — a poll failure must never kill the loop.
            pass
        # Keep cadence steady regardless of request latency.
        elapsed = time.monotonic() - start
        time.sleep(max(0.0, args.interval - elapsed))


if __name__ == "__main__":
    main()
