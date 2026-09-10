# SPDX-FileCopyrightText: Copyright (c) 2026 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Assert that a ``runs create --dry-run`` payload is actually well-formed.

``runs create --dry-run`` exits 0 as long as the three required files parse, so
exit status alone would still pass on a run that measured nothing. This checks
the payload's contents, which is what makes the CI gate meaningful:

* the four API fields the submission API needs are present and non-empty,
* the run window is ordered and non-zero,
* the config that reached the payload is the concurrency point we intended,
* the performance phase actually completed samples, with no failures.

    python .github/submission-cli-test/scripts/check_payload.py --payload payload.json --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REQUIRED_TOP_LEVEL = (
    "benchmark_version",
    "started_at",
    "finished_at",
    "system_info",
    "config",
    "result_summary",
)


def check(payload: dict[str, Any], concurrency: int) -> list[str]:
    """Return a list of problems with *payload*; empty means it passed."""
    problems: list[str] = []

    for key in _REQUIRED_TOP_LEVEL:
        if key not in payload:
            problems.append(f"payload is missing {key!r}")
        elif payload[key] in (None, "", {}, []):
            problems.append(f"payload field {key!r} is empty")

    if problems:
        # Everything below indexes into these fields; bail rather than pile on
        # cascading KeyErrors.
        return problems

    if payload["benchmark_version"] == "unknown":
        problems.append(
            "benchmark_version is 'unknown' (git_sha missing from result_summary)"
        )

    try:
        started = datetime.fromisoformat(payload["started_at"])
        finished = datetime.fromisoformat(payload["finished_at"])
        if finished <= started:
            problems.append(f"finished_at {finished} is not after started_at {started}")
    except ValueError as exc:
        problems.append(f"unparseable run window: {exc}")

    config = payload["config"]
    actual = config.get("settings", {}).get("load_pattern", {})
    if actual.get("type") != "concurrency":
        problems.append(
            f"load_pattern.type is {actual.get('type')!r}, expected 'concurrency' (rules §6.1)"
        )
    if actual.get("target_concurrency") != concurrency:
        problems.append(
            f"target_concurrency is {actual.get('target_concurrency')!r}, expected {concurrency}"
        )

    client = config.get("settings", {}).get("client", {})
    if client.get("stream_all_chunks") is not True:
        problems.append(
            "settings.client.stream_all_chunks is not true; per-token timing "
            "(and therefore TPOT) is not measurable (rules §6.5)"
        )

    summary = payload["result_summary"]
    completed = summary.get("n_samples_completed") or 0
    failed = summary.get("n_samples_failed") or 0
    if completed <= 0:
        problems.append(
            f"n_samples_completed is {completed}; the point measured nothing"
        )
    if failed:
        problems.append(f"n_samples_failed is {failed}; the endpoint dropped requests")
    if not summary.get("complete"):
        problems.append(
            "result_summary.complete is false; the run did not finish cleanly"
        )

    system_info = payload["system_info"]
    for key in ("system_name", "division", "max_supported_concurrency"):
        if not system_info.get(key):
            problems.append(f"system_desc.json is missing {key!r}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload", type=Path, required=True, help="dry-run payload JSON"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="the point's expected concurrency",
    )
    args = parser.parse_args()

    try:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(
            f"FAIL c={args.concurrency}: dry-run output is not valid JSON: {exc}",
            file=sys.stderr,
        )
        return 1

    problems = check(payload, args.concurrency)
    if problems:
        print(f"FAIL c={args.concurrency}: payload did not validate", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    summary = payload["result_summary"]

    def fmt(value: Any) -> str:
        # tps is None whenever no tokenizer was attached, so this must not
        # assume a number is present.
        return f"{value:,.1f}" if isinstance(value, int | float) else "n/a"

    print(
        f"OK c={args.concurrency}: payload valid "
        f"({summary['n_samples_completed']} samples, qps={fmt(summary.get('qps'))}, "
        f"tps={fmt(summary.get('tps'))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
