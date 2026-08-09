#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator tool for a distributed SWE-bench work queue.

    swe_bench_wq.py status  REPORT_DIR
    swe_bench_wq.py merge   REPORT_DIR --run-id RUN
    swe_bench_wq.py requeue REPORT_DIR UNIT_ID [UNIT_ID ...]
    swe_bench_wq.py reap    REPORT_DIR [--apply]

Two deliberate omissions:

* There is no ``merge --all``. A merge is always scoped to one run id; merging
  "everything that looks finished" once combined hundreds of banked results
  from unrelated configurations into a single number.
* There is no way to re-run a unit other than ``requeue``. Deleting a result
  file does not requeue anything, because the claim tombstone still hides the
  unit; ``requeue`` removes the result, the claim and the attempt records
  together and prints exactly what it removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_endpoint.evaluation.swe_bench_distributed.fleet import (  # noqa: E402
    QUEUE_DIRNAME,
)
from inference_endpoint.evaluation.swe_bench_distributed.merge import (  # noqa: E402
    MergeRefusal,
    merge_run,
    verify_inventory,
)
from inference_endpoint.evaluation.swe_bench_distributed.queue import (  # noqa: E402
    WorkQueue,
)
from inference_endpoint.evaluation.swe_bench_distributed.reaper import (  # noqa: E402
    LocalProcessLiveness,
    SlurmStepLiveness,
    reap,
)


def _open(report_dir: Path) -> WorkQueue:
    root = report_dir / QUEUE_DIRNAME
    if not root.exists():
        root = report_dir
    return WorkQueue.open(root)


def cmd_status(args: argparse.Namespace) -> int:
    queue = _open(args.report_dir)
    results = queue.results()
    claimed = queue.claimed_unit_ids()
    inventory = verify_inventory(queue)
    print(f"run:            {queue.plan.run_id}")
    print(f"plan digest:    {queue.plan.digest[:16]}")
    print(f"units:          {len(queue.plan.units)}")
    print(f"  with result:  {len(results)}")
    print(f"  claimed:      {len(claimed)}")
    print(f"  available:    {len(queue.available_unit_ids())}")
    abandoned = [uid for uid, result in results.items() if result.abandoned]
    if abandoned:
        print(f"  ABANDONED:    {len(abandoned)} -> {', '.join(sorted(abandoned)[:8])}")
    infra = [uid for uid, result in results.items() if result.infra_error_count]
    if infra:
        print(f"  infra-damaged:{len(infra)} -> {', '.join(sorted(infra)[:8])}")
    if not inventory.consistent:
        print("\nINVENTORY DISAGREEMENT (claims, results and ids do not agree):")
        for label, values in (
            ("missing results", inventory.missing_units),
            ("results outside the plan", inventory.foreign_units),
            ("unreadable results", inventory.unreadable_units),
            ("ownerless claims", inventory.ownerless_claims),
        ):
            if values:
                print(f"  {label}: {len(values)} -> {', '.join(values[:8])}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    queue = _open(args.report_dir)
    try:
        result = merge_run(queue, args.run_id)
    except MergeRefusal as exc:
        print(f"REFUSED to score run {exc.run_id}:")
        for reason in exc.reasons:
            print(f"  - {reason}")
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_requeue(args: argparse.Namespace) -> int:
    queue = _open(args.report_dir)
    for unit_id in args.unit_ids:
        removed = queue.requeue(unit_id)
        total = sum(len(paths) for paths in removed.values())
        print(f"{unit_id}: removed {total} record(s)")
        for kind, paths in removed.items():
            for path in paths:
                print(f"    {kind}: {path}")
        if total == 0:
            print("    (nothing to remove; the unit was already runnable)")
    return 0


def cmd_reap(args: argparse.Namespace) -> int:
    queue = _open(args.report_dir)
    liveness = SlurmStepLiveness() if args.slurm else LocalProcessLiveness()
    report = reap(
        queue,
        liveness,
        stale_after_s=args.stale,
        step_stale_after_s=args.step_stale,
        apply=args.apply,
    )
    verb = "released" if args.apply else "would release"
    print(f"{verb} {len(report.released)} claim(s)")
    for unit_id in report.released:
        print(f"  {unit_id}")
    if args.verbose:
        for unit_id, reason in sorted(report.kept.items()):
            print(f"  kept {unit_id}: {reason}")
    if not args.apply and report.released:
        print("\nthis was a dry run; pass --apply to release")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="summarise the queue")
    status.add_argument("report_dir", type=Path)
    status.set_defaults(func=cmd_status)

    merge = sub.add_parser("merge", help="score exactly one run")
    merge.add_argument("report_dir", type=Path)
    merge.add_argument(
        "--run-id",
        required=True,
        help="required; a merge is always scoped to one run",
    )
    merge.set_defaults(func=cmd_merge)

    requeue = sub.add_parser(
        "requeue", help="make units runnable again (result + claim + attempts)"
    )
    requeue.add_argument("report_dir", type=Path)
    requeue.add_argument("unit_ids", nargs="+")
    requeue.set_defaults(func=cmd_requeue)

    reap_parser = sub.add_parser("reap", help="release claims whose owner is gone")
    reap_parser.add_argument("report_dir", type=Path)
    reap_parser.add_argument("--apply", action="store_true", help="actually release")
    reap_parser.add_argument("--slurm", action="store_true", help="use SLURM liveness")
    reap_parser.add_argument("--stale", type=float, default=3600.0)
    reap_parser.add_argument("--step-stale", type=float, default=900.0)
    reap_parser.add_argument("--verbose", action="store_true")
    reap_parser.set_defaults(func=cmd_reap)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
