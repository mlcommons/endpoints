# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect cohort policies or validate submission artifacts."""

import argparse
import json
from pathlib import Path

from . import (
    bundled_policy_path,
    load_policy,
    validate_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate submission artifacts or inspect a cohort policy bundle.",
        epilog=(
            "Examples:\n"
            "  inference-endpoint-validation --submission ./submission\n"
            "    Read artifacts, classify runs, and execute validation checks.\n"
            "  inference-endpoint-validation\n"
            "    Inspect the selected policy bundle."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("policy", nargs="?", type=Path, default=bundled_policy_path())
    parser.add_argument(
        "--submission",
        type=Path,
        metavar="DIRECTORY",
        help=(
            "Read a submission directory, automatically classify its runs, and "
            "execute checks; output validation findings"
        ),
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat validation warnings as failures"
    )
    arguments = parser.parse_args()
    try:
        if arguments.submission:
            policy = load_policy(arguments.policy)
            report = validate_submission(
                arguments.submission,
                policy=policy,
            )
            output = report.model_dump(mode="json")
            output["engine"] = "policy"
            output["version"] = policy.release.version
            output["revision"] = policy.release.revision
            output["policy_digest"] = policy.digest
            print(json.dumps(output, indent=2))
            return 1 if report.errors or (arguments.strict and report.warnings) else 0
        if arguments.strict:
            raise ValueError("Execution options require --submission")
        policy = load_policy(arguments.policy)
        output = {
            "version": policy.release.version,
            "revision": policy.release.revision,
            "policy_digest": policy.digest,
            "rules": len(policy.checks),
            "files": {
                file.value: digest for file, digest in policy.source_digests.items()
            },
        }
        print(json.dumps(output, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
