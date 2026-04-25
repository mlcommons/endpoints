#!/usr/bin/env python3
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

"""Validate multi-turn JSONL dataset files against multi_turn_dataset_schema.json.

Checks each row's structure against the JSON schema (field types, required fields,
tool_results shape, etc.). Does NOT check cross-row invariants such as turn
numbering or role sequences — those are enforced by MultiTurnDataset at load time.

Usage:
    python scripts/validate_jsonl_schema.py FILE [FILE ...]
    python scripts/validate_jsonl_schema.py examples/09_MultiTurn/datasets/agentic_coding_flat.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print(
        "Error: jsonschema not installed. Run: pip install jsonschema", file=sys.stderr
    )
    sys.exit(1)


def validate_file(path: Path, schema: dict, max_errors: int = 50) -> int:
    """Validate every row in a JSONL file against the schema.

    Returns the number of validation errors found.
    """
    errors: list[str] = []
    validator = jsonschema.Draft7Validator(schema)

    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  line {lineno}: JSON parse error: {e}")
                if len(errors) >= max_errors:
                    break
                continue

            conv_id = row.get("conversation_id", "<unknown>")
            turn = row.get("turn", "?")
            role = row.get("role", "?")

            row_errors = list(validator.iter_errors(row))
            for err in row_errors:
                path_str = " -> ".join(str(p) for p in err.absolute_path) or "(root)"
                errors.append(
                    f"  line {lineno} [{conv_id} turn={turn} role={role}] "
                    f"@ {path_str}: {err.message}"
                )

            if len(errors) >= max_errors:
                errors.append(f"  ... stopping after {max_errors} errors")
                break

    if errors:
        print(f"FAIL {path.name}: {len(errors)} error(s)")
        for msg in errors:
            print(msg)
    else:
        print(f"OK   {path.name}")

    return len(errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate multi-turn JSONL files against multi_turn_dataset_schema.json."
    )
    parser.add_argument("files", nargs="+", type=Path, help="JSONL files to validate")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).parent.parent / "multi_turn_dataset_schema.json",
        help="Path to the JSON schema file (default: multi_turn_dataset_schema.json)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Stop reporting after this many errors per file (default: 50)",
    )
    args = parser.parse_args()

    if not args.schema.exists():
        print(f"Error: schema not found: {args.schema}", file=sys.stderr)
        sys.exit(1)

    schema = json.load(args.schema.open())

    total_errors = 0
    for path in args.files:
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            total_errors += 1
            continue
        total_errors += validate_file(path, schema, max_errors=args.max_errors)

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
