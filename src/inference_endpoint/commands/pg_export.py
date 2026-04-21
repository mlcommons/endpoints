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

"""Export a Postgres events table to CSV using the StorageBackend interface."""

import csv
import os
import sys
from pathlib import Path

from ..storage.base import StorageBackend


def _get_conninfo(args) -> str:
    conninfo = args.conninfo or os.environ.get("DATABASE_URL")
    if not conninfo:
        print("Error: provide --conninfo or set DATABASE_URL", file=sys.stderr)
        sys.exit(1)
    return conninfo


def run_pg_export_command(args) -> None:
    from ..storage.db import PostgresBackend

    conninfo = _get_conninfo(args)
    table = args.table
    output = Path(args.output)

    backend: StorageBackend = PostgresBackend(conninfo=conninfo)

    with backend:
        if not backend.exists(
            "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
            params=(table,),
        ):
            print(f"Error: table '{table}' not found", file=sys.stderr)
            sys.exit(1)

        col_names = list(
            backend.list(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s ORDER BY ordinal_position",
                params=(table,),
            )
        )

        rows = backend.read(f"SELECT * FROM {table}")

        if not rows:
            print(f"No rows found in table '{table}'")
            return

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(col_names)
            writer.writerows(rows)

    print(f"Exported {len(rows)} rows from '{table}' → {output}")


def add_pg_export_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "pg-export",
        help="Export a Postgres events table to CSV",
    )
    parser.add_argument(
        "--conninfo",
        default=None,
        help="PostgreSQL connection string (default: $DATABASE_URL)",
    )
    parser.add_argument(
        "--table",
        required=True,
        help="Table name to export (e.g. events_<session_id>)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file path",
    )
    parser.set_defaults(func=run_pg_export_command)
