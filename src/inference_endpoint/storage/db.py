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

"""Database storage backends: SQLite and PostgreSQL."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

from .base import StorageBackend


class SQLiteBackend(StorageBackend):
    """SQLite backend.

    write(key=sql, data=params) — single row insert; pass batch=True in kwargs for executemany.
    read(key=sql, params=()) — returns list of rows.
    exists/delete/list are no-ops (not applicable to relational tables).
    """

    def __init__(self, path: str, read_only: bool = False) -> None:
        self.path = path
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None
        self._cur: sqlite3.Cursor | None = None

    def connect(self) -> None:
        uri = f"file:{self.path}?mode=ro" if self.read_only else self.path
        self._conn = sqlite3.connect(uri, uri=self.read_only)
        self._cur = self._conn.cursor()

    def write(self, key: str, data: Any, **kwargs) -> None:
        assert self._cur is not None and self._conn is not None
        if kwargs.get("batch"):
            self._cur.executemany(key, data)
        else:
            self._cur.execute(key, data)
        self._conn.commit()

    def read(self, key: str, **kwargs) -> list[Any]:
        assert self._cur is not None
        self._cur.execute(key, kwargs.get("params", ()))
        return self._cur.fetchall()

    def delete(self, key: str) -> None:
        raise NotImplementedError("Use write() with a DELETE statement")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("Use read() with a SELECT statement")

    def list(self, prefix: str = "") -> Iterator[str]:
        raise NotImplementedError("Use read() with a SELECT statement")

    def close(self) -> None:
        if self._cur:
            self._cur.close()
        if self._conn:
            self._conn.close()
        self._cur = None
        self._conn = None


class PostgresBackend(StorageBackend):
    """PostgreSQL backend via psycopg3.

    write(key=sql, data=params) — single row insert; pass batch=True in kwargs for executemany.
    read(key=sql, params=()) — returns list of rows.
    exists/delete/list are no-ops (not applicable to relational tables).
    """

    def __init__(self, conninfo: str, autocommit: bool = True) -> None:
        self.conninfo = conninfo
        self.autocommit = autocommit
        self._conn = None
        self._cur = None

    def connect(self) -> None:
        import psycopg

        self._conn = psycopg.connect(self.conninfo, autocommit=self.autocommit)
        self._cur = self._conn.cursor()

    def write(self, key: str, data: Any, **kwargs) -> None:
        assert self._cur is not None
        if kwargs.get("batch"):
            self._cur.executemany(key, data)
        else:
            self._cur.execute(key, data)
        if not self.autocommit:
            self._conn.commit()

    def read(self, key: str, **kwargs) -> list[Any]:
        assert self._cur is not None
        self._cur.execute(key, kwargs.get("params", ()))
        return self._cur.fetchall()

    def delete(self, key: str) -> None:
        raise NotImplementedError("Use write() with a DELETE statement")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("Use read() with a SELECT statement")

    def list(self, prefix: str = "") -> Iterator[str]:
        raise NotImplementedError("Use read() with a SELECT statement")

    def close(self) -> None:
        if self._cur:
            self._cur.close()
        if self._conn:
            self._conn.close()
        self._cur = None
        self._conn = None
