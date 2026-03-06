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

"""Tests for PostgreSQL backend support in EventRecorder and EventRow.

Unit tests (EventRow SQL generation) run without a database.
Integration tests require DATABASE_URL and are marked run_explicitly.
"""

import os

import pytest
from inference_endpoint.load_generator.events import SessionEvent
from inference_endpoint.metrics.recorder import EventRecorder, EventRow

# ---------------------------------------------------------------------------
# Unit tests: EventRow SQL generation for postgres (no DB required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventRowPostgresSQL:
    def test_to_table_query_uses_postgres_types(self):
        query = EventRow.to_table_query(table_name="events_abc123", backend="postgres")
        assert "events_abc123" in query
        assert "BIGINT" in query
        assert "BYTEA" in query
        # Should NOT contain sqlite types
        assert "INTEGER" not in query
        assert "BLOB" not in query

    def test_to_table_query_sqlite_default_unchanged(self):
        query = EventRow.to_table_query()
        assert "CREATE TABLE IF NOT EXISTS events (" in query
        assert "INTEGER" in query
        assert "BLOB" in query

    def test_insert_query_uses_percent_s_placeholders(self):
        query = EventRow.insert_query(table_name="events_test123", backend="postgres")
        assert "%s" in query
        assert "?" not in query
        assert "events_test123" in query

    def test_insert_query_sqlite_default_unchanged(self):
        query = EventRow.insert_query()
        assert "?" in query
        assert "%s" not in query
        assert "INSERT INTO events (" in query

    def test_custom_table_name_sqlite(self):
        query = EventRow.to_table_query(table_name="my_events", backend="sqlite")
        assert "my_events" in query
        assert "INTEGER" in query


# ---------------------------------------------------------------------------
# Unit tests: EventRecorder backend parameter validation (no DB required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventRecorderBackendValidation:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            EventRecorder(backend="mysql")

    def test_postgres_without_conninfo_or_env_raises(self):
        # Ensure DATABASE_URL is not set for this test
        env = os.environ.copy()
        env.pop("DATABASE_URL", None)
        with pytest.raises(ValueError, match="connection string"):
            EventRecorder(backend="postgres", pg_conninfo=None)

    def test_table_name_sqlite(self):
        """table_name for sqlite is always 'events'."""
        rec = EventRecorder(
            session_id="test_sid",
            backend="sqlite",
            min_memory_req_bytes=128 * 1024 * 1024,
        )
        assert rec.table_name == "events"

    def test_table_name_postgres(self):
        """table_name for postgres includes the session_id."""
        rec = EventRecorder(
            session_id="abc123",
            backend="postgres",
            pg_conninfo="postgresql://fake:fake@localhost/fake",
        )
        assert rec.table_name == "events_abc123"


# ---------------------------------------------------------------------------
# Integration tests: require a real Postgres database (run_explicitly)
# ---------------------------------------------------------------------------

pg_conninfo = os.environ.get("DATABASE_URL")
requires_postgres = pytest.mark.skipif(not pg_conninfo, reason="DATABASE_URL not set")


@pytest.mark.run_explicitly
@requires_postgres
class TestEventRecorderPostgresIntegration:
    def test_write_and_read_events(self):
        """Full write/read cycle against a real Postgres database."""
        import psycopg

        session_id = "pytest_pg_integration"
        table_name = f"events_{session_id}"

        # Clean up from any previous failed run
        with psycopg.connect(pg_conninfo) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()

        try:
            rec = EventRecorder(
                session_id=session_id,
                backend="postgres",
                pg_conninfo=pg_conninfo,
            )
            with rec:
                EventRecorder.record_event(
                    SessionEvent.TEST_STARTED, 1000000, force_commit=True
                )
                EventRecorder.record_event(
                    SessionEvent.TEST_ENDED, 2000000, force_commit=True
                )
                rec.wait_for_writes()

            # Read back and verify
            with psycopg.connect(pg_conninfo) as conn:
                rows = conn.execute(
                    f"SELECT event_type, timestamp_ns FROM {table_name} ORDER BY timestamp_ns"
                ).fetchall()

            assert len(rows) == 2
            assert rows[0][0] == SessionEvent.TEST_STARTED.value
            assert rows[0][1] == 1000000
            assert rows[1][0] == SessionEvent.TEST_ENDED.value
            assert rows[1][1] == 2000000

        finally:
            # Clean up
            with psycopg.connect(pg_conninfo) as conn:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
