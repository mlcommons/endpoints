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

"""SQL writer for event records using SQLAlchemy (swappable SQL backends, default sqlite)."""

from pathlib import Path

import orjson
from inference_endpoint.async_utils.transport.record import EventRecord
from sqlalchemy import BigInteger, Integer, LargeBinary, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .writer import RecordWriter


class Base(DeclarativeBase):
    """Declarative base for event logger SQL models."""

    pass


class EventRowModel(Base):
    """SQLAlchemy model for event rows.

    Schema aligned with metrics/recorder.EventRow but uses EventType topic strings
    (e.g. 'session.ended', 'sample.complete') for event_type instead of legacy Event enum values.
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_uuid: Mapped[str] = mapped_column(String, nullable=False, default="")
    """UUID string identifier for the sample."""

    event_type: Mapped[str] = mapped_column(String, nullable=False)
    """Event type as topic string (e.g. 'session.ended', 'sample.complete')."""

    timestamp_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    """Monotonic timestamp in nanoseconds."""

    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, default=b"")
    """JSON-encoded event data."""


def _record_to_row(record: EventRecord) -> EventRowModel:
    """Convert an EventRecord to an EventRowModel using EventType topic strings."""
    data_bytes = b""
    if record.data:
        data_bytes = orjson.dumps(record.data)
    # event_type.topic is set by EventTypeMeta on each enum member
    topic = record.event_type.topic  # type: ignore[attr-defined]
    return EventRowModel(
        sample_uuid=record.sample_uuid,
        event_type=topic,
        timestamp_ns=record.timestamp_ns,
        data=data_bytes,
    )


class SQLWriter(RecordWriter):
    """Writes event records to a SQL database via SQLAlchemy.

    Uses SQLAlchemy so the backend can be swapped (e.g. sqlite, postgresql).
    Default URL is sqlite at the given path with .db suffix.
    """

    def __init__(
        self,
        path: Path,
        url: str | None = None,
        flush_interval: int | None = None,
        **kwargs: object,
    ):
        """Initialize the SQL writer.

        Args:
            path: Base path for the database. For sqlite default, the file will be path.with_suffix(".db").
            url: Optional SQLAlchemy database URL. If None, uses sqlite at path.with_suffix(".db").
            flush_interval: If set, flush (commit) after every this many records.
        """
        super().__init__(flush_interval=flush_interval)
        if url is None:
            db_path = Path(path).with_suffix(".db")
            url = f"sqlite:///{db_path}"
        self._engine = create_engine(url)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(
            bind=self._engine, autoflush=False, expire_on_commit=False
        )
        self._session = self._session_factory()

    def write(self, record: EventRecord) -> None:
        if self._session is None:
            return
        row = _record_to_row(record)
        self._session.add(row)

    def flush(self) -> None:
        if self._session is not None:
            self._session.commit()
        super().flush()

    def close(self) -> None:
        if self._session is not None:
            try:
                self.flush()
                self._session.close()
            finally:
                self._session = None
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
