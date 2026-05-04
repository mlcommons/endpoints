import uuid
from datetime import datetime
from sqlalchemy import Boolean, CheckConstraint, Date, Integer, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
    finished_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    system_info: Mapped[dict] = mapped_column(JSONB, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    result_summary: Mapped[dict] = mapped_column(JSONB, nullable=False)
    archive_uri: Mapped[str] = mapped_column(Text, nullable=False)


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
    status: Mapped[str] = mapped_column(Text, nullable=False, default="COMPLIANCE_CHECKING")
    benchmark_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    division: Mapped[str] = mapped_column(Text, nullable=False)
    availability: Mapped[str] = mapped_column(Text, nullable=False)
    early_publish: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    publication_cycle: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_availability_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    availability_qualified_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    compliance_passed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    first_published_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    peer_review_started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    objection_resolution_started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    finalized_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    withdrawn_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    run_ids: Mapped[list] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=False)
    archive_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    pr_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('COMPLIANCE_CHECKING','PEER_REVIEW_PENDING','CONFIDENTIAL_REVIEW',"
            "'FINALIZED','PUBLISHED','WITHDRAWN','WITHDRAWN_INCOMPLETE','EXPIRED','SUPERSEDED')",
            name="valid_status",
        ),
        CheckConstraint("division IN ('standardized','serviced','rdi')", name="valid_division"),
        CheckConstraint("availability IN ('available','preview','rdi')", name="valid_availability"),
        CheckConstraint(
            "availability != 'preview' OR target_availability_date IS NOT NULL",
            name="preview_requires_target",
        ),
    )


class Submitter(Base):
    __tablename__ = "submitters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    organization: Mapped[str] = mapped_column(Text, nullable=False)
