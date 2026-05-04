-- Reference schema — tables are created automatically by SQLAlchemy on startup.
-- Run this manually if you need to seed a fresh database outside the app.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS runs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT        NOT NULL,
    started_at      TIMESTAMP   NOT NULL,
    finished_at     TIMESTAMP   NOT NULL,
    expires_at      TIMESTAMP,
    pinned          BOOLEAN     NOT NULL DEFAULT FALSE,
    system_info     JSONB       NOT NULL,
    config          JSONB       NOT NULL,
    result_summary  JSONB       NOT NULL,
    archive_uri     TEXT        NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs (user_id);

CREATE TABLE IF NOT EXISTS submissions (
    id                                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                             TEXT        NOT NULL,
    created_at                          TIMESTAMP   NOT NULL DEFAULT NOW(),
    status                              TEXT        NOT NULL DEFAULT 'COMPLIANCE_CHECKING',
    benchmark_version                   INT         NOT NULL DEFAULT 1,
    division                            TEXT        NOT NULL,
    availability                        TEXT        NOT NULL,
    early_publish                       BOOLEAN     NOT NULL DEFAULT FALSE,
    publication_cycle                   TEXT,
    target_availability_date            DATE,
    availability_qualified_at           TIMESTAMP,
    compliance_passed_at                TIMESTAMP,
    first_published_at                  TIMESTAMP,
    peer_review_started_at              TIMESTAMP,
    objection_resolution_started_at     TIMESTAMP,
    finalized_at                        TIMESTAMP,
    withdrawn_at                        TIMESTAMP,
    run_ids                             UUID[]      NOT NULL,
    archive_uri                         TEXT,
    pr_url                              TEXT,
    pr_number                           INT,

    CONSTRAINT valid_status CHECK (status IN (
        'COMPLIANCE_CHECKING', 'PEER_REVIEW_PENDING', 'CONFIDENTIAL_REVIEW',
        'FINALIZED', 'PUBLISHED', 'WITHDRAWN', 'WITHDRAWN_INCOMPLETE',
        'EXPIRED', 'SUPERSEDED'
    )),
    CONSTRAINT valid_division     CHECK (division     IN ('standardized', 'serviced', 'rdi')),
    CONSTRAINT valid_availability CHECK (availability IN ('available', 'preview', 'rdi')),
    CONSTRAINT preview_requires_target CHECK (
        availability != 'preview' OR target_availability_date IS NOT NULL
    )
);

CREATE TABLE IF NOT EXISTS submitters (
    id           UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT  NOT NULL,
    organization TEXT  NOT NULL
);
