from __future__ import annotations
from datetime import date, datetime
from typing import Any, Optional, List
from uuid import UUID
from pydantic import BaseModel, model_validator

VALID_STATUSES = {
    "COMPLIANCE_CHECKING", "PEER_REVIEW_PENDING", "CONFIDENTIAL_REVIEW",
    "FINALIZED", "PUBLISHED", "WITHDRAWN", "WITHDRAWN_INCOMPLETE",
    "EXPIRED", "SUPERSEDED",
}
VALID_DIVISIONS = {"standardized", "serviced", "rdi"}
VALID_AVAILABILITIES = {"available", "preview", "rdi"}

# ── Shared example data (from real benchmark runs) ────────────────────────────

_EXAMPLE_SYSTEM_INFO = {
    "cpu": "Intel Xeon W9-3595X",
    "gpu": "8x NVIDIA H100 SXM5 80GB",
    "memory_gb": 512,
    "os": "Ubuntu 22.04.4 LTS",
    "framework": "vllm==0.4.2",
}

_EXAMPLE_CONFIG = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "scenario": "llama-3.1-8b_vllm_perf",
    "concurrency": 4,
    "target_qps": 4.2,
    "dataset": "cnn_dailymail",
    "n_samples": 2000,
    "max_output_tokens": 128,
    "temperature": 0.0,
}

_EXAMPLE_RESULT_SUMMARY = {
    "qps": 4.2,
    "completed": 2000,
    "failed": 0,
    "elapsed_time_secs": 475.7,
    "latency_ms": {
        "min": 870.1,
        "max": 11470.3,
        "median": 927.3,
        "mean": 944.7,
        "std_dev": 400.6,
        "p90": 1283.1,
        "p99": 2134.5,
    },
    "ttft_ms": {
        "min": 31.2,
        "max": 335.8,
        "median": 80.5,
        "mean": 80.9,
    },
    "tpot_ms": {
        "median": 8.3,
        "mean": 8.7,
    },
    "accuracy_scores": {
        "rouge1": 38.83,
        "rouge2": 15.96,
        "rougeL": 24.55,
        "rougeLsum": 35.88,
    },
}

_EXAMPLE_RUN_ID = "d5d9873e-5eca-4f8d-a487-4be1cb8b440c"
_EXAMPLE_SUB_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_EXAMPLE_USER_ID = "u_prism_7f3a9b"
_EXAMPLE_ARCHIVE_URI = (
    "s3://mlperf-submissions/runs/2025-04/u_prism_7f3a9b/"
    "llama-3.1-8b_vllm_perf_concurrency4.tar.gz"
)


# ── Runs ──────────────────────────────────────────────────────────────────────

class RunSummary(BaseModel):
    id: UUID
    model: Optional[str]
    concurrency: Optional[int]
    started_at: datetime
    finished_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": _EXAMPLE_RUN_ID,
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "concurrency": 4,
                "started_at": "2025-04-28T09:15:00",
                "finished_at": "2025-04-28T11:42:17",
            }
        }
    }


class RunCreate(BaseModel):
    started_at: datetime
    finished_at: datetime
    expires_at: Optional[datetime] = None
    pinned: bool = False
    system_info: dict[str, Any]
    config: dict[str, Any]
    result_summary: dict[str, Any]
    archive_uri: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "started_at": "2025-04-28T09:15:00",
                "finished_at": "2025-04-28T11:42:17",
                "system_info": _EXAMPLE_SYSTEM_INFO,
                "config": _EXAMPLE_CONFIG,
                "result_summary": _EXAMPLE_RESULT_SUMMARY,
                "archive_uri": _EXAMPLE_ARCHIVE_URI,
            }
        }
    }


class RunOut(BaseModel):
    id: UUID
    user_id: str
    started_at: datetime
    finished_at: datetime
    expires_at: Optional[datetime]
    pinned: bool
    system_info: dict[str, Any]
    config: dict[str, Any]
    result_summary: dict[str, Any]
    archive_uri: str

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": _EXAMPLE_RUN_ID,
                "user_id": _EXAMPLE_USER_ID,
                "started_at": "2025-04-28T09:15:00",
                "finished_at": "2025-04-28T11:42:17",
                "expires_at": "2026-04-28T09:15:00",
                "pinned": False,
                "system_info": _EXAMPLE_SYSTEM_INFO,
                "config": _EXAMPLE_CONFIG,
                "result_summary": _EXAMPLE_RESULT_SUMMARY,
                "archive_uri": _EXAMPLE_ARCHIVE_URI,
            }
        },
    }


# ── Submissions ───────────────────────────────────────────────────────────────

class SubmissionCreate(BaseModel):
    benchmark_version: int = 1
    division: str
    availability: str
    early_publish: bool = False
    publication_cycle: Optional[str] = None
    target_availability_date: Optional[date] = None
    run_ids: List[UUID]

    model_config = {
        "json_schema_extra": {
            "example": {
                "benchmark_version": 1,
                "division": "standardized",
                "availability": "available",
                "early_publish": False,
                "publication_cycle": "2025-04-C1",
                "run_ids": [_EXAMPLE_RUN_ID],
            }
        }
    }

    @model_validator(mode="after")
    def validate_fields(self) -> SubmissionCreate:
        if self.division not in VALID_DIVISIONS:
            raise ValueError(f"division must be one of {VALID_DIVISIONS}")
        if self.availability not in VALID_AVAILABILITIES:
            raise ValueError(f"availability must be one of {VALID_AVAILABILITIES}")
        if self.availability == "preview" and self.target_availability_date is None:
            raise ValueError("target_availability_date required when availability is 'preview'")
        return self


class SubmissionUpdate(BaseModel):
    status: Optional[str] = None
    availability_qualified_at: Optional[datetime] = None
    compliance_passed_at: Optional[datetime] = None
    first_published_at: Optional[datetime] = None
    peer_review_started_at: Optional[datetime] = None
    objection_resolution_started_at: Optional[datetime] = None
    finalized_at: Optional[datetime] = None
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    archive_uri: Optional[str] = None
    publication_cycle: Optional[str] = None
    target_availability_date: Optional[date] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "PEER_REVIEW_PENDING",
                "compliance_passed_at": "2025-05-01T14:30:00",
                "pr_url": "https://github.com/mlcommons/submissions/pull/42",
                "pr_number": 42,
            }
        }
    }

    @model_validator(mode="after")
    def validate_status(self) -> SubmissionUpdate:
        if self.status is not None and self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}")
        return self


class SubmissionOut(BaseModel):
    id: UUID
    user_id: str
    created_at: datetime
    status: str
    benchmark_version: int
    division: str
    availability: str
    early_publish: bool
    publication_cycle: Optional[str]
    target_availability_date: Optional[date]
    availability_qualified_at: Optional[datetime]
    compliance_passed_at: Optional[datetime]
    first_published_at: Optional[datetime]
    peer_review_started_at: Optional[datetime]
    objection_resolution_started_at: Optional[datetime]
    finalized_at: Optional[datetime]
    withdrawn_at: Optional[datetime]
    run_ids: List[UUID]
    archive_uri: Optional[str]
    pr_url: Optional[str]
    pr_number: Optional[int]

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": _EXAMPLE_SUB_ID,
                "user_id": _EXAMPLE_USER_ID,
                "created_at": "2025-04-28T12:00:00",
                "status": "COMPLIANCE_CHECKING",
                "benchmark_version": 1,
                "division": "standardized",
                "availability": "available",
                "early_publish": False,
                "publication_cycle": "2025-04-C1",
                "target_availability_date": None,
                "availability_qualified_at": None,
                "compliance_passed_at": None,
                "first_published_at": None,
                "peer_review_started_at": None,
                "objection_resolution_started_at": None,
                "finalized_at": None,
                "withdrawn_at": None,
                "run_ids": [_EXAMPLE_RUN_ID],
                "archive_uri": None,
                "pr_url": None,
                "pr_number": None,
            }
        },
    }


class SubmissionWithRuns(SubmissionOut):
    runs: List[RunOut] = []

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": _EXAMPLE_SUB_ID,
                "user_id": _EXAMPLE_USER_ID,
                "created_at": "2025-04-28T12:00:00",
                "status": "COMPLIANCE_CHECKING",
                "benchmark_version": 1,
                "division": "standardized",
                "availability": "available",
                "early_publish": False,
                "publication_cycle": "2025-04-C1",
                "target_availability_date": None,
                "availability_qualified_at": None,
                "compliance_passed_at": None,
                "first_published_at": None,
                "peer_review_started_at": None,
                "objection_resolution_started_at": None,
                "finalized_at": None,
                "withdrawn_at": None,
                "run_ids": [_EXAMPLE_RUN_ID],
                "archive_uri": None,
                "pr_url": None,
                "pr_number": None,
                "runs": [
                    {
                        "id": _EXAMPLE_RUN_ID,
                        "user_id": _EXAMPLE_USER_ID,
                        "started_at": "2025-04-28T09:15:00",
                        "finished_at": "2025-04-28T11:42:17",
                        "expires_at": "2026-04-28T09:15:00",
                        "pinned": False,
                        "system_info": _EXAMPLE_SYSTEM_INFO,
                        "config": _EXAMPLE_CONFIG,
                        "result_summary": _EXAMPLE_RESULT_SUMMARY,
                        "archive_uri": _EXAMPLE_ARCHIVE_URI,
                    }
                ],
            }
        },
    }
