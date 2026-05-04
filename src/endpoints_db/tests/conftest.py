import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.main import app
from app.database import Base, get_db

TEST_DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5433/mlperf_test"

# NullPool gives each context a fresh connection — avoids asyncpg "operation in progress" errors
test_engine = create_async_engine(TEST_DB_URL, echo=False, poolclass=NullPool)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture(autouse=True)
async def reset_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(lambda c: Base.metadata.create_all(c, checkfirst=False))
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client():
    async def override_get_db():
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ── Shared payloads ───────────────────────────────────────────────────────────

RUN_PAYLOAD = {
    "started_at": "2025-04-28T09:15:00",
    "finished_at": "2025-04-28T11:42:17",
    "system_info": {
        "cpu": "Intel Xeon W9-3595X",
        "gpu": "8x NVIDIA H100 SXM5 80GB",
        "memory_gb": 512,
        "os": "Ubuntu 22.04.4 LTS",
        "framework": "vllm==0.4.2",
    },
    "config": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "scenario": "llama-3.1-8b_vllm_perf",
        "concurrency": 4,
        "target_qps": 4.2,
        "dataset": "cnn_dailymail",
        "n_samples": 2000,
        "max_output_tokens": 128,
        "temperature": 0.0,
    },
    "result_summary": {
        "qps": 4.2,
        "completed": 2000,
        "failed": 0,
        "elapsed_time_secs": 475.7,
        "latency_ms": {"min": 870.1, "max": 11470.3, "median": 927.3, "mean": 944.7, "p99": 2134.5},
        "ttft_ms": {"min": 31.2, "max": 335.8, "median": 80.5, "mean": 80.9},
        "accuracy_scores": {"rouge1": 38.83, "rouge2": 15.96, "rougeL": 24.55},
    },
    "archive_uri": "s3://mlperf-submissions/runs/2025-04/u_alice/llama-3.1-8b_vllm_perf_concurrency4.tar.gz",
}

SUBMISSION_PAYLOAD = {
    "benchmark_version": 1,
    "division": "standardized",
    "availability": "available",
    "early_publish": False,
}
