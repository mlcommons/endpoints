from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import engine, Base
import app.models  # noqa: F401 — registers models with Base
from app.routers import runs, submissions


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="MLPerf Submission API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(runs.router)
app.include_router(submissions.router)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}
