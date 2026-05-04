from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_user
from app.database import get_db
from app.models import Run
from app.schemas import RunCreate, RunOut, RunSummary

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunOut, status_code=status.HTTP_201_CREATED)
async def create_run(
    body: RunCreate,
    user_id: str = Query(..., description="PRISM user ID of the submitter"),
    db: AsyncSession = Depends(get_db),
):
    expires_at = body.expires_at
    if expires_at is None:
        expires_at = datetime.utcnow() + timedelta(days=365)

    run = Run(
        user_id=user_id,
        started_at=body.started_at,
        finished_at=body.finished_at,
        expires_at=expires_at,
        pinned=body.pinned,
        system_info=body.system_info,
        config=body.config,
        result_summary=body.result_summary,
        archive_uri=body.archive_uri,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


@router.get("", response_model=list[RunSummary])
async def list_runs(
    user_id: str = Query(..., description="PRISM user ID"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Run).where(Run.user_id == user_id))
    runs = result.scalars().all()
    return [
        RunSummary(
            id=r.id,
            model=r.config.get("model") if r.config else None,
            concurrency=r.config.get("concurrency") if r.config else None,
            started_at=r.started_at,
            finished_at=r.finished_at,
        )
        for r in runs
    ]


@router.get("/{run_id}", response_model=RunOut)
async def get_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(require_user),
):
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if run.user_id != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return run


@router.patch("/{run_id}/pin")
async def pin_run(
    run_id: UUID,
    user_id: str = Query(..., description="PRISM user ID"),
    db: AsyncSession = Depends(get_db),
):
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if run.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    run.pinned = True
    run.expires_at = None
    await db.commit()
    return {"status": "ok"}


@router.patch("/{run_id}/unpin")
async def unpin_run(
    run_id: UUID,
    user_id: str = Query(..., description="PRISM user ID"),
    db: AsyncSession = Depends(get_db),
):
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if run.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    run.pinned = False
    run.expires_at = datetime.utcnow() + timedelta(days=365)
    await db.commit()
    return {"status": "ok"}


@router.delete("/{run_id}")
async def delete_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(require_user),
):
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if run.user_id != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    await db.delete(run)
    await db.commit()
    return {"status": "deleted"}
