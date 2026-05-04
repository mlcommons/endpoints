from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_user
from app.database import get_db
from app.models import Run, Submission
from app.schemas import RunOut, SubmissionCreate, SubmissionOut, SubmissionUpdate, SubmissionWithRuns

router = APIRouter(prefix="/submissions", tags=["submissions"])


@router.post("", response_model=SubmissionOut, status_code=status.HTTP_201_CREATED)
async def create_submission(
    body: SubmissionCreate,
    user_id: str = Query(..., description="PRISM user ID of the submitter"),
    db: AsyncSession = Depends(get_db),
):
    submission = Submission(
        user_id=user_id,
        benchmark_version=body.benchmark_version,
        division=body.division,
        availability=body.availability,
        early_publish=body.early_publish,
        publication_cycle=body.publication_cycle,
        target_availability_date=body.target_availability_date,
        run_ids=body.run_ids,
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)
    return submission


@router.get("", response_model=list[SubmissionOut])
async def list_submissions(
    user_id: str = Query(..., description="PRISM user ID"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Submission).where(Submission.user_id == user_id))
    return result.scalars().all()


@router.get("/{submission_id}", response_model=SubmissionWithRuns)
async def get_submission(
    submission_id: UUID,
    user_id: str = Query(..., description="PRISM user ID"),
    include_runs: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    submission = await db.get(Submission, submission_id)
    if not submission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")
    if submission.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    runs: list[RunOut] = []
    if include_runs and submission.run_ids:
        result = await db.execute(select(Run).where(Run.id.in_(submission.run_ids)))
        runs = [RunOut.model_validate(r) for r in result.scalars().all()]

    out = SubmissionWithRuns.model_validate(submission)
    out.runs = runs
    return out


@router.patch("/{submission_id}", response_model=SubmissionOut)
async def update_submission(
    submission_id: UUID,
    body: SubmissionUpdate,
    user_id: str = Query(..., description="PRISM user ID"),
    db: AsyncSession = Depends(get_db),
):
    submission = await db.get(Submission, submission_id)
    if not submission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")
    if submission.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    for field, value in body.model_dump(exclude_none=True).items():
        setattr(submission, field, value)

    await db.commit()
    await db.refresh(submission)
    return submission


@router.delete("/{submission_id}", response_model=SubmissionOut)
async def withdraw_submission(
    submission_id: UUID,
    user_id: str = Query(..., description="PRISM user ID"),
    db: AsyncSession = Depends(get_db),
):
    submission = await db.get(Submission, submission_id)
    if not submission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")
    if submission.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if submission.withdrawn_at is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Submission already withdrawn")

    submission.withdrawn_at = datetime.utcnow()
    submission.status = "WITHDRAWN"
    await db.commit()
    await db.refresh(submission)
    return submission
