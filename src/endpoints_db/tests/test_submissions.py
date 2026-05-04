import uuid

import pytest
from httpx import AsyncClient

from tests.conftest import RUN_PAYLOAD, SUBMISSION_PAYLOAD

USER_ALICE = "u_alice"
USER_BOB = "u_bob"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def create_run(client: AsyncClient, user_id: str = USER_ALICE) -> str:
    res = await client.post("/runs", params={"user_id": user_id}, json=RUN_PAYLOAD)
    assert res.status_code == 201, res.text
    return res.json()["id"]


async def create_submission(client: AsyncClient, user_id: str = USER_ALICE, **overrides) -> dict:
    run_id = await create_run(client, user_id=user_id)
    payload = {**SUBMISSION_PAYLOAD, "run_ids": [run_id], **overrides}
    res = await client.post("/submissions", params={"user_id": user_id}, json=payload)
    assert res.status_code == 201, res.text
    return res.json()


# ── CREATE ────────────────────────────────────────────────────────────────────

class TestCreateSubmission:
    async def test_happy_path(self, client: AsyncClient):
        data = await create_submission(client)
        assert uuid.UUID(data["id"])
        assert data["user_id"] == USER_ALICE
        assert data["status"] == "COMPLIANCE_CHECKING"
        assert data["division"] == "standardized"
        assert data["availability"] == "available"
        assert len(data["run_ids"]) == 1

    async def test_missing_user_id_param(self, client: AsyncClient):
        run_id = await create_run(client)
        res = await client.post("/submissions", json={**SUBMISSION_PAYLOAD, "run_ids": [run_id]})
        assert res.status_code == 422

    async def test_invalid_division(self, client: AsyncClient):
        run_id = await create_run(client)
        payload = {**SUBMISSION_PAYLOAD, "run_ids": [run_id], "division": "invalid"}
        res = await client.post("/submissions", params={"user_id": USER_ALICE}, json=payload)
        assert res.status_code == 422

    async def test_invalid_availability(self, client: AsyncClient):
        run_id = await create_run(client)
        payload = {**SUBMISSION_PAYLOAD, "run_ids": [run_id], "availability": "bad"}
        res = await client.post("/submissions", params={"user_id": USER_ALICE}, json=payload)
        assert res.status_code == 422

    async def test_preview_requires_target_date(self, client: AsyncClient):
        run_id = await create_run(client)
        payload = {**SUBMISSION_PAYLOAD, "run_ids": [run_id], "availability": "preview"}
        res = await client.post("/submissions", params={"user_id": USER_ALICE}, json=payload)
        assert res.status_code == 422

    async def test_preview_with_target_date_ok(self, client: AsyncClient):
        run_id = await create_run(client)
        payload = {
            **SUBMISSION_PAYLOAD,
            "run_ids": [run_id],
            "availability": "preview",
            "target_availability_date": "2025-06-01",
        }
        res = await client.post("/submissions", params={"user_id": USER_ALICE}, json=payload)
        assert res.status_code == 201
        assert res.json()["target_availability_date"] == "2025-06-01"

    async def test_multiple_run_ids(self, client: AsyncClient):
        run1 = await create_run(client)
        run2 = await create_run(client)
        payload = {**SUBMISSION_PAYLOAD, "run_ids": [run1, run2]}
        res = await client.post("/submissions", params={"user_id": USER_ALICE}, json=payload)
        assert res.status_code == 201
        assert len(res.json()["run_ids"]) == 2

    async def test_all_divisions(self, client: AsyncClient):
        for division in ("standardized", "serviced", "rdi"):
            run_id = await create_run(client)
            payload = {**SUBMISSION_PAYLOAD, "run_ids": [run_id], "division": division}
            res = await client.post("/submissions", params={"user_id": USER_ALICE}, json=payload)
            assert res.status_code == 201, f"division={division} failed: {res.text}"

    async def test_all_availabilities(self, client: AsyncClient):
        for avail in ("available", "rdi"):
            run_id = await create_run(client)
            payload = {**SUBMISSION_PAYLOAD, "run_ids": [run_id], "availability": avail}
            res = await client.post("/submissions", params={"user_id": USER_ALICE}, json=payload)
            assert res.status_code == 201, f"availability={avail} failed: {res.text}"


# ── LIST ──────────────────────────────────────────────────────────────────────

class TestListSubmissions:
    async def test_list_by_user_id(self, client: AsyncClient):
        await create_submission(client, user_id=USER_ALICE)
        await create_submission(client, user_id=USER_ALICE)
        res = await client.get("/submissions", params={"user_id": USER_ALICE})
        assert res.status_code == 200
        assert len(res.json()) == 2

    async def test_required_user_id_param(self, client: AsyncClient):
        res = await client.get("/submissions")
        assert res.status_code == 422

    async def test_isolation_between_users(self, client: AsyncClient):
        await create_submission(client, user_id=USER_ALICE)
        await create_submission(client, user_id=USER_BOB)
        alice = (await client.get("/submissions", params={"user_id": USER_ALICE})).json()
        bob = (await client.get("/submissions", params={"user_id": USER_BOB})).json()
        assert len(alice) == 1 and alice[0]["user_id"] == USER_ALICE
        assert len(bob) == 1 and bob[0]["user_id"] == USER_BOB


# ── GET ───────────────────────────────────────────────────────────────────────

class TestGetSubmission:
    async def test_returns_submission_with_runs(self, client: AsyncClient):
        sub = await create_submission(client)
        res = await client.get(f"/submissions/{sub['id']}",
                               params={"user_id": USER_ALICE, "include_runs": True})
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == sub["id"]
        assert len(data["runs"]) == 1

    async def test_include_runs_false_returns_empty_runs(self, client: AsyncClient):
        sub = await create_submission(client)
        res = await client.get(f"/submissions/{sub['id']}",
                               params={"user_id": USER_ALICE, "include_runs": False})
        assert res.status_code == 200
        assert res.json()["runs"] == []

    async def test_wrong_user_forbidden(self, client: AsyncClient):
        sub = await create_submission(client, user_id=USER_ALICE)
        res = await client.get(f"/submissions/{sub['id']}", params={"user_id": USER_BOB})
        assert res.status_code == 403

    async def test_not_found(self, client: AsyncClient):
        res = await client.get(f"/submissions/{uuid.uuid4()}", params={"user_id": USER_ALICE})
        assert res.status_code == 404

    async def test_lifecycle_fields_initially_null(self, client: AsyncClient):
        sub = await create_submission(client)
        data = (await client.get(f"/submissions/{sub['id']}",
                                 params={"user_id": USER_ALICE})).json()
        for field in ("compliance_passed_at", "first_published_at", "withdrawn_at",
                      "peer_review_started_at", "finalized_at"):
            assert data[field] is None


# ── UPDATE (PATCH) ────────────────────────────────────────────────────────────

class TestUpdateSubmission:
    async def test_update_status(self, client: AsyncClient):
        sub = await create_submission(client)
        res = await client.patch(f"/submissions/{sub['id']}",
                                 params={"user_id": USER_ALICE},
                                 json={"status": "PEER_REVIEW_PENDING"})
        assert res.status_code == 200
        assert res.json()["status"] == "PEER_REVIEW_PENDING"

    async def test_update_pr_fields(self, client: AsyncClient):
        sub = await create_submission(client)
        res = await client.patch(f"/submissions/{sub['id']}",
                                 params={"user_id": USER_ALICE},
                                 json={"pr_url": "https://github.com/org/repo/pull/42",
                                       "pr_number": 42})
        assert res.status_code == 200
        data = res.json()
        assert data["pr_url"] == "https://github.com/org/repo/pull/42"
        assert data["pr_number"] == 42

    async def test_update_lifecycle_timestamps(self, client: AsyncClient):
        sub = await create_submission(client)
        ts = "2024-06-01T12:00:00"
        res = await client.patch(f"/submissions/{sub['id']}",
                                 params={"user_id": USER_ALICE},
                                 json={"compliance_passed_at": ts, "peer_review_started_at": ts})
        assert res.status_code == 200
        data = res.json()
        assert data["compliance_passed_at"] is not None
        assert data["peer_review_started_at"] is not None

    async def test_invalid_status_rejected(self, client: AsyncClient):
        sub = await create_submission(client)
        res = await client.patch(f"/submissions/{sub['id']}",
                                 params={"user_id": USER_ALICE},
                                 json={"status": "INVALID_STATUS"})
        assert res.status_code == 422

    async def test_wrong_user_forbidden(self, client: AsyncClient):
        sub = await create_submission(client, user_id=USER_ALICE)
        res = await client.patch(f"/submissions/{sub['id']}",
                                 params={"user_id": USER_BOB},
                                 json={"status": "PEER_REVIEW_PENDING"})
        assert res.status_code == 403

    async def test_not_found(self, client: AsyncClient):
        res = await client.patch(f"/submissions/{uuid.uuid4()}",
                                 params={"user_id": USER_ALICE},
                                 json={"status": "PEER_REVIEW_PENDING"})
        assert res.status_code == 404

    async def test_all_valid_statuses(self, client: AsyncClient):
        valid = [
            "COMPLIANCE_CHECKING", "PEER_REVIEW_PENDING", "CONFIDENTIAL_REVIEW",
            "FINALIZED", "PUBLISHED", "WITHDRAWN", "WITHDRAWN_INCOMPLETE",
            "EXPIRED", "SUPERSEDED",
        ]
        for s in valid:
            sub = await create_submission(client)
            res = await client.patch(f"/submissions/{sub['id']}",
                                     params={"user_id": USER_ALICE},
                                     json={"status": s})
            assert res.status_code == 200, f"status={s} rejected: {res.text}"


# ── DELETE (withdraw) ─────────────────────────────────────────────────────────

class TestWithdrawSubmission:
    async def test_happy_path(self, client: AsyncClient):
        sub = await create_submission(client)
        res = await client.delete(f"/submissions/{sub['id']}",
                                  params={"user_id": USER_ALICE})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "WITHDRAWN"
        assert data["withdrawn_at"] is not None

    async def test_already_withdrawn_returns_409(self, client: AsyncClient):
        sub = await create_submission(client)
        await client.delete(f"/submissions/{sub['id']}", params={"user_id": USER_ALICE})
        res = await client.delete(f"/submissions/{sub['id']}", params={"user_id": USER_ALICE})
        assert res.status_code == 409

    async def test_wrong_user_forbidden(self, client: AsyncClient):
        sub = await create_submission(client, user_id=USER_ALICE)
        res = await client.delete(f"/submissions/{sub['id']}", params={"user_id": USER_BOB})
        assert res.status_code == 403

    async def test_not_found(self, client: AsyncClient):
        res = await client.delete(f"/submissions/{uuid.uuid4()}",
                                  params={"user_id": USER_ALICE})
        assert res.status_code == 404
