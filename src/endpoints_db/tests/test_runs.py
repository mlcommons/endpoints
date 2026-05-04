import uuid
from datetime import datetime, timedelta

import pytest
from httpx import AsyncClient

from tests.conftest import RUN_PAYLOAD

USER_ALICE = "u_alice"
USER_BOB = "u_bob"
AUTH_ALICE = {"X-User-Id": USER_ALICE}
AUTH_BOB = {"X-User-Id": USER_BOB}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def create_run(client: AsyncClient, user_id: str = USER_ALICE, **overrides) -> dict:
    payload = {**RUN_PAYLOAD, **overrides}
    res = await client.post("/runs", params={"user_id": user_id}, json=payload)
    assert res.status_code == 201, res.text
    return res.json()


# ── Health ────────────────────────────────────────────────────────────────────

async def test_health(client: AsyncClient):
    res = await client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# ── CREATE ────────────────────────────────────────────────────────────────────

class TestCreateRun:
    async def test_happy_path(self, client: AsyncClient):
        data = await create_run(client)
        assert uuid.UUID(data["id"])
        assert data["user_id"] == USER_ALICE
        assert data["pinned"] is False
        assert data["archive_uri"] == RUN_PAYLOAD["archive_uri"]

    async def test_user_id_from_query_param(self, client: AsyncClient):
        data = await create_run(client, user_id="u_charlie")
        assert data["user_id"] == "u_charlie"

    async def test_missing_user_id_param(self, client: AsyncClient):
        res = await client.post("/runs", json=RUN_PAYLOAD)
        assert res.status_code == 422

    async def test_default_expires_at_is_365_days(self, client: AsyncClient):
        data = await create_run(client)
        expires = datetime.fromisoformat(data["expires_at"])
        expected = datetime.utcnow() + timedelta(days=365)
        assert abs((expires - expected).total_seconds()) < 60

    async def test_explicit_expires_at_is_respected(self, client: AsyncClient):
        explicit = "2030-06-01T00:00:00"
        data = await create_run(client, expires_at=explicit)
        assert data["expires_at"].startswith("2030-06-01")

    async def test_pinned_run(self, client: AsyncClient):
        data = await create_run(client, pinned=True)
        assert data["pinned"] is True

    async def test_missing_required_fields(self, client: AsyncClient):
        res = await client.post("/runs", params={"user_id": USER_ALICE},
                                json={"started_at": "2024-01-01T00:00:00"})
        assert res.status_code == 422


# ── LIST ──────────────────────────────────────────────────────────────────────

class TestListRuns:
    async def test_returns_summary_fields(self, client: AsyncClient):
        await create_run(client)
        res = await client.get("/runs", params={"user_id": USER_ALICE})
        assert res.status_code == 200
        item = res.json()[0]
        assert set(item.keys()) == {"id", "model", "concurrency", "started_at", "finished_at"}

    async def test_model_and_concurrency_extracted_from_config(self, client: AsyncClient):
        await create_run(client)
        item = (await client.get("/runs", params={"user_id": USER_ALICE})).json()[0]
        assert item["model"] == RUN_PAYLOAD["config"]["model"]
        assert item["concurrency"] == RUN_PAYLOAD["config"]["concurrency"]

    async def test_multiple_runs_listed(self, client: AsyncClient):
        await create_run(client)
        await create_run(client)
        res = await client.get("/runs", params={"user_id": USER_ALICE})
        assert len(res.json()) == 2

    async def test_required_user_id_param(self, client: AsyncClient):
        res = await client.get("/runs")
        assert res.status_code == 422

    async def test_empty_list_for_unknown_user(self, client: AsyncClient):
        res = await client.get("/runs", params={"user_id": "nobody"})
        assert res.status_code == 200
        assert res.json() == []

    async def test_isolation_between_users(self, client: AsyncClient):
        await create_run(client, user_id=USER_ALICE)
        await create_run(client, user_id=USER_BOB)
        alice = (await client.get("/runs", params={"user_id": USER_ALICE})).json()
        bob = (await client.get("/runs", params={"user_id": USER_BOB})).json()
        assert len(alice) == 1
        assert len(bob) == 1


# ── GET ───────────────────────────────────────────────────────────────────────

class TestGetRun:
    async def test_happy_path(self, client: AsyncClient):
        created = await create_run(client)
        res = await client.get(f"/runs/{created['id']}", headers=AUTH_ALICE)
        assert res.status_code == 200
        assert res.json()["id"] == created["id"]

    async def test_returns_full_payload(self, client: AsyncClient):
        created = await create_run(client)
        data = (await client.get(f"/runs/{created['id']}", headers=AUTH_ALICE)).json()
        assert data["system_info"] == RUN_PAYLOAD["system_info"]
        assert data["config"] == RUN_PAYLOAD["config"]
        assert data["result_summary"] == RUN_PAYLOAD["result_summary"]

    async def test_wrong_user_forbidden(self, client: AsyncClient):
        created = await create_run(client, user_id=USER_ALICE)
        res = await client.get(f"/runs/{created['id']}", headers=AUTH_BOB)
        assert res.status_code == 403

    async def test_missing_auth_header(self, client: AsyncClient):
        created = await create_run(client)
        res = await client.get(f"/runs/{created['id']}")
        assert res.status_code == 422

    async def test_not_found(self, client: AsyncClient):
        res = await client.get(f"/runs/{uuid.uuid4()}", headers=AUTH_ALICE)
        assert res.status_code == 404


# ── PIN ───────────────────────────────────────────────────────────────────────

class TestPinRun:
    async def test_pin_returns_ok(self, client: AsyncClient):
        created = await create_run(client)
        res = await client.patch(f"/runs/{created['id']}/pin", params={"user_id": USER_ALICE})
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}

    async def test_pin_actually_pins(self, client: AsyncClient):
        created = await create_run(client)
        await client.patch(f"/runs/{created['id']}/pin", params={"user_id": USER_ALICE})
        data = (await client.get(f"/runs/{created['id']}", headers=AUTH_ALICE)).json()
        assert data["pinned"] is True
        assert data["expires_at"] is None

    async def test_pin_nonowner_forbidden(self, client: AsyncClient):
        created = await create_run(client, user_id=USER_ALICE)
        res = await client.patch(f"/runs/{created['id']}/pin", params={"user_id": USER_BOB})
        assert res.status_code == 403

    async def test_pin_missing_user_id(self, client: AsyncClient):
        created = await create_run(client)
        res = await client.patch(f"/runs/{created['id']}/pin")
        assert res.status_code == 422

    async def test_pin_not_found(self, client: AsyncClient):
        res = await client.patch(f"/runs/{uuid.uuid4()}/pin", params={"user_id": USER_ALICE})
        assert res.status_code == 404


# ── UNPIN ─────────────────────────────────────────────────────────────────────

class TestUnpinRun:
    async def test_unpin_returns_ok(self, client: AsyncClient):
        created = await create_run(client)
        await client.patch(f"/runs/{created['id']}/pin", params={"user_id": USER_ALICE})
        res = await client.patch(f"/runs/{created['id']}/unpin", params={"user_id": USER_ALICE})
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}

    async def test_unpin_restores_expiry(self, client: AsyncClient):
        created = await create_run(client)
        run_id = created["id"]
        await client.patch(f"/runs/{run_id}/pin", params={"user_id": USER_ALICE})
        await client.patch(f"/runs/{run_id}/unpin", params={"user_id": USER_ALICE})
        data = (await client.get(f"/runs/{run_id}", headers=AUTH_ALICE)).json()
        assert data["pinned"] is False
        assert data["expires_at"] is not None
        expires = datetime.fromisoformat(data["expires_at"])
        assert abs((expires - (datetime.utcnow() + timedelta(days=365))).total_seconds()) < 60

    async def test_unpin_nonowner_forbidden(self, client: AsyncClient):
        created = await create_run(client, user_id=USER_ALICE)
        res = await client.patch(f"/runs/{created['id']}/unpin", params={"user_id": USER_BOB})
        assert res.status_code == 403

    async def test_unpin_not_found(self, client: AsyncClient):
        res = await client.patch(f"/runs/{uuid.uuid4()}/unpin", params={"user_id": USER_ALICE})
        assert res.status_code == 404


# ── DELETE ────────────────────────────────────────────────────────────────────

class TestDeleteRun:
    async def test_happy_path(self, client: AsyncClient):
        created = await create_run(client)
        run_id = created["id"]

        res = await client.delete(f"/runs/{run_id}", headers=AUTH_ALICE)
        assert res.status_code == 200
        assert res.json() == {"status": "deleted"}

        res = await client.get(f"/runs/{run_id}", headers=AUTH_ALICE)
        assert res.status_code == 404

    async def test_delete_nonowner_forbidden(self, client: AsyncClient):
        created = await create_run(client, user_id=USER_ALICE)
        res = await client.delete(f"/runs/{created['id']}", headers=AUTH_BOB)
        assert res.status_code == 403

    async def test_delete_not_found(self, client: AsyncClient):
        res = await client.delete(f"/runs/{uuid.uuid4()}", headers=AUTH_ALICE)
        assert res.status_code == 404

    async def test_delete_missing_auth(self, client: AsyncClient):
        created = await create_run(client)
        res = await client.delete(f"/runs/{created['id']}")
        assert res.status_code == 422
