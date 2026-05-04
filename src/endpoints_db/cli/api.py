"""Thin httpx wrapper around the MLPerf Submission API."""
from __future__ import annotations

import sys
from typing import Any

import httpx
import typer

from cli.config import API_URL


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=30)


def _handle(res: httpx.Response) -> dict:
    if res.status_code >= 400:
        typer.echo(f"Error {res.status_code}: {res.text}", err=True)
        raise typer.Exit(code=1)
    return res.json()


# ── Runs ──────────────────────────────────────────────────────────────────────

def runs_create(user_id: str, payload: dict) -> dict:
    with _client() as c:
        return _handle(c.post("/runs", params={"user_id": user_id}, json=payload))


def runs_list(user_id: str) -> list:
    with _client() as c:
        return _handle(c.get("/runs", params={"user_id": user_id}))


def runs_get(run_id: str, user_id: str) -> dict:
    with _client() as c:
        return _handle(c.get(f"/runs/{run_id}", headers={"X-User-Id": user_id}))


def runs_pin(run_id: str, user_id: str) -> dict:
    with _client() as c:
        return _handle(c.patch(f"/runs/{run_id}/pin", params={"user_id": user_id}))


def runs_unpin(run_id: str, user_id: str) -> dict:
    with _client() as c:
        return _handle(c.patch(f"/runs/{run_id}/unpin", params={"user_id": user_id}))


def runs_delete(run_id: str, user_id: str) -> dict:
    with _client() as c:
        return _handle(c.delete(f"/runs/{run_id}", headers={"X-User-Id": user_id}))


# ── Submissions ───────────────────────────────────────────────────────────────

def submissions_create(user_id: str, payload: dict) -> dict:
    with _client() as c:
        return _handle(c.post("/submissions", params={"user_id": user_id}, json=payload))


def submissions_list(user_id: str) -> list:
    with _client() as c:
        return _handle(c.get("/submissions", params={"user_id": user_id}))


def submissions_get(submission_id: str, user_id: str, include_runs: bool = True) -> dict:
    with _client() as c:
        return _handle(
            c.get(
                f"/submissions/{submission_id}",
                params={"user_id": user_id, "include_runs": include_runs},
            )
        )


def submissions_update(submission_id: str, user_id: str, payload: dict) -> dict:
    with _client() as c:
        return _handle(
            c.patch(f"/submissions/{submission_id}", params={"user_id": user_id}, json=payload)
        )


def submissions_withdraw(submission_id: str, user_id: str) -> dict:
    with _client() as c:
        return _handle(
            c.delete(f"/submissions/{submission_id}", params={"user_id": user_id})
        )
