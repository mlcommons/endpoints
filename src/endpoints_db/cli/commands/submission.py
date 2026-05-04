"""inference-endpoint submission <subcommand>"""
from __future__ import annotations

import json
from typing import Annotated, Optional

import typer

from cli import api as API
from cli.auth import user_id_from_token

app = typer.Typer(help="Manage benchmark submissions.")

TOKEN_OPT = Annotated[Optional[str], typer.Option("--token", "-t", envvar="PRISM_TOKEN", help="PRISM auth token (prsm_<user>_<org>)")]

VALID_DIVISIONS = ("standardized", "serviced", "rdi")
VALID_AVAILABILITIES = ("available", "preview", "rdi")


def _print(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command("submit")
def submit(
    benchmark_version: Annotated[str, typer.Option("--benchmark-version", "-b", help="MLPerf benchmark version, e.g. v5.0")],
    run_ids: Annotated[list[str], typer.Option("--run-id", "-r", help="Run UUID (repeat for multiple)")],
    division: Annotated[str, typer.Option("--division", "-d", help="standardized | serviced | rdi")] = "standardized",
    availability: Annotated[str, typer.Option("--availability", "-a", help="available | preview | rdi")] = "available",
    target_availability_date: Annotated[Optional[str], typer.Option("--target-date", help="Required for preview (YYYY-MM-DD)")] = None,
    publication_cycle: Annotated[Optional[str], typer.Option("--cycle", help="Publication cycle, e.g. 2025h1")] = None,
    early_publish: Annotated[bool, typer.Option("--early-publish/--no-early-publish", help="Request early publication")] = False,
    token: TOKEN_OPT = None,
):
    """Create a new submission."""
    if division not in VALID_DIVISIONS:
        typer.echo(f"Error: --division must be one of {VALID_DIVISIONS}", err=True)
        raise typer.Exit(1)
    if availability not in VALID_AVAILABILITIES:
        typer.echo(f"Error: --availability must be one of {VALID_AVAILABILITIES}", err=True)
        raise typer.Exit(1)
    if availability == "preview" and not target_availability_date:
        typer.echo("Error: --target-date is required when --availability=preview", err=True)
        raise typer.Exit(1)

    user_id = user_id_from_token(token)
    payload: dict = {
        "benchmark_version": benchmark_version,
        "run_ids": run_ids,
        "division": division,
        "availability": availability,
        "early_publish": early_publish,
    }
    if publication_cycle:
        payload["publication_cycle"] = publication_cycle
    if target_availability_date:
        payload["target_availability_date"] = target_availability_date

    data = API.submissions_create(user_id, payload)
    typer.echo(f"Submission created: {data['id']}  status={data['status']}")


@app.command("list")
def list_submissions(
    token: TOKEN_OPT = None,
):
    """List all submissions for the authenticated user."""
    user_id = user_id_from_token(token)
    subs = API.submissions_list(user_id)
    if not subs:
        typer.echo("No submissions found.")
        return
    for s in subs:
        typer.echo(
            f"  {s['id']}  version={s.get('benchmark_version')}  "
            f"status={s['status']}  division={s['division']}  availability={s['availability']}"
        )


@app.command("view")
def view_submission(
    submission_id: Annotated[str, typer.Argument(help="Submission UUID")],
    include_runs: Annotated[bool, typer.Option("--runs/--no-runs", help="Include nested run details")] = True,
    token: TOKEN_OPT = None,
):
    """Show full details of a submission (with nested runs by default)."""
    user_id = user_id_from_token(token)
    _print(API.submissions_get(submission_id, user_id, include_runs=include_runs))


@app.command("withdraw")
def withdraw(
    submission_id: Annotated[str, typer.Argument(help="Submission UUID")],
    token: TOKEN_OPT = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
):
    """Withdraw a submission (sets status to WITHDRAWN)."""
    if not yes:
        typer.confirm(f"Withdraw submission {submission_id}?", abort=True)
    user_id = user_id_from_token(token)
    data = API.submissions_withdraw(submission_id, user_id)
    typer.echo(f"Submission {submission_id} withdrawn at {data['withdrawn_at']}.")


@app.command("availability-qualify")
def availability_qualify(
    submission_id: Annotated[str, typer.Argument(help="Submission UUID")],
    target_date: Annotated[str, typer.Option("--target-date", help="Target availability date (YYYY-MM-DD)")],
    token: TOKEN_OPT = None,
):
    """Mark a submission as preview and set its target availability date."""
    user_id = user_id_from_token(token)
    data = API.submissions_update(
        submission_id,
        user_id,
        {"availability": "preview", "target_availability_date": target_date},
    )
    typer.echo(
        f"Submission {submission_id} set to preview, available from {data['target_availability_date']}."
    )
