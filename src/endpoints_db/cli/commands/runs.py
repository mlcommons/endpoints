"""inference-endpoint runs <subcommand>"""
from __future__ import annotations

from typing import Annotated, Optional
import json

import typer

from cli import api as API
from cli.auth import user_id_from_token

app = typer.Typer(help="Manage benchmark runs.")

TOKEN_OPT = Annotated[Optional[str], typer.Option("--token", "-t", envvar="PRISM_TOKEN", help="PRISM auth token (prsm_<user>_<org>)")]


def _print(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command("list")
def list_runs(
    token: TOKEN_OPT = None,
):
    """List all runs for the authenticated user."""
    user_id = user_id_from_token(token)
    runs = API.runs_list(user_id)
    if not runs:
        typer.echo("No runs found.")
        return
    for r in runs:
        finished = r.get("finished_at") or "running"
        typer.echo(f"  {r['id']}  model={r.get('model')}  concurrency={r.get('concurrency')}  started={r['started_at']}  finished={finished}")


@app.command("view")
def view_run(
    run_id: Annotated[str, typer.Argument(help="Run UUID")],
    token: TOKEN_OPT = None,
):
    """Show full details of a run."""
    user_id = user_id_from_token(token)
    _print(API.runs_get(run_id, user_id))


@app.command("pin")
def pin_run(
    run_id: Annotated[str, typer.Argument(help="Run UUID")],
    token: TOKEN_OPT = None,
):
    """Pin a run so it never expires."""
    user_id = user_id_from_token(token)
    API.runs_pin(run_id, user_id)
    typer.echo(f"Run {run_id} pinned.")


@app.command("unpin")
def unpin_run(
    run_id: Annotated[str, typer.Argument(help="Run UUID")],
    token: TOKEN_OPT = None,
):
    """Unpin a run (resets expiry to 365 days from now)."""
    user_id = user_id_from_token(token)
    API.runs_unpin(run_id, user_id)
    typer.echo(f"Run {run_id} unpinned.")


@app.command("delete")
def delete_run(
    run_id: Annotated[str, typer.Argument(help="Run UUID")],
    token: TOKEN_OPT = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
):
    """Delete a run permanently."""
    if not yes:
        typer.confirm(f"Delete run {run_id}?", abort=True)
    user_id = user_id_from_token(token)
    API.runs_delete(run_id, user_id)
    typer.echo(f"Run {run_id} deleted.")
