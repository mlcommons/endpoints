"""inference-endpoint benchmark <subcommand>"""
from __future__ import annotations

from typing import Annotated, Optional

import typer

app = typer.Typer(help="Run MLPerf inference benchmarks.")

TOKEN_OPT = Annotated[Optional[str], typer.Option("--token", "-t", envvar="PRISM_TOKEN", help="PRISM auth token (prsm_<user>_<org>)")]


@app.command("online")
def online(
    model: Annotated[str, typer.Option("--model", "-m", help="Model name or HuggingFace ID")],
    endpoint: Annotated[str, typer.Option("--endpoint", "-e", help="Inference endpoint URL")],
    scenario: Annotated[str, typer.Option("--scenario", "-s", help="LoadGen scenario (Server|Offline)")] = "Server",
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", help="Number of concurrent requests")] = 1,
    duration: Annotated[int, typer.Option("--duration", help="Target duration in seconds")] = 600,
    token: TOKEN_OPT = None,
):
    """Run an online inference benchmark against an endpoint (stub)."""
    typer.echo(f"[stub] Running {scenario} benchmark:")
    typer.echo(f"  model      = {model}")
    typer.echo(f"  endpoint   = {endpoint}")
    typer.echo(f"  concurrency= {concurrency}")
    typer.echo(f"  duration   = {duration}s")
    typer.echo("")
    typer.echo("Benchmark execution not yet implemented — this is a stub.")
    typer.echo("Once implemented, results will be uploaded and a run UUID returned.")
