"""inference-endpoint — MLPerf Endpoints CLI"""
import typer

from cli.commands import benchmark, runs, submission

app = typer.Typer(
    name="inference-endpoint",
    help="MLPerf Endpoints CLI — manage runs, submissions, and benchmarks.",
    no_args_is_help=True,
)

app.add_typer(runs.app, name="runs")
app.add_typer(submission.app, name="submission")
app.add_typer(benchmark.app, name="benchmark")
