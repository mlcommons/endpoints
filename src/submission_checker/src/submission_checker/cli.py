"""Command-line interface for the submission checker."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .checker import SubmissionChecker
from .models import Severity, compute_regions

console = Console()

_SEVERITY_STYLE: dict[Severity, str] = {
    Severity.ERROR: "bold red",
    Severity.WARNING: "yellow",
    Severity.INFO: "dim",
}


@click.group()
@click.version_option(package_name="submission-checker")
def main() -> None:
    """MLPerf Endpoints submission checker — validate a submission directory."""


@main.command()
@click.argument("path", type=click.Path(exists=False, path_type=Path))
@click.option("--strict", is_flag=True, default=False, help="Treat warnings as errors.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Hide INFO-level results.")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write full results as JSON to FILE (in addition to the terminal table).",
)
def check(path: Path, strict: bool, quiet: bool, output: Path | None) -> None:
    """Check the submission at PATH for §9.1 compliance.

    PATH is the submitting organisation's root directory (contains systems/ and pareto/).

    Exit codes:

    \b
      0  All checks passed (no errors; warnings ignored unless --strict).
      1  One or more errors found (or warnings when --strict is active).
    """
    checker = SubmissionChecker(path)
    report = checker.run()

    if output is not None:
        output.write_text(report.model_dump_json(indent=2))

    table = Table(title=f"Submission Check — {path}", show_lines=True)
    table.add_column("Rule", style="cyan", no_wrap=True)
    table.add_column("§ Ref", style="dim", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Message")
    table.add_column("Path", style="dim")

    for result in report.results:
        if quiet and result.severity == Severity.INFO:
            continue
        style = _SEVERITY_STYLE[result.severity]
        table.add_row(
            result.rule,
            result.spec_ref,
            f"[{style}]{result.severity.value}[/{style}]",
            result.message,
            str(result.path.relative_to(path))
            if result.path and result.path.is_relative_to(path)
            else str(result.path or ""),
        )

    console.print(table)

    error_count = len(report.errors)
    warn_count = len(report.warnings)
    total_failures = error_count + (warn_count if strict else 0)

    if total_failures:
        console.print(f"[bold red]FAILED[/] — {error_count} error(s), {warn_count} warning(s)")
    else:
        console.print(f"[bold green]PASSED[/] — {error_count} error(s), {warn_count} warning(s)")

    raise SystemExit(1 if total_failures else 0)


@main.command()
@click.option(
    "--max-concurrency",
    "-M",
    required=True,
    type=int,
    help="Declared Maximum Supported Concurrency.",
)
def regions(max_concurrency: int) -> None:
    """Show computed region boundaries for a given Maximum Supported Concurrency.

    Uses the §5.5 reference algorithm (banker's rounding).
    """
    try:
        r = compute_regions(max_concurrency)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from None

    table = Table(title=f"Region Boundaries for M = {max_concurrency}", show_lines=True)
    table.add_column("Region", style="cyan")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")

    rows = [
        ("Low Latency", r.low_latency),
        ("Low Throughput", r.low_throughput),
        ("Medium Throughput", r.med_throughput),
        ("High Throughput (incl. 10% margin)", r.high_throughput),
    ]
    for label, bounds in rows:
        table.add_row(label, str(bounds.start), str(bounds.end))

    console.print(table)
