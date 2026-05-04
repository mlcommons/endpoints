import os
import re
import sys

import typer

TOKEN_ENV = "PRISM_TOKEN"
_TOKEN_RE = re.compile(r"^prsm_([^_]+)_([^_].*)$")


def parse_token(token: str | None) -> tuple[str, str]:
    """Return (user_id, org) from a prsm_<user>_<org> token."""
    raw = token or os.environ.get(TOKEN_ENV)
    if not raw:
        typer.echo(
            f"Error: no token provided. Pass --token or set {TOKEN_ENV}.", err=True
        )
        raise typer.Exit(code=1)
    m = _TOKEN_RE.match(raw)
    if not m:
        typer.echo(
            f"Error: invalid token format. Expected prsm_<user>_<org>, got: {raw!r}",
            err=True,
        )
        raise typer.Exit(code=1)
    return m.group(1), m.group(2)


def user_id_from_token(token: str | None) -> str:
    user_id, _ = parse_token(token)
    return user_id
