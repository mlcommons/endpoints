# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rules that decide which token counts a run uses.

- ``classify_server_usage``: the one rule for whether a request's server-reported
  usage is usable, shared by the live aggregator and every post-run tool;
  ``read_server_usage`` also reads the completion count it allows.
- ``effective_token_count_source``: the configured basis after the endpoint's
  API type is taken into account.
- ``token_count_source_for_run``: the basis a finished run used, for standalone
  tools that only see its report directory; ``replay_basis`` adds a
  ``--token-count-source`` override (``TOKEN_COUNT_SOURCE_METAVAR``).
- ``missing_usage_message``: the one message for samples without usable usage.

This module is imported by the metrics-aggregator subprocess, so it must not
import ``config.schema`` (pydantic and cyclopts startup cost).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import Any, Final, TypeVar, cast

import yaml

from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    has_countable_output,
)
from inference_endpoint.core.types import (
    APIType,
    TextModelOutput,
    TokenCountSource,
    well_formed_count,
)

_E = TypeVar("_E", bound=Enum)


class ServerUsage(Enum):
    """How one request's server-reported usage reads under the usage rule."""

    USABLE = auto()  # both counts well-formed, prompt >= 1, completion >= 1
    TRUE_ZERO = auto()  # completion 0 beside empty output: a count, not a gap
    MISSING = auto()  # anything else, including 0 beside non-empty output


def classify_server_usage(
    prompt_tokens: object, completion_tokens: object, output: TextModelOutput
) -> ServerUsage:
    """Classify one request's server usage; the one rule for every consumer.

    ``USABLE`` iff both counts are well-formed (``well_formed_count``) and at
    least 1. ``TRUE_ZERO`` iff the prompt count is well-formed and the server
    reports 0 completion tokens for an output with no countable content
    (``has_countable_output``). Everything else is ``MISSING``, including 0
    completion tokens beside non-empty output. ``output`` is inspected only
    when the counts are not usable.
    """
    prompt = well_formed_count(prompt_tokens)
    completion = well_formed_count(completion_tokens)
    if prompt is None or completion is None:
        return ServerUsage.MISSING
    if prompt >= 1 and completion >= 1:
        return ServerUsage.USABLE
    if completion == 0 and not has_countable_output(output):
        return ServerUsage.TRUE_ZERO
    return ServerUsage.MISSING


def read_server_usage(
    prompt_tokens: object, completion_tokens: object, output: TextModelOutput
) -> tuple[ServerUsage, int | None]:
    """``classify_server_usage``, with the completion count it allows.

    The count is ``completion_tokens`` when the usage is ``USABLE`` (a
    well-formed int), else ``None``.
    """
    usage = classify_server_usage(prompt_tokens, completion_tokens, output)
    if usage is ServerUsage.USABLE:
        return usage, cast(int, completion_tokens)
    return usage, None


def effective_token_count_source(
    configured: TokenCountSource, api_type: APIType
) -> TokenCountSource:
    """The basis a run counts on: videogen endpoints report no usage, so client."""
    if api_type is APIType.VIDEOGEN:
        return TokenCountSource.CLIENT
    return configured


# The basis of a run that recorded none (written before the setting existed).
# Deliberately separate from the schema default so changing the default never
# reinterprets archived runs.
ARCHIVED_RUN_BASIS: Final = TokenCountSource.CLIENT

# argparse metavar of every ``--token-count-source`` flag.
TOKEN_COUNT_SOURCE_METAVAR: Final = (
    "{" + ",".join(source.value for source in TokenCountSource) + "}"
)


class RunBasisError(Exception):
    """A run's recorded token-count basis could not be read."""


def _read_mapping(path: Path, parse: Callable[[str], Any]) -> dict[str, Any] | None:
    """Parse ``path`` as a mapping; None when the file does not exist."""
    try:
        parsed = parse(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RunBasisError(
            f"cannot read the token-count basis from {path} ({exc}); "
            "pass --token-count-source to choose it"
        ) from exc
    if not isinstance(parsed, dict):
        raise RunBasisError(
            f"cannot read the token-count basis from {path} (not a mapping); "
            "pass --token-count-source to choose it"
        )
    return parsed


def _section(config: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    section = config.get(key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RunBasisError(
            f"cannot read the token-count basis from {path} ({key!r} is not a "
            "mapping); pass --token-count-source to choose it"
        )
    return section


def _parse_enum(enum_type: type[_E], value: object, field: str, path: Path) -> _E:
    try:
        return enum_type(value)
    except ValueError:
        raise RunBasisError(
            f"{field} in {path} is {value!r}, not one of "
            f"{[member.value for member in enum_type]}; "
            "pass --token-count-source to choose the basis"
        ) from None


def _recorded_run_basis(run_dir: Path, summary_path: Path) -> TokenCountSource | None:
    """The basis ``run_dir`` records, or None when it records none."""
    summary = _read_mapping(summary_path, json.loads)
    if summary is not None and summary.get("token_count_source") is not None:
        return _parse_enum(
            TokenCountSource,
            summary["token_count_source"],
            "token_count_source",
            summary_path,
        )

    config_path = run_dir / "config.yaml"
    config = _read_mapping(config_path, yaml.safe_load)
    if config is None:
        return None
    configured = _section(config, "settings", config_path).get("token_count_source")
    if configured is None:
        return None
    source = _parse_enum(
        TokenCountSource,
        configured,
        "settings.token_count_source",
        config_path,
    )
    api_type = _section(config, "endpoint_config", config_path).get("api_type")
    if api_type is None:
        return source
    return effective_token_count_source(
        source,
        _parse_enum(APIType, api_type, "endpoint_config.api_type", config_path),
    )


def _default_summary_path(run_dir: Path) -> Path:
    return run_dir / "performance" / "result_summary.json"


def token_count_source_for_run(
    run_dir: Path, *, summary_path: Path | None = None
) -> TokenCountSource:
    """The token-count basis of the run that wrote ``run_dir``.

    Resolution order:

    1. top-level ``token_count_source`` in ``summary_path``, by default
       ``performance/result_summary.json`` under ``run_dir``;
    2. ``settings.token_count_source`` in ``config.yaml`` (written at setup, so
       it survives a run that died before the summary), resolved with
       ``effective_token_count_source`` against ``endpoint_config.api_type``;
    3. ``ARCHIVED_RUN_BASIS``, with a note on stderr.

    Raises ``RunBasisError`` for a file that exists but cannot be read or an
    unknown basis value.
    """
    if summary_path is None:
        summary_path = _default_summary_path(run_dir)
    recorded = _recorded_run_basis(run_dir, summary_path)
    if recorded is not None:
        return recorded
    print(
        f"no token-count basis recorded (neither token_count_source in "
        f"{summary_path} nor settings.token_count_source in "
        f"{run_dir / 'config.yaml'}); using the {ARCHIVED_RUN_BASIS.value} basis, "
        "override with --token-count-source",
        file=sys.stderr,
    )
    return ARCHIVED_RUN_BASIS


def replay_basis(
    run_dir: Path,
    *,
    override: TokenCountSource | None,
    summary_path: Path | None = None,
) -> tuple[TokenCountSource | None, TokenCountSource]:
    """``(primary, basis)`` for a post-run replay of the run in ``run_dir``.

    ``primary`` is the run's own basis (``token_count_source_for_run``) and
    ``basis`` the one to replay on: ``override`` when given, else ``primary``.
    Beside an override, a run that records no basis is ``ARCHIVED_RUN_BASIS``
    without the stderr note, and one whose basis cannot be read has a ``None``
    primary; each caller decides what that means for its output. Without an
    override, ``RunBasisError`` propagates.
    """
    if override is None:
        primary = token_count_source_for_run(run_dir, summary_path=summary_path)
        return primary, primary
    if summary_path is None:
        summary_path = _default_summary_path(run_dir)
    try:
        recorded = _recorded_run_basis(run_dir, summary_path)
    except RunBasisError:
        return None, override
    return (ARCHIVED_RUN_BASIS if recorded is None else recorded), override


def missing_usage_message(
    n_missing: int,
    n_empty_output: int,
    n_succeeded: int,
    *,
    primary: TokenCountSource,
) -> str:
    """Why samples lack usable server usage, and what that means for the run.

    On the server basis the run is incomplete; the remedy is client counting,
    which fixes samples that produced output but no usage and cannot fix
    samples with empty output (server errors or cut streams). On the client
    basis nothing failed: the ``*_server`` blocks just cover fewer samples.
    """
    head = (
        f"{n_missing} of {n_succeeded} successful tracked samples returned no "
        "server token usage"
    )
    if primary is TokenCountSource.CLIENT:
        return f"{head}; the *_server blocks are partial and the run is not affected"
    parts = [head]
    if n_missing > n_empty_output:
        # `benchmark from-config` takes no --token-count-source; name the key too.
        parts.append(
            "set --token-count-source client if this server cannot report usage "
            "(YAML: settings.token_count_source: client)"
        )
    if n_empty_output > 0:
        parts.append(
            f"{n_empty_output} had empty output — likely server errors or cut "
            "streams; client mode will not fix these"
        )
    return "; ".join(parts)
