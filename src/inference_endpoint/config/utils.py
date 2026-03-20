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

"""Config utilities — env var interpolation, dataset string parsing, error formatting."""

from __future__ import annotations

import os
import re

import cyclopts
from pydantic import ValidationError
from rich.panel import Panel

_ENV_VAR_PATTERN = re.compile(
    r"(?<!\$)\$\{([A-Za-z_]\w*)(?::-(.*?))?\}|(?<!\$)\$([A-Za-z_]\w*)"
)


def resolve_env_var_string(text: str) -> str:
    """Resolve ``${VAR}`` and ``$VAR`` references in a single string value.

    Supports ``${VAR:-default}`` for fallback values.
    Use ``$$`` to escape a literal ``$`` (e.g. ``$$HOME`` → ``$HOME``).
    Unresolved variables with no default raise ``ValueError``.
    """

    def _replace(match: re.Match) -> str:
        # ${VAR:-default} or ${VAR}
        if match.group(1) is not None:
            var = match.group(1)
            default = match.group(2)  # None if no :- separator
            value = os.environ.get(var)
            if value is not None:
                return value
            if default is not None:
                return default
            raise ValueError(
                f"Environment variable ${{{var}}} is not set and has no default"
            )
        # $VAR (bare)
        var = match.group(3)
        value = os.environ.get(var)
        if value is not None:
            return value
        raise ValueError(f"Environment variable ${var} is not set")

    result = _ENV_VAR_PATTERN.sub(_replace, text)
    return result.replace("$$", "$")


def resolve_env_vars(obj: object) -> None:
    """Walk parsed YAML tree and resolve env vars in string values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                obj[k] = resolve_env_var_string(v)
            elif isinstance(v, dict | list):
                resolve_env_vars(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = resolve_env_var_string(v)
            elif isinstance(v, dict | list):
                resolve_env_vars(v)


def parse_dataset_string(s: str) -> dict[str, object]:
    """Parse a CLI dataset string into a dict suitable for Dataset construction.

    Grammar (TOML-style dotted keys)::

        [perf|acc:]<path>[,key=value...]

    Dotted keys build nested dicts. Pydantic validates fields and coerces types.

    Examples::

        "data.pkl"
        "acc:eval.jsonl"
        "data.csv,samples=500"
        "data.csv,parser.prompt=article,parser.system=inst"
        "perf:d.jsonl,samples=500,parser.prompt=article"
        "acc:e.pkl,accuracy_config.eval_method=pass_at_1,accuracy_config.ground_truth=answer"
    """
    parts = s.split(",")
    path_part = parts[0]

    dtype = "performance"
    if ":" in path_part and path_part.split(":")[0] in ("perf", "acc"):
        tag, path_part = path_part.split(":", 1)
        dtype = "accuracy" if tag == "acc" else "performance"

    result: dict[str, object] = {"path": path_part, "type": dtype}

    for kv in parts[1:]:
        if "=" not in kv:
            hint = " (did you mean '=' instead of ':'?)" if ":" in kv else ""
            raise ValueError(
                f"Invalid option '{kv}'{hint}. "
                f"Format: key=value (e.g. samples=500,parser.prompt=article)"
            )
        key, value = kv.split("=", 1)
        # Build nested dict from dotted path
        segments = key.split(".")
        target = result
        for seg in segments[:-1]:
            if seg not in target:
                target[seg] = {}  # type: ignore[index]
            target = target[seg]  # type: ignore[assignment]
        target[segments[-1]] = value  # type: ignore[index]

    # Validate parser remap targets (CLI only — YAML validated in factory)
    if "parser" in result and isinstance(result["parser"], dict):
        # Lazy import to avoid circular dep: schema_utils → dataset_manager → schema
        from inference_endpoint.dataset_manager.transforms import (
            MakeAdapterCompatible,
        )

        valid = set(MakeAdapterCompatible().remap.values())
        invalid = set(result["parser"].keys()) - valid
        if invalid:
            raise ValueError(
                f"Unknown parser remap target(s): {invalid}. "
                f"Valid targets: {sorted(valid)}"
            )

    return result


def _fmt_names(names: tuple[str, ...]) -> str:
    """Format as '--full.path [--alias]' or just '--flag' if no alias."""
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{names[0]} [{names[1]}]"


def cli_error_formatter(e: cyclopts.CycloptsError) -> Panel:
    """Clean error formatting — resolve aliases, strip Pydantic boilerplate."""
    # cyclopts argument errors — show full path [alias]
    if hasattr(e, "argument") and e.argument is not None:
        arg = e.argument
        # If parent arg (e.g. --endpoint-config), show required children
        children = getattr(arg, "children", [])
        required_children = [
            c
            for c in children
            if getattr(c, "required", False) and not getattr(c, "has_tokens", False)
        ]
        if required_children:
            names = [
                _fmt_names(getattr(c, "names", (c.name,))) for c in required_children
            ]
            return Panel(
                f"Required: {', '.join(names)}",
                title="Error",
                border_style="red",
            )
        # Leaf arg — show all names
        name = arg.name
        if name and name != "*":
            all_names = getattr(arg, "names", (name,))
            display = _fmt_names(all_names)
            return Panel(
                f"Required: {display}",
                title="Error",
                border_style="red",
            )

    # Pydantic ValidationError — show field name + message
    cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    if cause and isinstance(cause, ValidationError):
        lines = []
        for err in cause.errors():
            loc = ".".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", str(err))
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            lines.append(f"  {loc}: {msg}" if loc else f"  {msg}")
        return Panel("\n".join(lines), title="Error", border_style="red")

    return Panel(str(e), title="Error", border_style="red")
