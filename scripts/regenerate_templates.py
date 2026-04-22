#!/usr/bin/env python3
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

"""Regenerate YAML config templates from Pydantic schema field defaults.

Used by pre-commit to keep templates in sync when schema.py changes.

Generates two variants per template:
  - ``<name>_template.yaml``      — minimal: only required fields + placeholders
  - ``<name>_template_full.yaml`` — all fields with schema defaults + placeholders
"""

from __future__ import annotations

import enum
import os
import re
import sys
import types
import typing
from pathlib import Path

import cyclopts
import yaml
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    OfflineBenchmarkConfig,
    OnlineBenchmarkConfig,
    TestType,
)
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

TEMPLATES_DIR = Path(__file__).parent.parent / "src/inference_endpoint/config/templates"

# Template name → (test_type, extra overrides merged on top).
TEMPLATES: dict[str, tuple[TestType, dict]] = {
    "offline": (TestType.OFFLINE, {}),
    "online": (
        TestType.ONLINE,
        {"settings": {"load_pattern": {"type": "poisson", "target_qps": 10.0}}},
    ),
    "concurrency": (
        TestType.ONLINE,
        {
            "name": "concurrency_benchmark",
            "settings": {
                "load_pattern": {"type": "concurrency", "target_concurrency": 32}
            },
        },
    ),
    # TODO(vir): eval/submission raise CLIError in schema, generate templates when support is added
}

MODEL_FOR_TYPE: dict[TestType, type[BenchmarkConfig]] = {
    TestType.OFFLINE: OfflineBenchmarkConfig,
    TestType.ONLINE: OnlineBenchmarkConfig,
}

PERF_DATASET = {
    "name": "perf",
    "type": "performance",
    "path": "<DATASET_PATH eg: tests/datasets/dummy_1k.jsonl>",
    "parser": {"prompt": "text_input"},
}

ACC_DATASET = {
    "name": "accuracy",
    "type": "accuracy",
    "path": "<DATASET_PATH eg: tests/datasets/ds_samples.jsonl>",
    "eval_method": "exact_match",
    "parser": {"prompt": "question", "system": "system_prompt"},
    "accuracy_config": {
        "eval_method": "pass_at_1",
        "ground_truth": "ground_truth",
        "extractor": "boxed_math_extractor",
        "num_repeats": 1,
    },
}

PLACEHOLDER_MODEL = "<MODEL_NAME eg: meta-llama/Llama-3.1-8B-Instruct>"
PLACEHOLDER_ENDPOINT = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unwrap(annotation: object) -> object:
    """Unwrap Optional/Annotated/Union/ForwardRef to the core type."""
    # Evaluate string forward refs (e.g. ForwardRef('AccuracyConfig | None'))
    if isinstance(annotation, typing.ForwardRef):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _unwrap(typing.get_args(annotation)[0])
    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _unwrap(args[0]) if len(args) == 1 else annotation
    return annotation


def _resolved_hints(model: type[BaseModel]) -> dict[str, object]:
    """Get type hints with forward refs resolved."""
    try:
        return typing.get_type_hints(model, include_extras=True)
    except Exception:
        return {n: i.annotation for n, i in model.model_fields.items()}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _dump_defaults(model: type[BaseModel]) -> dict:
    """Extract field defaults from a model WITHOUT constructing it.

    Avoids model validators (e.g. num_workers=-1 → CPU count).
    Recurses into nested BaseModel fields.  Excluded fields are omitted.
    """
    hints = _resolved_hints(model)
    out: dict[str, object] = {}
    for name, info in model.model_fields.items():
        if info.exclude is True:
            continue
        core = _unwrap(hints.get(name, info.annotation))
        # Get raw default
        if info.default is not PydanticUndefined:
            default = info.default
        elif info.default_factory is not None:
            # If the factory is itself a BaseModel subclass (e.g.
            # default_factory=HTTPClientConfig), recurse into it instead of
            # calling it — calling would run validators, defeating the point
            # of this function. Factories that dynamically pick a concrete
            # subclass (e.g. TransportConfig.create_default → ZMQTransportConfig)
            # aren't types, so they fall through and get called as before.
            if isinstance(info.default_factory, type) and issubclass(
                info.default_factory, BaseModel
            ):
                out[name] = _dump_defaults(info.default_factory)
                continue
            default = info.default_factory()
        else:
            # Required field — recurse if BaseModel, else None
            if isinstance(core, type) and issubclass(core, BaseModel):
                out[name] = _dump_defaults(core)
            else:
                out[name] = None
            continue
        # Serialize
        if isinstance(default, BaseModel):
            out[name] = _dump_defaults(type(default))
        elif isinstance(default, list):
            out[name] = [
                _dump_defaults(type(i)) if isinstance(i, BaseModel) else i
                for i in default
            ]
        elif isinstance(default, enum.Enum):
            out[name] = default.value
        elif isinstance(default, Path):
            out[name] = str(default)
        else:
            out[name] = default
    return out


def _list_item_model(info: object) -> type[BaseModel] | None:
    """For a list[SomeModel] field, return SomeModel."""
    if not isinstance(info, FieldInfo):
        return None
    core = _unwrap(info.annotation)
    if typing.get_origin(core) is not list:
        return None
    args = typing.get_args(core)
    if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
        return args[0]
    return None


# ---------------------------------------------------------------------------
# Inline comments — auto-discovered from Enum/Literal/description
# ---------------------------------------------------------------------------


def _collect_comments(model: type[BaseModel]) -> dict[str, str]:
    """Walk model tree, build {yaml_key: "# comment"} for described/enum fields.

    For ambiguous field names (same name, different descriptions across models),
    falls back to value-specific keys so each enum value gets the right comment.
    """

    def _enum_vals(tp: object) -> list[str] | None:
        origin = typing.get_origin(tp)
        if origin is typing.Literal:
            return [
                a.value if isinstance(a, enum.Enum) else str(a)
                for a in typing.get_args(tp)
            ]
        if isinstance(tp, type) and issubclass(tp, enum.Enum):
            return [str(m.value) for m in tp]
        return None

    def _help(info: object) -> str | None:
        if not isinstance(info, FieldInfo):
            return None
        if info.description:
            return info.description
        for m in info.metadata or []:
            if isinstance(m, cyclopts.Parameter) and m.help:
                return m.help
        return None

    result: dict[str, str] = {}
    by_name: dict[str, list[str]] = {}

    def _walk(m: type[BaseModel]) -> None:
        hints = _resolved_hints(m)
        for name, info in m.model_fields.items():
            if info.annotation is None:
                continue
            core = _unwrap(hints.get(name, info.annotation))
            vals = _enum_vals(core)
            parts: list[str] = []
            desc = _help(info)
            if desc:
                parts.append(desc)
            if vals:
                parts.append(f"options: {', '.join(vals)}")
            if parts:
                comment = "# " + " | ".join(parts)
                by_name.setdefault(name, []).append(comment)
                if vals:
                    for v in vals:
                        result[f"{name}: {v}"] = comment
            # Recurse into nested models
            if isinstance(core, type) and issubclass(core, BaseModel):
                _walk(core)
            elif typing.get_origin(core) is list:
                args = typing.get_args(core)
                if (
                    args
                    and isinstance(args[0], type)
                    and issubclass(args[0], BaseModel)
                ):
                    _walk(args[0])

    _walk(model)
    for name, comments in by_name.items():
        if len(set(comments)) == 1:
            result[f"{name}: "] = comments[0]
            # Also match block-style (no trailing space, e.g. "parser:\n")
            result[f"{name}:"] = comments[0]

    return result


def _add_comments(text: str, comments: dict[str, str]) -> str:
    """Inject inline # comments into YAML text."""
    for key, comment in sorted(comments.items(), key=lambda x: -len(x[0])):
        text = re.sub(
            rf"^(\s*{re.escape(key)}.*)$",
            lambda m, c=comment: m.group(0)
            if "#" in m.group(0)
            else f"{m.group(0)}  {c}",
            text,
            count=0,
            flags=re.MULTILINE,
        )
    return text


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------


def _build_full(model_cls: type[BenchmarkConfig], overrides: dict) -> dict:
    """All fields with schema defaults + placeholders.  2 dataset examples."""
    data = _dump_defaults(model_cls)

    # Fill empty list[BaseModel] fields with one default entry
    for name, info in model_cls.model_fields.items():
        if isinstance(data.get(name), list) and len(data[name]) == 0:
            item_model = _list_item_model(info)
            if item_model is not None:
                data[name] = [_dump_defaults(item_model)]

    data = _deep_merge(
        data,
        {
            "model_params": {"name": PLACEHOLDER_MODEL},
            "endpoint_config": {"endpoints": [PLACEHOLDER_ENDPOINT]},
        },
    )

    # 2 dataset examples: perf + accuracy
    ds_defaults = _dump_defaults(
        _list_item_model(model_cls.model_fields["datasets"])  # type: ignore[arg-type]
    )
    data["datasets"] = [
        _deep_merge(ds_defaults, PERF_DATASET),
        _deep_merge(ds_defaults, ACC_DATASET),
    ]

    if overrides:
        data = _deep_merge(data, overrides)

    # Resolve streaming AUTO → off/on (mirrors schema validator)
    test_type = data.get("type")
    mp = data.get("model_params", {})
    if isinstance(mp, dict) and mp.get("streaming") == "auto":
        mp["streaming"] = "off" if test_type == "offline" else "on"

    if not data.get("name") and test_type:
        data["name"] = f"{test_type}_benchmark"

    return data


def _build_minimal(test_type: TestType, overrides: dict) -> dict:
    """Only required fields + placeholders.  1 dataset example."""
    name = overrides.get("name") or f"{test_type.value}_benchmark"
    data: dict[str, object] = {
        "name": name,
        "type": test_type.value,
        "model_params": {"name": PLACEHOLDER_MODEL},
        "datasets": [PERF_DATASET],
        "settings": {
            "runtime": {
                "min_duration_ms": 600000,
                "max_duration_ms": 0,
                "n_samples_to_issue": None,
            },
        },
        "endpoint_config": {"endpoints": [PLACEHOLDER_ENDPOINT]},
    }
    if overrides:
        data = _deep_merge(data, overrides)
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(check_only: bool = False):
    """Regenerate templates, or check they're up to date.

    Locally (pre-commit): regenerates files, pre-commit detects the diff.
    CI: auto-detects ``CI`` env var and switches to check-only mode.
    Explicit: ``--check`` flag forces check-only mode.
    """
    if os.environ.get("CI"):
        check_only = True

    comments = _collect_comments(BenchmarkConfig)
    stale = False

    for name, (test_type, overrides) in TEMPLATES.items():
        model_cls = MODEL_FOR_TYPE[test_type]
        variants = {
            f"{name}_template.yaml": _build_minimal(test_type, overrides),
            f"{name}_template_full.yaml": _build_full(model_cls, overrides),
        }
        for filename, data in variants.items():
            raw = yaml.dump(data, default_flow_style=False, sort_keys=False)
            expected = _add_comments(raw, comments)
            path = TEMPLATES_DIR / filename

            if check_only:
                current = path.read_text() if path.exists() else ""
                if current != expected:
                    print(f"  STALE: {filename}")
                    stale = True
                else:
                    print(f"  OK: {filename}")
            else:
                path.write_text(expected)
                print(f"  Generated: {filename}")

    if stale:
        print("\nRun: python scripts/regenerate_templates.py")
        raise SystemExit(1)


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
