# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/regenerate_templates.py.

`_dump_defaults` must extract defaults without constructing nested
BaseModels that appear as default_factory, because construction runs
validators (which may have platform-dependent side effects).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "regenerate_templates.py"


def _load_regenerate_templates():
    """Load scripts/regenerate_templates.py as a module (it is not a package)."""
    if "regenerate_templates" in sys.modules:
        return sys.modules["regenerate_templates"]
    spec = importlib.util.spec_from_file_location("regenerate_templates", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["regenerate_templates"] = module
    spec.loader.exec_module(module)
    return module


class TestDumpDefaultsSkipsBaseModelFactory:
    def test_basemodel_factory_does_not_run_validator(self):
        """default_factory=<BaseModel subclass> must not invoke the model's validators."""
        rt = _load_regenerate_templates()

        call_count = 0

        class Inner(BaseModel):
            x: int = 42

            @model_validator(mode="after")
            def _count(self):
                nonlocal call_count
                call_count += 1
                return self

        class Outer(BaseModel):
            inner: Inner = Field(default_factory=Inner)

        # Sanity: constructing Inner() directly does invoke the validator.
        Inner()
        assert call_count == 1

        call_count = 0
        result = rt._dump_defaults(Outer)

        assert call_count == 0, (
            "Inner validator was invoked — _dump_defaults called the factory "
            "instead of recursing."
        )
        assert result == {"inner": {"x": 42}}

    def test_callable_factory_is_still_invoked(self):
        """Factories that are callables (not BaseModel subclasses) must still be called."""
        rt = _load_regenerate_templates()

        class Config(BaseModel):
            tags: list[str] = Field(default_factory=lambda: ["default-tag"])

        result = rt._dump_defaults(Config)
        assert result == {"tags": ["default-tag"]}
