# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extensible callable-class dispatch for closed policy evaluator kinds."""

from abc import ABC, abstractmethod
from typing import ClassVar

from .artifacts import Artifacts
from .planner import PlannedCheck
from .results import CheckResult
from .vocabulary import CheckKind


class Evaluator(ABC):
    registry: ClassVar[dict[CheckKind, type["Evaluator"]]] = {}

    def __init_subclass__(cls, *, kind: CheckKind, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if kind in cls.registry:
            raise ValueError(f"Duplicate evaluator registration for {kind}")
        cls.registry[kind] = cls

    @abstractmethod
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        """Return findings from this rule's effective requirements."""
        raise NotImplementedError

    @classmethod
    def lookup(cls, kind: CheckKind) -> type["Evaluator"] | None:
        return cls.registry.get(kind)
