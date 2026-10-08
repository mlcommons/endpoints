# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed immutable check parameters retained across parsing and planning."""

from collections.abc import Mapping
from typing import Self

from .types import FrozenModel


class CheckRequirements(FrozenModel):
    def wire(self) -> dict[str, object]:
        return self.model_dump(mode="python", by_alias=True, exclude_unset=True)

    def with_updates(self, updates: Mapping[str, object]) -> Self:
        return type(self).model_validate({**self.wire(), **updates})
