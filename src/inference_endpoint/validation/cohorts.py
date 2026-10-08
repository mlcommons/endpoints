# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Publication cohort identifiers and adoption windows."""

from __future__ import annotations

import re
from dataclasses import dataclass

COHORT_RE = re.compile("(?P<year>\\d{4})-(?P<month>\\d{2})-C(?P<index>[01])")
_COHORTS_PER_MONTH = 2


@dataclass(frozen=True, order=True)
class Cohort:
    """One publication cohort."""

    year: int
    month: int
    index: int

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-C{self.index}"

    @classmethod
    def parse(cls, value: str) -> Cohort | None:
        """Parse ``YYYY-MM-C0`` / ``YYYY-MM-C1``, returning None when malformed."""
        match = COHORT_RE.fullmatch(value.strip()) if value else None
        if match is None:
            return None
        month = int(match["month"])
        if not 1 <= month <= 12:
            return None
        return cls(year=int(match["year"]), month=month, index=int(match["index"]))

    def next(self) -> Cohort:
        """The cohort immediately following this one."""
        if self.index + 1 < _COHORTS_PER_MONTH:
            return Cohort(self.year, self.month, self.index + 1)
        if self.month == 12:
            return Cohort(self.year + 1, 1, 0)
        return Cohort(self.year, self.month + 1, 0)


def adoption_window(published: Cohort, length: int = 4) -> tuple[Cohort, ...]:
    """The cohorts during which a set published in *published* may be adopted."""
    window = [published]
    while len(window) < length:
        window.append(window[-1].next())
    return tuple(window)
