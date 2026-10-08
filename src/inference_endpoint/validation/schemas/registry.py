# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit cohort and inclusive revision-range parser selection."""

from abc import ABC, abstractmethod
from typing import ClassVar

from ..types import Release


class BundleParser(ABC):
    registry: ClassVar[list[tuple[str, int, int | None, type["BundleParser"]]]] = []

    def __init_subclass__(
        cls, *, version: str, first_revision: int, last_revision: int | None, **kwargs
    ):
        super().__init_subclass__(**kwargs)
        Release(version=version, revision=first_revision)
        if last_revision is not None and type(last_revision) is not int:
            raise ValueError("Parser revision boundaries must be integers")
        if last_revision is not None and last_revision < first_revision:
            raise ValueError("Invalid parser revision interval")
        for known_version, first, last, _ in cls.registry:
            if known_version == version and max(first, first_revision) <= min(
                last if last is not None else float("inf"),
                last_revision if last_revision is not None else float("inf"),
            ):
                raise ValueError("Overlapping parser revision intervals")
        cls.registry.append((version, first_revision, last_revision, cls))

    @abstractmethod
    def __call__(self, documents, digests):
        """Normalize a complete document set into an immutable policy."""


def parser_for(release: Release) -> BundleParser:
    matches = [
        parser
        for version, first, last, parser in BundleParser.registry
        if version == release.version
        and first <= release.revision
        and (last is None or release.revision <= last)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Unsupported policy release {release.version} revision {release.revision}"
        )
    return matches[0]()
