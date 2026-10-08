# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed inclusion and exclusion predicates over classified subjects."""

from pydantic import Field, model_validator

from inference_endpoint.config.schema import LoadPatternType

from .types import (
    Declaration,
    Division,
    EvidenceKey,
    FrozenModel,
    Identifier,
    Match,
    OfflineMode,
    Scope,
)


class Context(FrozenModel):
    """A subject classified by the artifact adapter; None denotes unknown facts.

    Known absence is represented by empty sets or False. Available dependencies
    must have passed artifact parsing/computation; presence alone is insufficient.
    """

    id: Identifier
    scope: Scope
    model_id: Identifier | None = None
    load_pattern: LoadPatternType | None = None
    offline: OfflineMode | None = None
    division: Division | None = None
    declared: frozenset[Declaration] | None = None
    speculative_decoding: bool | None = Field(default=None, strict=True)
    available: frozenset[EvidenceKey] = frozenset()
    invalid: frozenset[EvidenceKey] | None = None
    members: tuple["Context", ...] | None = None

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.invalid and self.invalid & self.available:
            raise ValueError("Invalid evidence cannot also be available")
        if self.members is not None:
            if self.scope is not Scope.CURVE or any(
                member.scope is not Scope.POINT for member in self.members
            ):
                raise ValueError("Curve members must be point contexts")
            if len({member.id for member in self.members}) != len(self.members):
                raise ValueError("Duplicate curve-member IDs")
        return self


class Conditions(FrozenModel):
    """Different fields are ANDed; enum lists are alternatives except fact sets."""

    models: tuple[Identifier, ...] | None = Field(default=None, min_length=1)
    load_pattern: tuple[LoadPatternType, ...] | None = Field(default=None, min_length=1)
    offline: tuple[OfflineMode, ...] | None = Field(default=None, min_length=1)
    division: tuple[Division, ...] | None = Field(default=None, min_length=1)
    declared: frozenset[Declaration] | None = Field(default=None, min_length=1)
    not_declared: frozenset[Declaration] | None = Field(default=None, min_length=1)
    invalid: frozenset[EvidenceKey] | None = Field(default=None, min_length=1)
    speculative_decoding: bool | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def validate_predicates(self):
        if not self.model_fields_set or any(
            getattr(self, field) is None for field in self.model_fields_set
        ):
            raise ValueError("Conditions require non-null predicates")
        if self.declared and self.not_declared and self.declared & self.not_declared:
            raise ValueError("A declaration cannot be both required and excluded")
        return self

    def __call__(self, context: Context) -> Match:
        results: list[Match] = []
        for required, actual in (
            (self.models, context.model_id),
            (self.load_pattern, context.load_pattern),
            (self.offline, context.offline),
            (self.division, context.division),
        ):
            if required is not None:
                results.append(
                    Match.UNKNOWN
                    if actual is None
                    else Match.YES
                    if actual in required
                    else Match.NO
                )
        for required_set, actual_set, exclude in (
            (self.declared, context.declared, False),
            (self.not_declared, context.declared, True),
            (self.invalid, context.invalid, False),
        ):
            if required_set is not None:
                if actual_set is None:
                    results.append(Match.UNKNOWN)
                else:
                    matches = (
                        not bool(required_set.intersection(actual_set))
                        if exclude
                        else required_set <= actual_set
                    )
                    results.append(Match.YES if matches else Match.NO)
        if self.speculative_decoding is not None:
            results.append(
                Match.UNKNOWN
                if context.speculative_decoding is None
                else Match.YES
                if context.speculative_decoding == self.speculative_decoding
                else Match.NO
            )
        if Match.NO in results:
            return Match.NO
        return Match.UNKNOWN if Match.UNKNOWN in results else Match.YES
