# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict check requirement contracts for cohort revision 1."""

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from inference_endpoint.config.schema import LoadPatternType

from ..operations import (
    AccuracyOperation,
    BindingMatch,
    ComparisonOperator,
    CountOperator,
    DerivedOperation,
    DrafterOperation,
    DurationBasis,
    OfflineOperation,
    OperandKind,
    PowerOperation,
    RegionPlacementOperation,
    SeedOperation,
    SteadyOperation,
    WarmupOperation,
)
from ..types import AccuracyKind, FrozenModel, Identifier, OfflineMode, SampleUnit
from ..vocabulary import CheckKind

Number = Annotated[StrictInt | StrictFloat, Field(allow_inf_nan=False)]
NonnegativeNumber = Annotated[Number, Field(ge=0)]
PositiveNumber = Annotated[Number, Field(gt=0)]
NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]


class Requirements(FrozenModel):
    @model_validator(mode="before")
    @classmethod
    def reject_null_fields(cls, value: object) -> object:
        if isinstance(value, Mapping):
            nulls = [name for name, child in value.items() if child is None]
            if nulls:
                raise ValueError(
                    f"Requirement fields cannot be null: {', '.join(nulls)}"
                )
        return value

    def require_fields(self, *names: str) -> None:
        missing = [name for name in names if name not in self.model_fields_set]
        if missing:
            raise ValueError(f"Required fields: {', '.join(missing)}")

    def require_exactly_one(self, left: str, right: str) -> None:
        if (left in self.model_fields_set) == (right in self.model_fields_set):
            raise ValueError(f"Requires exactly one of {left}, {right}")

    @model_validator(mode="after")
    def nonempty_sequences(self) -> Self:
        for name in (
            "accept",
            "sources",
            "targets",
            "allowed",
            "fields",
            "basis_precedence",
            "replica_factors",
        ):
            if name in self.model_fields_set and not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        return self


class Bounds(Requirements):
    minimum: NonnegativeInt
    maximum: NonnegativeInt

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class Tolerance(Requirements):
    absolute: NonnegativeNumber


class ComparisonWhen(Requirements):
    field: Identifier
    greater_than: Number


class FieldOperand(Requirements):
    kind: Literal[OperandKind.FIELD]
    field: Identifier


class ConstantOperand(Requirements):
    kind: Literal[OperandKind.CONSTANT]
    value: Number


Operand = Annotated[FieldOperand | ConstantOperand, Field(discriminator="kind")]


class CountWhen(Requirements):
    curve_type: Literal[AccuracyKind.SINGLE_TURN]
    has_offline: Literal[OfflineMode.DEDICATED, OfflineMode.ELECTED]


class CountOverride(Requirements):
    when: CountWhen
    minimum: NonnegativeInt


class AccuracyCoverageRequirements(Requirements):
    mandatory_bands: Identifier
    margin_counts: StrictBool | None = None
    minimum_results_per_band: NonnegativeInt
    single_turn_requires_offline_accuracy: StrictBool | None = None


class AccuracyGateRequirements(Requirements):
    aggregation: Identifier | None = None
    bands: Identifier | None = None
    bounds: Identifier | None = None
    count_scope: Identifier | None = None
    dataset: Identifier | None = None
    dataset_catalog: Identifier | None = None
    fraction_conversion: Identifier | None = None
    inclusive: StrictBool | None = None
    input_range: tuple[Number, Number] | None = None
    metric_matching: Identifier | None = None
    minimum: NonnegativeNumber | Identifier | None = None
    minimum_repeats: PositiveInt | None = None
    missing_repeats: PositiveInt | None = None
    models: Identifier
    on_missing: Identifier | None = None
    on_missing_bands: Identifier | None = None
    on_missing_metric: Identifier | None = None
    on_missing_required_dataset: Identifier | None = None
    on_missing_weights: Identifier | None = None
    on_multiple_results_per_band: Identifier | None = None
    on_no_results: Identifier | None = None
    on_no_sample_counts: Identifier | None = None
    on_unknown_model: Identifier | None = None
    on_unknown_or_unpublished: Identifier | None = None
    on_unrelated_dataset: Identifier | None = None
    operation: AccuracyOperation
    output_scale: PositiveNumber | None = None
    required_band_count: PositiveInt | None = None
    required_datasets: Identifier | None = None
    score: Identifier | None = None
    source: Identifier | None = None
    threshold: Identifier | None = None
    units: Identifier | None = None
    unnamed_scalar: Identifier | None = None
    windowed_fallback: StrictBool | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        if self.operation in (
            AccuracyOperation.SINGLE_TURN_METRICS,
            AccuracyOperation.ISSUED_COUNT,
        ):
            self.require_fields("required_datasets", "dataset_catalog")
        if self.operation is AccuracyOperation.ISSUED_COUNT:
            self.require_fields("threshold", "missing_repeats", "minimum_repeats")
        elif self.operation in (
            AccuracyOperation.PER_POINT_FRACTION,
            AccuracyOperation.MEAN_OF_BAND_MEANS,
        ):
            self.require_fields("minimum", "dataset", "input_range", "output_scale")
            assert self.input_range is not None
            if self.input_range[0] > self.input_range[1]:
                raise ValueError("input_range must be ordered")
            if self.operation is AccuracyOperation.MEAN_OF_BAND_MEANS:
                self.require_fields("bands", "required_band_count")
        elif self.operation is AccuracyOperation.PER_POINT_RANGE:
            self.require_fields("source", "bounds")
        return self


class AccuracyPresenceRequirements(Requirements):
    accept: list[Identifier]
    minimum_models: NonnegativeInt


class ArtifactBindingRequirements(Requirements):
    catalog: Identifier
    format: Identifier | None = None
    matching: BindingMatch
    on_empty_approval_list: Identifier | None = None
    on_empty_model_catalog: Identifier | None = None
    on_missing: Identifier | None = None
    repository: Identifier | None = None
    source: Identifier | None = None
    sources: list[Identifier] | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        self.require_fields(
            "sources" if self.matching is BindingMatch.CHECKPOINT else "source"
        )
        if self.matching is BindingMatch.CHECKPOINT:
            assert self.sources is not None
            if len(self.sources) != 2:
                raise ValueError(
                    "Checkpoint sources must contain repository and revision"
                )
        return self


class ArtifactSchemaRequirements(Requirements):
    accept: list[Identifier] | None = None
    artifact: Identifier | None = None
    precedence: Identifier | None = None
    reject_empty: StrictBool | None = None
    artifact_schema: Identifier = Field(alias="schema")
    when_present: StrictBool | None = None


class CatalogIntegrityRequirements(Requirements):
    catalog: Identifier
    on_unavailable: Identifier | None = None


class CohortIdentifierRequirements(Requirements):
    format: Identifier
    target: Identifier


class CollectionSizeRequirements(Requirements):
    maximum: NonnegativeInt | None = None
    minimum: NonnegativeInt | None = None
    overrides: list[CountOverride] | None = None
    source: Identifier

    @model_validator(mode="after")
    def check_form(self) -> Self:
        if self.minimum is None and self.maximum is None:
            raise ValueError("Requires minimum or maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum exceeds maximum")
        return self


class ComparisonRequirements(Requirements):
    left: Operand
    operator: ComparisonOperator
    right: Operand
    when: ComparisonWhen | None = None


class ConsistencyRequirements(Requirements):
    comparison: Identifier | None = None
    exclude_fields: list[Identifier] | None = None
    left: Identifier | None = None
    on_disagreement_curve_type: Identifier | None = None
    on_missing: Identifier | None = None
    require_nonempty: StrictBool | None = None
    required_artifact: Identifier | None = None
    right: Identifier | None = None
    target: Identifier | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        self.require_exactly_one("target", "left")
        if "left" in self.model_fields_set:
            self.require_fields("right")
        elif "right" in self.model_fields_set:
            raise ValueError("right requires left")
        return self


class CoverageRequirements(Requirements):
    band: Identifier | Bounds
    margin_counts: StrictBool | None = None
    minimum_count: NonnegativeInt
    target: Identifier


class CountRequirements(Requirements):
    catalog: Identifier
    dataset: Identifier | None = None
    dataset_source: Identifier | None = None
    on_missing: Identifier | None = None
    on_unknown_or_null_threshold: Identifier | None = None
    on_unsupported_sample_unit: Identifier | None = None
    operator: CountOperator
    require_all: StrictBool | None = None
    sample_unit: SampleUnit | None = None
    source: Identifier | None = None
    sources: list[Identifier] | None = None
    supported_sample_units: list[SampleUnit] | None = None
    threshold: Identifier | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        self.require_exactly_one("source", "sources")
        self.require_exactly_one("dataset", "dataset_source")
        return self


class DerivedMetricRequirements(Requirements):
    constants: Identifier | None = None
    denominator: Identifier | None = None
    numerator: PositiveNumber | None = None
    on_invalid_tpot: Identifier | None = None
    on_missing_or_invalid_file: Identifier | None = None
    on_missing_or_nonpositive_power: Identifier | None = None
    on_missing_stored: Identifier | None = None
    on_no_inputs_and_no_stored: Identifier | None = None
    on_nonpositive_peak: Identifier | None = None
    on_stored_without_derivable_inputs: Identifier | None = None
    operation: DerivedOperation
    peak_source: Identifier | None = None
    source: Identifier | None = None
    stored: Identifier
    tolerance: Tolerance | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        if self.operation is DerivedOperation.CURVE_PEAK_RATIO:
            self.require_fields("tolerance")
        elif self.operation is DerivedOperation.INVERSE_TPOT_P90_MS:
            self.require_fields("numerator")
        elif self.operation is DerivedOperation.THROUGHPUT_PER_PROVISIONED_KW:
            self.require_fields("denominator")
        return self


class DisclosureRequirements(Requirements):
    fields: Identifier


class DrafterBindingRequirements(Requirements):
    catalog: Identifier
    matching: BindingMatch | None = None
    minimum_cohorts: PositiveInt | None = None
    on_empty_catalog: Identifier | None = None
    on_missing_approval_cohort: Identifier | None = None
    on_missing_repository_or_revision: Identifier | None = None
    on_unmatched_drafter: Identifier | None = None
    on_unparseable_cohort: Identifier | None = None
    operation: DrafterOperation

    @model_validator(mode="after")
    def check_form(self) -> Self:
        if self.operation is DrafterOperation.APPROVAL_AGE:
            self.require_fields("minimum_cohorts")
        return self


class DurationRequirements(Requirements):
    basis_precedence: list[DurationBasis]
    on_unclassifiable_concurrency: Identifier | None = None
    thresholds: Identifier
    window_status_required: StrictBool | None = None


class FieldConstraintsRequirements(Requirements):
    constraints: Identifier | None = None
    constraints_by_model: Identifier | None = None
    on_missing: Identifier | None = None
    target: Identifier | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        self.require_exactly_one("constraints", "constraints_by_model")
        if self.constraints_by_model is not None:
            self.require_fields("target")
        return self


class IssuanceRequirements(Requirements):
    allowed: list[LoadPatternType]
    require_positive_concurrency: StrictBool | None = None


class MembershipRequirements(Requirements):
    catalog: Identifier
    deduplicate: StrictBool | Identifier | None = None
    matching: Identifier | None = None
    required: StrictBool | None = None
    skip_missing: StrictBool | None = None
    target: Identifier


class NumericValidityRequirements(Requirements):
    finite: StrictBool | None = None
    require_present: StrictBool | None = None
    source: Identifier
    strictly_positive: StrictBool | None = None
    verify_underlying_distribution: StrictBool | None = None


class OfflineRequirements(Requirements):
    agentic_count: NonnegativeInt | None = None
    concurrency_floor: NonnegativeNumber | Identifier | None = None
    elected_must_equal: PositiveInt | Identifier | None = None
    on_missing_summary: Identifier | None = None
    operation: OfflineOperation
    single_turn_count: NonnegativeInt | None = None
    throughput_minimum_multiplier: NonnegativeNumber | None = None
    throughput_reference: Identifier | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        if self.operation is OfflineOperation.DECLARATION_COUNT:
            self.require_fields(
                "agentic_count", "single_turn_count", "elected_must_equal"
            )
        else:
            self.require_fields("concurrency_floor", "throughput_minimum_multiplier")
        return self


class PathResolutionRequirements(Requirements):
    allow_absolute: StrictBool | None = None
    allow_escape_via_symlink: StrictBool | None = None
    allow_parent_traversal: StrictBool | None = None
    base: Identifier
    on_missing: Identifier | None = None
    require_directory: StrictBool | None = None
    targets: list[Identifier]


class PowerRequirements(Requirements):
    artifact: Identifier | None = None
    constants: Identifier | None = None
    cooling_must_match_system_description: StrictBool | None = None
    declared_total_scaling: Identifier | None = None
    on_disaggregated: Identifier | None = None
    on_missing_parallelism: Identifier | None = None
    on_spare_nodes: Identifier | None = None
    on_undeclared_nodes: Identifier | None = None
    on_unknown_accelerator_count: Identifier | None = None
    operation: PowerOperation
    parallelism: Identifier | None = None
    reject_duplicate_ensemble_ids: StrictBool | None = None
    replica_factors: list[Identifier] | None = None
    require_accelerator_capacity: StrictBool | None = None
    require_derivable_total: StrictBool | None = None
    require_known_ensemble_ids: StrictBool | None = None
    require_maximum_data_parallel: StrictBool | None = None
    require_nodes_within_provisioned: StrictBool | None = None
    round_kw_decimals: NonnegativeInt | None = None
    artifact_schema: Identifier | None = Field(default=None, alias="schema")
    shortfall: Identifier | None = None
    source: Identifier | None = None
    switch_scaling: Identifier | None = None
    tag: Identifier | None = None
    validate_computed_values: StrictBool | None = None
    validate_node_sets: StrictBool | None = None
    validate_public_sources: StrictBool | None = None
    validate_scale_out: StrictBool | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        if self.operation is PowerOperation.DESCRIPTOR:
            self.require_fields("artifact", "constants")
        elif self.operation is PowerOperation.ENGAGED_NODES:
            self.require_fields("round_kw_decimals")
        elif self.operation is PowerOperation.MAXIMAL_ENGAGEMENT:
            self.require_fields("replica_factors")
        return self


class PresenceRequirements(Requirements):
    case_sensitive: StrictBool | None = None
    minimum_count: NonnegativeInt | None = None
    object: Identifier | None = None
    pattern: Identifier | None = None
    target: Identifier | None = None
    targets: list[Identifier] | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        self.require_exactly_one("target", "targets")
        return self


class RegionBasisRequirements(Requirements):
    minimum_concurrency: Identifier | None = None
    on_no_parsed_points: Identifier | None = None
    on_partial_parse: Identifier | None = None
    source: Identifier
    upper_clamp: PositiveInt


class RegionBoundariesRequirements(Requirements):
    constants: Identifier
    require_c_max_greater_than: NonnegativeInt
    require_c_min: tuple[PositiveInt, PositiveInt]


class RegionPlacementRequirements(Requirements):
    ignore_declared: list[Identifier] | None = None
    include_margin: StrictBool | None = None
    on_missing_or_unclassifiable: Identifier | None = None
    operation: RegionPlacementOperation


class ReportRequirements(Requirements):
    interpretation: Identifier | None = None
    operation: Identifier | None = None
    target: Identifier


class SeedBindingRequirements(Requirements):
    adoption_window_cohorts: PositiveInt | None = None
    aliases: dict[Identifier, Identifier] | None = None
    catalog: Identifier | None = None
    fields: list[Identifier] | None = None
    on_legacy_registry_without_cohorts: Identifier | None = None
    on_unavailable: Identifier | None = None
    operation: SeedOperation
    require_all: StrictBool | None = None
    warn_legacy_only_when_model_seed_missing: StrictBool | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        if self.operation is SeedOperation.LEGACY_NAMES:
            self.require_fields("aliases")
        else:
            self.require_fields("catalog")
        if self.operation is SeedOperation.RUNTIME_VALUES:
            self.require_fields("fields")
        elif self.operation is SeedOperation.ADOPTION_WINDOW:
            self.require_fields("adoption_window_cohorts")
        return self


class SteadyStateRequirements(Requirements):
    check_reported_super_pass_count: StrictBool | None = None
    constants: Identifier
    fallback: Identifier | None = None
    official_requires_plateau: StrictBool | None = None
    on_drift: Identifier | None = None
    on_missing_or_nonofficial: Identifier | None = None
    on_missing_window_extent: Identifier | None = None
    operation: SteadyOperation


class WarmupRequirements(Requirements):
    disabled_warmup: Identifier | None = None
    on_missing: Identifier | None = None
    operation: WarmupOperation
    require_declared_retained: StrictBool | None = None
    source: Identifier | None = None
    verify_archive_contents: StrictBool | None = None

    @model_validator(mode="after")
    def check_form(self) -> Self:
        if self.operation is WarmupOperation.SALT_DISABLED:
            self.require_fields("source")
        return self


REQUIREMENT_MODELS: dict[CheckKind, type[Requirements]] = {
    CheckKind.ACCURACY_COVERAGE: AccuracyCoverageRequirements,
    CheckKind.ACCURACY_GATE: AccuracyGateRequirements,
    CheckKind.ACCURACY_PRESENCE: AccuracyPresenceRequirements,
    CheckKind.ARTIFACT_BINDING: ArtifactBindingRequirements,
    CheckKind.ARTIFACT_SCHEMA: ArtifactSchemaRequirements,
    CheckKind.CATALOG_INTEGRITY: CatalogIntegrityRequirements,
    CheckKind.COHORT_IDENTIFIER: CohortIdentifierRequirements,
    CheckKind.COLLECTION_SIZE: CollectionSizeRequirements,
    CheckKind.COMPARISON: ComparisonRequirements,
    CheckKind.CONSISTENCY: ConsistencyRequirements,
    CheckKind.COVERAGE: CoverageRequirements,
    CheckKind.COUNT: CountRequirements,
    CheckKind.DERIVED_METRIC: DerivedMetricRequirements,
    CheckKind.DISCLOSURE: DisclosureRequirements,
    CheckKind.DRAFTER_BINDING: DrafterBindingRequirements,
    CheckKind.DURATION: DurationRequirements,
    CheckKind.FIELD_CONSTRAINTS: FieldConstraintsRequirements,
    CheckKind.ISSUANCE: IssuanceRequirements,
    CheckKind.MEMBERSHIP: MembershipRequirements,
    CheckKind.NUMERIC_VALIDITY: NumericValidityRequirements,
    CheckKind.OFFLINE: OfflineRequirements,
    CheckKind.PATH_RESOLUTION: PathResolutionRequirements,
    CheckKind.POWER: PowerRequirements,
    CheckKind.PRESENCE: PresenceRequirements,
    CheckKind.REGION_BASIS: RegionBasisRequirements,
    CheckKind.REGION_BOUNDARIES: RegionBoundariesRequirements,
    CheckKind.REGION_PLACEMENT: RegionPlacementRequirements,
    CheckKind.REPORT: ReportRequirements,
    CheckKind.SEED_BINDING: SeedBindingRequirements,
    CheckKind.STEADY_STATE: SteadyStateRequirements,
    CheckKind.WARMUP: WarmupRequirements,
}


def validate_requirements(
    kind: CheckKind, parameters: Mapping[str, object]
) -> Requirements:
    """Validate a complete rule or merged override using its revision's contract."""
    return REQUIREMENT_MODELS[kind].model_validate(dict(parameters))
