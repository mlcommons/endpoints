# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operation vocabulary shared by policy contracts and callable evaluators."""

from enum import StrEnum


class BindingMatch(StrEnum):
    CHECKPOINT = "model_and_repository_and_revision"
    CLIENT_ALLOWLIST = "exact_revision"


class AccuracyOperation(StrEnum):
    SINGLE_TURN_METRICS = "single_turn_metrics"
    ISSUED_COUNT = "issued_count"
    AGENTIC_PROFILE_AVAILABLE = "agentic_profile_available"
    PER_POINT_FRACTION = "per_point_fraction"
    MEAN_OF_BAND_MEANS = "mean_of_band_means"
    PER_POINT_RANGE = "per_point_range"


class RegionPlacementOperation(StrEnum):
    IN_RANGE = "in_range"
    DECLARED_MATCHES_COMPUTED = "declared_matches_computed"


class OfflineOperation(StrEnum):
    DECLARATION_COUNT = "declaration_count"
    ORDERING = "ordering"


class DrafterOperation(StrEnum):
    MEMBERSHIP = "membership"
    APPROVAL_AGE = "approval_age"


class ComparisonOperator(StrEnum):
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    EQUAL = "equal"


class OperandKind(StrEnum):
    FIELD = "field"
    CONSTANT = "constant"


class DerivedOperation(StrEnum):
    CURVE_PEAK_RATIO = "curve_peak_ratio"
    OUTPUT_TOKENS_PER_ELAPSED_SECOND = "output_tokens_per_elapsed_second"
    INVERSE_TPOT_P90_MS = "inverse_tpot_p90_ms"
    THROUGHPUT_PER_PROVISIONED_KW = "throughput_per_provisioned_kw"
    OUTPUT_TOKENS_PER_COMPLETED_TURN_SECOND = "output_tokens_per_completed_turn_second"


class PowerOperation(StrEnum):
    DESCRIPTOR = "descriptor"
    APPENDIX_D_DEFAULTS = "appendix_d_defaults"
    ENGAGED_NODES = "engaged_nodes"
    NODE_DECLARATION = "node_declaration"
    MAXIMAL_ENGAGEMENT = "maximal_engagement"


class SeedOperation(StrEnum):
    LEGACY_NAMES = "legacy_names"
    MEMBERSHIP = "membership"
    RUNTIME_VALUES = "runtime_values"
    ADOPTION_WINDOW = "adoption_window"


class SteadyOperation(StrEnum):
    VOCABULARY = "vocabulary"
    STATUS_WINDOW_AGREEMENT = "status_window_agreement"
    REPORTING_BASIS = "reporting_basis"


class WarmupOperation(StrEnum):
    LOG_RETENTION = "log_retention"
    SALT_DISABLED = "salt_disabled"


class DurationBasis(StrEnum):
    REPORTED_WINDOW_ISSUE_SPAN = "reported_window_issue_span"
    WHOLE_RUN_DURATION = "whole_run_duration"


class DatasetCountOperator(StrEnum):
    EQUAL = "equal"
    POSITIVE_MULTIPLE = "positive_multiple"
