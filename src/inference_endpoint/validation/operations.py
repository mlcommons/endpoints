# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operation vocabulary shared by policy contracts and callable evaluators."""

from enum import StrEnum


class BindingMatch(StrEnum):
    CHECKPOINT = "model_and_repository_and_revision"
    CLIENT_ALLOWLIST = "exact_revision"
    SPEC_DECODE_HEAD = "model_and_identity"


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


class SpecDecodeHeadOperation(StrEnum):
    MEMBERSHIP = "membership"
    APPROVAL_AGE = "approval_age"


class ComparisonOperator(StrEnum):
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    EQUAL = "equal"


class OperandKind(StrEnum):
    FIELD = "field"
    CONSTANT = "constant"
    SUM = "sum"


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


class CountOperator(StrEnum):
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    EQUAL = "equal"
    POSITIVE_MULTIPLE = "positive_multiple"


class FractionConversion(StrEnum):
    LEGACY_VALUE_AND_THRESHOLD_HEURISTIC = "legacy_value_and_threshold_heuristic"


class AccuracyMode(StrEnum):
    ARITHMETIC_MEAN = "arithmetic_mean"
    CASE_INSENSITIVE = "case_insensitive"
    PER_REQUIRED_DATASET_OR_SUITE = "per_required_dataset_or_suite"
    SAMPLE_WEIGHTED_MEAN = "sample_weighted_mean"
    SAMPLES_TIMES_REPEATS = "samples_times_repeats"
    SCORE_OR_FIRST_METRIC = "score_or_first_metric"
    SOLE_METRIC_ONLY_WITH_WARNING = "sole_metric_only_with_warning"


class ArtifactPrecedence(StrEnum):
    STANDALONE_FIRST = "standalone_first"


class ArtifactSchema(StrEnum):
    ACCURACY_RESULT = "accuracy_result"
    APPENDIX_E_SYSTEM_POWER = "appendix_e_system_power"
    POINT_CONFIG = "point_config"
    RESULT_SUMMARY = "result_summary"
    SYSTEM_DESCRIPTION = "system_description"


class ConsistencyMode(StrEnum):
    EXACT = "exact"
    WHOLE_DOCUMENT = "whole_document"


class CurveTypePolicy(StrEnum):
    SINGLE_TURN = "single_turn"


class Deduplication(StrEnum):
    DECLARED_NAME = "declared_name"


class IdentifierFormat(StrEnum):
    GIT_COMMIT_SHA1 = "git_commit_sha1"
    YEAR_MONTH_C0_OR_C1 = "year_month_c0_or_c1"


class PolicyAction(StrEnum):
    AVERAGE_AND_WARN = "average_and_warn"
    BLOCK_DEPENDENT_CHECKS = "block_dependent_checks"
    BLOCKED = "blocked"
    ERROR = "error"
    EXCLUDE = "exclude"
    EXEMPT = "exempt"
    GATE_AVAILABLE_MEAN_AND_WARN = "gate_available_mean_and_warn"
    REPORT_DERIVED = "report_derived"
    SKIP = "skip"
    SKIP_COUNT_COMPARISON = "skip_count_comparison"
    WARN_RANGE_OR_SLOPE_REQUIRED = "warn_range_or_slope_required"
    WARNING = "warning"


class PowerMode(StrEnum):
    ENGAGED_NODE_FRACTION = "engaged_node_fraction"
    FULLY_ENGAGED = "fully_engaged"
    MAXIMUM_ENGAGED_FRACTION = "maximum_engaged_fraction"


class PowerTag(StrEnum):
    ESTIMATED_POWER = "estimated_power"


class PresenceObject(StrEnum):
    DECLARATION = "declaration"
    DIRECTORY = "directory"
    FILE = "file"
    PATH = "path"


class RegionSource(StrEnum):
    C_MAX_POINT = "c_max_point"
    POINTS_WITH_READABLE_THROUGHPUT_AND_UTILIZATION = (
        "points_with_readable_throughput_and_utilization"
    )
    SMALLEST_SUBMITTED = "smallest_submitted"


class ReportInterpretation(StrEnum):
    CLIENT_IPC_FORWARDING_ONLY = "client_ipc_forwarding_only"


class ReportOperation(StrEnum):
    BLOCKED_CONFIG_DEPENDENTS = "blocked_config_dependents"


class ReportingBasis(StrEnum):
    WHOLE_RUN_TOTAL = "whole_run_total"


class ChecksumAlgorithm(StrEnum):
    GIT_SHA1 = "git-sha1"


class AccuracyPresenceSource(StrEnum):
    STANDALONE_ACCURACY_RESULTS = "standalone_accuracy_results"
    EMBEDDED_NONEMPTY_ACCURACY_SCORES = "embedded_nonempty_accuracy_scores"


class AccuracyArtifactSource(StrEnum):
    ACCURACY_RESULTS = "accuracy_results.json"
    INLINE_ACCURACY_SCORES = "results.json.accuracy_scores"
