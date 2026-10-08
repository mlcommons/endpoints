# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator kinds and explicit evidence addresses admitted by policy parsers."""

from enum import StrEnum


class CheckKind(StrEnum):
    ACCURACY_COVERAGE = "accuracy_coverage"
    ACCURACY_GATE = "accuracy_gate"
    ACCURACY_PRESENCE = "accuracy_presence"
    ARTIFACT_BINDING = "artifact_binding"
    ARTIFACT_SCHEMA = "artifact_schema"
    CATALOG_INTEGRITY = "catalog_integrity"
    COHORT_IDENTIFIER = "cohort_identifier"
    COLLECTION_SIZE = "collection_size"
    COMPARISON = "comparison"
    CONSISTENCY = "consistency"
    COVERAGE = "coverage"
    COUNT = "count"
    DERIVED_METRIC = "derived_metric"
    DISCLOSURE = "disclosure"
    SPEC_DECODE_HEAD = "spec_decode_head"
    DURATION = "duration"
    FIELD_CONSTRAINTS = "field_constraints"
    ISSUANCE = "issuance"
    MEMBERSHIP = "membership"
    NUMERIC_VALIDITY = "numeric_validity"
    OFFLINE = "offline"
    PATH_RESOLUTION = "path_resolution"
    POWER = "power"
    PRESENCE = "presence"
    REGION_BASIS = "region_basis"
    REGION_BOUNDARIES = "region_boundaries"
    REGION_PLACEMENT = "region_placement"
    REPORT = "report"
    SEED_BINDING = "seed_binding"
    STEADY_STATE = "steady_state"
    WARMUP = "warmup"


class EvidenceReference(StrEnum):
    ACCURACY_SWE_BENCH_EVALUATED_INSTANCE_COUNT = (
        "accuracy.swe_bench.evaluated_instance_count"
    )
    ACCURACY_SWE_BENCH_EXTRAS_NUM_INSTANCES = "accuracy.swe_bench.extras.num_instances"
    ACCURACY_SWE_BENCH_EXTRAS_SWEBENCH_TEMPLATE = (
        "accuracy.swe_bench.extras.swebench_template"
    )
    CURVE_FIRST_DECLARED_MODEL_NAME = "curve.first_declared_model_name"
    CURVE_MAX_SUPPORTED_CONCURRENCY = "curve.max_supported_concurrency"
    CURVE_MODEL_DIRECTORY_NAME = "curve.model_directory_name"
    CURVE_PARSED_POINTS = "curve.parsed_points"
    CURVE_POINT_DIRECTORIES = "curve.point_directories"
    IMPLEMENTATION_README = "implementation.readme"
    MODEL_ACCURACY_DATASETS = "model.accuracy.datasets"
    MODEL_ACCURACY_FULL_RUN_OSL_RANGE = "model.accuracy.full_run_osl_range"
    MODEL_ACCURACY_INLINE_MINIMUM_PERCENT = "model.accuracy.inline_minimum_percent"
    MODEL_ACCURACY_SWEBENCH_MEAN_MINIMUM_PERCENT = (
        "model.accuracy.swebench_mean_minimum_percent"
    )
    POINT_CHECKPOINT_REPOSITORY = "point.checkpoint.repository"
    POINT_CHECKPOINT_REVISION = "point.checkpoint.revision"
    POINT_CONCURRENCY = "point.concurrency"
    POINT_DATASET = "point.dataset"
    POINT_DIRECTORY = "point.directory"
    POINT_DIRECTORY_CONCURRENCY = "point.directory_concurrency"
    POINT_DP_SHORTFALL = "point.dp_shortfall"
    POINT_MODEL_NAME = "point.model_name"
    POINT_NODES_USED = "point.nodes_used"
    POINT_OFFLINE = "point.offline"
    POINT_POWER_KW = "point.power_kw"
    POINT_REGION = "point.region"
    POINT_RESULT_SUMMARY_JSON = "point.result_summary.json"
    POINT_RUNTIME_SETTINGS_AGENTIC_INFERENCE_NUM_TRAJECTORIES_TO_ISSUE = (
        "point.runtime_settings.agentic_inference.num_trajectories_to_issue"
    )
    POINT_RUNTIME_SETTINGS_LOAD_PATTERN = "point.runtime_settings.load_pattern"
    POINT_RUNTIME_SETTINGS_STREAM_ALL_CHUNKS = (
        "point.runtime_settings.stream_all_chunks"
    )
    POINT_RUNTIME_SETTINGS_WARMUP_SALT = "point.runtime_settings.warmup.salt"
    POINT_SEED_SET = "point.seed_set"
    POINT_SHARED_DOCS = "point.shared_docs"
    POINT_SHARED_SRC = "point.shared_src"
    POINT_SYSTEM_DESC_JSON = "point.system_desc.json"
    POINT_SYSTEM_DESCRIPTION = "point.system_description"
    POINT_SYSTEM_DESCRIPTION_SYSTEM_NAME = "point.system_description.system_name"
    POINT_SYSTEM_DESCRIPTION_TPS_UTILIZATION = (
        "point.system_description.tps_utilization"
    )
    POINT_TARGET_COHORT = "point.target_cohort"
    POINT_WARMUP = "point.warmup"
    POINT_YAML = "point.yaml"
    RESULT_SUMMARY_DURATION_NS = "result_summary.duration_ns"
    RESULT_SUMMARY_E2E_AVG_INTERACTIVITY = "result_summary.e2e_avg_interactivity"
    RESULT_SUMMARY_GIT_SHA = "result_summary.git_sha"
    RESULT_SUMMARY_JSON = "result_summary.json"
    RESULT_SUMMARY_N_SAMPLES_COMPLETED = "result_summary.n_samples_completed"
    RESULT_SUMMARY_N_SAMPLES_FAILED = "result_summary.n_samples_failed"
    RESULT_SUMMARY_N_SAMPLES_ISSUED = "result_summary.n_samples_issued"
    RESULT_SUMMARY_OUTPUT_SEQUENCE_LENGTHS_TOTAL = (
        "result_summary.output_sequence_lengths.total"
    )
    RESULT_SUMMARY_OUTPUT_SEQUENCE_LENGTHS_FULL_RUN_OUTPUT_SEQUENCE_LENGTHS_AVG = (
        "result_summary.output_sequence_lengths_full_run.output_sequence_lengths.avg"
    )
    RESULT_SUMMARY_RAW_DERIVED_SYSTEM_TPS = "result_summary.raw_derived_system_tps"
    RESULT_SUMMARY_SYSTEM_TPS = "result_summary.system_tps"
    RESULT_SUMMARY_SYSTEM_TPS_PER_KW = "result_summary.system_tps_per_kw"
    RESULT_SUMMARY_TPOT_PERCENTILES_90 = "result_summary.tpot.percentiles.90"
    RESULT_SUMMARY_TPS_PER_USER = "result_summary.tps_per_user"
    SUBMISSION_IMPLEMENTATION_DIRECTORIES = "submission.implementation_directories"
    SUBMISSION_ROOT = "submission.root"
    SUBMISSION_SYSTEM_DIRECTORIES = "submission.system_directories"
    SYSTEM_MODEL_DIRECTORIES = "system.model_directories"
