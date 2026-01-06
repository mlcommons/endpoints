# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global constants and mappings for the inference endpoint package."""

# Mapping from endpoints results keys to MLPerf loadgen and submission checker supported keys
# This ensures compatibility when generating user.conf and mlperf_log_details.txt for submission checker
# Format: {"endpoints_key": "loadgen_key"}
ENDPOINTS_TO_LOADGEN_KEY_MAPPING = {
    "endpoints_version": "loadgen_version",
    "endpoints_git_commit_date": "loadgen_git_commit_date",
    "endpoints_git_commit_hash": "loadgen_git_commit_hash",
    "test_datetime": "test_datetime",
    "n_samples_issued": "qsl_reported_total_count",
    "n_samples_from_dataset": "qsl_reported_performance_count",
    "effective_scenario": "effective_scenario",
    "mode": "effective_test_mode",
    "streaming": "streaming",
    "output_sequence_lengths.min": "min_output_tokens",
    "output_sequence_lengths.max": "max_output_tokens",
    "load_pattern": "load_pattern",
    "min_duration_ms": "effective_min_duration_ms",
    "max_duration_ms": "effective_max_duration_ms",
    "effective_target_duration_ms": "effective_target_duration_ms",
    "min_sample_count": "effective_min_query_count",
    "effective_sample_index_rng_seed": "effective_sample_index_rng_seed",
    "effective_schedule_rng_seed": "effective_schedule_rng_seed",
    "effective_sample_concatenate_permutation": "effective_sample_concatenate_permutation",
    "effective_samples_per_query": "effective_samples_per_query",
    "generated_query_count": "generated_query_count",
    "generated_query_duration": "generated_query_duration",
    "target_qps": "effective_target_qps",  # (results_summary.json)
    "result_scheduled_samples_per_sec": "result_scheduled_samples_per_sec",
    "qps": "result_completed_samples_per_sec",
    "results_sample_per_second": "results_sample_per_second",
    "effective_max_concurrency": "effective_max_async_queries",
    "effective_target_latency_ns": "effective_target_latency_ns",
    "effective_target_latency_percentile": "effective_target_latency_percentile",
    "latency.min": "result_min_latency_ns",
    "latency.max": "result_max_latency_ns",
    "latency.avg": "result_mean_latency_ns",
    "latency.percentiles.50": "result_50.00_percentile_latency_ns",
    "latency.percentiles.90": "result_90.00_percentile_latency_ns",
    "latency.percentiles.95": "result_95.00_percentile_latency_ns",
    "latency.percentiles.99": "result_99.00_percentile_latency_ns",
    "latency.percentiles.99.9": "result_99.90_percentile_latency_ns",
    "ttft.min": "result_first_token_min_latency_ns",
    "ttft.max": "result_first_token_max_latency_ns",
    "ttft.avg": "result_first_token_mean_latency_ns",
    "ttft.percentiles.50": "result_first_token_50.00_percentile_latency_ns",
    "ttft.percentiles.90": "result_first_token_90.00_percentile_latency_ns",
    "ttft.percentiles.95": "result_first_token_95.00_percentile_latency_ns",
    "ttft.percentiles.99": "result_first_token_99.00_percentile_latency_ns",
    "ttft.percentiles.99.9": "result_first_token_99.90_percentile_latency_ns",
    "tpot.percentiles.50": "result_time_per_output_token_50.00_percentile_ns",
    "tpot.percentiles.90": "result_time_per_output_token_90.00_percentile_ns",
    "tpot.percentiles.95": "result_time_per_output_token_95.00_percentile_ns",
    "tpot.percentiles.99": "result_time_per_output_token_99.00_percentile_ns",
    "tpot.percentiles.99.9": "result_time_per_output_token_99.90_percentile_ns",
    "tpot.min": "result_time_to_output_token_min",
    "tpot.max": "result_time_to_output_token_max",
    "tpot.avg": "result_time_to_output_token_mean",
    "tps": "result_completed_tokens_per_second",
    "result_validity": "result_validity",
    "result_perf_constraints_met": "result_perf_constraints_met",
    "result_min_duration_met": "result_min_duration_met",
    "result_min_queries_met": "result_min_queries_met",
    "early_stopping_met": "early_stopping_met",
    "early_stopping_ttft_result": "early_stopping_ttft_result",
    "early_stopping_tpot_result": "early_stopping_tpot_result",
    "result.total": "result_query_count",
    "result_overlatency_query_count": "result_overlatency_query_count",
    "result.failed": "num_errors",
}
