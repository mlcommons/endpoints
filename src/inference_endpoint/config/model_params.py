# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Model/generation parameter models (``model_params:`` block)."""

from __future__ import annotations

from typing import Annotated, Any, Self

import cyclopts
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import OSLDistributionType, StreamingMode
from .ruleset_base import BenchmarkSuiteRuleset


class OSLDistribution(BaseModel):
    """Output Sequence Length distribution configuration.

    Distribution types:
    - ORIGINAL: Use the natural distribution from the dataset (default)
    - FIXED: All outputs have the same length (uses mean value)
    - UNIFORM: Uniformly distributed between min and max
    - NORMAL: Normal/Gaussian distribution with mean and std
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: OSLDistributionType = Field(
        OSLDistributionType.ORIGINAL, description="Distribution type"
    )
    mean: int | None = Field(None, description="Mean length (FIXED/NORMAL)")
    std: int | None = Field(None, description="Std deviation (NORMAL)")
    min: Annotated[
        int,
        cyclopts.Parameter(alias="--min-output-tokens", help="Minimum output length"),
    ] = 1
    max: int = Field(2048, description="Maximum output length")


class ModelParams(BaseModel):
    """Model generation parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: Annotated[
        str,
        cyclopts.Parameter(alias="--model", help="Model name", required=True),
    ] = ""
    temperature: float | None = Field(None, description="Sampling temperature")
    seed: Annotated[
        int | None,
        cyclopts.Parameter(
            alias="--seed", help="Random seed for reproducible sampling"
        ),
    ] = Field(None, description="Random seed for reproducible sampling")
    top_k: int | None = Field(None, description="Top-K sampling")
    top_p: float | None = Field(None, description="Top-P (nucleus) sampling")
    repetition_penalty: float | None = Field(None, description="Repetition penalty")
    presence_penalty: float | None = Field(None, description="Presence penalty")
    frequency_penalty: float | None = Field(None, description="Frequency penalty")
    chat_template_kwargs: dict[str, Any] | None = Field(
        None,
        description="Per-request chat-template kwargs forwarded to compatible servers.",
    )
    max_new_tokens: Annotated[
        int, cyclopts.Parameter(alias="--max-output-tokens", help="Max output tokens")
    ] = 1024
    min_new_tokens: int = Field(
        1,
        ge=0,
        description="Minimum output tokens for OpenAI text-completions servers",
    )
    skip_special_tokens: bool = Field(
        True,
        description=(
            "Whether OpenAI text-completions servers omit special tokens from decoded output"
        ),
    )
    osl_distribution: OSLDistribution | None = Field(
        None, description="Output sequence length distribution"
    )
    streaming: Annotated[
        StreamingMode,
        cyclopts.Parameter(alias="--streaming", help="Streaming mode: auto/on/off"),
    ] = StreamingMode.AUTO
    tokenizer_name: Annotated[
        str | None,
        cyclopts.Parameter(
            alias="--tokenizer",
            help="HF repo ID or local path for the tokenizer. Overrides model name for client-side token metrics (ISL/OSL/TPOT).",
        ),
    ] = None

    @model_validator(mode="after")
    def _validate_generation_lengths(self) -> Self:
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                "min_new_tokens must be less than or equal to max_new_tokens"
            )
        return self


class SubmissionReference(BaseModel):
    """Reference configuration for official benchmark submissions.

    Links a submission to a specific model and ruleset (competition rules).
    The ruleset defines constraints like min duration, sample counts, and
    performance targets that must be met for a valid submission.

    Example:
        submission_ref:
          model: "llama-2-70b"
          ruleset: "mlperf-inference-v5.1"
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    model: str  # Model identifier (e.g., "llama-2-70b")
    ruleset: str  # Ruleset name/version (e.g., "mlperf-inference-v5.1")

    def get_ruleset_instance(self) -> BenchmarkSuiteRuleset:
        """Get the actual ruleset instance from registry.

        Returns:
            BenchmarkSuiteRuleset instance

        Raises:
            KeyError: If ruleset not found in registry
        """
        from .ruleset_registry import get_ruleset

        return get_ruleset(self.ruleset)


# ModelParams fields that drive the single global tokenizer / MetricsAggregator
# (launched once from top-level model_params), so a per-dataset override would
# desync ISL/OSL/TTFT/TPOT accounting without changing what is measured. Rejected
# as generation_config_override keys — they are per-run/identity, not per-dataset.
_METRICS_DECOUPLED_OVERRIDE_KEYS = frozenset({"name", "streaming", "tokenizer_name"})


def _non_default_completion_controls(mp: ModelParams) -> list[str]:
    """Completion-only ModelParams controls set to a non-default value.

    ``min_new_tokens``/``skip_special_tokens`` are only honored by the
    ``openai_completions`` adapter; ``BenchmarkConfig`` rejects them for other
    ``api_type``s. Shared by the top-level and per-dataset-override checks so
    both config surfaces validate identically.
    """
    checks = {
        "min_new_tokens": mp.min_new_tokens != 1,
        "skip_special_tokens": not mp.skip_special_tokens,
    }
    return [name for name, non_default in checks.items() if non_default]
