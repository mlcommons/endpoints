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

"""Dataset configuration models (``datasets:`` entries)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Self

import cyclopts
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import DatasetType, EvalMethod, ScorerMethod
from .model_params import _METRICS_DECOUPLED_OVERRIDE_KEYS, ModelParams


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return the result.

    For overlapping keys whose values are both dicts, recurse; otherwise the
    override value wins. Mutates a *copy* — callers can safely pass model_dump()
    output. Used by ``Dataset.effective_generation_config`` so a sparse nested
    override (e.g. ``{osl_distribution: {max: 512}}``) preserves siblings.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class AgenticInferenceConfig(BaseModel):
    """Agentic inference conversation configuration.

    Configuration for benchmarking conversational AI workloads with turn sequencing.
    Enables testing agentic inference conversations where each turn depends on previous responses.
    Presence of this block in the dataset config enables agentic inference mode.

    Attributes:
        turn_timeout_s: Deadline between issuing a turn and receiving its
            response. A timeout aborts that turn and all remaining client
            turns of the same conversation because subsequent turns depend
            on the timed-out response.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_timeout_s: float = Field(
        default=86400.0,
        gt=0,
        description=(
            "Per-turn timeout in seconds. A timeout aborts that turn and all "
            "remaining turns in the same conversation."
        ),
    )
    enable_salt: bool = Field(
        False,
        description=(
            "Add deterministic salt markers before and after the system prompt "
            "to prevent KV cache reuse across trajectories in agentic inference setting."
        ),
    )
    inject_tool_delay: bool = Field(
        False,
        description=(
            "Pause for a predefined duration between turns. Duration is defined "
            "in dataset."
        ),
    )
    num_trajectories_to_issue: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Number of conversation trajectories to start. Defaults to one pass "
            "over the dataset; values above the dataset size repeat trajectories "
            "with unique logical conversation ids."
        ),
    )
    stop_issuing_on_first_user_complete: bool = Field(
        False,
        description=(
            "When performance tracking stops because the first concurrency slot "
            "has no next trajectory left to assign, also stop issuing future "
            "turns. If false, replay continues outside the performance window "
            "for accuracy/log coverage."
        ),
    )


class Dataset(BaseModel):
    """Dataset configuration.

    Name and type have smart defaults: name is auto-derived from path,
    type defaults to PERFORMANCE.

    Accepts CLI strings via BeforeValidator on BenchmarkConfig.datasets:
    ``[perf|acc:]<path>[,key=value...]``
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field("", description="Dataset name (auto-derived from path if empty)")
    type: DatasetType = Field(
        DatasetType.PERFORMANCE, description="Dataset purpose: performance or accuracy"
    )
    path: Annotated[
        str | None, cyclopts.Parameter(alias="--dataset", help="Dataset file path")
    ] = None
    format: str | None = Field(None, description="Dataset format (auto-detected)")
    samples: int | None = Field(None, gt=0, description="Number of samples to use")
    eval_method: EvalMethod | None = Field(
        None, description="Accuracy evaluation method"
    )
    parser: dict[str, str] | None = Field(
        None, description="Column remapping: {prompt: <col>, system: <col>}"
    )
    generate_params: dict[str, Any] | None = Field(
        None, description="Dataset-specific parameters passed to the generate() method"
    )
    accuracy_config: AccuracyConfig | None = Field(
        None, description="Accuracy evaluation settings"
    )
    agentic_inference: AgenticInferenceConfig | None = Field(
        None, description="Agentic inference conversation configuration"
    )
    # Per-dataset generation config is a first-class capability: different
    # accuracy datasets legitimately want different generation settings (e.g.
    # per-dataset max OSL or top_p, as seen in DS-V4), and dataset-scoping also
    # enables per-dataset dynamic OSL distributions. Only generation knobs are
    # overridable — per-run/identity fields (`_METRICS_DECOUPLED_OVERRIDE_KEYS`:
    # name / streaming / tokenizer_name) drive the single global tokenizer and
    # MetricsAggregator, so overriding them per-dataset would desync ISL/OSL/
    # TTFT/TPOT accounting; they are rejected at validation.
    #
    # TODO(post-mortem): split ModelParams into a per-run ModelIdentity and a
    # GenerationConfig, so the override surface is exactly the generation fields
    # and identity fields cannot be named here at all. Field/method names use
    # "generation_config" to keep that migration mechanical.
    #
    # Nested dicts (`osl_distribution`, `chat_template_kwargs`) are deep-merged
    # so sparse overrides preserve sibling defaults.
    generation_config_override: dict[str, Any] | None = Field(
        None,
        description=(
            "Per-dataset overrides for the top-level model_params (sparse — "
            "only the fields you want to override). Merged on top of "
            "BenchmarkConfig.model_params at dataset-load time. Useful for "
            "MLPerf-style runs where accuracy and performance use different "
            "output budgets in the same fleet, e.g. "
            "generation_config_override: {max_new_tokens: 32768, "
            "temperature: 0.0}. NOTE: per-run/identity keys (`name`, "
            "`streaming`, `tokenizer_name`) are rejected here — set them on "
            "top-level model_params."
        ),
    )

    @model_validator(mode="after")
    def _auto_derive_name(self) -> Self:
        """Derive name from path stem if not explicitly provided."""
        if not self.name and self.path:
            object.__setattr__(self, "name", Path(self.path).stem)
        return self

    @model_validator(mode="after")
    def _validate_generation_config_override(self) -> Self:
        """Fail fast on unknown keys and on per-run/identity keys the single
        global tokenizer / MetricsAggregator would ignore. Override *values*
        are validated at merge time (see ``effective_generation_config``)
        because cross-field validation needs the base ``ModelParams`` from
        ``BenchmarkConfig``.
        """
        if self.generation_config_override:
            keys = set(self.generation_config_override)
            valid = set(ModelParams.model_fields)
            bad = sorted(keys - valid)
            if bad:
                raise ValueError(
                    f"Dataset '{self.name}': unknown keys in "
                    f"generation_config_override: {bad}. "
                    f"Valid keys: {sorted(valid)}"
                )
            decoupled = sorted(keys & _METRICS_DECOUPLED_OVERRIDE_KEYS)
            if decoupled:
                raise ValueError(
                    f"Dataset '{self.name}': generation_config_override keys "
                    f"{decoupled} are not honored per-dataset — the single "
                    "global tokenizer / metrics aggregator is launched from "
                    "top-level model_params, so a per-dataset value would "
                    "desync ISL/OSL/TTFT/TPOT accounting. Set them on "
                    "top-level model_params instead."
                )
        return self

    def effective_generation_config(self, base: ModelParams) -> ModelParams:
        """Return base merged with this dataset's generation-config overrides.

        Nested dicts are deep-merged so a sparse nested override preserves
        sibling defaults (e.g. ``{osl_distribution: {max: 512}}`` keeps the
        base ``type/mean/std/min``). The merged dict is re-validated through
        ``ModelParams.model_validate`` so type-invalid scalar overrides (e.g.
        ``temperature: 'hot'``) are rejected. Note that this only catches
        scalar invalidity — a sparse nested override whose merged result
        passes default-validation will not raise (callers that need stricter
        nested validation should set ``base`` to an explicit instance).
        """
        if not self.generation_config_override:
            return base
        merged = _deep_merge(base.model_dump(), self.generation_config_override)
        return ModelParams.model_validate(merged)


class AccuracyConfig(BaseModel):
    """Accuracy configuration.

    eval_method: Scorer to use (see ScorerMethod enum for options).
    ground_truth: Column in the dataset containing ground truth. Defaults to "ground_truth".
    extractor: Post-processor to extract answers from model output
        (abcd_extractor, boxed_math_extractor, identity_extractor, python_code_extractor).
        Optional for scorers that declare REQUIRES_EXTRACTOR = False (e.g. vbench).
    num_repeats: Number of times to repeat the dataset for evaluation. Defaults to 1.
    extras: Free-form keyword args forwarded to the scorer's ``__init__`` —
        used for scorer-specific knobs that don't warrant a top-level field
        (e.g. ``vbench_project_path``, ``subprocess_timeout_s`` for VBench).

    Example:
        accuracy_config:
          eval_method: "pass_at_1"
          ground_truth: "answer"
          extractor: "boxed_math_extractor"
          num_repeats: 5
          extras:
            vbench_project_path: "/path/to/accuracy"
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    eval_method: ScorerMethod | None = Field(None, description="Scorer method")
    ground_truth: str | None = Field(None, description="Ground truth column name")
    extractor: str | None = Field(
        None,
        description="Answer extractor (abcd_extractor, boxed_math_extractor, identity_extractor, python_code_extractor)",
    )
    num_repeats: int = Field(
        1, ge=1, description="Repeat dataset N times for evaluation"
    )
    extras: dict[str, Any] | None = Field(
        None,
        description="Free-form scorer kwargs (e.g. vbench_project_path, subprocess_timeout_s)",
    )
