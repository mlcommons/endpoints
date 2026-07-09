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

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _Dataset:
    # TODO: Expand this class to include more dataset metadata
    description: str
    """The name of the dataset, or a description if it is a custom dataset"""

    size: int
    """The number of unique samples in the dataset"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the dataset"""

    def __hash__(self) -> int:
        # Datasets are used as pseudo-'Enum' values, so we can use id as hash
        return id(self)


OpenOrca = _Dataset(description="OpenOrca", size=24576, metadata={"max_seq_len": 1024})

CNNDailyMail = _Dataset(
    description="CNNDailyMail v3.0.0", size=13368, metadata={"max_seq_len": 2048}
)

TextGenLongSeqLen = _Dataset(
    description="Subset of LongBench, LongDataCollections, Ruler, GovReport",
    size=8313,
)

TextGenComplex = _Dataset(
    description="Subset of OpenOrca (5k samples), GSM8K (5k samples from train split), MBXP (5k samples) for QA, Math, and Code Generation",
    size=15000,
    metadata={"max_seq_len": 2048},
)

MLPerfDeepseekR1 = _Dataset(
    description="Custom dataset curated by MLCommons for DeepSeek R1, specifically for the MLPerf Inference benchmark",
    size=4388,
)

# --- Edge-Agentic (BFCL v4) benchmark datasets ---

BFCLv4SingleTurn = _Dataset(
    description=(
        "BFCL v4 single-turn function-calling accuracy set (non_live, live, "
        "hallucination), per-category sampled to a stable ~995-sample point "
        "estimate. This is the gated accuracy workload for the Edge-Agentic "
        "benchmark."
    ),
    size=995,
    metadata={
        "categories": ["non_live", "live", "hallucination"],
        "category_sample_pct": {"non_live": 62, "live": 10, "hallucination": 10},
        "subset_floor": 25,
        "max_seq_len": 32768,
        "scorer": "bfcl_v4",
    },
)

AgenticCodingPerf = _Dataset(
    description=(
        "Recorded multi-turn agentic-coding trajectories (SWE-bench-style) "
        "replayed as the Edge-Agentic performance workload. The dataset is both "
        "the performance workload and its own ground truth for the inline online "
        "checker (multiset IoU of executables). Sized so no conversation "
        "overflows a 32K served context (peak ISL ~23.5K)."
    ),
    size=1007,
    metadata={
        "conversations": 20,
        "generated_turns": 1007,
        "peak_isl": 23456,
        "max_seq_len": 32768,
        "scorer": "agentic_inference_inline",
    },
)


# Note this isn't completely robust, but will prevent simple cases of defining new instances
def _disallow_instantiation(cls, *args, **kwargs):
    raise TypeError(
        "Cannot instantiate _Dataset directly. Use a pre-defined dataset from this module."
    )


_Dataset.__new__ = _disallow_instantiation  # type: ignore[method-assign]
