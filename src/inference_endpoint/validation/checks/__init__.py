# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register built-in callable evaluators by their policy kind."""

from . import accuracy, power, seeds, spec_decode_heads, steady_state, warmup

__all__ = ["accuracy", "spec_decode_heads", "power", "seeds", "steady_state", "warmup"]
