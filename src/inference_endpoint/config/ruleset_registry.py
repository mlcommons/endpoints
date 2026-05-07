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

"""
TODO: PoC only, subject to change!
Ruleset registry for benchmark competitions.

This module provides a central registry for discovering and loading rulesets
by name or version. This allows configs to reference rulesets by string
(e.g., "mlperf-inference-v5.1") and get the actual ruleset instance.

Note: Uses string forward references for type hints to avoid circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rulesets.mlcommons.rules import CURRENT as mlcommons_current

if TYPE_CHECKING:
    from .ruleset_base import BenchmarkSuiteRuleset

# Global ruleset registry
_RULESET_REGISTRY: dict[str, BenchmarkSuiteRuleset] = {}


def register_ruleset(name: str, ruleset: BenchmarkSuiteRuleset) -> None:
    """Register a ruleset in the global registry.

    Args:
        name: Unique name/version identifier (e.g., "mlperf-inference-v5.1")
        ruleset: The ruleset instance to register
    """
    _RULESET_REGISTRY[name] = ruleset


def get_ruleset(name: str) -> BenchmarkSuiteRuleset:
    """Get a ruleset by name from the registry.

    Args:
        name: Ruleset name/version identifier

    Returns:
        The registered ruleset instance

    Raises:
        KeyError: If ruleset not found in registry
    """
    if name not in _RULESET_REGISTRY:
        available = list(_RULESET_REGISTRY.keys())
        raise KeyError(
            f"Ruleset '{name}' not found in registry. Available: {available}"
        )
    return _RULESET_REGISTRY[name]


def list_rulesets() -> list[str]:
    """List all registered ruleset names.

    Returns:
        List of ruleset name/version identifiers
    """
    return list(_RULESET_REGISTRY.keys())


# Auto-register MLCommons rulesets
def _auto_register_mlcommons():
    """Auto-register MLCommons rulesets."""
    # Register with version-specific name
    register_ruleset(f"mlperf-inference-{mlcommons_current.version}", mlcommons_current)
    # Also register as "mlcommons-current" for convenience
    register_ruleset("mlcommons-current", mlcommons_current)


# Auto-register on import
_auto_register_mlcommons()
