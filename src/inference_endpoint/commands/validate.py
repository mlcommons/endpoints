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

"""Validate command — validate YAML config files."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..config.schema import BenchmarkConfig
from ..exceptions import InputValidationError

logger = logging.getLogger(__name__)


def execute_validate(config_path: Path) -> None:
    """Validate YAML configuration file."""
    logger.info(f"Validating: {config_path}")

    try:
        config = BenchmarkConfig.from_yaml_file(config_path)
        logger.info(f"Config valid: {config.name}")
        logger.info(f"  Type: {config.type}")
        logger.info(f"  Datasets: {len(config.datasets)}")

        if config.submission_ref:
            logger.info(
                f"  Submission: model={config.submission_ref.model}, "
                f"ruleset={config.submission_ref.ruleset}"
            )

    except (ValidationError, ValueError, FileNotFoundError, yaml.YAMLError) as e:
        logger.error("Validation failed")
        raise InputValidationError(f"Config validation failed: {e}") from e
