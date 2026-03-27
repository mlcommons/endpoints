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

"""Init command — generate config templates."""

import logging
import shutil
from pathlib import Path

import yaml

from ..config.schema import TEMPLATE_TYPE_MAP, BenchmarkConfig
from ..exceptions import InputValidationError, SetupError

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "config" / "templates"

TEMPLATE_FILES = {
    "offline": "offline_template.yaml",
    "online": "online_template.yaml",
    "eval": "eval_template.yaml",
    "submission": "submission_template.yaml",
}


def execute_init(template_type: str) -> None:
    """Generate example YAML configuration template."""
    output_path = f"{template_type}_template.yaml"

    if template_type not in TEMPLATE_FILES:
        raise InputValidationError(
            f"Unknown template type: {template_type}. "
            f"Available: {', '.join(TEMPLATE_FILES.keys())}"
        )

    template_file = TEMPLATES_DIR / TEMPLATE_FILES[template_type]
    output_file = Path(output_path)
    if output_file.exists():
        logger.warning(f"File exists: {output_path} (will be overwritten)")

    try:
        if not template_file.exists():
            logger.info("Generating from BenchmarkConfig.create_default_config...")
            config = BenchmarkConfig.create_default_config(
                TEMPLATE_TYPE_MAP[template_type]
            )
            config_dict = config.model_dump(exclude_none=True)
            with open(output_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Generated: {output_path}")
        else:
            shutil.copy(template_file, output_path)
            logger.info(f"Created from template: {output_path}")

    except NotImplementedError as e:
        logger.error(str(e))
        if template_file.exists():
            shutil.copy(template_file, output_path)
            logger.info(f"Created from template: {output_path}")
        else:
            raise SetupError(f"Template file not found: {template_file}") from e
    except (OSError, PermissionError) as e:
        raise SetupError(f"Failed to create template: {e}") from e
