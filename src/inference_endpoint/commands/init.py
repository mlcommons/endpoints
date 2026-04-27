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

from ..config.schema import (
    BenchmarkConfig,
    LoadPattern,
    LoadPatternType,
    OnlineSettings,
    TestType,
)
from ..exceptions import InputValidationError, SetupError

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "config" / "templates"

VALID_TYPES = {"offline", "online", "concurrency", "eval", "submission"}

# eval/submission not yet supported in create_default_config — copy handwritten templates
_HANDWRITTEN = {
    "eval": "eval_template.yaml",
    "submission": "submission_template.yaml",
}

_TYPE_MAP = {
    "offline": TestType.OFFLINE,
    "online": TestType.ONLINE,
    "concurrency": TestType.ONLINE,
}


def execute_init(template_type: str) -> None:
    """Generate YAML config template.

    For offline/online/concurrency: generates via model_dump(exclude_none=True).
    For eval/submission: copies handwritten template files.

    Args:
        template_type: One of "offline", "online", "concurrency", "eval", "submission".
    """
    if template_type not in VALID_TYPES:
        raise InputValidationError(
            f"Unknown template type: {template_type}. "
            f"Available: {', '.join(sorted(VALID_TYPES))}"
        )

    output_path = f"{template_type}_template.yaml"
    output_file = Path(output_path)

    if output_file.exists():
        logger.warning(f"File exists: {output_path} (will be overwritten)")

    try:
        # TODO(vir):
        # generate these automatically when support is added
        # for now just copy over hand-written templates
        if template_type in _HANDWRITTEN:
            template_file = TEMPLATES_DIR / _HANDWRITTEN[template_type]
            if not template_file.exists():
                raise SetupError(f"Template file not found: {template_file}")
            shutil.copy(template_file, output_path)
        else:
            config = BenchmarkConfig.create_default_config(_TYPE_MAP[template_type])
            if template_type == "concurrency":
                config = config.with_updates(
                    name="concurrency_benchmark",
                    settings=OnlineSettings(
                        load_pattern=LoadPattern(
                            type=LoadPatternType.CONCURRENCY,
                            target_concurrency=32,
                        ),
                    ),
                )
            data = config.model_dump(mode="json", exclude_none=True)
            with open(output_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Created: {output_path}")
    except (OSError, PermissionError) as e:
        raise SetupError(f"Failed to create template: {e}") from e
