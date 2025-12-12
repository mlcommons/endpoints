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

"""Utility commands: info, validate, init."""

import argparse
import datetime
import logging
import os
import platform
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import psutil
import yaml
from pydantic import ValidationError as PydanticValidationError

from .. import __version__
from ..config.schema import TEMPLATE_TYPE_MAP, BenchmarkConfig
from ..config.yaml_loader import ConfigError, ConfigLoader
from ..exceptions import InputValidationError, SetupError

logger = logging.getLogger(__name__)

# Path to template files
TEMPLATES_DIR = Path(__file__).parent.parent / "config" / "templates"

# Template mapping
TEMPLATE_FILES = {
    "offline": "offline_template.yaml",
    "online": "online_template.yaml",
    "eval": "eval_template.yaml",
    "submission": "submission_template.yaml",
}


async def run_info_command(args: argparse.Namespace) -> None:
    """Display system information and tool version.

    Shows version, status, and detailed system information:
    - Python version and implementation
    - Operating system details
    - CPU information
    - Memory information
    - Available disk space
    - Network configuration

    Args:
        args: Command arguments.
    """
    # Collect all system information
    lines = [
        f"Inference Endpoint Benchmarking Tool v{__version__}",
        "",
        "=== System Information ===",
        "",
        "Python Environment:",
        f"  Version: {sys.version}",
        f"  Implementation: {platform.python_implementation()}",
        f"  Compiler: {platform.python_compiler()}",
        "",
        "Operating System:",
        f"  System: {platform.system()}",
        f"  Release: {platform.release()}",
        f"  Version: {platform.version()}",
        f"  Architecture: {platform.machine()}",
    ]

    # Hostname - used in multiple places
    hostname = socket.gethostname()
    lines.append(f"  Hostname: {hostname}")
    lines.append("")

    # CPU information
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    lines.extend(
        [
            "CPU:",
            f"  Physical cores: {cpu_physical}",
            f"  Logical cores: {cpu_logical}",
        ]
    )

    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        lines.append(
            f"  Frequency: {cpu_freq.current:.2f} MHz (max: {cpu_freq.max:.2f} MHz)"
        )
    lines.append("")

    # Memory information
    mem = psutil.virtual_memory()
    lines.extend(
        [
            "Memory:",
            f"  Total: {mem.total / (1024**3):.2f} GB",
            f"  Available: {mem.available / (1024**3):.2f} GB",
            f"  Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)",
            "",
        ]
    )

    # Disk information
    disk = psutil.disk_usage("/")
    lines.extend(
        [
            "Disk (root):",
            f"  Total: {disk.total / (1024**3):.2f} GB",
            f"  Used: {disk.used / (1024**3):.2f} GB ({disk.percent}%)",
            f"  Free: {disk.free / (1024**3):.2f} GB",
            "",
        ]
    )

    # Network information
    lines.append("Network:")
    try:
        ip_address = socket.gethostbyname(hostname)
        lines.append(f"  IP Address: {ip_address}")
    except Exception:
        lines.append("  IP Address: Unable to determine")

    # Environment information
    lines.extend(
        [
            "",
            "Environment:",
            f"  Working Directory: {os.getcwd()}",
            f"  User: {os.getenv('USER', 'unknown')}",
        ]
    )

    if os.getenv("VIRTUAL_ENV"):
        lines.append(f"  Virtual Env: {os.getenv('VIRTUAL_ENV')}")

    # Output all at once
    output = "\n".join(lines)
    logger.info(output)


async def run_validate_command(args: argparse.Namespace) -> None:
    """Validate YAML configuration file for correctness and completeness.

    Performs comprehensive validation including:
    - YAML syntax parsing
    - Pydantic schema validation
    - Required field checks
    - Type validation
    - Cross-field consistency checks
    - Ruleset-specific validation (if submission config)

    Displays:
    - Config name and type
    - Dataset count
    - Submission reference info (if applicable)
    - Model parameters and load pattern (if --verbose)

    Args:
        args: Command arguments containing config file path.
              Required: --config PATH
              Optional: --verbose for extended info

    Raises:
        InputValidationError: If config file is missing, unreadable, or invalid.
    """
    config_path = args.config

    if not config_path:
        raise InputValidationError("Config file required: --config PATH")

    logger.info(f"Validating: {config_path}")

    try:
        config = ConfigLoader.load_yaml(config_path)
        logger.info(f"✓ Config valid: {config.name}")
        logger.info(f"  Type: {config.type}")
        logger.info(f"  Datasets: {len(config.datasets)}")

        if config.submission_ref:
            logger.info(
                f"  Submission: model={config.submission_ref.model}, ruleset={config.submission_ref.ruleset}"
            )

        if args.verbose:
            logger.info(
                f"  Model params: temp={config.model_params.temperature}, max_tokens={config.model_params.max_new_tokens}"
            )
            logger.info(f"  Load pattern: {config.settings.load_pattern.type}")
            logger.info(f"  Workers: {config.settings.client.workers}")

    except (ConfigError, PydanticValidationError, FileNotFoundError) as e:
        logger.error("✗ Validation failed")
        raise InputValidationError(f"Config validation failed: {e}") from e


def _fallback_to_template(template_file: Path, output_path: str) -> None:
    """Helper to fallback to template file when dynamic generation fails.

    Args:
        template_file: Path to the template file.
        output_path: Path where output should be written.

    Raises:
        SetupError: If template file doesn't exist.
    """
    logger.info("Falling back to template file if available...")
    if template_file.exists():
        shutil.copy(template_file, output_path)
        logger.info(f"✓ Created from template: {output_path}")
    else:
        raise SetupError(f"Template file not found: {template_file}")


async def run_init_command(args: argparse.Namespace) -> None:
    """Generate example YAML configuration template from built-in defaults.

    Creates a ready-to-use config file with:
    - Sensible default values
    - Comprehensive inline documentation
    - Example settings for common use cases

    Available templates:
    - offline: Max throughput benchmark template
    - online: Sustained QPS benchmark template
    - eval: Accuracy evaluation template (stub)
    - submission: Official submission template with baseline

    Templates are generated from BenchmarkConfig.create_default_config() which serves
    as the single source of truth. If a template file doesn't exist, it will be
    generated dynamically from the config creation method.

    The generated file can be edited and used with:
        benchmark from-config --config <filename>

    Args:
        args: Command arguments.
              Required: --template TYPE (offline/online/eval/submission)
              Optional: --output PATH (default: <type>_template.yaml)

    Raises:
        InputValidationError: If template type is unknown.
        SetupError: If template generation/writing fails.
    """
    template_type = args.template
    output_path = getattr(args, "output", None) or f"{template_type}_template.yaml"

    if template_type not in TEMPLATE_FILES:
        logger.error(f"Unknown template: {template_type}")
        logger.info(f"Available: {', '.join(TEMPLATE_FILES.keys())}")
        raise InputValidationError(f"Unknown template type: {template_type}")

    template_file = TEMPLATES_DIR / TEMPLATE_FILES[template_type]

    # Warn if file exists
    output_file = Path(output_path)
    if output_file.exists():
        logger.warning(f"⚠ File exists: {output_path} (will be overwritten)")

    try:
        # Check if template file exists, if not generate from BenchmarkConfig.create_default_config
        if not template_file.exists():
            logger.info(
                "Template file not found, generating from BenchmarkConfig.create_default_config..."
            )

            # Generate config using BenchmarkConfig as source of truth (TEMPLATE_TYPE_MAP from schema)
            config = BenchmarkConfig.create_default_config(
                TEMPLATE_TYPE_MAP[template_type]
            )

            # Convert config to dict and write as YAML
            config_dict = config.model_dump(exclude_none=True)
            with open(output_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"✓ Generated and saved: {output_path}")
        else:
            # Use existing template file
            shutil.copy(template_file, output_path)
            logger.info(f"✓ Created from template: {output_path}")

    except NotImplementedError as e:
        logger.error(f"✗ {e}")
        _fallback_to_template(template_file, output_path)
    except (OSError, PermissionError) as e:
        logger.error("✗ Failed to write template file")
        raise SetupError(f"Failed to create template: {e}") from e
    except Exception as e:
        logger.error(f"✗ Failed to generate config: {e}")
        _fallback_to_template(template_file, output_path)


def get_default_report_path() -> Path:
    """Get the default report path.

    Returns:
        The default report path as a Path object.
    """
    return Path(
        f"{tempfile.gettempdir()}/reports_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
