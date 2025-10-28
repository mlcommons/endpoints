"""Utility commands: info, validate, init."""

import argparse
import logging
import platform
import shutil
import sys
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from .. import __version__
from ..config.yaml_config import ConfigError, ConfigLoader
from ..exceptions import InputValidationError, SetupError

logger = logging.getLogger(__name__)

# Path to template files
TEMPLATES_DIR = Path(__file__).parent.parent / "config" / "defaults"

# Template mapping
TEMPLATE_FILES = {
    "offline": "offline_template.yaml",
    "online": "online_template.yaml",
    "eval": "eval_template.yaml",
    "submission": "submission_template.yaml",
}


async def run_info_command(args: argparse.Namespace) -> None:
    """Show system information."""
    logger.info(f"Inference Endpoint Benchmarking Tool v{__version__}")
    logger.info("Status: Operational")

    if args.verbose:
        logger.info("")
        logger.info("Architecture:")
        logger.info(f"  Python: {sys.version.split()[0]}")
        logger.info(f"  Platform: {platform.system()} {platform.machine()}")
        logger.info("  Event Loop: uvloop")
        logger.info("  Async: asyncio + multiprocessing")
        logger.info("")
        logger.info("Capabilities:")
        logger.info("  Performance: offline (max throughput), online (Poisson)")
        logger.info(
            "  Schedulers: MaxThroughputScheduler, PoissonDistributionScheduler"
        )
        logger.info("  Metrics: QPS, latency (p50-p999), TTFT, TPOT")
        logger.info("  Accuracy: Built-in datasets support (stub)")


async def run_validate_command(args: argparse.Namespace) -> None:
    """Validate YAML config file."""
    config_path = args.config

    if not config_path:
        raise InputValidationError("Config file required: --config PATH")

    logger.info(f"Validating: {config_path}")

    try:
        config = ConfigLoader.load_yaml(config_path)
        logger.info(f"✓ Config valid: {config.name}")
        logger.info(f"  Type: {config.type}")
        logger.info(f"  Datasets: {len(config.datasets)}")

        if config.is_locked():
            logger.info(f"  Baseline: locked ({config.baseline.model})")

        if args.verbose:
            logger.info(
                f"  Model params: temp={config.model_params.temperature}, max_tokens={config.model_params.max_new_tokens}"
            )
            logger.info(f"  Load pattern: {config.settings.load_pattern.type}")
            logger.info(f"  Workers: {config.settings.client.workers}")

    except (ConfigError, PydanticValidationError, FileNotFoundError) as e:
        logger.error("✗ Validation failed")
        raise InputValidationError(f"Config validation failed: {e}") from e


async def run_init_command(args: argparse.Namespace) -> None:
    """Generate example YAML config templates from defaults."""
    template_type = args.template
    output_path = getattr(args, "output", None) or f"{template_type}_template.yaml"

    if template_type not in TEMPLATE_FILES:
        logger.error(f"Unknown template: {template_type}")
        logger.info(f"Available: {', '.join(TEMPLATE_FILES.keys())}")
        raise InputValidationError(f"Unknown template type: {template_type}")

    template_file = TEMPLATES_DIR / TEMPLATE_FILES[template_type]

    if not template_file.exists():
        logger.error(f"Template not found: {template_file}")
        raise SetupError(f"Template file missing: {template_file}")

    # Warn if file exists
    output_file = Path(output_path)
    if output_file.exists():
        logger.warning(f"⚠ File exists: {output_path} (will be overwritten)")

    try:
        shutil.copy(template_file, output_path)
        logger.info(f"✓ Created: {output_path}")
    except (OSError, PermissionError) as e:
        logger.error("✗ Failed to create template")
        raise SetupError(f"Failed to create template: {e}") from e
