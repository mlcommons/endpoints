#!/usr/bin/env python3
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

"""
Main entry point for the MLPerf Inference Endpoint Benchmarking System.

This module provides the main application logic and can be run directly
or imported as a module.
"""

import asyncio
import logging
import sys

from inference_endpoint.cli import main as cli_main
from inference_endpoint.utils.logging import setup_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main application entry point."""
    logger.info("Starting MLPerf Inference Endpoint Benchmarking System")
    await cli_main()


def run() -> None:
    """Entry point for setuptools."""
    # Setup logging first
    setup_logging()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(130)  # 128 + SIGINT
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
