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

"""Evaluation command implementation for accuracy testing."""

import argparse
import logging
import pickle
import tempfile
from pathlib import Path

from ..config.schema import BenchmarkConfig, ClientSettings, Dataset, DatasetType, EndpointConfig, LoadPattern, LoadPatternType, ModelParams, RuntimeConfig, Settings, StreamingMode, TestMode, TestType
from ..exceptions import InputValidationError
from ..eval.evaluate import evaluate_results
from .benchmark import _run_benchmark

logger = logging.getLogger(__name__)


def _build_eval_config_from_cli(args: argparse.Namespace) -> BenchmarkConfig:
    """Build BenchmarkConfig for eval command from CLI arguments.
    
    Args:
        args: Parsed CLI arguments
    
    Returns:
        BenchmarkConfig for running eval benchmark
    """
    return BenchmarkConfig(
        name="cli_eval",
        version="1.0",
        type=TestType.OFFLINE,  # Use offline mode for eval (max throughput)
        datasets=[
            Dataset(
                name=args.dataset.stem,
                type=DatasetType.ACCURACY,
                path=str(args.dataset),
                format=None,  # Will be inferred
            )
        ],
        settings=Settings(
            load_pattern=LoadPattern(
                type=LoadPatternType.MAX_THROUGHPUT,
                target_qps=None,
            ),
            runtime=RuntimeConfig(
                min_duration_ms=0,  # Run all samples
                max_duration_ms=1800000,
                n_samples_to_issue=None,  # Will be calculated from dataset size * repeats
                scheduler_random_seed=42,
                dataloader_random_seed=42,
            ),
            client=ClientSettings(
                workers=args.workers if args.workers else 4,
                max_concurrency=-1,
            ),
        ),
        model_params=ModelParams(
            name=args.model,
            temperature=0.0,  # Greedy decoding for eval
            max_new_tokens=2048,
            streaming=StreamingMode.OFF,  # Disable streaming for eval
        ),
        endpoint_config=EndpointConfig(
            endpoint=args.endpoint,
            api_key=args.api_key
        ),
    )


async def run_eval_command(args: argparse.Namespace) -> None:
    """Run benchmark + accuracy evaluation.
    
    This command:
    1. Validates input parameters
    2. Runs benchmark in accuracy mode (TestMode.ACC)
    3. Collects model responses during benchmark
    4. Evaluates responses using specified evaluator
    5. Reports accuracy metrics
    6. Prints repro command for re-evaluation
    
    Args:
        args: Command line arguments with:
            - endpoint: Endpoint URL
            - model: Model name
            - dataset: Dataset file path
            - evaluator: Evaluator name
            - repeats: Number of times to run dataset (default: 1)
            - pass_k: Optional pass@k value
            - api_key: Optional API key
            - workers: Optional worker count
            - output: Optional output path
            - timeout: Timeout in seconds
    
    Raises:
        InputValidationError: If parameters invalid
        SetupError: If benchmark setup fails
        ExecutionError: If benchmark or evaluation fails
    """
    logger.info("Running benchmark + accuracy evaluation")
    
    # Get k value (default: 1 if not specified)
    k = args.pass_k if args.pass_k is not None else 1
    
    # Validate dataset exists
    if not args.dataset.exists():
        raise InputValidationError(f"Dataset not found: {args.dataset}")
    
    logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Evaluator: {args.evaluator}")
    logger.info(f"Repeats: {args.repeats}")
    logger.info(f"Pass@k: {k}")
    
    # Build benchmark config from CLI args
    config = _build_eval_config_from_cli(args)
    
    # Create temporary file for results
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp_file:
        tmp_results_path = Path(tmp_file.name)
    
    try:
        # Run benchmark with TestMode.ACC to collect responses
        # TODO: Modify _run_benchmark to return results instead of using ResponseCollector
        # For now, we need to implement the benchmark run and collect responses
        
        logger.error("Eval command not yet fully implemented")
        logger.info("TODO: Integrate with benchmark command to collect responses")
        logger.info("TODO: Call evaluate_results() with collected responses")
        logger.info("TODO: Print repro command for eval-results")
        
        raise NotImplementedError(
            "Eval command requires benchmark integration. "
            "Will be implemented in next phase."
        )
    
    finally:
        # Clean up temporary file
        if tmp_results_path.exists():
            tmp_results_path.unlink()
