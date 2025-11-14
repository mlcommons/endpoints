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

"""Eval-results command for post-processing existing benchmark results."""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from ..eval.evaluate import evaluate_results
from ..exceptions import InputValidationError

logger = logging.getLogger(__name__)


async def run_eval_results_command(args: argparse.Namespace) -> None:
    """Evaluate existing benchmark results (post-processing only).
    
    This command loads previously collected benchmark results and evaluates them
    using the specified evaluator. Useful for:
    - Debugging evaluation logic
    - Experimenting with different pass@k values
    - Re-evaluating results without re-running benchmarks
    
    Args:
        args: Command line arguments with:
            - results: Path to results file (pickle)
            - evaluator: Evaluator name
            - pass_k: Optional pass@k value
            - output: Optional output path for detailed results
    
    Raises:
        InputValidationError: If results file invalid or pass_k invalid
        KeyError: If evaluator not found in registry
    """
    logger.info("Evaluating existing benchmark results")
    
    # Validate results file exists
    if not args.results.exists():
        raise InputValidationError(f"Results file not found: {args.results}")
    
    logger.info(f"Results file: {args.results}")
    logger.info(f"Evaluator: {args.evaluator}")
    if args.pass_k:
        logger.info(f"Pass@k: {args.pass_k}")
    
    # Load results
    try:
        logger.info("Loading results...")
        with open(args.results, 'rb') as f:
            results = pickle.load(f)
        
        # Convert to DataFrame if needed
        if isinstance(results, dict):
            # Handle different result formats
            if "responses" in results:
                # Format from ResponseCollector
                logger.info("Converting ResponseCollector format to DataFrame")
                # TODO: Need to determine the exact format from ResponseCollector
                raise NotImplementedError(
                    "ResponseCollector format not yet supported. "
                    "Results must be a DataFrame with 'response_output' and 'ground_truth' columns."
                )
            else:
                results = pd.DataFrame([results])
        
        if not isinstance(results, pd.DataFrame):
            raise InputValidationError(
                f"Results must be a pandas DataFrame or dict, got {type(results)}"
            )
        
        logger.info(f"Loaded {len(results)} samples")
        
        # Validate required columns
        if "response_output" not in results.columns and "response" not in results.columns:
            raise InputValidationError(
                "Results must contain 'response_output' or 'response' column"
            )
        
        if "ground_truth" not in results.columns:
            raise InputValidationError("Results must contain 'ground_truth' column")
        
    except pickle.UnpicklingError as e:
        raise InputValidationError(f"Invalid pickle file: {e}")
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        raise
    
    # Get k value (default: 1 if not specified)
    k = args.pass_k if args.pass_k is not None else 1
    
    # Validate k is at least 1
    if k < 1:
        raise InputValidationError(f"k must be >= 1, got {k}")
    
    # Evaluate results (validation of k against repeats happens in scorer)
    try:
        report = evaluate_results(
            results=results,
            evaluator_name=args.evaluator,
            k=k,
            output_path=args.output
        )
    except ValueError as e:
        # Re-raise validation errors as InputValidationError
        if "k must be in" in str(e):
            raise InputValidationError(str(e))
        raise
    
    logger.info("\nEvaluation complete!")
    
    # Print metrics (generic approach - works for any evaluator)
    for metric_name, metric_value in report.metrics.items():
        if isinstance(metric_value, float):
            if metric_value <= 1.0:  # Likely a percentage metric
                logger.info(f"{metric_name}: {metric_value:.2%}")
            else:
                logger.info(f"{metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value}")

