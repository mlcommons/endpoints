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

"""Generate GPQA dataset from HuggingFace.

GPQA: A Graduate-Level Google-Proof Q&A Benchmark
Reference: https://arxiv.org/abs/2311.12022
OpenAI implementation: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py

Usage:
    python -m inference_endpoint.eval.dataset_generation.generate_gpqa \\
        --output datasets/gpqa.pkl \\
        --variant diamond \\
        --seed 42
"""

import argparse
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Query template from OpenAI's GPQA implementation
# Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py
QUERY_TEMPLATE_MULTICHOICE = """
{Question}

(A) {A}
(B) {B}
(C) {C}
(D) {D}

Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'.
""".strip()


def format_multichoice_question(row: dict) -> str:
    """Format question with multiple choice options.

    Based on OpenAI's format_multichoice_question.
    Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py

    Args:
        row: Dictionary with Question, A, B, C, D keys

    Returns:
        Formatted question string
    """
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def generate_gpqa_dataset(
    output_path: str | Path,
    variant: str = "diamond",
    seed: int = 0,  # GPT-OSS uses 0 as default seed
    num_samples: int | None = None,
) -> None:
    """Generate GPQA dataset with shuffled options.

    Args:
        output_path: Path to save pickle file
        variant: GPQA variant ("diamond", "extended", "main")
        seed: Random seed for reproducibility
        num_samples: Optional limit on number of samples (None = all)

    Raises:
        ValueError: If variant is invalid
    """
    # Validate variant
    valid_variants = ["diamond", "extended", "main"]
    if variant not in valid_variants:
        raise ValueError(
            f"Invalid variant '{variant}'. Must be one of: {', '.join(valid_variants)}"
        )

    print(f"Loading GPQA {variant} dataset from HuggingFace...")

    # Load dataset from HuggingFace
    # Note: May require login with `huggingface-cli login` for some variants
    try:
        dataset = load_dataset("Idavidrein/gpqa", f"gpqa_{variant}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: This dataset may require HuggingFace authentication.")
        print("Run: huggingface-cli login")
        raise

    # Convert to pandas DataFrame
    df = dataset["train"].to_pandas()
    print(f"Loaded {len(df)} questions")

    # Limit samples if requested
    if num_samples is not None and num_samples < len(df):
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(len(df)), num_samples)
        df = df.iloc[sampled_indices].reset_index(drop=True)
        print(f"Sampled {num_samples} questions")

    # Process each question following OpenAI's pattern
    # Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py
    rng = random.Random(seed)
    processed_rows = []

    for _, row in df.iterrows():
        # Create permutation for this example (following OpenAI's approach)
        # This shuffles the order of the 4 choices
        permutation = rng.sample(range(4), 4)

        # Extract original choices (following OpenAI's exact order)
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]

        # Apply permutation to shuffle choices
        choices = [choices[i] for i in permutation]

        # Find where the correct answer ended up after permutation
        correct_index = choices.index(row["Correct Answer"])
        correct_answer = "ABCD"[correct_index]  # Convert to letter

        # Create formatted question with permuted choices
        choices_dict = {
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "Question": row["Question"],
        }

        formatted_question = format_multichoice_question(choices_dict)

        # Create processed row
        processed_row = {
            "question": row["Question"],  # Original question
            "text_input": formatted_question,  # Formatted with shuffled options
            "ground_truth": correct_answer,  # A, B, C, or D (after permutation)
            "dataset": f"gpqa_{variant}",  # Dataset identifier
            # Keep metadata for reference
            "domain": row.get("High-level domain", ""),
            "subdomain": row.get("Subdomain", ""),
        }

        processed_rows.append(processed_row)

    # Create final DataFrame
    result_df = pd.DataFrame(processed_rows)

    # Save as pickle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_pickle(output_path)

    print(f"\n{'=' * 60}")
    print("Dataset generated successfully!")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(result_df)}")
    print(f"Variant: {variant}")
    print(f"Seed: {seed}")
    print(f"Columns: {list(result_df.columns)}")
    print(f"{'=' * 60}")

    # Show sample
    if len(result_df) > 0:
        print("\nSample question:")
        print(result_df["text_input"].iloc[0])
        print(f"\nGround truth: {result_df['ground_truth'].iloc[0]}")


def main():
    """Main entry point for GPQA dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate GPQA dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full diamond variant
  python -m inference_endpoint.eval.dataset_generation.generate_gpqa \\
      --output datasets/gpqa_diamond.pkl --variant diamond

  # Generate 50 samples for testing
  python -m inference_endpoint.eval.dataset_generation.generate_gpqa \\
      --output tests/datasets/gpqa_50.pkl --variant diamond --num-samples 50

  # Generate with custom seed
  python -m inference_endpoint.eval.dataset_generation.generate_gpqa \\
      --output datasets/gpqa.pkl --variant extended --seed 123
        """,
    )

    parser.add_argument(
        "--output", type=Path, required=True, help="Output path for pickle file"
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="diamond",
        choices=["diamond", "extended", "main"],
        help="GPQA variant to generate (default: diamond)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--num-samples", type=int, help="Limit number of samples (default: all)"
    )

    args = parser.parse_args()

    try:
        generate_gpqa_dataset(
            output_path=args.output,
            variant=args.variant,
            seed=args.seed,
            num_samples=args.num_samples,
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
