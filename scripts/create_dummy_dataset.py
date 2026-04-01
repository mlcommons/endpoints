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

"""Create dummy dataset for CLI testing.

This script generates a dummy dataset (not real benchmark data) with a configurable
number of samples (default 1000) in the same format as real datasets, useful for
local testing without requiring large production datasets.

Usage:
    python scripts/create_dummy_dataset.py  # Creates 1000 samples
    python scripts/create_dummy_dataset.py --samples 500  # Creates 500 samples
    python scripts/create_dummy_dataset.py --samples 5000 --output custom.jsonl
"""

import argparse
from pathlib import Path

import pandas as pd


def create_dummy_dataset(num_samples: int = 1000, output_path: str = None):
    """Create dummy dataset with varied prompts.

    Args:
        num_samples: Number of samples to generate
        output_path: Output file path (default: tests/datasets/dummy_1k.jsonl)
    """
    # Create varied prompts
    prompt_templates = [
        "Write a short story about {}",
        "Explain the concept of {} in simple terms",
        "Create a poem about {}",
        "Describe {} in detail",
        "What are the key features of {}?",
        "Compare and contrast {}",
        "Analyze the importance of {}",
        "Provide examples of {}",
        "Discuss the history of {}",
        "Summarize the main points about {}",
    ]

    topics = [
        "artificial intelligence",
        "quantum computing",
        "renewable energy",
        "space exploration",
        "biotechnology",
        "climate change",
        "machine learning",
        "neural networks",
        "data science",
        "robotics",
    ]

    prompts = []
    outputs = []

    for i in range(num_samples):
        template = prompt_templates[i % len(prompt_templates)]
        topic = topics[i % len(topics)]
        # Add variation with numbers
        prompt = template.format(f"{topic} (case {i})")
        prompts.append(prompt)

        # Generate corresponding reference output
        output = (
            f"Response for prompt {i}: This is a detailed response about {topic}..."
        )
        outputs.append(output)

    # Create DataFrame matching the expected format
    data = {"text_input": prompts, "ref_output": outputs}
    df = pd.DataFrame(data)

    # Determine output path
    if output_path is None:
        repo_root = Path(__file__).parent.parent
        output_path = repo_root / "tests" / "datasets" / "dummy_1k.jsonl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    df.to_json(output_path, orient="records", lines=True)

    print(f"✓ Created {output_path} with {len(df)} samples")
    print(f"✓ Sample prompt: {df['text_input'][0]}")
    print(f"✓ Sample output: {df['ref_output'][0]}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Create dummy dataset for CLI testing")
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: tests/datasets/dummy_1k.jsonl)",
    )

    args = parser.parse_args()

    create_dummy_dataset(num_samples=args.samples, output_path=args.output)


if __name__ == "__main__":
    main()
