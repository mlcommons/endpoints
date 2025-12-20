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

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from ..dataset import Dataset, DatasetFormat, load_from_huggingface

logger = logging.getLogger(__name__)


class GPQA(
    Dataset,
    columns=[
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
        "ground_truth",
        "domain",
        "subdomain",
    ],
    format=DatasetFormat.PARQUET,
    ground_truth_column="ground_truth",
):
    """GPQA: A Graduate-Level Google-Proof Q&A Benchmark
    Reference: https://arxiv.org/abs/2311.12022
    OpenAI implementation: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py
    """

    def __init__(
        self,
        *args,
        variant: str = "diamond",
        seed: int = 0,  # GPT-OSS uses 0 as default seed
        max_samples: int | None = None,
        **kwargs,
    ):
        """
        Args:
            variant: The variant of the dataset to load. Must be one of "diamond", "extended", or "main".
            seed: The seed to use for the random number generator.
            max_samples: The maximum number of samples to load. If None, all samples will be loaded.
        """
        super().__init__(*args, variant=variant, **kwargs)
        if self.variant == "full":
            self.variant = "main"

        if self.variant not in ("diamond", "extended", "main"):
            raise ValueError(f"Invalid variant: {self.variant}")

        self.seed = seed
        self.max_samples = max_samples
        self._variant_name = self.variant
        if max_samples is not None and max_samples > 0:
            # Update the variant tag to also include the number of samples
            self.variant = f"{self._variant_name}_{max_samples}"

    def generate(self, datasets_dir: Path) -> pd.DataFrame:
        # Load the variant from HuggingFace
        try:
            df = load_from_huggingface(
                "Idavidrein/gpqa",
                dataset_name=f"gpqa_{self.variant}",
                split="train",
                cache_dir=datasets_dir / "hf_cache" / f"gpqa_{self.variant}",
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Note: This dataset may require HuggingFace authentication.")
            print("Run: huggingface-cli login")
            raise

        logger.info(f"Loaded {len(df)} samples from {self.variant} variant of GPQA")

        # If max_samples is specified, sample 'max_samples' rows from the dataset
        if self.max_samples is not None and self.max_samples < len(df):
            rng = random.Random(self.seed)
            sampled_indices = rng.sample(range(len(df)), self.max_samples)
            df = df.iloc[sampled_indices].reset_index(drop=True)
            logger.info(f"Sampled {self.max_samples} questions")

        rng = random.Random(self.seed)

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
            correct_answer = f"choice{correct_index + 1}"

            # Create processed row
            processed_row = {
                "question": row["Question"],  # Original question
                "choice1": choices[0],
                "choice2": choices[1],
                "choice3": choices[2],
                "choice4": choices[3],
                "ground_truth": correct_answer,
                # Keep metadata for reference
                "domain": row.get("High-level domain", ""),
                "subdomain": row.get("Subdomain", ""),
            }

            processed_rows.append(processed_row)
        df = pd.DataFrame(processed_rows)

        # Save to pickle file
        if not datasets_dir.exists():
            datasets_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(datasets_dir / self.filename)
        logger.info(f"Saved {len(df)} samples to {datasets_dir / self.filename}")

        return df


class RowToSGLangHarmony:
    def __init__(self, tokenizer_name: str = "openai/gpt-oss-120b"):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.harmony = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        self.system_message = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort.HIGH)
            .with_conversation_start_date(
                "2025-09-30"
            )  # Same as in official MLCommons GPT-OSS implementation
        )
        self.user_prompt_format = (
            "{question}\n\n"
            "(A) {choice1}\n"
            "(B) {choice2}\n"
            "(C) {choice3}\n"
            "(D) {choice4}\n\n"
            "Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'."
        )

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert a row of the Inference Endpoints GPQA dataset to the format that is compatible with
        the SGLang implementation of GPT-OSS for legacy LoadGen.
        """
        # First generate the user prompt
        user_prompt = self.user_prompt_format.format(**row)
        gt = {
            "choice1": "A",
            "choice2": "B",
            "choice3": "C",
            "choice4": "D",
        }[row["ground_truth"]]

        # Harmonize with OpenAI Harmony
        conv = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, self.system_message),
                Message.from_role_and_content(Role.USER, user_prompt),
            ]
        )

        toks = self.harmony.render_conversation_for_completion(conv, Role.ASSISTANT)
        harmonized_text = self.tokenizer.decode(toks, skip_special_tokens=False)

        return {
            "input_tokens": toks,
            "num_tokens": len(toks),
            "text_input": harmonized_text,
            "original_prompt": user_prompt,
            "ground_truth": gt,
            "dataset": "gpqa_diamond",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GPQA dataset for accuracy evaluation"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="diamond",
        choices=["diamond", "extended", "main"],
        help="The variant of the dataset to generate (default: diamond)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets"),
        help="Output directory for the dataset (default: ./datasets)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling choices (default: 0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate (default: all samples)",
    )
    parser.add_argument(
        "--make-legacy-lg-compat",
        action="store_true",
        help="Make the dataset compatible with the legacy LG format",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="openai/gpt-oss-120b",
        help="The name of the tokenizer to use for the dataset (default: openai/gpt-oss-120b)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create GPQA dataset instance
    gpqa = GPQA(variant=args.variant, seed=args.seed, max_samples=args.max_samples)

    # Generate the dataset
    logger.info(f"Generating GPQA {args.variant} dataset...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Seed: {args.seed}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")

    df = gpqa.generate(args.output_dir)

    logger.info(f"✓ Successfully generated dataset: {args.output_dir / gpqa.filename}")

    if args.make_legacy_lg_compat:
        # Legacy LG dataset file uses the GPT-OSS tokenizer, and pre-harmonizes the input text
        # The output columns are:
        # - question: The original question as plaintext
        # - ground_truth: The ABCD letter choice for the correct answer
        # - dataset: This is always "gpqa" for GPQA
        # - tok_input: The tokenized input text as a list of integers
        # - tok_input_len: The length of the tokenized input text
        # - text_input: The harmonized input text as a string
        from openai_harmony import (
            Conversation,
            HarmonyEncodingName,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
            load_harmony_encoding,
        )
        from transformers import AutoTokenizer

        from ..dataloader import ParquetLoader

        loader = ParquetLoader(
            gpqa,
            datasets_dir=Path(args.output_dir),
            process_row=RowToSGLangHarmony(tokenizer_name=args.tokenizer_name),
        )
        # Use existing dataframe instead of re-reading it
        loader.data = df
        loader.loaded = True

        new_rows = []
        n_repeats = 5
        for _ in range(n_repeats):
            for i in range(loader.num_samples()):
                row = loader.load_sample(i)
                new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        new_df.to_parquet(args.output_dir / "gpqa_diamond_legacy_lg_compat.parquet")
        logger.info(
            f"Saved {len(new_df)} samples to {args.output_dir / "gpqa_diamond_legacy_lg_compat.parquet"}"
        )
