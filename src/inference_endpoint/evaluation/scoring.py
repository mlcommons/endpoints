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
# See the License for the specific permissions and
# limitations under the License.


import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import orjson
import pandas as pd

from ..dataset_manager.dataset import Dataset
from ..load_generator.events import SampleEvent
from .extractor import Extractor


class Scorer(ABC):
    """Scorers will read in a dataset and outputs from a log and compute an accuracy score.
    An optional extractor can be provided to post-process the output to extract values that
    can be compared against the ground truth.
    """

    PREDEFINED: ClassVar[dict[str, type["Scorer"]]] = {}

    def __init_subclass__(
        cls,
        scorer_id: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        if scorer_id is not None:
            cls.SCORER_ID = scorer_id
            Scorer.PREDEFINED[scorer_id] = cls

    @classmethod
    def get(cls, scorer_id: str) -> type["Scorer"]:
        """Look up a Scorer subclass by its registered name.

        Args:
            scorer_id: The registered name of the scorer

        Returns:
            Scorer subclass
        """
        if scorer_id not in cls.PREDEFINED:
            raise ValueError(
                f"Unknown scorer: {scorer_id}. Available: {list(cls.PREDEFINED.keys())}"
            )
        return cls.PREDEFINED[scorer_id]

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str = "ground_truth",
    ):
        self.dataset = dataset
        self.report_dir = Path(report_dir)
        self.extractor = extractor
        self.dataset_name = dataset_name
        self.ground_truth_column = (
            ground_truth_column if ground_truth_column is not None else "ground_truth"
        )
        self.sample_index_map = self._load_sample_index_map()

    def _load_sample_index_map(self):
        sample_index_map_path = self.report_dir / "sample_idx_map.json"
        if not sample_index_map_path.exists():
            raise FileNotFoundError(
                f"Sample index map file not found at {sample_index_map_path}"
            )

        with sample_index_map_path.open("r") as f:
            d = orjson.loads(f.read())
            return d[self.dataset_name]  # Implicitly raises KeyError

    def get_outputs(self):
        # TODO: Currently, the outputs are only saved in the events.jsonl file, which is quite
        # large, and only saved optionally. Later, we should move to saving the outputs in a
        # separate file for easier compute.
        events_log_path = self.report_dir / "events.jsonl"
        if not events_log_path.exists():
            raise FileNotFoundError(f"Events log file not found at {events_log_path}")

        outputs = []
        with events_log_path.open("r") as f:
            for line in f:
                event = orjson.loads(line.strip())
                if event["event_type"] == SampleEvent.COMPLETE.value:
                    outputs.append(event)
        df = pd.DataFrame(outputs, columns=["sample_uuid", "value"])
        df.rename(columns={"value": "output"}, inplace=True)
        return df

    def match_sample_index(self, row: pd.Series) -> pd.Series:
        # Pandas Apply function to create a new 'sample_index' column
        row["sample_index"] = self.sample_index_map[row["sample_uuid"]]
        return row

    @abstractmethod
    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise NotImplementedError

    def score(self) -> tuple[float, int]:
        """Scores the dataset and returns the mean score and the number of repeats.

        Returns:
            tuple[float, int]: The mean score and the number of repeats.
        """
        df = self.get_outputs()

        # Outputs are for all samples, not just the target dataset
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]

        # Match to sample index from dataset
        df = df.apply(self.match_sample_index, axis=1)

        empirical = df["output"]
        if self.extractor is not None:
            empirical = empirical.apply(self.extractor.extract)
        empirical = empirical.to_numpy()

        # Get ground truths
        order = df["sample_index"].to_numpy()
        assert (
            self.ground_truth_column in self.dataset.dataframe.columns
        ), f"Ground truth column {self.ground_truth_column} not found in dataset {self.dataset}"
        ground_truths = self.dataset.dataframe[self.ground_truth_column].to_numpy()[
            order
        ]

        scores = []
        for i in range(len(empirical)):
            scores.append(self.score_single_sample(empirical[i], ground_truths[i]))

        n_repeats = len(scores) // self.dataset.num_samples()
        return np.mean(scores), n_repeats


class PassAt1Scorer(Scorer, scorer_id="pass_at_1"):
    """Implements pass@1 scoring as defined by Artificial Analysis.
    pass@1 means the model gets exactly one attempt to produce the correct answer.
    The score is 1 if the output matches the ground truth exactly, 0 otherwise.
    This is the standard scoring method for multiple-choice questions and other
    tasks where there is a single correct answer.
    Reference: https://artificialanalysis.ai/methodology/intelligence-benchmarking

    This is equivalent to Exact Match Scoring.
    """

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        return 1.0 if value == ground_truth else 0.0


ExactMatchScorer = PassAt1Scorer


class RougeScorer(Scorer, scorer_id="rouge"):
    """Implements ROUGE scoring for text generation evaluation.
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap
    between generated text and reference text. Returns the ROUGE-L F1 score.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            # from rouge_score import rouge_scorer
            # self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            import evaluate
            import nltk

            self.metric = evaluate.load("rouge")
            nltk.download("punkt")
            nltk.download("punkt_tab")
        except ImportError:
            raise ImportError(
                "nltk and evaluate is required for ROUGE scoring. "
                "Install it with: pip install nltk evaluate"
            ) from None

    def postprocess_text(self, texts):
        import nltk

        texts = [text.strip() for text in texts]
        # rougeLSum expects newline after each sentence
        texts = ["\n".join(nltk.sent_tokenize(text)) for text in texts]
        return texts

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        """Compute ROUGE-L F1 score between value and ground_truth."""
        # print("ground_truth:", ground_truth)
        # print("value:", value)
        # scores = self.rouge_scorer.score(ground_truth, value)
        # return scores['rougeL'].fmeasure

    def score(self) -> tuple[float, int]:
        df = self.get_outputs()

        # Outputs are for all samples, not just the target dataset
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]

        # Match to sample index from dataset
        df = df.apply(self.match_sample_index, axis=1)

        empirical = df["output"].tolist()

        order = df["sample_index"].to_numpy().astype(int)
        assert (
            self.ground_truth_column in self.dataset.dataframe.columns
        ), f"Ground truth column {self.ground_truth_column} not found in dataset {self.dataset}"

        ground_truths = list(
            self.dataset.dataframe[self.ground_truth_column].to_numpy()[order]
        )

        empirical = self.postprocess_text(empirical)
        ground_truths = self.postprocess_text(ground_truths)

        result = self.metric.compute(
            predictions=empirical,
            references=ground_truths,
            use_stemmer=True,
            use_aggregator=False,
        )

        result = {k: f"{round(np.mean(v) * 100, 4)}" for k, v in result.items()}
        prediction_lens = [len(pred) for pred in empirical]
        gen_num = len(empirical)

        result = {
            **result,
            "gen_len": f"{np.sum(prediction_lens)}",
            "gen_num": gen_num,
            # "gen_tok_len": gen_tok_len,
            # "tokens_per_sample": round(np.sum(prediction_lens) / gen_num, 1),
        }

        print("\nResults\n")
        print(result)

        return result, 1
