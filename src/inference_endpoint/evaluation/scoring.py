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


import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, ClassVar

import msgspec
import msgspec.json
import numpy as np
import pandas as pd
import yaml
from pydantic import ValidationError
from tqdm import tqdm

try:
    import websocket
except ImportError:
    websocket = None

try:
    import evaluate as _evaluate
    import nltk as _nltk
except ImportError:
    _evaluate = None
    _nltk = None

from ..core.record import EventRecord, EventType, SampleEventType
from ..core.types import TextModelOutput
from ..dataset_manager.agentic_inference_dataset import AgenticInferenceDataset
from ..dataset_manager.dataset import Dataset
from ..dataset_manager.predefined.shopify_product_catalogue import ProductMetadata
from ..dataset_manager.predefined.swe_bench import _REPO_MAP as _SWE_BENCH_HF_MAP
from ..exceptions import SetupError
from .extractor import Extractor, PythonCodeExtractor

logger = logging.getLogger(__name__)


class Scorer(ABC):
    """Scorers will read in a dataset and outputs from a log and compute an accuracy score.
    An optional extractor can be provided to post-process the output to extract values that
    can be compared against the ground truth.
    """

    PREDEFINED: ClassVar[dict[str, type["Scorer"]]] = {}
    SCORER_ID: ClassVar[str]
    REQUIRES_EXTRACTOR: ClassVar[bool] = True
    SKIP_ENDPOINT_PHASE: ClassVar[bool] = False

    def __init_subclass__(
        cls,
        scorer_id: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            if scorer_id is None:
                scorer_id = cls.__name__
            cls.SCORER_ID = scorer_id
            Scorer.PREDEFINED[scorer_id] = cls

    @classmethod
    def get(cls, name: str) -> type["Scorer"]:
        """Look up an Scorer subclass by its registered name.

        Args:
            name: str, the registered scorer name

        Returns:
            Scorer subclass

        Raises:
            KeyError: If no scorer with the given name is found
        """
        try:
            return Scorer.PREDEFINED[name]
        except KeyError as e:
            raise KeyError(
                f"Scorer '{name}' is not registered - available scorers: {Scorer.available_scorers()}"
            ) from e

    @classmethod
    def available_scorers(cls) -> list[str]:
        """Return the list of registered scorer names."""
        return list(Scorer.PREDEFINED.keys())

    @classmethod
    def external_sample_count(cls, extras: dict[str, Any]) -> int | None:
        """Return the number of samples the scorer will evaluate externally, or None.

        Used to surface sample counts for scorers that skip the endpoint phase and
        manage their own evaluation (e.g. SWEBenchScorer running via mini-extra).
        The default returns None (scorer uses the endpoint accuracy phase normally).
        """
        return None

    @classmethod  # noqa: B027
    def preflight(cls, extras: dict[str, Any]) -> None:
        """Verify external dependencies before the benchmark starts. No-op by default."""

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = "ground_truth",
    ):
        self.dataset = dataset
        self.report_dir = Path(report_dir)
        self.extractor = extractor
        self.dataset_name = dataset_name

        self.ground_truth_column = (
            ground_truth_column if ground_truth_column is not None else "ground_truth"
        )
        self.sample_index_map = (
            None if self.SKIP_ENDPOINT_PHASE else self._load_sample_index_map()
        )

    def _load_sample_index_map(self):
        sample_index_map_path = self.report_dir / "sample_idx_map.json"
        if not sample_index_map_path.exists():
            raise FileNotFoundError(
                f"Sample index map file not found at {sample_index_map_path}"
            )

        with sample_index_map_path.open("r") as f:
            d = msgspec.json.decode(f.read())
            return d[self.dataset_name]  # Implicitly raises KeyError

    def get_outputs(self):
        """Read COMPLETE events from events.jsonl and extract response text.

        The EventLoggerService writes EventRecord objects serialized via msgspec.
        We decode them using the EventRecord decoder and extract the response
        text from TextModelOutput data.
        """
        events_log_path = self.report_dir / "events.jsonl"
        if not events_log_path.exists():
            raise FileNotFoundError(f"Events log file not found at {events_log_path}")

        decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
        outputs: list[dict[str, str]] = []
        with events_log_path.open("r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                record = decoder.decode(stripped)
                if record.event_type == SampleEventType.COMPLETE:
                    output_text = str(record.data) if record.data is not None else ""
                    outputs.append(
                        {"sample_uuid": record.sample_uuid, "output": output_text}
                    )
        return pd.DataFrame(outputs)

    def match_sample_index(self, row: pd.Series) -> pd.Series:
        # Pandas Apply function to create a new 'sample_index' column
        row["sample_index"] = self.sample_index_map[row["sample_uuid"]]
        return row

    @abstractmethod
    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise NotImplementedError

    def score(self) -> tuple[float | None, int]:
        """Scores the dataset and returns the mean score and the number of repeats.

        Returns:
            tuple[float | None, int]: The mean score and the number of repeats.
                Returns None as the score if evaluation fails.
        """
        assert self.sample_index_map is not None, (
            f"{self.__class__.__name__} sets SKIP_ENDPOINT_PHASE=True but did not "
            "override score(); either override score() or set SKIP_ENDPOINT_PHASE=False."
        )
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
            self.dataset.dataframe is not None
        ), f"Dataset {self.dataset} has no dataframe loaded"
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


class StringMatchScorer(Scorer, scorer_id="string_match"):
    """Implements exact string match scoring.
    The score is 1 if the output matches the ground truth exactly, 0 otherwise.
    This is useful for debugging and development.
    """

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        return 1.0 if value.strip() == ground_truth.strip() else 0.0


ExactMatchScorer = PassAt1Scorer


class RougeScorer(Scorer, scorer_id="rouge"):
    """Implements ROUGE scoring for text generation evaluation.
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap
    between generated text and reference text. Returns the ROUGE-L F1 score.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _evaluate is None or _nltk is None:
            raise ImportError(
                "nltk, evaluate, and rouge_score are required for ROUGE scoring. "
                "Install with: pip install nltk evaluate rouge_score"
            )
        self.metric = _evaluate.load("rouge")
        self.nltk = _nltk

    def postprocess_text(self, texts):
        texts = [text.strip() for text in texts]
        # rougeLSum expects newline after each sentence
        texts = ["\n".join(self.nltk.sent_tokenize(text)) for text in texts]
        return texts

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        # This method is not used
        raise RuntimeError(
            "ROUGE scoring requires batch processing for accurate aggregation. "
            "Call score() to compute metrics across the entire dataset instead of "
            "per-sample scoring."
        )

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
            self.dataset.dataframe is not None
        ), f"Dataset {self.dataset} has no dataframe loaded"
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
        }

        # TODO: return only rouge1 for now to align with other scorers
        # Return the rest of the metrics later
        return result, 1


class AgenticInferenceInlineScorer(Scorer, scorer_id="agentic_inference_inline"):
    """Score agentic inference performance replay outputs without issuing another phase."""

    REQUIRES_EXTRACTOR = False
    _EXECUTABLE_ALIASES: ClassVar[dict[str, str]] = {
        "python": "python",
        "python2": "python",
        "python3": "python",
        "py": "python",
        "pip": "pip",
        "pip3": "pip",
        "pytest": "pytest",
        "pylint": "pylint",
        "sphinx-build": "sphinx",
        "sphinx-quickstart": "sphinx",
        "cython": "cython",
        "make": "make",
        "conda": "conda",
        "cat": "cat",
        "head": "head",
        "tail": "tail",
        "less": "cat",
        "more": "cat",
        "wc": "wc",
        "diff": "diff",
        "grep": "grep",
        "egrep": "grep",
        "fgrep": "grep",
        "rg": "grep",
        "ag": "grep",
        "sed": "sed",
        "awk": "awk",
        "gawk": "awk",
        "tr": "tr",
        "sort": "sort",
        "uniq": "uniq",
        "cut": "cut",
        "find": "find",
        "ls": "ls",
        "locate": "find",
        "xargs": "xargs",
        "cp": "cp",
        "mv": "mv",
        "rm": "rm",
        "mkdir": "mkdir",
        "touch": "touch",
        "tee": "tee",
        "source": "source",
        ".": "source",
        "which": "which",
        "alias": "alias",
        "unset": "unset",
        "export": "export",
        "git": "git",
        "curl": "curl",
        "wget": "curl",
        "true": "true",
        "false": "false",
        "timeout": "timeout",
        "date": "date",
        "apt-get": "apt",
        "apt": "apt",
        "yum": "yum",
    }
    _SHELL_WRAPPERS: ClassVar[set[str]] = {
        "env",
        "time",
        "nice",
        "sudo",
        "exec",
        "command",
    }
    _REPEAT_SUFFIX_RE: ClassVar[re.Pattern[str]] = re.compile(r"__repeat_(\d+)$")
    _WORKFLOW_CONVERSATION_RE: ClassVar[re.Pattern[str]] = re.compile(r"^sim_\d+$")
    _INTENT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\bintent:\s*(I\d{3})\b", re.IGNORECASE
    )
    _BARE_INTENT_RE: ClassVar[re.Pattern[str]] = re.compile(r"\bI(\d{3})\b")
    _COMMAND_SEPARATOR_RE: ClassVar[re.Pattern[str]] = re.compile(r"\|\||\||&&|;|\n")
    _QUOTED_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"'[^']*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`"
    )
    _ENV_ASSIGNMENT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^[A-Za-z_][A-Za-z0-9_]*="
    )
    _PY_VERSION_SUFFIX_RE: ClassVar[re.Pattern[str]] = re.compile(r"\.\d+(\.\d+)?$")

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = None,
        scores_filename: str = "scores.json",
    ):
        """Initialize a scorer for already-issued agentic inference performance events.

        The scorer intentionally does not use an extractor or a single
        ``ground_truth`` column. Ground truth is derived from expected assistant
        turns in the loaded ``AgenticInferenceDataset`` dataframe.

        Example:
            A performance dataset config such as
            ``accuracy_config.eval_method: agentic_inference_inline`` instantiates this
            scorer with ``dataset_name="performance"`` so it reads the
            performance phase's entries from ``sample_idx_map.json``.
        """
        if extractor is not None:
            raise ValueError("AgenticInferenceInlineScorer does not use an extractor")
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=None,
            ground_truth_column=ground_truth_column,
        )
        self.scores_filename = scores_filename

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        """Reject single-sample scoring for the conversation-level scorer.

        Agentic inference accuracy depends on neighboring turns and conversation ids,
        so a single output string cannot be scored in isolation.

        Example:
            ``score_single_sample("answer", "expected")`` raises
            ``RuntimeError``; callers should use ``score()``.
        """
        raise RuntimeError(
            "AgenticInferenceInlineScorer scores whole conversations; call score()."
        )

    def score(self) -> tuple[float | None, int]:
        """Score completed agentic inference performance outputs.

        The method builds expected assistant turns from the loaded dataset,
        reads issued turns and model assistant completions from ``events.jsonl``,
        identifies each conversation as workflow or coding, and averages issued
        turns with scorable ground truth. Issued turns without a model output
        contribute score ``0``.

        Examples:
            A workflow turn with ``intent_codes=["I042"]`` scores ``1.0`` when
            the model text contains ``intent: I042``.

            A coding turn with expected bash command ``{"cmd": "python test.py"}``
            is scored by comparing normalized executables such as ``["python"]``
            against the model's bash tool calls.
        """
        if not isinstance(self.dataset, AgenticInferenceDataset):
            raise TypeError(
                "AgenticInferenceInlineScorer requires an AgenticInferenceDataset"
            )
        assert (
            self.dataset.dataframe is not None
        ), f"Dataset {self.dataset} has no dataframe loaded"

        expected = self._expected_assistant_turns()
        scorable_expected: dict[tuple[str, int], dict[str, Any]] = {}
        excluded_turns: list[dict[str, Any]] = []
        for (conversation_id, client_turn), ground_truth in sorted(expected.items()):
            domain = (
                "workflow"
                if self._WORKFLOW_CONVERSATION_RE.match(conversation_id)
                else "coding"
            )
            has_ground_truth = (
                bool(self._ground_truth_intents(ground_truth))
                if domain == "workflow"
                else bool(self._bash_actions(ground_truth))
            )
            if has_ground_truth:
                scorable_expected[(conversation_id, client_turn)] = ground_truth
            else:
                excluded_turns.append(
                    {
                        "conversation_id": conversation_id,
                        "turn": ground_truth["_assistant_turn"],
                        "domain": domain,
                        "exclude_reason": "no ground truth",
                    }
                )

        issued_turns, model_turns = self._issued_and_completed_model_turns(
            set(expected)
        )
        issued_repeats = sorted({key[1] for key in issued_turns})
        scorable_issued_turns = sorted(
            key for key in issued_turns if (key[0], key[2]) in scorable_expected
        )

        total_score = 0.0
        n_scored = 0
        domain_totals = {"coding": 0.0, "workflow": 0.0}
        domain_counts = {"coding": 0, "workflow": 0}
        per_turn: list[dict[str, Any]] = []

        for conversation_id, repeat_id, client_turn in scorable_issued_turns:
            ground_truth = scorable_expected[(conversation_id, client_turn)]
            key = (conversation_id, repeat_id, client_turn)
            model = model_turns.get(key)
            domain = (
                "workflow"
                if self._WORKFLOW_CONVERSATION_RE.match(conversation_id)
                else "coding"
            )
            row: dict[str, Any] = {
                "conversation_id": conversation_id,
                "repeat": repeat_id,
                "turn": ground_truth["_assistant_turn"],
                "domain": domain,
            }

            if model is None:
                row["missing"] = True
                model = {"role": "assistant"}

            score: float
            if domain == "workflow":
                gt_intents = self._ground_truth_intents(ground_truth)
                model_intent = self._model_intent(model)
                row["gt_intents"] = sorted(gt_intents)
                row["model_intent"] = model_intent
                score = 1.0 if model_intent in gt_intents else 0.0
            else:
                gt_actions = self._bash_actions(ground_truth)
                model_actions = self._bash_actions(model)
                row["gt_actions"] = gt_actions
                row["model_actions"] = model_actions
                gt_counts = Counter(gt_actions)
                model_counts = Counter(model_actions)
                union = sum((gt_counts | model_counts).values())
                score = sum((gt_counts & model_counts).values()) / union

            row["score"] = round(score, 4)
            per_turn.append(row)
            total_score += score
            n_scored += 1
            domain_totals[domain] += score
            domain_counts[domain] += 1

        expected_outputs = set(scorable_issued_turns)
        observed_outputs = {
            key
            for key, model in model_turns.items()
            if model and key in expected_outputs
        }
        missing_outputs = len(expected_outputs - observed_outputs)
        final_score = round(total_score / n_scored, 4) if n_scored else None
        result: dict[str, Any] = {
            "score": final_score,
            "turns": {
                "issued": len(issued_turns),
                "expected": len(expected_outputs),
                "observed": len(observed_outputs),
                "missing": missing_outputs,
                "scored": n_scored,
            },
            "domains": {
                domain: {
                    "score": round(domain_totals[domain] / domain_counts[domain], 4),
                    "scored": domain_counts[domain],
                }
                for domain in ("coding", "workflow")
                if domain_counts[domain]
            },
            "per_turn": per_turn,
        }
        if excluded_turns:
            result["excluded_turns"] = excluded_turns

        out_path = self.report_dir / self.scores_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        return final_score, len(issued_repeats)

    def _expected_assistant_turns(self) -> dict[tuple[str, int], dict[str, Any]]:
        """Return expected assistant turns keyed by source conversation and turn.

        The dataset stores alternating client-side rows and expected assistant
        rows. This method pairs each ``user`` or ``tool`` row with the following
        ``assistant`` row and uses the client row's turn as the event-log turn
        to match.

        Example:
            Rows ``conv1/user/turn=1`` followed by ``conv1/assistant/turn=2``
            produce ``expected[("conv1", 1)]`` with ``"_assistant_turn": 2``.
        """
        assert (
            self.dataset.dataframe is not None
        ), f"Dataset {self.dataset} has no dataframe loaded"

        rows_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw_row in self.dataset.dataframe.to_dict("records"):
            row: dict[str, Any] = {}
            for field, value in raw_row.items():
                try:
                    row[field] = None if value != value else value
                except (TypeError, ValueError):
                    row[field] = value
            conversation_id = row.get("conversation_id")
            if conversation_id is not None:
                rows_by_conversation[str(conversation_id)].append(row)

        expected: dict[tuple[str, int], dict[str, Any]] = {}
        for conversation_id, rows in rows_by_conversation.items():
            rows.sort(key=lambda row: int(row.get("turn") or 0))
            for row, next_row in zip(rows, rows[1:], strict=False):
                if row.get("role") not in ("user", "tool"):
                    continue
                if next_row.get("role") != "assistant":
                    continue
                try:
                    client_turn = int(row.get("turn") or 0)
                    assistant_turn = int(next_row.get("turn") or 0)
                except (TypeError, ValueError):
                    continue
                expected[(conversation_id, client_turn)] = {
                    **next_row,
                    "_assistant_turn": assistant_turn,
                }
        return expected

    def _issued_and_completed_model_turns(
        self, expected_keys: set[tuple[str, int]]
    ) -> tuple[
        set[tuple[str, int, int]],
        dict[tuple[str, int, int], dict[str, Any] | None],
    ]:
        """Read issued turns and completed model outputs from ``events.jsonl``.

        ISSUED records define the scoring denominator. COMPLETE records are
        joined by ``sample_uuid`` and may carry ``None`` data for failed turns,
        which keeps those turns in the denominator with score ``0``.

        Example:
            ISSUED conversation id ``"conv1__repeat_3"`` and turn ``1`` becomes
            issued key ``("conv1", 3, 1)``. A matching COMPLETE record with
            ``data=None`` is returned as ``model_turns[("conv1", 3, 1)] = None``.
        """
        events_path = self.report_dir / "events.jsonl"
        if not events_path.exists():
            raise FileNotFoundError(f"Events log file not found at {events_path}")

        decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
        uuid_to_key: dict[str, tuple[str, int, int]] = {}
        completed_by_uuid: dict[str, dict[str, Any] | None] = {}
        issued_turns: set[tuple[str, int, int]] = set()
        model_turns: dict[tuple[str, int, int], dict[str, Any] | None] = {}
        with events_path.open() as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = decoder.decode(stripped)
                except msgspec.DecodeError as exc:
                    logger.warning(
                        "Skipping malformed event log line %d in %s: %s",
                        line_no,
                        events_path,
                        exc,
                    )
                    continue
                if record.event_type not in (
                    SampleEventType.ISSUED,
                    SampleEventType.COMPLETE,
                ):
                    continue
                if record.turn is None or not record.conversation_id:
                    continue

                conversation_id = record.conversation_id
                repeat_id = 1
                repeat_match = self._REPEAT_SUFFIX_RE.search(conversation_id)
                if repeat_match is not None:
                    conversation_id = conversation_id[: repeat_match.start()]
                    repeat_id = int(repeat_match.group(1))
                turn = int(record.turn)
                if (conversation_id, turn) not in expected_keys:
                    continue

                key = (conversation_id, repeat_id, turn)
                if record.event_type == SampleEventType.ISSUED:
                    uuid_to_key[record.sample_uuid] = key
                    issued_turns.add(key)
                    if record.sample_uuid in completed_by_uuid:
                        model_turns[key] = completed_by_uuid[record.sample_uuid]
                    continue

                model: dict[str, Any] | None = None
                if isinstance(record.data, TextModelOutput):
                    content, reasoning, tool_calls = record.data.as_message_parts()
                    model = {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                        "tool_calls": list(tool_calls) if tool_calls else None,
                    }
                if model is not None or record.sample_uuid not in completed_by_uuid:
                    completed_by_uuid[record.sample_uuid] = model
                if record.sample_uuid in uuid_to_key:
                    key = uuid_to_key[record.sample_uuid]
                    if model is not None or key not in model_turns:
                        model_turns[key] = model
        return issued_turns, model_turns

    def _ground_truth_intents(self, turn: dict[str, Any]) -> set[str]:
        """Extract valid workflow intent codes from a ground-truth turn.

        Example:
            ``{"intent_codes": ["i001", "I002", None]}`` returns
            ``{"I001", "I002"}``.
        """
        codes = turn.get("intent_codes")
        if not isinstance(codes, list | tuple):
            return set()
        return {code.upper() for code in codes if isinstance(code, str) and code}

    def _model_intent(self, turn: dict[str, Any]) -> str | None:
        """Extract the model's workflow intent code from text fields.

        The explicit ``intent: I123`` form is preferred. If absent, the last bare
        ``I123`` token in ``reasoning_content`` or ``content`` is used.

        Example:
            ``{"content": "final intent: I042"}`` returns ``"I042"``.
        """
        for field in ("reasoning_content", "content"):
            text = turn.get(field) or ""
            if not isinstance(text, str):
                continue
            match = self._INTENT_RE.search(text)
            if match is not None:
                return match.group(1).upper()
        for field in ("reasoning_content", "content"):
            text = turn.get(field) or ""
            if not isinstance(text, str):
                continue
            matches = list(self._BARE_INTENT_RE.finditer(text))
            if matches:
                return f"I{matches[-1].group(1)}"
        return None

    def _bash_actions(self, turn: dict[str, Any]) -> list[str]:
        """Extract normalized bash executable names from assistant tool calls.

        Only ``bash`` function tool calls are considered. Shell wrappers,
        leading environment assignments, command paths, and common aliases are
        normalized before scoring.

        Example:
            A tool call with ``{"cmd": "CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -m pytest"}``
            returns ``["python"]``.
        """
        tool_calls = turn.get("tool_calls")
        if not isinstance(tool_calls, list | tuple):
            return []

        actions: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function") or {}
            if not isinstance(fn, dict) or fn.get("name") != "bash":
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if not isinstance(args, dict):
                continue
            command = args.get("command") or args.get("cmd")
            if not isinstance(command, str):
                continue

            command = self._QUOTED_RE.sub(" ", command)
            for stage in self._COMMAND_SEPARATOR_RE.split(command):
                tokens = stage.split()
                while tokens and (
                    self._ENV_ASSIGNMENT_RE.match(tokens[0])
                    or tokens[0] in self._SHELL_WRAPPERS
                ):
                    tokens = tokens[1:]
                if not tokens:
                    continue
                executable = tokens[0].rsplit("/", 1)[-1].lower()
                executable = self._PY_VERSION_SUFFIX_RE.sub("", executable)
                action = self._EXECUTABLE_ALIASES.get(executable)
                if action:
                    actions.append(action)
        return actions


class LiveCodeBenchScorer(Scorer, scorer_id="code_bench_scorer"):
    """Scorer for LiveCodeBench code generation tasks.

    Uses the lcb_runner evaluation framework to execute generated code against test cases.
    Can connect to a containerized WebSocket evaluation service or fall back to subprocess.

    The scorer:
    1. Extracts Python code from model outputs (using PythonCodeExtractor)
    2. Attempts to use WebSocket service if lcb_websocket_port is provided
    3. Falls back to subprocess execution if WebSocket is unavailable
    4. Returns pass@1 score based on test results

    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object containing problems
        report_dir: Directory containing evaluation logs
        extractor: Extractor class (defaults to PythonCodeExtractor)
        lcb_version: LiveCodeBench version tag (e.g., "release_v5", "release_v6")
        timeout: Timeout in seconds for each test execution
        question_id_column: Column name in dataset containing question IDs
        show_lcb_runner_output: Whether to show output during evaluation
        lcb_websocket_port: Port for WebSocket service on localhost (default: 13835)
                            Set to None to disable WebSocket and use subprocess only.
                            Why is the default port 13835? It's short for LCB WebSocket:
                            1=L, 3rd letter=C, 8=B, 3 rotated sideways=W, 5=S
    """

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] = PythonCodeExtractor,
        ground_truth_column: str | None = None,
        lcb_version: str = "release_v6",
        timeout: int = 60,
        question_id_column: str = "question_id",
        show_lcb_runner_output: bool = True,
        lcb_websocket_port: int | None = 13835,
    ):
        # Note: LiveCodeBench doesn't use ground_truth_column the same way
        # but we need to pass something to the parent
        assert (
            ground_truth_column is None
        ), "ground_truth_column should be None for LiveCodeBenchScorer"
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=question_id_column,
        )

        self.lcb_version = lcb_version
        self.timeout = timeout
        self.question_id_column = question_id_column
        self.show_lcb_runner_output = show_lcb_runner_output

        # Construct WebSocket URL from port if provided
        self.lcb_websocket_url = (
            f"ws://localhost:{lcb_websocket_port}/evaluate"
            if lcb_websocket_port is not None
            else None
        )

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "This method should not be called. Use the score() method instead, which invokes lcb_runner."
        )

    def _evaluate_via_websocket(self, codes_dict: dict[str, list[str]]) -> dict | None:
        """Attempt to evaluate via WebSocket service (synchronous).

        Configured for long-running connections (minutes to hours) with:
        - Extended timeouts for send/receive operations
        - Automatic ping/pong for connection keep-alive
        - Proper error handling for network interruptions

        Returns:
            dict with evaluation results, or None if connection failed
        """
        if websocket is None:
            print(
                "Warning: websocket-client package not installed, falling back to subprocess"
            )
            print("Install with: pip install websocket-client")
            return None

        try:
            # Create WebSocket connection with settings for long-running operations
            # Timeout is set high for long evaluations (hours), but recv() will return
            # as soon as data is available (not blocking for the full timeout)
            ws = websocket.create_connection(
                self.lcb_websocket_url,
                timeout=7200,  # 2 hours connection timeout
                ping_interval=30,  # Send ping every 30 seconds to keep connection alive
                ping_timeout=10,  # Wait 10 seconds for pong response
            )

            # Setup progress tracking
            total_samples = sum(len(codes) for codes in codes_dict.values())
            pbar = None

            try:
                # Send evaluation request
                request = {
                    "codes_dict": codes_dict,
                    "timeout_sec": self.timeout,
                }
                ws.send(msgspec.json.encode(request).decode("utf-8"))

                print(f"Connected to WebSocket service: {self.lcb_websocket_url}")
                print(
                    f"Evaluating {len(codes_dict)} questions ({total_samples} samples)..."
                )
                pbar = tqdm(
                    total=total_samples,
                    desc="LCB Evaluation",
                    unit="sample",
                )

                # Process responses
                while True:
                    try:
                        message = ws.recv()
                        if not message:
                            # Connection closed cleanly
                            break

                        data = msgspec.json.decode(message)
                        status = data.get("status")

                        if status == "started":
                            # Initial message, progress bar already initialized
                            pass

                        elif status == "progress":
                            completed = data.get("completed_samples", 0)
                            # Update progress bar to current position
                            pbar.n = completed
                            pbar.refresh()

                        elif status == "completed":
                            pbar.n = total_samples
                            pbar.refresh()
                            return data.get("result")

                        elif status == "error":
                            error_msg = data.get("error", "Unknown error")
                            print(f"WebSocket evaluation error: {error_msg}")
                            return None

                    except websocket.WebSocketTimeoutException:
                        # This shouldn't happen with ping/pong, but handle gracefully
                        print("WebSocket timeout - connection lost")
                        return None

                # If we exit the loop without returning, something went wrong
                return None

            finally:
                # Ensure progress bar is always closed
                if pbar:
                    pbar.close()

                # Close WebSocket connection
                try:
                    ws.close()
                except Exception:
                    pass  # Ignore errors on close

        except (ConnectionRefusedError, OSError, Exception) as e:
            print(f"WebSocket connection failed: {e}, falling back to subprocess")
            return None

    def _evaluate_via_subprocess(self, df: pd.DataFrame) -> float | None:
        """Evaluate via subprocess (fallback method).

        Returns:
            pass@1 score or None if evaluation failed
        """
        # Check if local evaluation is allowed via environment variable
        allow_local_eval = os.environ.get("ALLOW_LCB_LOCAL_EVAL", "").lower() in (
            "true",
            "1",
            "yes",
        )
        if not allow_local_eval:
            raise RuntimeError(
                "Local LiveCodeBench evaluation via subprocess is disabled by default for security reasons. "
                "To enable it, set the environment variable ALLOW_LCB_LOCAL_EVAL=true. "
                "This will allow execution of generated code on your local machine."
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_name = f"{uuid.uuid4()}.parquet"
            parquet_path = Path(temp_dir) / parquet_name
            df.to_parquet(parquet_path)

            # Invoke lcb_serve.py as a subprocess to avoid importing LiveCodeBench dependencies
            # in the main inference endpoint environment, and also because LCB eval will
            # attempt to sandbox Python code execution by setting a bunch of core standard library
            # methods to None (i.e. most things in the os, sys, and other such modules), which would
            # impact the rest of the current Python process.
            cmd = [
                sys.executable,
                "-m",
                "inference_endpoint.dataset_manager.predefined.livecodebench.lcb_serve",
                str(parquet_path),
                "--version-tag",
                self.lcb_version,
                "--datasets-dir",
                f"datasets/livecodebench/{self.lcb_version}",
                "--timeout",
                str(self.timeout),
            ]

            try:
                # Run subprocess with output both captured and displayed (tee-like behavior)
                # Note: We let stderr pass through directly for real-time progress bars/logs
                proc_stderr = (
                    None if self.show_lcb_runner_output else subprocess.DEVNULL
                )

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=proc_stderr,
                    text=True,
                    bufsize=1,  # Line buffered
                )

                # Collect stdout while displaying it character-by-character to support
                # progress bars that use carriage returns
                if process.stdout is None:
                    raise RuntimeError("Failed to capture subprocess stdout")

                stdout_buffer = []
                while True:
                    char = process.stdout.read(1)
                    if not char:
                        break

                    if self.show_lcb_runner_output:
                        sys.stdout.write(char)
                        sys.stdout.flush()
                    stdout_buffer.append(char)

                # Wait for process to complete and check return code
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)

                # Parse the JSON output from the captured stdout
                # Look for JSON at the end (after any progress bar output)
                stdout_text = "".join(stdout_buffer)
                # Try to find the last line that looks like JSON
                lines = stdout_text.strip().split("\n")
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        output = msgspec.json.decode(line.encode("utf-8"))
                        return output["pass_at_1"]

                # No JSON found, try parsing the whole output
                output = msgspec.json.decode(stdout_text.encode("utf-8"))
                return output["pass_at_1"]

            except (subprocess.CalledProcessError, msgspec.DecodeError, KeyError):
                # Return None if subprocess fails or JSON parsing fails
                return None

    def score(self) -> tuple[float | None, int]:
        """Score the dataset using parallel evaluation.

        Attempts WebSocket evaluation first if configured, falls back to subprocess.

        Returns:
            tuple[float | None, int]: The pass@1 score and the number of repeats.
            Returns None as the score if evaluation fails.
        """
        df = self.get_outputs()

        # Outputs are for all samples, not just the target dataset
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]

        # Match to sample index from dataset
        df = df.apply(self.match_sample_index, axis=1)

        # Get question IDs
        assert (
            self.dataset.dataframe is not None
        ), f"Dataset {self.dataset} has no dataframe loaded"

        def get_question_id(sample_index: int) -> str:
            assert self.dataset.dataframe is not None
            return self.dataset.dataframe.iloc[sample_index][self.question_id_column]

        df["question_id"] = df["sample_index"].apply(get_question_id)

        # Extract code from outputs with default value for failed extractions
        # Use a comment that will fail all tests instead of None to maintain uniform list lengths
        assert self.extractor is not None, "Extractor must be set for code extraction"
        df["extracted_code"] = df["output"].apply(
            lambda x: self.extractor.extract(x, default="# FAILED TO EXTRACT CODE")
        )

        n_repeats = len(df) // self.dataset.num_samples()

        # Try WebSocket evaluation first if URL is provided
        if self.lcb_websocket_url:
            # Group codes by question ID for WebSocket API
            codes_dict = defaultdict(list)
            for _, row in df.iterrows():
                codes_dict[row["question_id"]].append(row["extracted_code"])

            # Attempt WebSocket evaluation (synchronous)
            result = self._evaluate_via_websocket(codes_dict)

            if result is not None:
                # Successfully evaluated via WebSocket
                total_samples = result.get("total_samples", 0)
                per_problem_results = result.get("results", {})
                if not per_problem_results and total_samples:
                    print(
                        f"Server evaluated {total_samples} samples but returned an empty summary"
                    )
                    return None, n_repeats

                total_passed = sum(
                    sum(code_passed) for code_passed in per_problem_results.values()
                )
                pass_at_1 = total_passed / total_samples if total_samples > 0 else 0.0
                return pass_at_1, n_repeats

        # Fall back to subprocess evaluation
        if self.show_lcb_runner_output and self.lcb_websocket_url:
            print(
                "WebSocket evaluation unavailable, using subprocess evaluation method"
            )

        pass_at_1 = self._evaluate_via_subprocess(df)
        return pass_at_1, n_repeats


_CATEGORY_SEPARATOR = " > "

# Pad tokens for unparsable responses (matches MLCommons Q3VL evaluation.py)
_PRED_CATEGORY_PAD = "<|__PRED_CATEGORY_PAD__|>"


def _create_pred_pad_category(ground_truth: str, separator: str) -> str:
    """Create dummy category with same depth as ground truth for unparsable responses.

    Matches MLCommons reference: unparsable responses get pred pad with matching depth
    so hierarchical F1 yields 0 intersection.
    """
    n_levels = len(ground_truth.split(separator))
    return separator.join([_PRED_CATEGORY_PAD] * n_levels) if n_levels > 0 else ""


def _parse_response_to_category(
    response: str,
    ground_truth: str,
    separator: str = _CATEGORY_SEPARATOR,
) -> str:
    """Parse model output to category, or use pred pad fallback for unparsable responses.

    Aligns with MLCommons Q3VL evaluation.py: validates with ProductMetadata directly,
    on ValidationError uses pred pad category with same depth as ground truth.
    No markdown/code-block stripping - reference passes raw string to model_validate_json.
    """
    try:
        parsed = ProductMetadata.model_validate_json(response)
        return parsed.category.strip()
    except ValidationError:
        return _create_pred_pad_category(ground_truth, separator)


def _match_hierarchical_paths(
    predicted_path: str,
    true_path: str,
    separator: str = _CATEGORY_SEPARATOR,
) -> tuple[int, int, int]:
    """Match two hierarchical category paths and return precision/recall components.

    Splits both paths on ``separator``, then counts consecutive matching levels
    from the root, stopping at the first mismatch. Returns the intersection
    count and the length of each path for use in hierarchical P/R calculation.

    Reference: https://github.com/mlcommons/inference/blob/master/multimodal/qwen3-vl/src/mlperf_inf_mm_q3vl/evaluation.py

    Example::

        data = [
            ("Clothing > Shirts > Polo",  "Clothing > Shirts > Polo"),   # exact match
            ("Clothing > Shirts > Dress", "Clothing > Shirts > Polo"),   # wrong leaf
        ]
        # Pair 1: intersection=3, pred_len=3, true_len=3
        # Pair 2: intersection=2 (stops at "Dress" != "Polo"), pred_len=3, true_len=3
        # HP = (3+2)/(3+3) = 5/6,  HR = (3+2)/(3+3) = 5/6
        # F1 = 2*(5/6)*(5/6) / (5/6+5/6) = 5/6 ≈ 0.833

    Args:
        predicted_path: Categories predicted by the VLM.
        true_path: Ground truth categories.
        separator: Separator for each level of the category (default " > ").

    Returns:
        Tuple of (intersection_count, predicted_length, true_length).
    """
    predicted_categories = [c.strip() for c in predicted_path.split(separator)]
    true_categories = [c.strip() for c in true_path.split(separator)]

    if not predicted_categories or not true_categories:
        return 0, len(predicted_categories), len(true_categories)

    intersection_count = 0
    for pred_cat, true_cat in zip(predicted_categories, true_categories, strict=False):
        if pred_cat == true_cat:
            intersection_count += 1
        else:
            break

    return intersection_count, len(predicted_categories), len(true_categories)


def _calculate_hierarchical_f1(
    data: list[tuple[str, str]],
    separator: str = _CATEGORY_SEPARATOR,
) -> float:
    """Calculate aggregate hierarchical F1 for a list of (predicted, true) pairs.

    Reference: https://github.com/mlcommons/inference/blob/master/multimodal/qwen3-vl/src/mlperf_inf_mm_q3vl/evaluation.py

    Args:
        data: List of (predicted_path_str, true_path_str) tuples.
        separator: Separator used to split paths into category levels.

    Returns:
        Hierarchical F1 score (0.0 to 1.0).
    """
    total_intersection = 0
    total_predicted_length = 0
    total_true_length = 0

    for pred_path, true_path in data:
        intersection, pred_len, true_len = _match_hierarchical_paths(
            predicted_path=pred_path,
            true_path=true_path,
            separator=separator,
        )
        total_intersection += intersection
        total_predicted_length += pred_len
        total_true_length += true_len

    hp = (
        total_intersection / total_predicted_length
        if total_predicted_length > 0
        else 0.0
    )
    hr = total_intersection / total_true_length if total_true_length > 0 else 0.0

    return 0.0 if hp + hr == 0 else 2 * (hp * hr) / (hp + hr)


class ShopifyCategoryF1Scorer(Scorer, scorer_id="shopify_category_f1"):
    """Hierarchical F1 scorer for Shopify product catalogue category classification.

    Implements the MLCommons Q3VL evaluation logic for category taxonomy.
    Model output must be JSON with category field (ProductMetadata format).
    Each category level is separated by " > " (e.g. "Clothing > Shirts > Polo").

    Reference: https://github.com/mlcommons/inference/blob/master/multimodal/qwen3-vl/src/mlperf_inf_mm_q3vl/evaluation.py
    """

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = "ground_truth_category",
        category_separator: str = _CATEGORY_SEPARATOR,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column,
        )
        self.category_separator = category_separator

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "ShopifyCategoryF1Scorer uses aggregate scoring. "
            "Call score() instead of score_single_sample."
        )

    def score(self) -> tuple[float, int]:
        df = self.get_outputs()

        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]
        df = df.apply(self.match_sample_index, axis=1)

        empirical = df["output"].tolist()

        order = df["sample_index"].to_numpy().astype(int)
        assert (
            self.dataset.dataframe is not None
        ), f"Dataset {self.dataset} has no dataframe loaded"
        assert (
            self.ground_truth_column in self.dataset.dataframe.columns
        ), f"Ground truth column {self.ground_truth_column} not found in dataset"

        ground_truths = list(
            self.dataset.dataframe[self.ground_truth_column].to_numpy()[order]
        )

        ground_truths = [str(g).strip() if g is not None else "" for g in ground_truths]

        predicted_categories = [
            _parse_response_to_category(out, gt, self.category_separator)
            for out, gt in zip(empirical, ground_truths, strict=False)
        ]

        data = list(zip(predicted_categories, ground_truths, strict=False))
        hf1 = _calculate_hierarchical_f1(data, separator=self.category_separator)

        n_repeats = len(data) // self.dataset.num_samples()
        return hf1, n_repeats


_VBENCH_DIMENSIONS: tuple[str, ...] = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "appearance_style",
    "scene",
)

_DEFAULT_VBENCH_PROJECT_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "09_Wan22_VideoGen_Example"
    / "accuracy"
)

_VBENCH_PROJECT_PATH_ENV = "VBENCH_PROJECT_PATH"

# Filenames in `vbench_standard` mode key on the prompt verbatim — VBench looks
# the filename's prompt-prefix up in vbench_full_info.json. We can therefore
# only reshape unsafe characters, not replace the prompt with a UUID. Slashes
# and `..` are turned into `_`; null bytes / control chars are rejected.
_UNSAFE_PROMPT_CHARS = re.compile(r"[\x00-\x1f/\\]")
_MAX_PROMPT_FILENAME_LEN = 200


def _sanitize_prompt_for_filename(prompt: str) -> str:
    """Make `prompt` safe to use as a filename component.

    Rejects `..` segments (path traversal) and replaces slashes and control
    characters with `_`. Truncates to `_MAX_PROMPT_FILENAME_LEN` to stay
    under ext4's 255-byte filename limit even after the `-{idx}.mp4` suffix.
    """
    if ".." in Path(prompt).parts or prompt == "..":
        raise ValueError(f"Refusing to stage video for prompt with '..': {prompt!r}")
    cleaned = _UNSAFE_PROMPT_CHARS.sub("_", prompt)
    if not cleaned or cleaned in (".", ".."):
        raise ValueError(f"Prompt sanitizes to an empty/invalid name: {prompt!r}")
    return cleaned[:_MAX_PROMPT_FILENAME_LEN]


class VBenchScorer(Scorer, scorer_id="vbench"):
    """VBench accuracy scorer for video generation outputs.

    Runs the six MLPerf WAN2.2 dimensions (subject_consistency,
    background_consistency, motion_smoothness, dynamic_degree,
    appearance_style, scene) on the produced videos and returns the mean
    of the per-dimension scores.

    VBench is invoked as a subprocess via `uv run --project <vbench_project_path>`
    so the main benchmark environment never imports vbench (which pins
    transformers==4.33.2 and numpy<2, incompatible with our core deps).
    The subproject lives at examples/09_Wan22_VideoGen_Example/accuracy/.

    Assumes the MLPerf WAN2.2 prompt set is a subset of VBench's standard
    prompt suite, so we use VBench's default evaluation flow: videos are
    staged into a directory with VBench's expected filename convention,
    `{prompt}-{index}.mp4`, and VBench looks each prompt up in its
    bundled `vbench_full_info.json`. Prompts are passed through
    `_sanitize_prompt_for_filename` first to keep the staged path inside
    `staged_dir`; VBench's prompt lookup tolerates the same `/`→`_`
    replacement applied here.

    The scorer reads each sample's video path from response_output (the
    VideoGenAdapter mirrors `video_path` into `TextModelOutput.output`)
    and the prompt from `dataset.dataframe[ground_truth_column]` — the
    prompt is the VBench input, not a comparison target, so callers should
    set `ground_truth_column: prompt` in `accuracy_config`.

    Returns `(None, n_repeats)` when no successful video was produced or
    when scoring fails to yield a usable per-dimension number — matching
    `LiveCodeBenchScorer` and the `Scorer.score()` contract.
    """

    REQUIRES_EXTRACTOR: ClassVar[bool] = False
    DIMENSIONS: ClassVar[tuple[str, ...]] = _VBENCH_DIMENSIONS
    DEFAULT_SUBPROCESS_TIMEOUT_S: ClassVar[int] = 4 * 60 * 60

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = "prompt",
        dimensions: tuple[str, ...] = _VBENCH_DIMENSIONS,
        full_info_json_path: str | None = None,
        vbench_project_path: os.PathLike | None = None,
        uv_executable: str = "uv",
        subprocess_timeout_s: int | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column,
        )
        self.dimensions = dimensions
        self.full_info_json_path = full_info_json_path
        self.vbench_project_path = self._resolve_project_path(vbench_project_path)
        self.uv_executable = uv_executable
        self.subprocess_timeout_s = (
            subprocess_timeout_s
            if subprocess_timeout_s is not None
            else self.DEFAULT_SUBPROCESS_TIMEOUT_S
        )
        runner = self.vbench_project_path / "vbench_runner.py"
        if not runner.exists():
            raise FileNotFoundError(
                f"vbench_runner.py not found at {runner}. "
                f"Run `uv sync` in the accuracy subproject, or set "
                f"${_VBENCH_PROJECT_PATH_ENV} to the synced subproject path."
            )

    @staticmethod
    def _resolve_project_path(
        explicit: os.PathLike | None,
    ) -> Path:
        """Resolve the VBench subproject path.

        Lookup order: explicit ctor arg → ``$VBENCH_PROJECT_PATH`` env var →
        editable-checkout fallback. The env var lets wheel-installed users
        point at a synced subproject without patching source.
        """
        if explicit is not None:
            return Path(explicit)
        from_env = os.environ.get(_VBENCH_PROJECT_PATH_ENV)
        if from_env:
            return Path(from_env)
        return Path(_DEFAULT_VBENCH_PROJECT_PATH)

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "VBench scoring requires batch processing; call score() instead."
        )

    def _stage_videos(
        self, staged_dir: Path, video_paths: list[str], prompts: list[str]
    ) -> None:
        """Symlink each video into a fresh staged_dir as `{prompt}-{index}.mp4`.

        Wipes `staged_dir` first so a re-score with fewer repeats can't leave
        stale `{prompt}-{M-1}.mp4` from a prior run for VBench to pick up.
        Indexing is per-prompt to disambiguate when the same prompt appears
        multiple times (num_repeats > 1).
        """
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        staged_dir.mkdir(parents=True)
        per_prompt_idx: dict[str, int] = defaultdict(int)
        for video_path, prompt in zip(video_paths, prompts, strict=True):
            safe_prompt = _sanitize_prompt_for_filename(prompt)
            idx = per_prompt_idx[safe_prompt]
            per_prompt_idx[safe_prompt] += 1
            src = Path(video_path)
            # strict=True surfaces missing/unmounted sources here, not as an
            # opaque decord read failure inside VBench 30 minutes later.
            resolved_src = src.resolve(strict=True)
            dst = staged_dir / f"{safe_prompt}-{idx}{src.suffix or '.mp4'}"
            dst.symlink_to(resolved_src)

    def _run_vbench_subprocess(
        self, staged_dir: Path, vbench_out: Path, run_name: str
    ) -> None:
        """Invoke vbench_runner.py via `uv run --project <subproject>`.

        Captures stdout+stderr into ``report_dir/vbench_subprocess.log`` and,
        on non-zero exit, raises with the tail of the captured log so the
        real failure (CUDA OOM, missing model, etc.) isn't lost.
        """
        cmd = [
            self.uv_executable,
            "run",
            "--project",
            str(self.vbench_project_path),
            "python",
            str(self.vbench_project_path / "vbench_runner.py"),
            "--videos-dir",
            str(staged_dir),
            "--out-dir",
            str(vbench_out),
            "--name",
            run_name,
            "--dims",
            ",".join(self.dimensions),
        ]
        if self.full_info_json_path is not None:
            cmd += ["--full-info-json", self.full_info_json_path]

        log_path = self.report_dir / "vbench_subprocess.log"
        _run_subprocess_with_log(
            cmd,
            log_path,
            timeout_s=self.subprocess_timeout_s,
            label="VBench",
        )

    def _extract_per_dim_scores(self, results: dict[str, Any]) -> list[float]:
        """Pull each requested dim's aggregate score, with clear errors.

        VBench's `_eval_results.json` is shaped `{dim: [aggregate, [per_video, ...]]}`.
        A missing dim (e.g. ``scene`` when the prompt set doesn't intersect
        VBench's scene suite) gets a named ValueError rather than the bare
        KeyError that propagates today.
        """
        missing = [d for d in self.dimensions if d not in results]
        if missing:
            raise ValueError(
                f"VBench results missing dimensions {missing}; "
                f"check that the prompt set overlaps vbench_standard for all "
                f"requested dimensions."
            )
        scores: list[float] = []
        for dim in self.dimensions:
            entry = results[dim]
            try:
                scores.append(float(entry[0]))
            except (IndexError, TypeError, ValueError) as e:
                raise ValueError(
                    f"VBench result for dimension {dim!r} is malformed: {entry!r}"
                ) from e
        return scores

    def score(self) -> tuple[float | None, int]:
        df = self.get_outputs()
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]
        # Drop failed queries: Scorer.get_outputs() emits "" when record.data
        # is None (workers set response_output=None on error). Passing "" to
        # _stage_videos would Path("").resolve() → cwd and symlink the repo
        # root as a "video", corrupting the entire VBench run. Failed samples
        # still count toward the denominator via n_total below.
        n_total = len(df)
        df = df[df["output"].astype(bool)]
        n_dropped = n_total - len(df)
        if n_dropped:
            logger.warning(
                "VBenchScorer: dropped %d failed/empty-output sample(s) before staging",
                n_dropped,
            )
        # n_repeats reflects the *issued* sample count (n_total), not the
        # surviving subset, so a single failure on a 1-repeat run still
        # reports n_repeats == 1.
        num_samples = self.dataset.num_samples()
        n_repeats = n_total // num_samples if num_samples else 0
        if df.empty:
            logger.warning(
                "VBenchScorer: no successful video outputs; returning None score."
            )
            return None, n_repeats

        df = df.apply(self.match_sample_index, axis=1)

        video_paths: list[str] = df["output"].tolist()
        order = df["sample_index"].to_numpy().astype(int)
        assert (
            self.dataset.dataframe is not None
        ), f"Dataset {self.dataset} has no dataframe loaded"
        assert (
            self.ground_truth_column in self.dataset.dataframe.columns
        ), f"Prompt column {self.ground_truth_column} not found in dataset"
        prompts: list[str] = [
            str(p)
            for p in self.dataset.dataframe[self.ground_truth_column].to_numpy()[order]
        ]

        # Stage videos for VBench in a per-run scratch dir under report_dir
        # so artifacts survive after the benchmark for re-evaluation.
        staged_dir = self.report_dir / "vbench_videos"
        self._stage_videos(staged_dir, video_paths, prompts)

        vbench_out = self.report_dir / "vbench_results"
        vbench_out.mkdir(parents=True, exist_ok=True)
        run_name = f"vbench_{self.dataset_name}"
        self._run_vbench_subprocess(staged_dir, vbench_out, run_name)

        # VBench writes `{run_name}_eval_results.json` to vbench_out. Each
        # dim entry is `[aggregate_score, [per_video_results, ...]]`.
        results_path = vbench_out / f"{run_name}_eval_results.json"
        results = msgspec.json.decode(results_path.read_bytes())
        per_dim_scores = self._extract_per_dim_scores(results)
        mean_score = float(np.mean(per_dim_scores))
        return mean_score, n_repeats


def _run_subprocess_with_log(
    cmd: list[str],
    log_path: Path,
    *,
    timeout_s: int,
    label: str,
    cwd: Path | None = None,
) -> None:
    """Run *cmd*, capture stdout+stderr to *log_path*, raise on timeout or non-zero exit."""
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd) if cwd is not None else None,
        )
    except subprocess.TimeoutExpired as e:
        partial = (
            e.stdout
            if isinstance(e.stdout, str)
            else (e.stdout or b"").decode("utf-8", errors="replace")
        )
        log_path.write_text(partial)
        raise RuntimeError(
            f"{label} subprocess timed out after {timeout_s}s; "
            f"see {log_path} for partial output."
        ) from e
    log_path.write_text(completed.stdout or "")
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-50:])
        raise RuntimeError(
            f"{label} subprocess exited with code {completed.returncode}; "
            f"full log at {log_path}. Last 50 lines:\n{tail}"
        )


_DEFAULT_SWE_BENCH_PROJECT_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "10_SWEBench_Example"
    / "accuracy"
)
_SWE_BENCH_PROJECT_PATH_ENV = "SWE_BENCH_PROJECT_PATH"
_DEFAULT_SWE_BENCH_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "10_SWEBench_Example"
    / "swebench_template.yaml"
)


class SWEBenchScorer(Scorer, scorer_id="swe_bench_scorer"):
    """SWE-bench accuracy scorer using mini-swe-agent.

    Invokes mini-extra and swebench via ``uv run --project <swe_bench_project_path>``
    so the parent process never imports them directly. Run ``uv sync`` in the subproject
    directory once before use.
    """

    REQUIRES_EXTRACTOR: ClassVar[bool] = False
    SKIP_ENDPOINT_PHASE: ClassVar[bool] = True
    DEFAULT_SUBPROCESS_TIMEOUT_S: ClassVar[int] = 8 * 60 * 60

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = "instance_id",
        swe_bench_project_path: str | os.PathLike | None = None,
        swebench_config_template: str | os.PathLike | None = None,
        subset: str = "verified",
        split: str = "test",
        num_instances: int = 100,
        workers: int = 10,
        max_eval_workers: int = 10,
        subprocess_timeout_s: int | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column,
        )
        self.report_dir = self.report_dir.resolve()
        self.swe_bench_project_path = self._resolve_project_path(swe_bench_project_path)
        self.swebench_config_template = (
            Path(swebench_config_template)
            if swebench_config_template is not None
            else _DEFAULT_SWE_BENCH_TEMPLATE
        )
        if subset not in _SWE_BENCH_HF_MAP:
            raise ValueError(
                f"Unknown SWE-bench subset {subset!r}; "
                f"choose from: {list(_SWE_BENCH_HF_MAP)}"
            )
        self.subset = subset
        self.split = split
        self.num_instances = num_instances
        self.workers = workers
        self.max_eval_workers = max_eval_workers
        self.subprocess_timeout_s = (
            subprocess_timeout_s
            if subprocess_timeout_s is not None
            else self.DEFAULT_SUBPROCESS_TIMEOUT_S
        )

        if not self.swebench_config_template.exists():
            raise FileNotFoundError(
                f"swebench template not found: {self.swebench_config_template}. "
                f"Pass swebench_config_template= in accuracy_config.extras."
            )
        with self.swebench_config_template.open() as _f:
            _tmpl = yaml.safe_load(_f)
        if not isinstance((_tmpl or {}).get("model", {}).get("model_kwargs"), dict):
            raise ValueError(
                f"swebench template {self.swebench_config_template} must have a "
                "'model.model_kwargs' dict; check the template structure."
            )
        pyproject = self.swe_bench_project_path / "pyproject.toml"
        if not pyproject.exists():
            raise FileNotFoundError(
                f"SWE-bench subproject not found at {self.swe_bench_project_path}. "
                f"Set ${_SWE_BENCH_PROJECT_PATH_ENV} to the subproject path, "
                f"then run: cd {self.swe_bench_project_path} && uv sync"
            )

    @staticmethod
    def _resolve_project_path(
        explicit: str | os.PathLike | None,
    ) -> Path:
        """Lookup order: explicit ctor arg → ``$SWE_BENCH_PROJECT_PATH`` env var → in-repo default."""
        if explicit is not None:
            return Path(explicit)
        from_env = os.environ.get(_SWE_BENCH_PROJECT_PATH_ENV)
        if from_env:
            return Path(from_env)
        return Path(_DEFAULT_SWE_BENCH_PROJECT_PATH)

    @classmethod
    def external_sample_count(cls, extras: dict[str, Any]) -> int | None:
        try:
            return int(extras["num_instances"])
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def preflight(cls, extras: dict[str, Any]) -> None:
        """Check uv, mini-extra, swebench, and Docker before the benchmark starts."""
        swe_bench_project_path = cls._resolve_project_path(
            extras.get("swe_bench_project_path")
        )

        if shutil.which("uv") is None:
            raise SetupError(
                "uv is not on PATH; install it with: "
                "curl -LsSf https://astral.sh/uv/install.sh | sh"
            )

        result = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(swe_bench_project_path),
                "mini-extra",
                "--help",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if result.returncode != 0:
            raise SetupError(
                f"mini-extra is not available in the SWE-bench subproject at "
                f"{swe_bench_project_path}. Run: cd {swe_bench_project_path} && uv sync"
            )

        swebench_result = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(swe_bench_project_path),
                "python",
                "-c",
                "import swebench",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if swebench_result.returncode != 0:
            raise SetupError(
                f"swebench is not available in the SWE-bench subproject at "
                f"{swe_bench_project_path}. Run: cd {swe_bench_project_path} && uv sync"
            )

        docker_result = subprocess.run(
            ["docker", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        if docker_result.returncode != 0:
            raise SetupError(
                "Docker daemon is not running or docker is not on PATH. "
                "Start Docker and retry."
            )

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "SWEBenchScorer uses subprocess evaluation; call score() instead."
        )

    def _patch_config(self, output_dir: Path, benchmark_config_dict: dict) -> Path:
        """Load template YAML, patch model fields from benchmark config, write to output_dir."""
        with self.swebench_config_template.open() as f:
            cfg = yaml.safe_load(f)

        model_params = benchmark_config_dict.get("model_params", {})
        endpoints = benchmark_config_dict.get("endpoint_config", {}).get(
            "endpoints", []
        )

        cfg["model"]["model_name"] = model_params["name"]
        if endpoints:
            base = endpoints[0].rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            cfg["model"]["model_kwargs"]["api_base"] = base + "/v1"
        else:
            cfg["model"]["model_kwargs"]["api_base"] = ""

        for field in (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            val = model_params.get(field)
            if val is not None:
                cfg["model"]["model_kwargs"][field] = val
            else:
                cfg["model"]["model_kwargs"].pop(field, None)

        chat_tmpl = model_params.get("chat_template_kwargs")
        if chat_tmpl is not None:
            cfg["model"]["model_kwargs"]["chat_template_kwargs"] = chat_tmpl
        else:
            cfg["model"]["model_kwargs"].pop("chat_template_kwargs", None)

        patched_path = output_dir / "swebench_patched.yaml"
        with patched_path.open("w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        return patched_path

    def _run_subprocess(self, cmd: list[str], log_path: Path, cwd: Path) -> None:
        """Run a command inside the accuracy subproject via ``uv run --project``."""
        full_cmd = [
            "uv",
            "run",
            "--project",
            str(self.swe_bench_project_path),
        ] + cmd
        _run_subprocess_with_log(
            full_cmd,
            log_path,
            timeout_s=self.subprocess_timeout_s,
            label="SWE-bench",
            cwd=cwd,
        )

    def score(self) -> tuple[float | None, int]:
        """Run mini-swe-agent + swebench evaluation. Returns (resolved_rate, 1)."""
        config_path = self.report_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {config_path}. "
                "SWEBenchScorer.score() must be called from within a benchmark run "
                "that has already written its config, or the path must be pre-populated."
            )
        with config_path.open() as f:
            benchmark_cfg = yaml.safe_load(f)

        model_name: str = benchmark_cfg["model_params"]["name"]
        assert self.dataset.dataframe is not None

        n_rows = len(self.dataset.dataframe)
        slice_str = f"0:{min(self.num_instances, n_rows)}"

        output_dir = self.report_dir / "swe_bench_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        patched_config = self._patch_config(output_dir, benchmark_cfg)

        agent_cmd = [
            "mini-extra",
            "swebench",
            "--model",
            model_name,
            "--config",
            str(patched_config),
            "--subset",
            self.subset,
            "--split",
            self.split,
            "--slice",
            slice_str,
            "--workers",
            str(self.workers),
            "--output",
            str(output_dir),
        ]
        logger.info("Running mini-swe-agent: %s", " ".join(agent_cmd))
        self._run_subprocess(
            agent_cmd,
            self.report_dir / "swe_bench_agent.log",
            cwd=output_dir,
        )

        preds_path = output_dir / "preds.json"
        if not preds_path.exists():
            logger.error(
                "preds.json not found after mini-swe-agent run; returning None score"
            )
            return None, 1

        hf_dataset_name = _SWE_BENCH_HF_MAP[self.subset]
        run_id = f"endpoints_{uuid.uuid4().hex[:8]}"
        eval_cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            hf_dataset_name,
            "--split",
            self.split,
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            str(self.max_eval_workers),
            "--run_id",
            run_id,
        ]
        logger.info("Running swebench evaluation: %s", " ".join(eval_cmd))
        self._run_subprocess(
            eval_cmd,
            self.report_dir / "swe_bench_eval.log",
            cwd=output_dir,
        )

        safe_model = model_name.replace("/", "__")
        result_path = output_dir / f"{safe_model}.{run_id}.json"
        if not result_path.exists():
            candidates = list(output_dir.glob(f"*{run_id}*.json"))
            if not candidates:
                logger.error(
                    "SWE-bench result file not found (run_id=%s); returning None",
                    run_id,
                )
                return None, 1
            result_path = candidates[0]

        shutil.copy2(result_path, self.report_dir / "swe_bench_results.json")

        result = msgspec.json.decode(result_path.read_bytes(), type=dict)
        submitted = result.get("submitted_instances", 0)
        resolved = result.get("resolved_instances", 0)
        if submitted == 0:
            logger.warning("SWE-bench: submitted_instances=0; returning None score")
            return None, 1

        resolved_rate = resolved / submitted
        logger.info(
            "SWE-bench: resolved %d / %d submitted (%.1f%%)",
            resolved,
            submitted,
            resolved_rate * 100,
        )
        return resolved_rate, 1
