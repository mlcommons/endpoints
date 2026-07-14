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
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urljoin, urlparse

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
from ..dataset_manager.predefined.swe_bench import SWEBench
from ..exceptions import SetupError
from .accuracy_results import build_breakdown
from .extractor import (
    Extractor,
    PythonCodeExtractor,
)

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
    def dataset_loader_kwargs(cls, extras: dict[str, Any]) -> dict[str, Any]:
        return {}

    @classmethod
    def external_sample_count(cls, extras: dict[str, Any]) -> int | None:
        return None

    @classmethod
    def preflight(
        cls, extras: dict[str, Any], *, loaded_sample_count: int | None = None
    ) -> None:
        return None

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
        self.sample_index_map = self._load_sample_index_map()

        # Whether the most recent score() covered every issued sample. Scorers
        # that can return a partial headline number (e.g. LegacyMLPerfDeepSeekR1Scorer when
        # the lcb-service container is unreachable) set this False so callers
        # can distinguish a partial result from a complete one. Default True.
        self.complete: bool = True

    def _load_sample_index_map(self):
        sample_index_map_path = self.report_dir / "sample_idx_map.json"
        if not sample_index_map_path.exists():
            raise FileNotFoundError(
                f"Sample index map file not found at {sample_index_map_path}"
            )

        with sample_index_map_path.open("r") as f:
            d = msgspec.json.decode(f.read())
            return d[self.dataset_name]  # Implicitly raises KeyError

    def _iter_complete(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(sample_uuid, data)`` for each COMPLETE event in events.jsonl.

        The single events reader shared by ``get_raw_outputs`` and any scorer
        that needs the structured event data (e.g. BFCL's ``tool_calls``). The
        EventLoggerService writes EventRecord objects serialized via msgspec.
        """
        events_log_path = self.report_dir / "events.jsonl"
        if not events_log_path.exists():
            raise FileNotFoundError(f"Events log file not found at {events_log_path}")

        decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
        with events_log_path.open("r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                record = decoder.decode(stripped)
                if record.event_type == SampleEventType.COMPLETE:
                    yield record.sample_uuid, record.data

    def get_raw_outputs(self, wanted_uuids: set[str] | None = None) -> pd.DataFrame:
        """uuid -> the text the model generated (``str(TextModelOutput)``).

        Identical for every scorer: the actual model output before any
        scoring-specific transform. Used for OSL and response accounting.
        ``wanted_uuids`` bounds the frame to a target population (one accuracy
        dataset's issued samples) so finalize need not hold the whole run's
        response-text corpus.
        """
        rows = [
            {"sample_uuid": u, "output": str(d) if d is not None else ""}
            for u, d in self._iter_complete()
            if wanted_uuids is None or u in wanted_uuids
        ]
        # Fixed columns so an all-filtered (empty) frame still exposes
        # "sample_uuid"/"output". Callers index those columns; a bare
        # pd.DataFrame([]) is column-less and would KeyError, dropping response
        # accounting for an all-missing phase — the exact masking the response
        # counts exist to prevent.
        return pd.DataFrame(rows, columns=["sample_uuid", "output"])

    def get_scoring_outputs(self) -> pd.DataFrame:
        """uuid -> the text to score. Base returns the raw model output; scorers
        that must transform before scoring (e.g. BFCL serializes ``tool_calls``)
        override this.
        """
        return self.get_raw_outputs()

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
        df = self.get_scoring_outputs()

        # Outputs are for all samples, not just the target dataset
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]

        # Denominator is the number of samples *issued* for this dataset
        # (``sample_index_map``, written at issue time), not the number that
        # produced a COMPLETE event. Samples lost to a drain-timeout or crash are
        # absent from ``df``; counting them as failures — dividing the summed
        # per-sample scores by the issued total — keeps a partial run from
        # reporting an accuracy inflated over only the surviving subset.
        issued = len(self.sample_index_map)
        if issued == 0:
            return None, 0
        n_repeats = issued // self.dataset.num_samples()
        if df.empty:
            # Nothing completed for this dataset: every issued sample failed.
            # 0 correct / issued == 0.0, never ``mean([]) == NaN`` (invalid JSON).
            return 0.0, n_repeats

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

        scores = [
            self.score_single_sample(empirical[i], ground_truths[i])
            for i in range(len(empirical))
        ]

        # ``float(...)`` yields a native Python float, not a numpy scalar, so the
        # score serializes cleanly into accuracy_results.json.
        mean = float(np.sum(scores)) / issued
        return (mean if np.isfinite(mean) else None), n_repeats

    def score_breakdown(self) -> dict[str, Any] | None:
        """Optional structured detail accompanying the scalar ``score()``.

        Most scorers report only the scalar mean from ``score()``. Scorers with a
        multi-metric result (e.g. per-subset / per-category accuracy) cache that
        breakdown and return it here, so ``accuracy_results.json``, compliance,
        plotting, and publishing read a typed dict without ``score()`` widening
        its scalar return contract. Returns ``None`` when there is no extra detail.
        """
        return None


def _exact_match(value: str, ground_truth: str) -> float:
    """pass@1 / exact-match: ``1.0`` if ``value == ground_truth`` else ``0.0``.

    Used by :class:`PassAt1Scorer` for per-sample match semantics.
    """
    return 1.0 if value == ground_truth else 0.0


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
        return _exact_match(value, ground_truth)


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
        df = self.get_scoring_outputs()

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


def _uv_subproject_env(project_path: os.PathLike | str) -> dict[str, str]:
    """Env for ``uv run --project <project_path>`` pinned to the subproject's OWN
    ``.venv``.

    Without this, an inherited ``UV_PROJECT_ENVIRONMENT`` (the dev image sets it
    to ``/opt/venv``) redirects ``uv`` to the parent environment instead of the
    isolated subproject venv - defeating the whole out-of-process isolation.
    Mirrors how ``setup_eval.sh`` provisioned it (``UV_PROJECT_ENVIRONMENT=$(pwd)/.venv``).
    """
    return {**os.environ, "UV_PROJECT_ENVIRONMENT": str(Path(project_path) / ".venv")}


def _lcb_ws_evaluate(
    url: str, codes_dict: dict[str, list[str]], timeout_sec: int
) -> dict | None:
    """Evaluate extracted code via the lcb-service WebSocket (synchronous).

    Sends ``{codes_dict, timeout_sec}`` and consumes progress frames until a
    terminal ``completed`` (returns ``result`` = ``{total_samples, results}``)
    or ``error``. Returns None on any failure so callers can fall back. Kept as
    a module function so both LiveCodeBenchScorer and LegacyMLPerfDeepSeekR1Scorer (which
    grades its livecodebench subset out-of-band) share one client.
    """
    if websocket is None:
        logger.warning(
            "websocket-client not installed; cannot reach lcb-service. "
            "Install with: pip install websocket-client"
        )
        return None
    try:
        ws = websocket.create_connection(
            url, timeout=7200, ping_interval=30, ping_timeout=10
        )
    except (OSError, websocket.WebSocketException) as e:
        logger.warning("lcb-service WebSocket connect failed (%s): %s", url, e)
        return None
    total = sum(len(c) for c in codes_dict.values())
    pbar = tqdm(total=total, desc="LCB Evaluation", unit="sample")
    try:
        ws.send(
            msgspec.json.encode(
                {"codes_dict": codes_dict, "timeout_sec": timeout_sec}
            ).decode("utf-8")
        )
        while True:
            message = ws.recv()
            if not message:
                return None
            data = msgspec.json.decode(message)
            status = data.get("status")
            if status == "progress":
                pbar.n = data.get("completed_samples", 0)
                pbar.refresh()
            elif status == "completed":
                pbar.n = total
                pbar.refresh()
                return data.get("result")
            elif status == "error":
                logger.error("lcb-service evaluation error: %s", data.get("error"))
                return None
    except Exception as e:  # noqa: BLE001 - network/protocol failure -> fall back
        logger.warning("lcb-service WebSocket evaluation failed: %s", e)
        return None
    finally:
        pbar.close()
        try:
            ws.close()
        except Exception:  # noqa: BLE001 - ignore close errors
            pass


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
        df = self.get_scoring_outputs()

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

            # Attempt WebSocket evaluation (synchronous) via the shared client.
            result = _lcb_ws_evaluate(
                self.lcb_websocket_url, dict(codes_dict), self.timeout
            )

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
        # F1 = 2*(5/6)*(5/6) / (5/6+5/6) = 5/6 ~ 0.833

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
        df = self.get_scoring_outputs()

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

# Filenames in `vbench_standard` mode key on the prompt verbatim - VBench looks
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
    `staged_dir`; VBench's prompt lookup tolerates the same `/`->`_`
    replacement applied here.

    The scorer reads each sample's video path from response_output (the
    VideoGenAdapter mirrors `video_path` into `TextModelOutput.output`)
    and the prompt from `dataset.dataframe[ground_truth_column]` - the
    prompt is the VBench input, not a comparison target, so callers should
    set `ground_truth_column: prompt` in `accuracy_config`.

    Returns `(None, n_repeats)` when no successful video was produced or
    when scoring fails to yield a usable per-dimension number - matching
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

        Lookup order: explicit ctor arg -> ``$VBENCH_PROJECT_PATH`` env var ->
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
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.subprocess_timeout_s,
                env=_uv_subproject_env(self.vbench_project_path),
            )
        except subprocess.TimeoutExpired as e:
            partial = (
                e.stdout
                if isinstance(e.stdout, str)
                else (e.stdout or b"").decode("utf-8", errors="replace")
            )
            log_path.write_text(partial)
            raise RuntimeError(
                f"VBench subprocess timed out after {self.subprocess_timeout_s}s; "
                f"see {log_path} for partial output."
            ) from e

        log_path.write_text(completed.stdout or "")
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout or "").splitlines()[-50:])
            raise RuntimeError(
                f"VBench subprocess exited with code {completed.returncode}; "
                f"full log at {log_path}. Last 50 lines:\n{tail}"
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
        df = self.get_scoring_outputs()
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]
        # Drop failed queries: Scorer.get_raw_outputs() emits "" when record.data
        # is None (workers set response_output=None on error). Passing "" to
        # _stage_videos would Path("").resolve() -> cwd and symlink the repo
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


class SWEBenchScorer(Scorer, scorer_id="swe_bench_scorer"):
    """SWE-bench accuracy scorer backed by a remote SWE-bench service."""

    REQUIRES_EXTRACTOR: ClassVar[bool] = False
    SKIP_ENDPOINT_PHASE: ClassVar[bool] = True
    DEFAULT_SUBSET: ClassVar[str] = "verified"
    DEFAULT_SPLIT: ClassVar[str] = "test"
    DEFAULT_NUM_INSTANCES: ClassVar[int] = 100
    DEFAULT_WORKERS: ClassVar[int] = 10
    DEFAULT_MAX_EVAL_WORKERS: ClassVar[int] = 10
    DEFAULT_SERVICE_TIMEOUT_S: ClassVar[int] = 24 * 60 * 60
    DEFAULT_POLL_INTERVAL_S: ClassVar[float] = 5.0
    SERVICE_API_VERSION: ClassVar[str] = "v1"
    REQUIRED_SERVICE_CAPABILITIES: ClassVar[set[str]] = {
        "swebench.run",
        "swebench.cancel",
        "artifacts.download",
    }
    SAFE_ARTIFACT_NAMES: ClassVar[set[str]] = {
        "preds.json",
        "swe_bench_agent.log",
        "swe_bench_eval.log",
        "swe_bench_results.json",
        "status.json",
    }
    TOOLCALL_PATCH_EXTRA: ClassVar[str] = "enable_swebench_toolcall_patch"
    SERVICE_TEMPLATES: ClassVar[set[str]] = {"default", "qwen_tools"}

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = "instance_id",
        swebench_service_url: str | None = None,
        swebench_service_auth_token: str | None = None,
        subset: str = "verified",
        split: str = "test",
        num_instances: int = 100,
        workers: int = 10,
        max_eval_workers: int = 10,
        enable_swebench_toolcall_patch: bool = False,
        swebench_template: str | None = None,
        service_timeout_s: int | None = None,
        poll_interval_s: float | None = None,
    ):
        ground_truth_column = ground_truth_column or "instance_id"
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column,
        )
        self.report_dir = self.report_dir.resolve()
        options = self._resolve_options(
            {
                "swebench_service_url": swebench_service_url,
                "swebench_service_auth_token": swebench_service_auth_token,
                "subset": subset,
                "split": split,
                "num_instances": num_instances,
                "workers": workers,
                "max_eval_workers": max_eval_workers,
                self.TOOLCALL_PATCH_EXTRA: enable_swebench_toolcall_patch,
                "swebench_template": swebench_template,
                "service_timeout_s": service_timeout_s,
                "poll_interval_s": poll_interval_s,
            }
        )
        self.swebench_service_url = options["swebench_service_url"]
        self.swebench_service_auth_token = options["swebench_service_auth_token"]
        self.subset = options["subset"]
        self.split = options["split"]
        self.num_instances = options["num_instances"]
        self.workers = options["workers"]
        self.max_eval_workers = options["max_eval_workers"]
        self.enable_swebench_toolcall_patch = options[self.TOOLCALL_PATCH_EXTRA]
        self.swebench_template = options["swebench_template"]
        self.service_timeout_s = options["service_timeout_s"]
        self.poll_interval_s = options["poll_interval_s"]

    @classmethod
    def _normalize_service_url(cls, value: Any) -> str:
        if value is None or str(value).strip() == "":
            raise SetupError(
                "accuracy_config.extras.swebench_service_url is required for "
                "swe_bench_scorer. Start the SWE-bench service and pass its URL."
            )
        return str(value).strip().rstrip("/") + "/"

    @classmethod
    def _http_json(
        cls,
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
        auth_token: str | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if payload is not None:
            data = msgspec.json.encode(payload)
            headers["Content-Type"] = "application/json"
        req = urllib_request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib_request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SetupError(
                f"SWE-bench service request failed: {url} returned HTTP "
                f"{exc.code}: {detail}"
            ) from exc
        except urllib_error.URLError as exc:
            raise SetupError(
                f"SWE-bench service is unreachable at {url}: {exc.reason}"
            ) from exc
        try:
            decoded = msgspec.json.decode(body, type=dict)
        except msgspec.DecodeError as exc:
            raise SetupError(
                f"SWE-bench service returned invalid JSON from {url}"
            ) from exc
        return decoded

    @classmethod
    def _check_health(
        cls, service_url: str, auth_token: str | None = None
    ) -> dict[str, Any]:
        health = cls._http_json(
            urljoin(service_url, "health"),
            timeout_s=10.0,
            auth_token=auth_token,
        )
        api_version = health.get("api_version")
        if api_version != cls.SERVICE_API_VERSION:
            raise SetupError(
                "SWE-bench service API version mismatch: expected "
                f"{cls.SERVICE_API_VERSION!r}, got {api_version!r}"
            )
        capabilities = set(health.get("capabilities") or [])
        missing = cls.REQUIRED_SERVICE_CAPABILITIES - capabilities
        if missing:
            raise SetupError(
                "SWE-bench service is missing required capabilities: "
                + ", ".join(sorted(missing))
            )
        return health

    @classmethod
    def _download_artifact(
        cls,
        service_url: str,
        artifact: dict[str, Any],
        report_dir: Path,
        run_id: str,
        auth_token: str | None = None,
    ) -> None:
        name = str(artifact.get("name") or "")
        href = str(artifact.get("url") or "")
        if name not in cls.SAFE_ARTIFACT_NAMES or not href:
            return
        parsed = urlparse(href)
        expected_path = f"/v1/runs/{run_id}/artifacts/{name}"
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.params
            or parsed.query
            or parsed.fragment
            or parsed.path != expected_path
        ):
            logger.warning("Ignoring unsafe SWE-bench artifact URL for %s", name)
            return
        target = report_dir / name
        url = urljoin(service_url, href.lstrip("/"))
        req = urllib_request.Request(
            url, headers={"Accept": "application/octet-stream"}
        )
        if auth_token:
            req.add_header("Authorization", f"Bearer {auth_token}")
        try:
            with urllib_request.urlopen(req, timeout=60.0) as resp:
                target.write_bytes(resp.read())
        except Exception:
            logger.warning(
                "Could not download SWE-bench artifact %s", name, exc_info=True
            )

    @classmethod
    def _download_artifacts(
        cls,
        service_url: str,
        status: dict[str, Any],
        report_dir: Path,
        auth_token: str | None = None,
    ) -> None:
        run_id = str(status.get("run_id") or "")
        if not run_id:
            return
        artifacts = status.get("artifacts") or []
        if isinstance(artifacts, dict):
            artifacts = [
                {"name": name, "url": url}
                for name, url in artifacts.items()
                if isinstance(name, str)
            ]
        if not isinstance(artifacts, list):
            return
        for artifact in artifacts:
            if isinstance(artifact, dict):
                cls._download_artifact(
                    service_url, artifact, report_dir, run_id, auth_token
                )

    @classmethod
    def _cancel_service_run(
        cls, service_url: str, run_id: str, auth_token: str | None = None
    ) -> None:
        try:
            cls._http_json(
                urljoin(service_url, f"v1/runs/{run_id}/cancel"),
                method="POST",
                timeout_s=10.0,
                auth_token=auth_token,
            )
        except SetupError:
            logger.warning("Could not cancel SWE-bench service run %s", run_id)

    @staticmethod
    def _write_service_status(report_dir: Path, status: dict[str, Any]) -> None:
        (report_dir / "swe_bench_service_status.json").write_bytes(
            msgspec.json.encode(status)
        )

    @staticmethod
    def _progress_int(status: dict[str, Any], key: str) -> int | None:
        value = status.get(key)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @classmethod
    def _update_progress_bars(
        cls, status: dict[str, Any], state: dict[str, Any]
    ) -> None:
        phase = str(status.get("phase") or "")
        agent_total = cls._progress_int(status, "agent_total")
        agent_completed = cls._progress_int(status, "agent_completed")
        eval_total = cls._progress_int(status, "eval_total")
        eval_completed = cls._progress_int(status, "eval_completed")
        if (
            not phase
            and agent_total is None
            and agent_completed is None
            and eval_total is None
            and eval_completed is None
        ):
            return

        agent_bar = state.get("agent_bar")
        if agent_bar is None and agent_total is not None and agent_total > 0:
            agent_bar = tqdm(
                total=agent_total,
                desc="SWE-bench agent",
                unit="inst",
            )
            state["agent_bar"] = agent_bar
            state["agent_completed"] = 0
        if agent_bar is not None:
            if agent_total is not None and agent_total > (agent_bar.total or 0):
                agent_bar.total = agent_total
                agent_bar.refresh()
            if agent_completed is not None:
                previous = int(state.get("agent_completed") or 0)
                current = max(previous, agent_completed)
                if current > previous:
                    agent_bar.update(current - previous)
                    state["agent_completed"] = current

        eval_bar = state.get("eval_bar")
        should_open_eval = (
            phase in {"eval", "succeeded"} and eval_total is not None and eval_total > 0
        )
        if eval_bar is None and should_open_eval:
            eval_bar = tqdm(
                total=eval_total,
                desc="SWE-bench eval",
                unit="inst",
            )
            state["eval_bar"] = eval_bar
            state["eval_completed"] = 0
        if eval_bar is not None:
            if eval_total is not None and eval_total > (eval_bar.total or 0):
                eval_bar.total = eval_total
                eval_bar.refresh()
            if eval_completed is not None:
                previous = int(state.get("eval_completed") or 0)
                current = max(previous, eval_completed)
                if current > previous:
                    eval_bar.update(current - previous)
                    state["eval_completed"] = current

    @staticmethod
    def _close_progress_bars(state: dict[str, Any]) -> None:
        for key in ("eval_bar", "agent_bar"):
            bar = state.get(key)
            if bar is not None:
                bar.close()

    @classmethod
    def _get_extra_int(
        cls, extras: dict[str, Any], key: str, *, default: int, min_value: int = 0
    ) -> int:
        value = extras.get(key)
        if value is None:
            value = default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise SetupError(
                f"accuracy_config.extras.{key} must be an integer; got {value!r}"
            ) from exc
        if parsed < min_value:
            raise SetupError(
                f"accuracy_config.extras.{key} must be >= {min_value}; got {parsed}"
            )
        return parsed

    @staticmethod
    def _validate_max_eval_workers(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"max_eval_workers must be an integer; got {value!r}"
            ) from exc
        if parsed < 1:
            raise ValueError(f"max_eval_workers must be >= 1; got {parsed}")
        return parsed

    @classmethod
    def _get_extra_bool(
        cls, extras: dict[str, Any], key: str, *, default: bool = False
    ) -> bool:
        value = extras.get(key)
        if value is None:
            value = default
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise SetupError(
            f"accuracy_config.extras.{key} must be a boolean; got {value!r}"
        )

    @classmethod
    def _get_extra_float(
        cls, extras: dict[str, Any], key: str, *, default: float, min_value: float = 0
    ) -> float:
        value = extras.get(key)
        if value is None:
            value = default
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise SetupError(
                f"accuracy_config.extras.{key} must be numeric; got {value!r}"
            ) from exc
        if parsed <= min_value:
            raise SetupError(
                f"accuracy_config.extras.{key} must be > {min_value:g}; got {parsed}"
            )
        return parsed

    @classmethod
    def _resolve_dataset_options(cls, extras: dict[str, Any]) -> dict[str, str]:
        subset = str(extras.get("subset", cls.DEFAULT_SUBSET))
        SWEBench.hf_dataset_name(subset)
        return {
            "subset": subset,
            "split": str(extras.get("split", cls.DEFAULT_SPLIT)),
        }

    @classmethod
    def _resolve_service_template(cls, extras: dict[str, Any]) -> str:
        raw = extras.get("swebench_template")
        if raw is None:
            raw = (
                "qwen_tools"
                if cls._get_extra_bool(extras, cls.TOOLCALL_PATCH_EXTRA)
                else "default"
            )
        template = str(raw)
        if template not in cls.SERVICE_TEMPLATES:
            raise SetupError(
                "accuracy_config.extras.swebench_template must be one of "
                f"{sorted(cls.SERVICE_TEMPLATES)}; got {template!r}"
            )
        return template

    @classmethod
    def _resolve_options(cls, extras: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = cls._resolve_dataset_options(extras)
        options["swebench_service_url"] = cls._normalize_service_url(
            extras.get("swebench_service_url")
        )
        auth_token = extras.get("swebench_service_auth_token")
        options["swebench_service_auth_token"] = (
            str(auth_token) if auth_token not in (None, "") else None
        )
        options["num_instances"] = cls._get_extra_int(
            extras,
            "num_instances",
            default=cls.DEFAULT_NUM_INSTANCES,
            min_value=1,
        )
        options["workers"] = cls._get_extra_int(
            extras,
            "workers",
            default=cls.DEFAULT_WORKERS,
            min_value=1,
        )
        options["max_eval_workers"] = cls._get_extra_int(
            extras,
            "max_eval_workers",
            default=cls.DEFAULT_MAX_EVAL_WORKERS,
            min_value=1,
        )
        options[cls.TOOLCALL_PATCH_EXTRA] = cls._get_extra_bool(
            extras,
            cls.TOOLCALL_PATCH_EXTRA,
        )
        options["swebench_template"] = cls._resolve_service_template(extras)
        options["service_timeout_s"] = cls._get_extra_int(
            extras,
            "service_timeout_s",
            default=cls.DEFAULT_SERVICE_TIMEOUT_S,
            min_value=1,
        )
        options["poll_interval_s"] = cls._get_extra_float(
            extras,
            "poll_interval_s",
            default=cls.DEFAULT_POLL_INTERVAL_S,
            min_value=0,
        )
        return options

    @staticmethod
    def _generation_params(model_params: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "max_new_tokens",
            "chat_template_kwargs",
        )
        return {field: model_params[field] for field in fields if field in model_params}

    @classmethod
    def dataset_loader_kwargs(cls, extras: dict[str, Any]) -> dict[str, Any]:
        return cls._resolve_dataset_options(extras)

    @classmethod
    def external_sample_count(cls, extras: dict[str, Any]) -> int | None:
        raw = extras.get("num_instances", cls.DEFAULT_NUM_INSTANCES)
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def preflight(
        cls, extras: dict[str, Any], *, loaded_sample_count: int | None = None
    ) -> None:
        """Check the SWE-bench service before the benchmark starts."""
        try:
            options = cls._resolve_options(extras)
        except ValueError as exc:
            raise SetupError(str(exc)) from exc
        cls._check_health(
            options["swebench_service_url"],
            options["swebench_service_auth_token"],
        )

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "SWEBenchScorer uses service evaluation; call score() instead."
        )

    def score(self) -> tuple[float | None, int]:
        """Submit a SWE-bench service run. Returns (resolved_rate, 1)."""
        self.complete = True
        config_path = self.report_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {config_path}. "
                "SWEBenchScorer.score() must be called from within a benchmark run "
                "that has already written its config, or the path must be pre-populated."
            )
        with config_path.open() as f:
            benchmark_cfg = yaml.safe_load(f)
        if not isinstance(benchmark_cfg, dict):
            raise ValueError(
                f"benchmark config at {config_path} must be a YAML mapping"
            )

        model_params = benchmark_cfg.get("model_params") or {}
        model_name = model_params.get("name")
        if not model_name:
            raise ValueError(
                "model_params.name is required in the benchmark config but is missing or empty"
            )
        if self.dataset.dataframe is None:
            raise RuntimeError(
                "SWEBench dataset must be loaded before scoring; call dataset.load() first."
            )

        n_rows = len(self.dataset.dataframe)
        if self.num_instances > n_rows:
            logger.warning(
                "num_instances=%d exceeds dataset size %d; evaluating %d instances",
                self.num_instances,
                n_rows,
                n_rows,
            )
        total_instances = min(self.num_instances, n_rows)
        evaluated_instance_ids = [
            str(instance_id)
            for instance_id in self.dataset.dataframe.iloc[:total_instances][
                self.ground_truth_column
            ].tolist()
        ]
        if not evaluated_instance_ids:
            logger.warning("SWE-bench: no evaluated instances; returning None score")
            self.complete = False
            return None, 1

        endpoint_config = benchmark_cfg.get("endpoint_config") or {}
        endpoint_urls = endpoint_config.get("endpoints") or []
        if len(endpoint_urls) != 1:
            raise SetupError(
                "SWE-bench service mode supports exactly one endpoint URL; "
                f"got {len(endpoint_urls)}."
            )
        payload: dict[str, Any] = {
            "model_name": model_name,
            "endpoint_urls": endpoint_urls,
            "endpoint_api_key": endpoint_config.get("api_key"),
            "generation_params": self._generation_params(model_params),
            "subset": self.subset,
            "split": self.split,
            "num_instances": total_instances,
            "workers": self.workers,
            "max_eval_workers": self.max_eval_workers,
            "evaluated_instance_ids": evaluated_instance_ids,
            self.TOOLCALL_PATCH_EXTRA: self.enable_swebench_toolcall_patch,
            "template": self.swebench_template,
        }

        run_id = ""
        progress_state: dict[str, Any] = {}
        try:
            submitted = type(self)._http_json(
                urljoin(self.swebench_service_url, "v1/runs"),
                method="POST",
                payload=payload,
                timeout_s=30.0,
                auth_token=self.swebench_service_auth_token,
            )
            run_id = str(submitted.get("run_id") or "")
            if not run_id:
                raise SetupError("SWE-bench service did not return run_id")
            type(self)._update_progress_bars(submitted, progress_state)

            deadline = time.monotonic() + self.service_timeout_s
            status = submitted
            while status.get("status") not in {"succeeded", "failed", "cancelled"}:
                if time.monotonic() >= deadline:
                    raise SetupError(
                        f"Timed out waiting for SWE-bench service run {run_id}"
                    )
                time.sleep(self.poll_interval_s)
                status = type(self)._http_json(
                    urljoin(self.swebench_service_url, f"v1/runs/{run_id}"),
                    timeout_s=30.0,
                    auth_token=self.swebench_service_auth_token,
                )
                type(self)._update_progress_bars(status, progress_state)
        except (KeyboardInterrupt, SystemExit):
            if run_id:
                type(self)._cancel_service_run(
                    self.swebench_service_url,
                    run_id,
                    self.swebench_service_auth_token,
                )
            raise
        except SetupError:
            if run_id:
                type(self)._cancel_service_run(
                    self.swebench_service_url,
                    run_id,
                    self.swebench_service_auth_token,
                )
            logger.error("SWE-bench service run failed", exc_info=True)
            self.complete = False
            return None, 1
        finally:
            type(self)._close_progress_bars(progress_state)

        type(self)._write_service_status(self.report_dir, status)
        type(self)._download_artifacts(
            self.swebench_service_url,
            status,
            self.report_dir,
            self.swebench_service_auth_token,
        )
        if status.get("status") != "succeeded":
            logger.error(
                "SWE-bench service run %s ended with status %s",
                run_id,
                status.get("status"),
            )
            self.complete = False
            return None, 1

        result = status.get("result")
        result_path = self.report_dir / "swe_bench_results.json"
        if result is None and result_path.exists():
            try:
                result = msgspec.json.decode(result_path.read_bytes(), type=dict)
            except msgspec.DecodeError:
                self.complete = False
                return None, 1
        if not isinstance(result, dict):
            logger.error("SWE-bench service run %s did not return a result", run_id)
            self.complete = False
            return None, 1
        if not result_path.exists():
            result_path.write_bytes(msgspec.json.encode(result))

        submitted_count = result.get("submitted_instances") or 0
        resolved = result.get("resolved_instances") or 0
        if submitted_count == 0:
            logger.warning("SWE-bench: submitted_instances=0; returning None score")
            self.complete = False
            return None, 1

        denominator = len(evaluated_instance_ids)
        if denominator == 0:
            logger.warning(
                "SWE-bench: evaluated instance count is 0; returning None score"
            )
            self.complete = False
            return None, 1
        if submitted_count != denominator:
            logger.warning(
                "SWE-bench: service submitted %d / %d evaluated instances; "
                "marking score incomplete",
                submitted_count,
                denominator,
            )
            self.complete = False

        resolved_rate = resolved / denominator
        logger.info(
            "SWE-bench: resolved %d / %d evaluated (%.1f%%)",
            resolved,
            denominator,
            resolved_rate * 100,
        )
        return resolved_rate, 1


class LegacyMLPerfDeepSeekR1Scorer(Scorer, scorer_id="legacy_mlperf_deepseek_r1"):
    """MLPerf DeepSeek-R1 combined-subset accuracy scorer.

    The MLPerf DeepSeek-R1 accuracy dataset is an ensemble of five subsets
    (``aime``, ``math500``, ``gpqa``, ``mmlu_pro``, ``livecodebench``), each
    parsed and graded differently. The official MLCommons ``eval_accuracy.py``
    routes each sample by its ``dataset`` column, then reports an aggregate
    ``exact_match`` (mean per-sample 100/0) plus ``tokens_per_sample``.

    That evaluator pulls in pinned/heavy deps (``transformers`` plus the
    ``prm800k`` math grader and ``LiveCodeBench`` code executor submodules)
    that are incompatible with the parent benchmark env, so - exactly like
    ``VBenchScorer`` - it runs out-of-process via ``uv run --project`` against
    the isolated subproject at
    ``src/inference_endpoint/evaluation/legacy_mlperf_deepseek_r1/``
    (a uv subproject, excluded from the parent wheel). The parent process never
    imports the evaluator.

    This scorer builds the DataFrame the evaluator expects - ``model_output``
    (the full raw generation, including the ``<think>`` trace, taken verbatim
    from the COMPLETE event), ``ground_truth``, ``dataset`` (the subset id),
    and ``question`` - writes it to a temp parquet, and shells out to
    ``deepseek_eval_runner.py``. Output token lengths are computed inside the
    subproject with the DeepSeek tokenizer so ``tokens_per_sample`` matches the
    MLPerf token accounting.

    Returns ``(exact_match, n_repeats)`` where ``exact_match`` is on the same
    0-100 scale as the MLPerf golden accuracy (81.3582), or ``(None, n)`` if
    no successful output was produced or the subprocess fails to yield a
    usable number - matching the ``Scorer.score()`` contract.

    Reads the subset id from ``dataset.dataframe[subset_column]`` (default
    column ``dataset``) and the per-sample question from ``question_column``
    (default ``question``); both are passed through ``accuracy_config.extras``.
    """

    REQUIRES_EXTRACTOR: ClassVar[bool] = False
    DEFAULT_SUBPROCESS_TIMEOUT_S: ClassVar[int] = 4 * 60 * 60

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = "ground_truth",
        subset_column: str = "dataset",
        question_column: str = "question",
        tokenizer_path: str = "deepseek-ai/DeepSeek-R1",
        deepseek_eval_project_path: os.PathLike | None = None,
        uv_executable: str = "uv",
        subprocess_timeout_s: int | None = None,
        lcb_subset: str = "livecodebench",
        lcb_websocket_port: int | None = 13835,
        lcb_timeout: int = 60,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column,
        )
        self._breakdown: dict[str, Any] | None = None
        self.subset_column = subset_column
        self.question_column = question_column
        self.tokenizer_path = tokenizer_path
        self.uv_executable = uv_executable
        # LiveCodeBench executes untrusted code, which the in-process MLCommons
        # executor can't sandbox. When a port is set, the livecodebench subset
        # is graded out-of-band against the lcb-service WebSocket container
        # (ws://localhost:<port>/evaluate); the rest go through the subprocess.
        # If the socket is unreachable, livecodebench is left UNSCORED and the
        # run is marked incomplete (it is never graded in-process). With no port
        # configured, grading LCB in-process requires an explicit ALLOW_LCB_LOCAL=1.
        self.lcb_subset = lcb_subset
        self.lcb_timeout = lcb_timeout
        self.lcb_websocket_url = (
            f"ws://localhost:{lcb_websocket_port}/evaluate"
            if lcb_websocket_port is not None
            else None
        )
        self.project_path = self._resolve_project_path(deepseek_eval_project_path)
        self.subprocess_timeout_s = (
            subprocess_timeout_s
            if subprocess_timeout_s is not None
            else self.DEFAULT_SUBPROCESS_TIMEOUT_S
        )
        runner = self.project_path / "deepseek_eval_runner.py"
        if not runner.exists():
            raise FileNotFoundError(
                f"deepseek_eval_runner.py not found at {runner}. "
                f"Run `uv sync` and `bash setup_eval.sh` in the accuracy "
                f"subproject, or set $DEEPSEEK_EVAL_PROJECT_PATH to the "
                f"synced subproject path."
            )

    @staticmethod
    def _resolve_project_path(explicit: os.PathLike | None) -> Path:
        """Resolve the DeepSeek eval subproject path.

        Lookup order: explicit ctor arg -> ``$DEEPSEEK_EVAL_PROJECT_PATH`` env
        var -> editable-checkout fallback. The env var lets wheel-installed
        users point at a synced subproject without patching source.
        """
        if explicit is not None:
            return Path(explicit)
        from_env = os.environ.get("DEEPSEEK_EVAL_PROJECT_PATH")
        if from_env:
            return Path(from_env)
        return Path(__file__).resolve().parent / "legacy_mlperf_deepseek_r1"

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "DeepSeek-R1 scoring requires batch processing; call score() instead."
        )

    def _run_eval_subprocess(
        self, input_parquet: Path, out_json: Path, external_subsets: str = ""
    ) -> None:
        """Invoke deepseek_eval_runner.py via ``uv run --project <subproject>``.

        Captures stdout+stderr into ``report_dir/deepseek_eval_subprocess.log``
        and, on non-zero exit, raises with the tail of the captured log so the
        real failure (missing submodule, tokenizer download, code-exec error)
        isn't lost. ``external_subsets`` (comma-separated) are tokenized but not
        graded by the runner - the caller grades them out-of-band.
        """
        cmd = [
            self.uv_executable,
            "run",
            "--project",
            str(self.project_path),
            "python",
            str(self.project_path / "deepseek_eval_runner.py"),
            "--input",
            str(input_parquet),
            "--output",
            str(out_json),
            "--tokenizer",
            self.tokenizer_path,
        ]
        if external_subsets:
            cmd += ["--external-subsets", external_subsets]
        log_path = self.report_dir / "deepseek_eval_subprocess.log"
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.subprocess_timeout_s,
                env=_uv_subproject_env(self.project_path),
            )
        except subprocess.TimeoutExpired as e:
            partial = (
                e.stdout
                if isinstance(e.stdout, str)
                else (e.stdout or b"").decode("utf-8", errors="replace")
            )
            log_path.write_text(partial)
            raise RuntimeError(
                f"DeepSeek eval subprocess timed out after "
                f"{self.subprocess_timeout_s}s; see {log_path} for partial output."
            ) from e

        log_path.write_text(completed.stdout or "")
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout or "").splitlines()[-50:])
            raise RuntimeError(
                f"DeepSeek eval subprocess exited with code "
                f"{completed.returncode}; full log at {log_path}. "
                f"Last 50 lines:\n{tail}"
            )

    def _score_lcb_via_container(self, lcb_df: pd.DataFrame) -> tuple[int, int] | None:
        """Grade the livecodebench rows against the lcb-service WebSocket.

        Extracts the python block from each ``model_output`` and keys it by
        question id (the ``ground_truth``), then evaluates via the container.
        Returns ``(passed, total)`` or None if the service is unreachable (so
        score() can fall back to the in-process path).
        """
        if self.lcb_websocket_url is None:
            raise ValueError(
                "lcb_websocket_url must be configured to score LCB via container"
            )
        codes_dict: dict[str, list[str]] = defaultdict(list)
        for _, row in lcb_df.iterrows():
            code = PythonCodeExtractor.extract(
                str(row["model_output"]), default="# FAILED TO EXTRACT CODE"
            )
            codes_dict[str(row["ground_truth"])].append(
                code or "# FAILED TO EXTRACT CODE"
            )
        result = _lcb_ws_evaluate(
            self.lcb_websocket_url, dict(codes_dict), self.lcb_timeout
        )
        if result is None:
            return None
        total_samples = int(result.get("total_samples", 0))
        per_problem = result.get("results", {})
        if not per_problem and total_samples:
            logger.error(
                "lcb-service evaluated %d samples but returned an empty summary",
                total_samples,
            )
            return None
        passed = sum(sum(code_passed) for code_passed in per_problem.values())
        return int(passed), total_samples

    def score(self) -> tuple[float | None, int]:
        df = self.get_scoring_outputs()
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]

        n_total = len(df)
        num_samples = self.dataset.num_samples()
        n_repeats = n_total // num_samples if num_samples else 0

        # Failed queries log "" (Scorer.get_raw_outputs() emits "" when
        # record.data is None). They are graded as incorrect by the evaluator
        # but still count toward the denominator, so keep them in.
        if df.empty:
            logger.warning(
                "LegacyMLPerfDeepSeekR1Scorer: no outputs to score; returning None score."
            )
            self.complete = False
            return None, n_repeats

        df = df.apply(self.match_sample_index, axis=1)
        order = df["sample_index"].to_numpy().astype(int)

        ref = self.dataset.dataframe
        if ref is None:
            raise RuntimeError(f"Dataset {self.dataset} has no dataframe loaded")
        for col in (self.ground_truth_column, self.subset_column, self.question_column):
            if col not in ref.columns:
                raise ValueError(
                    f"Column {col!r} not found in dataset {self.dataset}; "
                    f"available: {list(ref.columns)}"
                )

        eval_df = pd.DataFrame(
            {
                "model_output": df["output"].astype(str).to_numpy(),
                "ground_truth": ref[self.ground_truth_column].to_numpy()[order],
                "dataset": ref[self.subset_column].to_numpy()[order],
                "question": ref[self.question_column].to_numpy()[order],
            }
        )

        scratch = self.report_dir / "deepseek_eval"
        scratch.mkdir(parents=True, exist_ok=True)
        input_parquet = scratch / f"{self.dataset_name}_outputs.parquet"
        out_json = scratch / f"{self.dataset_name}_results.json"
        eval_df.to_parquet(input_parquet, index=False)

        n_lcb = int((eval_df["dataset"].astype(str) == self.lcb_subset).sum())
        use_container = self.lcb_websocket_url is not None and n_lcb > 0

        if not use_container:
            # Refuse to grade untrusted livecodebench code in-process without the
            # sandboxed container, unless explicitly opted in. (n_lcb == 0 is
            # safe: no LCB rows means no untrusted code to execute.)
            if n_lcb > 0 and os.environ.get("ALLOW_LCB_LOCAL") != "1":
                raise RuntimeError(
                    "livecodebench rows present but no lcb-service container is "
                    "configured (set lcb_websocket_port). Refusing to execute "
                    "untrusted model-generated code in-process; configure the "
                    "container, or set ALLOW_LCB_LOCAL=1 to override."
                )
            # No LCB rows (or explicit opt-in): grade every subset in-process.
            self._run_eval_subprocess(input_parquet, out_json)
            results = msgspec.json.decode(out_json.read_bytes())
            exact_match = results.get("exact_match")
            if exact_match is None:
                logger.warning(
                    "LegacyMLPerfDeepSeekR1Scorer: subprocess produced no exact_match; "
                    "returning None score. See %s",
                    out_json,
                )
                self.complete = False
                return None, n_repeats
            # The runner reports complete=False if any subset failed to grade.
            self.complete = bool(results.get("complete", True))
            self._cache_breakdown(results)
            return float(exact_match), n_repeats

        # Grade the text subsets in the subprocess and the livecodebench subset
        # against the lcb-service container, then merge into one 5-subset number
        # so no follow-up scorer is needed.
        self._run_eval_subprocess(
            input_parquet, out_json, external_subsets=self.lcb_subset
        )
        results = msgspec.json.decode(out_json.read_bytes())
        per_dataset = results.get("per_dataset", {})

        # Aggregate every text subset the runner graded. Track whether any
        # failed so a partial run is never reported as a complete score.
        text_correct = 0
        text_n = 0
        text_complete = True
        for sub, d in per_dataset.items():
            if sub == self.lcb_subset:
                continue
            em = d.get("exact_match")
            n = int(d.get("num_samples", 0))
            if em is None:  # subset failed to grade (status != "ok")
                text_complete = False
                continue
            # `em` is a per-subset mean of strictly-binary (0/100) per-sample
            # scores, so round(em/100*n) recovers the exact integer correct
            # count. If a future MLCommons subset emits fractional/partial
            # credit, sum raw per-sample counts from the runner instead.
            text_correct += round(em / 100.0 * n)
            text_n += n

        lcb_scored = self._score_lcb_via_container(
            eval_df[eval_df["dataset"].astype(str) == self.lcb_subset]
        )
        # Preserve the runner's external LCB entry (it carries tokens_per_sample).
        lcb_entry = dict(per_dataset.get(self.lcb_subset, {}))
        if lcb_scored is None:
            # Container unreachable: leave livecodebench UNSCORED. Do NOT re-run
            # the in-process executor - it can't sandbox runaway model code and
            # needs a ~21 GB dataset load. Launch the lcb-service container (see
            # src/inference_endpoint/evaluation/livecodebench/README.md) and re-run.
            logger.warning(
                "LegacyMLPerfDeepSeekR1Scorer: lcb-service unreachable at %s; livecodebench "
                "left unscored (reporting %d text samples only, run marked "
                "incomplete). Launch the lcb-service container (see "
                "evaluation/livecodebench/README.md) and re-run to score LCB.",
                self.lcb_websocket_url,
                text_n,
            )
            lcb_passed = 0
            lcb_total = 0
            lcb_entry["exact_match"] = None
            lcb_entry["status"] = "unscored"
            lcb_ok = False
        else:
            lcb_passed, lcb_total = lcb_scored
            lcb_entry["exact_match"] = (
                100.0 * lcb_passed / lcb_total if lcb_total else None
            )
            lcb_entry["num_samples"] = lcb_total
            lcb_entry["status"] = "lcb-service"
            lcb_ok = lcb_total > 0

        total_n = text_n + lcb_total
        combined = 100.0 * (text_correct + lcb_passed) / total_n if total_n else None

        # The headline number is only valid if it covers every issued sample;
        # a failed text subset or a diverging LCB count silently shrinks total_n.
        expected_n = len(eval_df)
        complete = bool(
            combined is not None and text_complete and lcb_ok and total_n == expected_n
        )
        if combined is not None and lcb_ok and total_n != expected_n:
            logger.warning(
                "LegacyMLPerfDeepSeekR1Scorer: scored %d of %d samples (LCB count diverged "
                "from the issued rows); marking the result incomplete.",
                total_n,
                expected_n,
            )

        self.complete = complete
        per_dataset[self.lcb_subset] = lcb_entry
        results["per_dataset"] = per_dataset
        results["exact_match"] = combined
        results["evaluated_samples"] = total_n
        results["complete"] = complete
        out_json.write_bytes(msgspec.json.encode(results))
        logger.info(
            "LegacyMLPerfDeepSeekR1Scorer: combined exact_match=%s (text %d/%d + LCB %d/%d, complete=%s)",
            f"{combined:.4f}" if combined is not None else "None",
            text_correct,
            text_n,
            lcb_passed,
            lcb_total,
            complete,
        )

        if combined is None:
            return None, n_repeats
        self._cache_breakdown(results)
        return float(combined), n_repeats

    def _cache_breakdown(self, results: dict[str, Any]) -> None:
        """Cache a per-subset breakdown from the runner's per-subset results.

        Subset ``exact_match`` values are already on the 0-100 scale, so they map
        straight onto ``subset_scores``. Subsets that failed to grade
        (``exact_match is None``) are omitted. The headline accuracy is the entry's
        scalar ``score`` (this scorer's :meth:`score` return), so it is not
        duplicated in the block.
        """
        per_dataset = results.get("per_dataset") or {}
        subset_scores: dict[str, float] = {}
        for sub, d in per_dataset.items():
            if not isinstance(d, dict):
                continue
            em = d.get("exact_match")
            if em is not None:
                subset_scores[sub] = float(em)
        total_samples = int(
            results.get("evaluated_samples") or results.get("num_samples") or 0
        )
        self._breakdown = build_breakdown(
            subset_scores=subset_scores,
            total_samples=total_samples,
            complete=self.complete,
        )

    def score_breakdown(self) -> dict[str, Any] | None:
        """Per-subset accuracy breakdown cached by :meth:`score` (BFCL-shaped).

        ``None`` until :meth:`score` runs, or if scoring produced no usable
        result.
        """
        return self._breakdown
