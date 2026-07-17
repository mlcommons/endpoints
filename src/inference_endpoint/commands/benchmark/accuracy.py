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

"""Accuracy scoring for benchmark finalization.

Off the hot path: scores each accuracy dataset (and the optional inline
perf-scored entry) into one ``accuracy_scores`` list entry, computes per-phase
response accounting and output-token-length rollups from ``events.jsonl``, and
writes the ``accuracy/accuracy_results.json`` artifact.
"""

from __future__ import annotations

import json
import logging
import numbers
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec.json

from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    encode_lengths,
    load_reference_backend,
)
from inference_endpoint.config.schema import DatasetType
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.evaluation import Extractor
from inference_endpoint.evaluation.accuracy_results import average_accuracy
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.exceptions import ExecutionError, InputValidationError
from inference_endpoint.load_generator.session import SessionResult
from inference_endpoint.metrics.report import series_metric_dict
from inference_endpoint.utils.atomic_write import atomic_write_bytes

if TYPE_CHECKING:
    from inference_endpoint.commands.benchmark.execute import BenchmarkContext

logger = logging.getLogger(__name__)


@dataclass
class AccuracyConfiguration:
    scorer: type[Scorer]
    extractor: type[Extractor] | None
    dataset_name: str
    dataset: Dataset
    report_dir: Path
    ground_truth_column: str | None
    num_repeats: int
    extras: dict[str, Any] = field(default_factory=dict)
    # Discriminates the inline perf-scored entry (PERFORMANCE) from real accuracy
    # datasets (ACCURACY). Branch on this, not on dataset_name == "performance":
    # a dataset legitimately named "performance" must not be misclassified.
    dataset_type: DatasetType = DatasetType.ACCURACY


def _phase_osl_stats(
    sample_uuids: Iterable[str],
    uuid_to_text: dict[str, str],
    backend: Any,
    batch_size: int = 256,
) -> dict[str, Any] | None:
    """Output-token-length rollup over one accuracy phase's completions.

    Counts tokens on each sample's response text via the shared reference
    tokenizer backend — the server's ``completion_tokens`` is not persisted, only
    the text is (in ``events.jsonl``) — then shapes the lengths via
    ``series_metric_dict`` so the block matches the perf report's
    ``output_sequence_lengths`` exactly. Returns ``None`` when the phase has no
    completed outputs.

    ``batch_size`` bounds each ``encode_batch`` pass: accuracy outputs can be tens
    of thousands of tokens each (e.g. gpt-oss lcb at 32768), so counting the whole
    population in one call would hold every Encoding in memory at once.
    """
    # Skip empty/failed completions (a failed request still logs a COMPLETE
    # event with output == ""). The perf-side OslTrigger does the same
    # (metrics_table.OslTrigger._extract_text returns None for empty text), so
    # accuracy OSL matches its population and a failure isn't counted as a
    # 0-token sample that would drag min/avg down.
    texts = [
        uuid_to_text[u] for u in sample_uuids if u in uuid_to_text and uuid_to_text[u]
    ]
    if not texts:
        return None
    lengths: list[int] = []
    for i in range(0, len(texts), batch_size):
        lengths.extend(encode_lengths(backend, texts[i : i + batch_size]))
    return series_metric_dict(lengths) or None


def _phase_response_counts(
    sample_uuids: Iterable[str],
    uuid_to_text: dict[str, str],
) -> dict[str, int]:
    """Per-phase response accounting over one accuracy phase's issued samples.

    Complements :func:`_phase_osl_stats`, which reports token lengths only over
    non-empty completions — on its own that can hide a run where the server
    returned blanks or dropped requests. Classifies each issued ``sample_uuid``
    as ``scored`` (COMPLETE, non-empty output — exactly the OSL population),
    ``empty`` (COMPLETE with blank output: a failed request the load generator
    logged as ERROR then an empty COMPLETE), or ``missing`` (no COMPLETE event).
    ``issued == scored + empty + missing`` always holds.

    Emptiness uses the same truthiness test as ``_phase_osl_stats`` so ``scored``
    is byte-for-byte the OSL population — the two blocks cannot disagree.
    """
    issued = scored = empty = missing = 0
    for u in sample_uuids:
        issued += 1
        if u not in uuid_to_text:
            missing += 1
        elif uuid_to_text[u]:
            scored += 1
        else:
            empty += 1
    return {"issued": issued, "scored": scored, "empty": empty, "missing": missing}


def _accuracy_uuid_bound(
    report_dir: Path | None, eval_configs: list[AccuracyConfiguration]
) -> set[str]:
    """Union of the accuracy datasets' issued uuids from ``sample_idx_map.json``.

    Bounds the finalize-side raw-output read to the accuracy population so it
    never holds the whole run's (incl. perf) response-text corpus. Returns an
    empty set (⇒ caller reads unbounded) when there is no report dir; a missing,
    corrupt, or wrong-shape map is warned and also falls back to unbounded.
    """
    if report_dir is None:
        return set()
    try:
        idx_map = msgspec.json.decode((report_dir / "sample_idx_map.json").read_bytes())
    except (OSError, msgspec.DecodeError) as e:
        logger.warning(
            "Accuracy OSL uuid bound unavailable (%s); reading outputs unbounded", e
        )
        return set()
    # A syntactically-valid map of the wrong shape must not crash finalize: this
    # runs outside the per-dataset try, so a raised AttributeError/TypeError would
    # fail scoring (OSL must never do that). Skip anything not dict-shaped.
    if not isinstance(idx_map, dict):
        logger.warning(
            "Accuracy OSL uuid bound: sample_idx_map.json is not an object; "
            "reading outputs unbounded"
        )
        return set()
    bound: set[str] = set()
    for ec in eval_configs:
        if ec.dataset_type == DatasetType.ACCURACY:
            per_dataset = idx_map.get(ec.dataset_name)
            if isinstance(per_dataset, dict):
                bound |= set(per_dataset)
    return bound


def _load_osl_backend(has_accuracy: bool, tokenizer_name: str | None) -> Any | None:
    """Load the reference tokenizer backend for accuracy OSL, or None to disable.

    Loaded only when a real accuracy dataset exists; a load failure or a tokenizer
    with no fast (Rust) backend disables OSL rather than failing scoring.
    """
    if not (has_accuracy and tokenizer_name is not None):
        return None
    try:
        osl_backend = load_reference_backend(tokenizer_name)
    except Exception as e:  # noqa: BLE001 - OSL is optional; never fail scoring
        logger.warning(
            "Accuracy OSL disabled: could not load tokenizer %r: %s",
            tokenizer_name,
            e,
        )
        return None
    # A tokenizer with no fast (Rust) backend disables OSL rather than falling
    # back to a slow Python-tokenizer count: the perf side
    # (token_metrics._setup_shards) requires a fast backend too and raises without
    # one, so OSL stays fast-only and consistent on both sides. Warn so the skip is
    # visible instead of silently dropping the block.
    if osl_backend is None:
        logger.warning(
            "Accuracy OSL disabled: tokenizer %r has no fast (Rust) backend "
            "(token counting requires one, as on the perf side)",
            tokenizer_name,
        )
    return osl_backend


def _score_accuracy(
    ctx: BenchmarkContext, result: SessionResult
) -> list[dict[str, Any]]:
    """Score each accuracy dataset into its own list entry.

    One entry per eval_config, in order; no cross-dataset consolidation. Each
    entry carries the scalar ``score`` plus sample accounting
    (``unit_samples`` × ``num_repeats`` = ``total_samples``); a scorer that
    returns a ``score_breakdown()`` (DeepSeek-R1, BFCL) also attaches
    ``breakdown``. The ``"performance"`` inline entry totals the perf phases'
    issued counts instead of unit × repeats (repeats is forced to 1 there).
    """
    accuracy_scores: list[dict[str, Any]] = []

    # Per-phase wall-clock (seconds) keyed by phase name. The accuracy phase name
    # is the dataset_name; the inline-scored perf entry keys on "performance".
    phase_durations: dict[str, float] = {}
    for pr in result.phase_results:
        phase_durations[pr.name] = phase_durations.get(pr.name, 0.0) + max(
            0.0, (pr.end_time_ns - pr.start_time_ns) / 1e9
        )

    # Accuracy-phase output-token lengths (finalize-side, off the hot path): the
    # aggregator only tokenizes perf-window samples, so count the accuracy
    # responses (already in events.jsonl) here, using the same reference tokenizer
    # as the perf side. (Counts still differ from perf for tool-call responses —
    # client-side OSL is approximate for structured output.)
    has_accuracy = any(
        ec.dataset_type == DatasetType.ACCURACY for ec in ctx.eval_configs
    )
    osl_backend = _load_osl_backend(has_accuracy, ctx.tokenizer_name)
    # Bound the raw-output read to the accuracy population so finalize never holds
    # the whole run's (incl. perf) response-text corpus.
    accuracy_uuids = (
        _accuracy_uuid_bound(ctx.report_dir, ctx.eval_configs)
        if has_accuracy
        else set()
    )
    uuid_to_text: dict[str, str] | None = None

    for eval_cfg in ctx.eval_configs:
        try:
            scorer_instance = eval_cfg.scorer(
                eval_cfg.dataset_name,
                eval_cfg.dataset,
                eval_cfg.report_dir,
                extractor=eval_cfg.extractor,
                ground_truth_column=eval_cfg.ground_truth_column,
                **eval_cfg.extras,
            )
        except TypeError as e:
            raise InputValidationError(
                f"Dataset '{eval_cfg.dataset_name}': invalid accuracy_config.extras "
                f"for scorer '{eval_cfg.scorer.__name__}': {e}"
            ) from e
        score, n_repeats = scorer_instance.score()
        # Coerce a numpy scalar score (np.float32/64, numpy ints — e.g. np.mean
        # from the base Scorer) to a native Python float so the entry stays
        # serializable by both msgspec (result_summary.json) and json
        # (accuracy_results.json). numbers.Real catches every numpy scalar (not
        # just np.float64, which isinstance(..., float) alone would miss) while
        # leaving None / dict (RougeScorer) untouched; bool is excluded.
        if isinstance(score, numbers.Real) and not isinstance(score, bool):
            score = float(score)
        unit_samples = eval_cfg.dataset.num_samples()
        num_repeats = eval_cfg.num_repeats
        if eval_cfg.dataset_type == DatasetType.PERFORMANCE:
            # A performance dataset always scores its already-issued outputs once
            # (enforced by the num_repeats == 1 guard in _load_datasets), so make
            # that locally provable rather than relying on eval_cfg carrying 1.
            num_repeats = 1
            total_samples = sum(phase.issued_count for phase in result.perf_results)
        else:
            total_samples = unit_samples * num_repeats
        entry: dict[str, Any] = {
            "dataset_name": eval_cfg.dataset_name,
            "extractor": (
                eval_cfg.extractor.__name__ if eval_cfg.extractor is not None else None
            ),
            "ground_truth_column": eval_cfg.ground_truth_column,
            "score": score,
            "unit_samples": unit_samples,
            "num_repeats": num_repeats,
            "total_samples": total_samples,
            # Wall-clock of this dataset's issue phase (seconds); 0.0 if the
            # phase left no timing (e.g. a scored-but-not-issued dataset).
            "duration_s": round(phase_durations.get(eval_cfg.dataset_name, 0.0), 3),
            # False when the scorer produced only a partial headline (e.g.
            # LegacyMLPerfDeepSeekR1Scorer when lcb-service was unreachable), so a
            # partial number is never mistaken for a complete one.
            "complete": scorer_instance.complete,
            # Persist the same DatasetType discriminator carried on the eval config
            # so consumers filter the inline perf-scored entry by type, not by
            # matching dataset_name == "performance".
            "dataset_type": eval_cfg.dataset_type.value,
        }
        breakdown = scorer_instance.score_breakdown()
        if breakdown is not None:
            entry["breakdown"] = breakdown

        # Response accounting + avg/min/max output-token length. Skipped for the
        # perf entry (its OSL / failure counts live in result_summary.json). The
        # counts are computed independent of the tokenizer and of OSL returning a
        # block — an all-failed phase must still publish scored=0 rather than
        # silently omitting everything. OSL stays tokenizer-gated. A read/tokenize
        # failure only drops these blocks — it never fails scoring.
        if eval_cfg.dataset_type == DatasetType.ACCURACY:
            try:
                if uuid_to_text is None:
                    # Built once from the first scorer and reused for every
                    # dataset. get_raw_outputs() returns the model's actual
                    # completion text (not the scorer's scoring-normalized form)
                    # for *all* phases' COMPLETE events, bounded to the accuracy
                    # population; intersecting it with each dataset's
                    # sample_index_map yields correct per-dataset counts.
                    out_df = scorer_instance.get_raw_outputs(accuracy_uuids or None)
                    uuid_to_text = dict(
                        zip(out_df["sample_uuid"], out_df["output"], strict=False)
                    )
                    # Drop the DataFrame so finalize doesn't hold both it and the
                    # dict (each carrying the response-text corpus).
                    del out_df
                entry["response_counts"] = _phase_response_counts(
                    scorer_instance.sample_index_map, uuid_to_text
                )
                if osl_backend is not None:
                    t0 = time.perf_counter()
                    osl = _phase_osl_stats(
                        scorer_instance.sample_index_map, uuid_to_text, osl_backend
                    )
                    if osl is not None:
                        # Same shape/key as the perf report output_sequence_lengths.
                        entry["output_sequence_lengths"] = osl
                        # Wall-clock of just this phase's tokenization (seconds);
                        # summed across datasets for the accuracy report's total.
                        entry["osl_tokenize_s"] = round(time.perf_counter() - t0, 3)
            except Exception as e:  # noqa: BLE001 - optional blocks; never fail scoring
                logger.warning(
                    "Accuracy response counts/OSL skipped for %s: %s",
                    eval_cfg.dataset_name,
                    e,
                )

        accuracy_scores.append(entry)
        logger.info(
            f"Score for {eval_cfg.dataset_name}: {score} "
            f"({n_repeats} repeats, complete={scorer_instance.complete})"
        )

    return accuracy_scores


def write_accuracy_results(
    report_dir: Path, accuracy_scores: list[dict[str, Any]]
) -> None:
    """Emit ``accuracy/accuracy_results.json`` from the per-dataset score entries.

    Perf rollups (qps/tps/latency percentiles) live in
    ``performance/result_summary.json`` and response/error text lives in
    ``events.jsonl``, so neither is duplicated here. A no-op when ``accuracy_scores``
    is empty — a perf-only run leaves no ``accuracy/`` folder.
    """
    if not accuracy_scores:
        return
    # Plain cross-component mean of the per-dataset scores (3 datasets for
    # gpt-oss, 1 for DeepSeek-R1); None when nothing numeric was scored.
    avg_accuracy = average_accuracy(accuracy_scores)
    # Total finalize-time spent tokenizing accuracy outputs for OSL (seconds),
    # summed across datasets. Emitted whenever OSL was computed for at least
    # one dataset — gating on the key's presence, not the rounded wall-clock,
    # so a sub-millisecond total (tiny outputs) still records 0.0 rather than
    # silently dropping the field.
    osl_computed = any("osl_tokenize_s" in e for e in accuracy_scores)
    osl_tokenization_s = round(
        sum(e.get("osl_tokenize_s", 0.0) for e in accuracy_scores), 3
    )
    accuracy_dir = report_dir / "accuracy"
    accuracy_results_path = accuracy_dir / "accuracy_results.json"
    accuracy_payload: dict[str, Any] = {}
    if avg_accuracy is not None:
        accuracy_payload["average_accuracy"] = avg_accuracy
    if osl_computed:
        accuracy_payload["osl_tokenization_s"] = osl_tokenization_s
    accuracy_payload["accuracy_scores"] = accuracy_scores
    # Atomic write so a crash mid-write can't leave truncated JSON the
    # compliance checker would read as corrupt. Not swallowed: if scoring
    # produced entries but they can't be persisted — the dir can't be made
    # (OSError), the payload won't serialize (TypeError/ValueError, e.g. a
    # numpy scalar left in a breakdown block), or the write fails — fail the
    # run loudly rather than exit 0 with no accuracy artifact.
    try:
        accuracy_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            accuracy_results_path,
            json.dumps(accuracy_payload, indent=2).encode(),
        )
    except (OSError, TypeError, ValueError) as e:
        raise ExecutionError(
            f"Failed to write accuracy results to {accuracy_results_path}: {e}"
        ) from e
    logger.info(f"Saved: {accuracy_results_path}")
