# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench accuracy scorer that fans out across a fleet of SWE-bench services.

:class:`~inference_endpoint.evaluation.swe_bench_scorer.SWEBenchScorer` runs the
whole instance list as one service run. This scorer shards it, runs the shards
concurrently on several services, refuses to score a run whose instances are not
all accounted for, and retries the shards that lost instances to infrastructure
rather than the model.

Configured entirely through ``accuracy_config.extras``::

    accuracy_config:
      eval_method: swe_bench_fleet
      extras:
        swebench_service_urls:
          - http://swe-host-1:18080
          - http://swe-host-2:18080
        shard_size: 10
        max_attempts: 3
        expected_model: Org/Model-FP8      # optional; gates the checkpoint
        min_prompt_tokens: 2000
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urljoin

import msgspec

from ..dataset_manager.dataset import Dataset
from ..exceptions import SetupError
from .extractor import Extractor
from .scoring import Scorer
from .swe_bench_distributed.fleet import (
    QUEUE_DIRNAME,
    UNITS_DIRNAME,
    FleetDispatcher,
    accounted_and_resolved,
    build_gates,
    load_benchmark_config,
    write_merge_artifacts,
)
from .swe_bench_distributed.gates import GateFailure, run_gates
from .swe_bench_distributed.merge import MergeRefusal, merge_run
from .swe_bench_distributed.queue import WorkQueue
from .swe_bench_distributed.units import Unit, plan_units
from .swe_bench_scorer import SWEBenchScorer

logger = logging.getLogger(__name__)


class SWEBenchFleetScorer(Scorer, scorer_id="swe_bench_fleet"):
    """Distributed SWE-bench scoring across N services."""

    REQUIRES_EXTRACTOR: ClassVar[bool] = False
    SKIP_ENDPOINT_PHASE: ClassVar[bool] = True
    DEFAULT_SHARD_SIZE: ClassVar[int] = 10
    DEFAULT_MAX_ATTEMPTS: ClassVar[int] = 3
    DEFAULT_MIN_PROMPT_TOKENS: ClassVar[int] = 2000
    DEFAULT_STALL_TIMEOUT_S: ClassVar[int] = 3 * 60 * 60
    DEFAULT_SERVICE_TIMEOUT_S: ClassVar[int] = 24 * 60 * 60
    DEFAULT_POLL_INTERVAL_S: ClassVar[float] = 5.0

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: Any,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = "instance_id",
        **extras: Any,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column or "instance_id",
        )
        self.report_dir = self.report_dir.resolve()
        self.options = self._resolve_options(extras)

    # --------------------------------------------------------------- config --

    @classmethod
    def _service_urls(cls, extras: dict[str, Any]) -> list[str]:
        raw = extras.get("swebench_service_urls")
        if raw is None:
            single = extras.get("swebench_service_url")
            raw = [single] if single else []
        if isinstance(raw, str):
            raw = [part.strip() for part in raw.split(",") if part.strip()]
        urls = [SWEBenchScorer._normalize_service_url(url) for url in raw or []]
        if not urls:
            raise SetupError(
                "accuracy_config.extras.swebench_service_urls is required for "
                "swe_bench_fleet; list one URL per SWE-bench service host."
            )
        duplicates = sorted({url for url in urls if urls.count(url) > 1})
        if duplicates:
            # Two entries for one host is not extra capacity; it is two
            # concurrent runs contending for the same Docker/Pyxis runtime.
            raise SetupError(
                "duplicate SWE-bench service URLs: " + ", ".join(duplicates)
            )
        return urls

    @classmethod
    def _resolve_options(cls, extras: dict[str, Any]) -> dict[str, Any]:
        options = dict(SWEBenchScorer._resolve_dataset_options(extras))
        options["service_urls"] = cls._service_urls(extras)
        options["auth_token"] = extras.get("swebench_service_auth_token") or None
        options["num_instances"] = SWEBenchScorer._get_extra_int(
            extras,
            "num_instances",
            default=SWEBenchScorer.DEFAULT_NUM_INSTANCES,
            min_value=1,
        )
        options["shard_size"] = SWEBenchScorer._get_extra_int(
            extras, "shard_size", default=cls.DEFAULT_SHARD_SIZE, min_value=1
        )
        options["workers"] = SWEBenchScorer._get_extra_int(
            extras, "workers", default=SWEBenchScorer.DEFAULT_WORKERS, min_value=1
        )
        options["max_eval_workers"] = SWEBenchScorer._get_extra_int(
            extras,
            "max_eval_workers",
            default=SWEBenchScorer.DEFAULT_MAX_EVAL_WORKERS,
            min_value=1,
        )
        options["max_attempts"] = SWEBenchScorer._get_extra_int(
            extras, "max_attempts", default=cls.DEFAULT_MAX_ATTEMPTS, min_value=1
        )
        options["min_prompt_tokens"] = SWEBenchScorer._get_extra_int(
            extras,
            "min_prompt_tokens",
            default=cls.DEFAULT_MIN_PROMPT_TOKENS,
            min_value=0,
        )
        options["stall_timeout_s"] = SWEBenchScorer._get_extra_int(
            extras,
            "stall_timeout_s",
            default=cls.DEFAULT_STALL_TIMEOUT_S,
            min_value=1,
        )
        options["service_timeout_s"] = SWEBenchScorer._get_extra_int(
            extras,
            "service_timeout_s",
            default=cls.DEFAULT_SERVICE_TIMEOUT_S,
            min_value=1,
        )
        options["poll_interval_s"] = SWEBenchScorer._get_extra_float(
            extras,
            "poll_interval_s",
            default=cls.DEFAULT_POLL_INTERVAL_S,
            min_value=0,
        )
        options["swebench_template"] = SWEBenchScorer._resolve_service_template(extras)
        options["expected_model"] = extras.get("expected_model") or None
        options["run_id"] = str(extras.get("run_id") or "swe_bench")
        return options

    @classmethod
    def dataset_loader_kwargs(cls, extras: dict[str, Any]) -> dict[str, Any]:
        return SWEBenchScorer._resolve_dataset_options(extras)

    @classmethod
    def external_sample_count(cls, extras: dict[str, Any]) -> int | None:
        return SWEBenchScorer.external_sample_count(extras)

    # ------------------------------------------------------------ preflight --

    @classmethod
    def preflight(
        cls, extras: dict[str, Any], *, loaded_sample_count: int | None = None
    ) -> None:
        """Health-check every service and run the pre-dispatch gates.

        Every problem is reported from one preflight. A run that starts against
        a mis-served checkpoint, or against an endpoint that cannot emit a tool
        call at SWE-bench prompt scale, produces a plausible-looking low score
        hours later and costs the whole run.
        """
        options = cls._resolve_options(extras)
        for url in options["service_urls"]:
            SWEBenchScorer._check_health(url, options["auth_token"])

        endpoints = extras.get("endpoint_urls") or []
        if not endpoints:
            logger.info(
                "swe_bench_fleet: no endpoint URLs available at preflight; "
                "checkpoint and tool-call gates run at dispatch instead"
            )
            return
        gates, _ = build_gates(
            expected_model=options["expected_model"],
            tool_call_model=extras.get("model_name"),
            min_prompt_tokens=options["min_prompt_tokens"],
            api_key=extras.get("endpoint_api_key"),
        )
        try:
            run_gates(gates, list(endpoints))
        except GateFailure as exc:
            raise SetupError(str(exc)) from exc

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "SWEBenchFleetScorer scores whole units through services; call score()."
        )

    # ---------------------------------------------------------------- score --

    def score(self) -> tuple[float | None, int]:
        self.complete = True
        config = load_benchmark_config(self.report_dir)
        model_params = config.get("model_params") or {}
        model_name = model_params.get("name")
        if not model_name:
            raise ValueError("model_params.name is required in the benchmark config")
        endpoint_config = config.get("endpoint_config") or {}
        endpoint_urls = list(endpoint_config.get("endpoints") or [])
        if not endpoint_urls:
            raise SetupError("the benchmark config lists no endpoint URLs")

        instance_ids = self._instance_ids()
        if not instance_ids:
            logger.warning("swe_bench_fleet: no instances selected")
            self.complete = False
            return None, 1

        gates, fingerprint_gate = build_gates(
            expected_model=self.options["expected_model"],
            tool_call_model=model_name,
            min_prompt_tokens=self.options["min_prompt_tokens"],
            api_key=endpoint_config.get("api_key"),
        )
        try:
            run_gates(gates, endpoint_urls)
        except GateFailure as exc:
            raise SetupError(str(exc)) from exc

        plan = plan_units(
            self.options["run_id"], instance_ids, shard_size=self.options["shard_size"]
        )
        queue = WorkQueue(self.report_dir / QUEUE_DIRNAME, plan)
        unit_root = self.report_dir / UNITS_DIRNAME
        unit_root.mkdir(parents=True, exist_ok=True)

        self._model_name = model_name
        self._endpoint_urls = endpoint_urls
        self._endpoint_api_key = endpoint_config.get("api_key")
        # load_benchmark_config() yaml.safe_load()s config.yaml, so model_params
        # is a plain mapping here, while _generation_params() expects the
        # pydantic ModelParams. Re-validate rather than re-implement the field
        # selection, so the fleet path and the single-service path agree.
        from ..config.schema import ModelParams

        self._generation_params = SWEBenchScorer._generation_params(
            ModelParams.model_validate(model_params)
        )
        self._unit_root = unit_root

        def fingerprint() -> str | None:
            values = [fingerprint_gate.fingerprint(url) for url in endpoint_urls]
            if any(value is None for value in values):
                return None
            return "|".join(v for v in values if v is not None)

        dispatcher = FleetDispatcher(
            queue=queue,
            service_urls=self.options["service_urls"],
            submit=self._submit_unit,
            poll=self._poll_unit,
            collect=self._collect_unit,
            fingerprint=fingerprint,
            max_attempts=self.options["max_attempts"],
            stall_timeout_s=self.options["stall_timeout_s"],
            unit_root=unit_root,
        )
        dispatcher.run()

        payload: dict[str, Any] = {
            "run_id": plan.run_id,
            "plan_digest": plan.digest,
            "services": self.options["service_urls"],
            "quarantined": dispatcher.quarantined,
        }
        try:
            merged = merge_run(queue, plan.run_id)
        except MergeRefusal as exc:
            payload["refused"] = exc.reasons
            write_merge_artifacts(self.report_dir, payload)
            logger.error("swe_bench_fleet: %s", exc)
            self.complete = False
            return None, 1

        payload["merge"] = merged.to_dict()
        write_merge_artifacts(self.report_dir, payload)
        logger.info(
            "swe_bench_fleet: resolved %d / %d (%.1f%%) across %d units",
            merged.resolved_instances,
            merged.total_instances,
            merged.resolved_rate * 100,
            merged.unit_count,
        )
        return merged.resolved_rate, 1

    # --------------------------------------------------------- service glue --

    def _instance_ids(self) -> list[str]:
        if self.dataset.dataframe is None:
            raise RuntimeError(
                "SWEBench dataset must be loaded before scoring; call dataset.load()."
            )
        frame = self.dataset.dataframe
        total = min(self.options["num_instances"], len(frame))
        return [
            str(instance_id)
            for instance_id in frame.iloc[:total][self.ground_truth_column].tolist()
        ]

    def _submit_unit(self, service_url: str, unit: Unit) -> str:
        # The service accepts exactly one endpoint URL per run, so a unit is
        # bound to exactly one endpoint. Sending every unit to endpoint 0 would
        # funnel the whole fleet through a single engine while the rest idle,
        # which is both a throughput ceiling and a measurement hazard: one
        # engine's behaviour would decide the entire run's accuracy. Binding by
        # shard index spreads units deterministically -- the same unit always
        # gets the same endpoint, so a retry is comparable to its first attempt.
        endpoint = self._endpoint_urls[unit.shard % len(self._endpoint_urls)]
        payload = {
            "model_name": self._model_name,
            "endpoint_urls": [endpoint],
            "endpoint_api_key": self._endpoint_api_key,
            "generation_params": self._generation_params,
            "subset": self.options["subset"],
            "split": self.options["split"],
            "num_instances": len(unit.instance_ids),
            "workers": self.options["workers"],
            "max_eval_workers": self.options["max_eval_workers"],
            "evaluated_instance_ids": list(unit.instance_ids),
            "template": self.options["swebench_template"],
        }
        submitted = SWEBenchScorer._http_json(
            urljoin(service_url, "v1/runs"),
            method="POST",
            payload=payload,
            timeout_s=30.0,
            auth_token=self.options["auth_token"],
        )
        run_id = str(submitted.get("run_id") or "")
        if not run_id:
            raise SetupError(f"{service_url} did not return a run_id")
        return run_id

    def _poll_unit(self, service_url: str, service_run_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.options["service_timeout_s"]
        status: dict[str, Any] = {"status": "queued"}
        while status.get("status") not in {"succeeded", "failed", "cancelled"}:
            if time.monotonic() >= deadline:
                SWEBenchScorer._cancel_service_run(
                    service_url, service_run_id, self.options["auth_token"]
                )
                raise SetupError(
                    f"timed out waiting for {service_url} run {service_run_id}"
                )
            time.sleep(self.options["poll_interval_s"])
            status = SWEBenchScorer._http_json(
                urljoin(service_url, f"v1/runs/{service_run_id}"),
                timeout_s=30.0,
                auth_token=self.options["auth_token"],
            )
        return status

    def _collect_unit(
        self,
        service_url: str,
        service_run_id: str,
        unit: Unit,
        status: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        target = self._unit_root / unit.unit_id
        target.mkdir(parents=True, exist_ok=True)
        SWEBenchScorer._download_artifacts(
            service_url, status, target, self.options["auth_token"]
        )
        report = status.get("result")
        if not isinstance(report, dict):
            results_path = target / "swe_bench_results.json"
            if results_path.exists():
                try:
                    report = msgspec.json.decode(results_path.read_bytes(), type=dict)
                except msgspec.DecodeError:
                    report = {}
            else:
                report = {}
        return report, target


__all__ = ["SWEBenchFleetScorer", "accounted_and_resolved"]
