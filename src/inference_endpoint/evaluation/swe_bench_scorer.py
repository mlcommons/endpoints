# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-backed SWE-bench accuracy scorer."""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urljoin, urlparse

import msgspec
from tqdm import tqdm

from ..dataset_manager.dataset import Dataset
from ..dataset_manager.predefined.swe_bench import SWEBench
from ..exceptions import SetupError
from ..utils.atomic_write import atomic_write_bytes
from .extractor import Extractor
from .scoring import Scorer

if TYPE_CHECKING:
    from ..config.schema import EndpointConfig, ModelParams

logger = logging.getLogger(__name__)

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class _RejectRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib_request.build_opener(_RejectRedirectHandler())


class SWEBenchScorer(Scorer, scorer_id="swe_bench_scorer"):
    """Run and grade SWE-bench through the configured external service.

    Unlike ordinary scorers, this class does not consume outputs from the
    benchmark endpoint phase. It submits the selected instance IDs and endpoint
    configuration to the service, follows the remote agent and evaluation
    phases, downloads run-scoped artifacts, and returns the resolved fraction.
    """

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
    REMOVED_TOOLCALL_PATCH_EXTRA: ClassVar[str] = "enable_swebench_toolcall_patch"
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
        swebench_template: str | None = None,
        service_timeout_s: int | None = None,
        poll_interval_s: float | None = None,
        model_params: ModelParams | None = None,
        endpoint_config: EndpointConfig | None = None,
    ):
        ground_truth_column = ground_truth_column or "instance_id"
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column,
        )
        self.report_dir: Path = Path(report_dir).resolve()
        options = self._resolve_options(
            {
                "swebench_service_url": swebench_service_url,
                "swebench_service_auth_token": swebench_service_auth_token,
                "subset": subset,
                "split": split,
                "num_instances": num_instances,
                "workers": workers,
                "max_eval_workers": max_eval_workers,
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
        self.swebench_template = options["swebench_template"]
        self.service_timeout_s = options["service_timeout_s"]
        self.poll_interval_s = options["poll_interval_s"]
        self.model_params = model_params
        self.endpoint_config = endpoint_config

    @classmethod
    def _normalize_service_url(cls, value: Any) -> str:
        if value is None or str(value).strip() == "":
            raise SetupError(
                "accuracy_config.extras.swebench_service_url is required for "
                "swe_bench_scorer. Start the SWE-bench service and pass its URL."
            )
        raw = str(value).strip()
        message = (
            "accuracy_config.extras.swebench_service_url must be an HTTP(S) "
            "service root URL with no path, query, or fragment"
        )
        try:
            parsed = urlparse(raw)
            _ = parsed.port
        except ValueError as exc:
            raise SetupError(message) from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise SetupError(message)
        return parsed._replace(path="/", params="", query="", fragment="").geturl()

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
            with _NO_REDIRECT_OPENER.open(req, timeout=timeout_s) as resp:
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

    @staticmethod
    def _validate_run_id(value: Any) -> str:
        if not isinstance(value, str) or not _SAFE_RUN_ID_RE.fullmatch(value):
            raise SetupError("SWE-bench service returned an invalid run_id")
        return value

    @classmethod
    def _download_artifact(
        cls,
        service_url: str,
        artifact: dict[str, Any],
        report_dir: Path,
        run_id: str,
        auth_token: str | None = None,
    ) -> None:
        run_id = cls._validate_run_id(run_id)
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
        target_dir = report_dir / "swe_bench_runs" / run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        url = urljoin(service_url, href.lstrip("/"))
        req = urllib_request.Request(
            url, headers={"Accept": "application/octet-stream"}
        )
        if auth_token:
            req.add_header("Authorization", f"Bearer {auth_token}")
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with (
                _NO_REDIRECT_OPENER.open(req, timeout=60.0) as resp,
                tmp.open("wb") as output,
            ):
                shutil.copyfileobj(resp, output, length=1024 * 1024)
            os.replace(tmp, target)
        except Exception:
            logger.warning(
                "Could not download SWE-bench artifact %s", name, exc_info=True
            )
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def _download_artifacts(
        cls,
        service_url: str,
        status: dict[str, Any],
        report_dir: Path,
        auth_token: str | None = None,
    ) -> None:
        run_id = cls._validate_run_id(status.get("run_id"))
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

    @classmethod
    def _write_service_status(cls, report_dir: Path, status: dict[str, Any]) -> None:
        run_id = cls._validate_run_id(status.get("run_id"))
        run_dir = report_dir / "swe_bench_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "swe_bench_service_status.json").write_bytes(
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
    def _result_counter(result: dict[str, Any], key: str) -> int:
        """Parse a required nonnegative service-result counter."""
        value = result.get(key)
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a nonnegative integer")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.isascii() and value.isdecimal():
            parsed = int(value)
        else:
            raise ValueError(f"{key} must be a nonnegative integer")
        if parsed < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        return parsed

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
            raw = "default"
        template = str(raw)
        if template not in cls.SERVICE_TEMPLATES:
            raise SetupError(
                "accuracy_config.extras.swebench_template must be one of "
                f"{sorted(cls.SERVICE_TEMPLATES)}; got {template!r}"
            )
        return template

    @classmethod
    def _resolve_options(cls, extras: dict[str, Any]) -> dict[str, Any]:
        if cls.REMOVED_TOOLCALL_PATCH_EXTRA in extras:
            raise SetupError(
                "accuracy_config.extras.enable_swebench_toolcall_patch has been "
                "removed; use swebench_template: qwen_tools instead"
            )
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
    def _generation_params(model_params: ModelParams) -> dict[str, Any]:
        fields = (
            "temperature",
            "seed",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "max_new_tokens",
            "chat_template_kwargs",
        )
        dumped = model_params.model_dump(exclude_none=True)
        return {field: dumped[field] for field in fields if field in dumped}

    @classmethod
    def dataset_loader_kwargs(cls, extras: dict[str, Any]) -> dict[str, Any]:
        """Select the configured SWE-bench subset and split for loading."""
        return cls._resolve_dataset_options(extras)

    @classmethod
    def external_sample_count(cls, extras: dict[str, Any]) -> int | None:
        """Return the requested service-run size before dataset-size clamping."""
        raw = extras.get("num_instances", cls.DEFAULT_NUM_INSTANCES)
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def preflight(cls, extras: dict[str, Any]) -> None:
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
        if self.model_params is None:
            raise ValueError(
                "model_params must be provided when constructing SWEBenchScorer"
            )
        if self.endpoint_config is None:
            raise ValueError(
                "endpoint_config must be provided when constructing SWEBenchScorer"
            )
        model_name = self.model_params.name
        if not model_name:
            raise ValueError("model_params.name is required but is missing or empty")
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

        endpoint_urls = self.endpoint_config.endpoints
        if len(endpoint_urls) != 1:
            raise SetupError(
                "SWE-bench service mode supports exactly one endpoint URL; "
                f"got {len(endpoint_urls)}."
            )
        payload: dict[str, Any] = {
            "model_name": model_name,
            "endpoint_urls": endpoint_urls,
            "endpoint_api_key": self.endpoint_config.api_key,
            "generation_params": self._generation_params(self.model_params),
            "subset": self.subset,
            "split": self.split,
            "num_instances": total_instances,
            "workers": self.workers,
            "max_eval_workers": self.max_eval_workers,
            "evaluated_instance_ids": evaluated_instance_ids,
            "template": self.swebench_template,
        }

        run_id = ""
        progress_state: dict[str, Any] = {}
        result_path = self.report_dir / "swe_bench_results.json"
        result_path.unlink(missing_ok=True)
        try:
            submitted = type(self)._http_json(
                urljoin(self.swebench_service_url, "v1/runs"),
                method="POST",
                payload=payload,
                timeout_s=30.0,
                auth_token=self.swebench_service_auth_token,
            )
            run_id = type(self)._validate_run_id(submitted.get("run_id"))
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
                status_run_id = type(self)._validate_run_id(status.get("run_id"))
                if status_run_id != run_id:
                    raise SetupError(
                        "SWE-bench service returned a mismatched run_id: "
                        f"expected {run_id!r}, got {status_run_id!r}"
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
        except Exception:
            if run_id:
                type(self)._cancel_service_run(
                    self.swebench_service_url,
                    run_id,
                    self.swebench_service_auth_token,
                )
            raise
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
        run_result_path = (
            self.report_dir / "swe_bench_runs" / run_id / "swe_bench_results.json"
        )
        if result is None and run_result_path.exists():
            try:
                result = msgspec.json.decode(run_result_path.read_bytes(), type=dict)
            except msgspec.DecodeError:
                self.complete = False
                return None, 1
        if not isinstance(result, dict):
            logger.error("SWE-bench service run %s did not return a result", run_id)
            self.complete = False
            return None, 1
        atomic_write_bytes(result_path, msgspec.json.encode(result))

        denominator = len(evaluated_instance_ids)
        if denominator == 0:
            logger.warning(
                "SWE-bench: evaluated instance count is 0; returning None score"
            )
            self.complete = False
            return None, 1
        try:
            submitted_count = type(self)._result_counter(result, "submitted_instances")
            resolved = type(self)._result_counter(result, "resolved_instances")
        except ValueError as exc:
            logger.error("SWE-bench: invalid service result: %s", exc)
            self.complete = False
            return None, 1
        if resolved > submitted_count or submitted_count > denominator:
            logger.error(
                "SWE-bench: inconsistent service counters: resolved=%d, "
                "submitted=%d, expected=%d",
                resolved,
                submitted_count,
                denominator,
            )
            self.complete = False
            return None, 1
        if submitted_count == 0:
            logger.warning("SWE-bench: submitted_instances=0; returning None score")
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
