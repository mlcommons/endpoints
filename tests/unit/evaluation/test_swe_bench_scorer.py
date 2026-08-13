# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for service-backed SWEBenchScorer."""

import io
import json
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import pandas as pd
import pytest

# isort: split
from inference_endpoint.config.schema import EndpointConfig, ModelParams
from inference_endpoint.evaluation import scoring as scoring_mod
from inference_endpoint.evaluation import swe_bench_scorer as swe_bench_mod
from inference_endpoint.evaluation.scoring import Scorer, SWEBenchScorer
from inference_endpoint.evaluation.swe_bench_scorer import (
    SWEBenchScorer as ExtractedSWEBenchScorer,
)
from inference_endpoint.evaluation.swebench_service.swebench_service.artifacts import (
    SAFE_ARTIFACT_NAMES,
)
from inference_endpoint.exceptions import SetupError

pytestmark = pytest.mark.unit

_DATASET_NAME = "swe_bench"
_MODEL_NAME = "TestOrg/test-model-7b"


class FakeTqdm:
    instances: list["FakeTqdm"] = []

    def __init__(self, *, total, desc, unit):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self.closed = False
        self.refreshes = 0
        self.updates: list[int] = []
        type(self).instances.append(self)

    def update(self, n):
        self.updates.append(n)
        self.n += n

    def refresh(self):
        self.refreshes += 1

    def close(self):
        self.closed = True


def _write_sample_idx_map(report_dir: Path, n: int = 3) -> None:
    idx_map = {_DATASET_NAME: {f"uuid-{i}": i for i in range(n)}}
    (report_dir / "sample_idx_map.json").write_bytes(msgspec.json.encode(idx_map))


def _make_dataset(n: int = 3) -> MagicMock:
    ds = MagicMock()
    ds.dataframe = pd.DataFrame(
        {
            "instance_id": [f"repo__repo-{i}" for i in range(n)],
            "prompt": ["placeholder"] * n,
        }
    )
    ds.num_samples.return_value = n
    return ds


def _make_scorer(report_dir: Path, *, dataset=None, **kwargs) -> SWEBenchScorer:
    model_params = kwargs.pop(
        "model_params",
        ModelParams(
            name=_MODEL_NAME,
            temperature=0.25,
            seed=17,
            max_new_tokens=4096,
            chat_template_kwargs={"enable_thinking": False},
        ),
    )
    endpoint_config = kwargs.pop(
        "endpoint_config",
        EndpointConfig(
            endpoints=["http://endpoint-host:30000"],
            api_key="secret-key",
        ),
    )
    return SWEBenchScorer(
        dataset_name=_DATASET_NAME,
        dataset=dataset or _make_dataset(),
        report_dir=report_dir,
        swebench_service_url="http://service-host:18080",
        poll_interval_s=0.01,
        model_params=model_params,
        endpoint_config=endpoint_config,
        **kwargs,
    )


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    d = tmp_path / "report"
    d.mkdir()
    _write_sample_idx_map(d)
    return d


class TestSWEBenchScorerRegistration:
    def test_registered(self):
        assert "swe_bench_scorer" in Scorer.PREDEFINED
        assert Scorer.get("swe_bench_scorer") is SWEBenchScorer
        assert SWEBenchScorer is ExtractedSWEBenchScorer
        assert scoring_mod.SWEBenchScorer is ExtractedSWEBenchScorer

    def test_skip_endpoint_phase(self):
        assert SWEBenchScorer.SKIP_ENDPOINT_PHASE is True

    def test_artifact_allowlist_matches_service(self):
        assert SWEBenchScorer.SAFE_ARTIFACT_NAMES == SAFE_ARTIFACT_NAMES

    def test_external_sample_count(self):
        assert SWEBenchScorer.external_sample_count({"num_instances": 50}) == 50
        assert (
            SWEBenchScorer.external_sample_count({})
            == SWEBenchScorer.DEFAULT_NUM_INSTANCES
        )
        assert SWEBenchScorer.external_sample_count({"num_instances": "bad"}) is None
        assert SWEBenchScorer.external_sample_count({"num_instances": 0}) is None

    def test_poll_interval_defaults_when_constructor_passes_none(self):
        options = SWEBenchScorer._resolve_options(
            {
                "swebench_service_url": "http://service-host:18080",
                "poll_interval_s": None,
            }
        )

        assert options["poll_interval_s"] == SWEBenchScorer.DEFAULT_POLL_INTERVAL_S


class TestSWEBenchScorerPreflight:
    def test_preflight_calls_health(self, monkeypatch):
        calls: list[tuple[str, str, str | None]] = []

        def fake_http_json(url, *, method="GET", **kwargs):
            calls.append((url, method, kwargs.get("auth_token")))
            return {
                "api_version": "v1",
                "capabilities": [
                    "swebench.run",
                    "swebench.cancel",
                    "artifacts.download",
                ],
            }

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)

        SWEBenchScorer.preflight(
            {
                "swebench_service_url": "http://service-host:18080",
                "swebench_service_auth_token": "tok",
                "subset": "lite",
                "num_instances": 1,
            }
        )

        assert calls == [("http://service-host:18080/health", "GET", "tok")]

    def test_preflight_never_calls_docker_or_subprocess(self, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {
                    "api_version": "v1",
                    "capabilities": [
                        "swebench.run",
                        "swebench.cancel",
                        "artifacts.download",
                    ],
                }
            ),
        )

        SWEBenchScorer.preflight({"swebench_service_url": "http://service-host:18080"})

    def test_missing_service_url_raises(self):
        with pytest.raises(SetupError, match="swebench_service_url is required"):
            SWEBenchScorer.preflight({})

    @pytest.mark.parametrize(
        "service_url",
        [
            "http://service-host:18080/v1",
            "http://user@service-host:18080",
            "http://user:password@service-host:18080",
        ],
    )
    def test_invalid_service_url_raises_before_network(self, service_url, monkeypatch):
        http_json = MagicMock()
        monkeypatch.setattr(SWEBenchScorer, "_http_json", http_json)

        with pytest.raises(SetupError, match="service root URL"):
            SWEBenchScorer.preflight({"swebench_service_url": service_url})

        http_json.assert_not_called()

    @pytest.mark.parametrize(
        ("error", "match"),
        [
            (
                swe_bench_mod.urllib_error.HTTPError(
                    "http://service-host:18080/health",
                    503,
                    "unavailable",
                    Message(),
                    io.BytesIO(b"service down"),
                ),
                "HTTP 503: service down",
            ),
            (
                swe_bench_mod.urllib_error.URLError("connection refused"),
                "unreachable.*connection refused",
            ),
        ],
    )
    def test_http_json_translates_transport_error(self, error, match, monkeypatch):
        opener = MagicMock()
        opener.open.side_effect = error
        monkeypatch.setattr(swe_bench_mod, "_NO_REDIRECT_OPENER", opener)

        with pytest.raises(SetupError, match=match):
            SWEBenchScorer._http_json("http://service-host:18080/health")

        opener.open.assert_called_once()

    def test_http_json_translates_invalid_json(self, monkeypatch):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"not-json"
        opener = MagicMock()
        opener.open.return_value = response
        monkeypatch.setattr(swe_bench_mod, "_NO_REDIRECT_OPENER", opener)

        with pytest.raises(SetupError, match="returned invalid JSON"):
            SWEBenchScorer._http_json("http://service-host:18080/health")

    def test_capability_mismatch_raises(self, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {"api_version": "v1", "capabilities": []}
            ),
        )

        with pytest.raises(SetupError, match="missing required capabilities"):
            SWEBenchScorer.preflight({"swebench_service_url": "http://service-host"})

    def test_api_version_mismatch_raises(self, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {
                    "api_version": "v2",
                    "capabilities": [
                        "swebench.run",
                        "swebench.cancel",
                        "artifacts.download",
                    ],
                }
            ),
        )

        with pytest.raises(SetupError, match="API version mismatch"):
            SWEBenchScorer.preflight({"swebench_service_url": "http://service-host"})


class TestSWEBenchScorerScore:
    def test_score_submits_exact_instance_ids_and_computes_rate(
        self, report_dir, monkeypatch
    ):
        payloads: list[dict] = []

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            if method == "POST":
                payloads.append(payload)
                return {
                    "run_id": "run-1",
                    "status": "succeeded",
                    "result": {"resolved_instances": 2, "submitted_instances": 3},
                    "artifacts": [],
                }
            raise AssertionError(f"unexpected GET {url}")

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        scorer = _make_scorer(report_dir)

        score, n_repeats = scorer.score()

        assert score == pytest.approx(2 / 3)
        assert n_repeats == 1
        assert scorer.complete is True
        assert payloads[0]["evaluated_instance_ids"] == [
            "repo__repo-0",
            "repo__repo-1",
            "repo__repo-2",
        ]
        assert "benchmark_config" not in payloads[0]
        assert payloads[0]["endpoint_urls"] == ["http://endpoint-host:30000"]
        assert payloads[0]["endpoint_api_key"] == "secret-key"
        assert payloads[0]["generation_params"] == {
            "temperature": 0.25,
            "seed": 17,
            "max_new_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        assert payloads[0]["template"] == "default"
        assert (report_dir / "swe_bench_results.json").exists()

    def test_score_polls_until_terminal(self, report_dir, monkeypatch):
        calls: list[str] = []

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            calls.append(url)
            if method == "POST":
                return {"run_id": "run-1", "status": "queued"}
            return {
                "run_id": "run-1",
                "status": "succeeded",
                "result": {"resolved_instances": 1, "submitted_instances": 3},
                "artifacts": [],
            }

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        score, _ = _make_scorer(report_dir).score()

        assert score == pytest.approx(1 / 3)
        assert calls[-1] == "http://service-host:18080/v1/runs/run-1"

    @pytest.mark.parametrize(
        ("responses", "expected_cancel"),
        [
            ([{"run_id": "../outside", "status": "succeeded"}], False),
            (
                [
                    {"run_id": "run-1", "status": "running"},
                    {"run_id": "run-2", "status": "succeeded"},
                ],
                True,
            ),
        ],
    )
    def test_score_rejects_unsafe_or_changed_run_id(
        self, report_dir, monkeypatch, responses, expected_cancel
    ):
        response_iter = iter(responses)
        cancel_service_run = MagicMock()
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            lambda *args, **kwargs: next(response_iter),
        )
        monkeypatch.setattr(SWEBenchScorer, "_cancel_service_run", cancel_service_run)
        scorer = _make_scorer(report_dir)

        assert scorer.score() == (None, 1)
        assert scorer.complete is False
        assert not (report_dir / "outside").exists()
        if expected_cancel:
            cancel_service_run.assert_called_once_with(
                "http://service-host:18080/", "run-1", None
            )
        else:
            cancel_service_run.assert_not_called()

    def test_score_forwards_auth_token_to_service_calls(self, report_dir, monkeypatch):
        calls: list[tuple[str, str, str | None]] = []
        downloads: list[tuple[str, str | None]] = []
        phase = "success"

        def fake_http_json(
            url, *, method="GET", payload=None, auth_token=None, **kwargs
        ):
            calls.append((url, method, auth_token))
            if phase == "success":
                if method == "POST":
                    return {"run_id": "run-1", "status": "running"}
                return {
                    "run_id": "run-1",
                    "status": "succeeded",
                    "result": {"resolved_instances": 1, "submitted_instances": 3},
                    "artifacts": [
                        {
                            "name": "preds.json",
                            "url": "/v1/runs/run-1/artifacts/preds.json",
                        }
                    ],
                }
            if method == "POST" and url.endswith("/cancel"):
                return {"run_id": "run-2", "status": "cancelled"}
            if method == "POST":
                return {"run_id": "run-2", "status": "running"}
            raise SetupError("poll failed")

        def fake_download(service_url, status, output_dir, auth_token=None):
            downloads.append((status["run_id"], auth_token))

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        monkeypatch.setattr(SWEBenchScorer, "_download_artifacts", fake_download)

        scorer = _make_scorer(report_dir, swebench_service_auth_token="tok")
        score, _ = scorer.score()
        assert score == pytest.approx(1 / 3)

        phase = "failure"
        failed_scorer = _make_scorer(report_dir, swebench_service_auth_token="tok")
        assert failed_scorer.score() == (None, 1)

        assert all(auth_token == "tok" for _, _, auth_token in calls)
        assert ("run-1", "tok") in downloads
        assert (
            "http://service-host:18080/v1/runs/run-2/cancel",
            "POST",
            "tok",
        ) in calls

    def test_score_rejects_multiple_endpoint_urls(self, report_dir, monkeypatch):
        endpoint_config = EndpointConfig(
            endpoints=["http://endpoint-a:30000", "http://endpoint-b:30000"]
        )
        monkeypatch.setattr(SWEBenchScorer, "_http_json", MagicMock())
        scorer = _make_scorer(report_dir, endpoint_config=endpoint_config)

        with pytest.raises(SetupError, match="exactly one endpoint URL"):
            scorer.score()

        SWEBenchScorer._http_json.assert_not_called()

    def test_score_renders_agent_and_eval_progress_bars(self, report_dir, monkeypatch):
        FakeTqdm.instances = []
        statuses = [
            {
                "run_id": "run-1",
                "status": "running",
                "phase": "agent",
                "agent_total": 3,
                "agent_completed": 1,
                "eval_total": 0,
                "eval_completed": 0,
            },
            {
                "run_id": "run-1",
                "status": "running",
                "phase": "agent",
                "agent_total": 3,
                "agent_completed": 1,
                "eval_total": 0,
                "eval_completed": 0,
            },
            {
                "run_id": "run-1",
                "status": "running",
                "phase": "eval",
                "agent_total": 3,
                "agent_completed": 3,
                "eval_total": 3,
                "eval_completed": 2,
            },
            {
                "run_id": "run-1",
                "status": "succeeded",
                "phase": "succeeded",
                "agent_total": 3,
                "agent_completed": 3,
                "eval_total": 3,
                "eval_completed": 3,
                "result": {"resolved_instances": 2, "submitted_instances": 3},
                "artifacts": [],
            },
        ]

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            return statuses.pop(0)

        monkeypatch.setattr(swe_bench_mod, "tqdm", FakeTqdm)
        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)

        score, _ = _make_scorer(report_dir).score()

        assert score == pytest.approx(2 / 3)
        assert [bar.desc for bar in FakeTqdm.instances] == [
            "SWE-bench agent",
            "SWE-bench eval",
        ]
        assert FakeTqdm.instances[0].n == 3
        assert FakeTqdm.instances[1].n == 3
        assert all(bar.closed for bar in FakeTqdm.instances)

    def test_score_without_progress_does_not_open_tqdm(self, report_dir, monkeypatch):
        statuses = [
            {"run_id": "run-1", "status": "running"},
            {
                "run_id": "run-1",
                "status": "succeeded",
                "result": {"resolved_instances": 1, "submitted_instances": 3},
                "artifacts": [],
            },
        ]

        def fake_tqdm(*args, **kwargs):
            pytest.fail("progress bar should not open without progress fields")

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            return statuses.pop(0)

        monkeypatch.setattr(swe_bench_mod, "tqdm", fake_tqdm)
        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)

        score, _ = _make_scorer(report_dir).score()

        assert score == pytest.approx(1 / 3)

    @pytest.mark.parametrize(
        "error", [KeyboardInterrupt(), RuntimeError("poll failed")]
    )
    def test_polling_exception_cancels_service_run(
        self, report_dir, monkeypatch, error
    ):
        calls: list[tuple[str, str]] = []

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            calls.append((url, method))
            if method == "POST" and url == "http://service-host:18080/v1/runs":
                return {"run_id": "run-1", "status": "running"}
            if (
                method == "POST"
                and url == "http://service-host:18080/v1/runs/run-1/cancel"
            ):
                return {"run_id": "run-1", "status": "cancelled"}
            raise error

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        scorer = _make_scorer(report_dir)

        with pytest.raises(type(error)):
            scorer.score()

        assert ("http://service-host:18080/v1/runs/run-1/cancel", "POST") in calls

    def test_service_timeout_cancels_service_run(self, report_dir, monkeypatch):
        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            if url == "http://service-host:18080/v1/runs":
                return {"run_id": "run-1", "status": "running"}
            raise AssertionError(f"unexpected polling request: {url}")

        cancel_service_run = MagicMock()
        monotonic = iter([0.0, 1.0])
        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        monkeypatch.setattr(SWEBenchScorer, "_cancel_service_run", cancel_service_run)
        monkeypatch.setattr(swe_bench_mod.time, "monotonic", lambda: next(monotonic))
        scorer = _make_scorer(report_dir, service_timeout_s=1)

        assert scorer.score() == (None, 1)
        assert scorer.complete is False
        cancel_service_run.assert_called_once_with(
            "http://service-host:18080/", "run-1", None
        )

    def test_service_failure_marks_incomplete(self, report_dir, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {
                    "run_id": "run-1",
                    "status": "failed",
                    "error": "boom",
                }
            ),
        )
        scorer = _make_scorer(report_dir)

        score, n_repeats = scorer.score()

        assert score is None
        assert n_repeats == 1
        assert scorer.complete is False

    def test_missing_result_marks_incomplete(self, report_dir, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {"run_id": "run-1", "status": "succeeded"}
            ),
        )
        scorer = _make_scorer(report_dir)

        score, _ = scorer.score()

        assert score is None
        assert scorer.complete is False

    def test_zero_denominator_marks_incomplete(self, report_dir, monkeypatch):
        monkeypatch.setattr(SWEBenchScorer, "_http_json", MagicMock())
        scorer = _make_scorer(report_dir, dataset=_make_dataset(n=0))

        score, _ = scorer.score()

        assert score is None
        assert scorer.complete is False
        SWEBenchScorer._http_json.assert_not_called()

    def test_partial_submission_marks_incomplete(self, report_dir, monkeypatch):
        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            return {
                "run_id": "run-1",
                "status": "succeeded",
                "result": {"resolved_instances": 1, "submitted_instances": 2},
                "artifacts": [],
            }

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        scorer = _make_scorer(report_dir)

        score, _ = scorer.score()

        assert score == pytest.approx(1 / 3)
        assert scorer.complete is False

    def test_decimal_string_result_counters_are_accepted(self, report_dir, monkeypatch):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            lambda *args, **kwargs: {
                "run_id": "run-1",
                "status": "succeeded",
                "result": {
                    "resolved_instances": "2",
                    "submitted_instances": "3",
                },
                "artifacts": [],
            },
        )
        scorer = _make_scorer(report_dir)

        score, _ = scorer.score()

        assert score == pytest.approx(2 / 3)
        assert scorer.complete is True

    @pytest.mark.parametrize(
        "result",
        [
            {"resolved_instances": True, "submitted_instances": 3},
            {"resolved_instances": 1, "submitted_instances": False},
            {"resolved_instances": -1, "submitted_instances": 3},
            {"resolved_instances": 1, "submitted_instances": -3},
            {"resolved_instances": " 1", "submitted_instances": 3},
            {"resolved_instances": "1.0", "submitted_instances": 3},
            {"resolved_instances": 1.0, "submitted_instances": 3},
            {"submitted_instances": 3},
            {"resolved_instances": 1},
            {"resolved_instances": 3, "submitted_instances": 2},
            {"resolved_instances": 1, "submitted_instances": 4},
        ],
    )
    def test_invalid_result_counters_mark_score_incomplete(
        self, report_dir, monkeypatch, result
    ):
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            lambda *args, **kwargs: {
                "run_id": "run-1",
                "status": "succeeded",
                "result": result,
                "artifacts": [],
            },
        )
        scorer = _make_scorer(report_dir)

        assert scorer.score() == (None, 1)
        assert scorer.complete is False

    def test_unsafe_artifact_url_is_not_downloaded(self, report_dir, monkeypatch):
        opener = MagicMock()
        monkeypatch.setattr(swe_bench_mod, "_NO_REDIRECT_OPENER", opener)

        SWEBenchScorer._download_artifact(
            "http://service-host:18080/",
            {"name": "preds.json", "url": "http://evil/preds.json"},
            report_dir,
            "run-1",
        )

        opener.open.assert_not_called()

        redirect = swe_bench_mod.urllib_error.HTTPError(
            "http://service-host:18080/v1/runs/run-1/artifacts/preds.json",
            302,
            "Found",
            Message(),
            io.BytesIO(),
        )
        opener.open.side_effect = redirect
        SWEBenchScorer._download_artifact(
            "http://service-host:18080/",
            {
                "name": "preds.json",
                "url": "/v1/runs/run-1/artifacts/preds.json",
            },
            report_dir,
            "run-1",
        )

        opener.open.assert_called_once()
        assert not (report_dir / "swe_bench_runs" / "run-1" / "preds.json").exists()
        assert (
            swe_bench_mod._RejectRedirectHandler().redirect_request(
                None, None, 302, "Found", Message(), "http://evil/preds.json"
            )
            is None
        )

    def test_downloaded_artifact_is_streamed_and_namespaced(
        self, report_dir, monkeypatch
    ):
        target = report_dir / "swe_bench_runs" / "run-1" / "preds.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"stale")
        response = MagicMock()
        response.__enter__.return_value.read.side_effect = [
            b'{"prediction":',
            b'"patch"}',
            b"",
        ]
        opener = MagicMock()
        opener.open.return_value = response
        monkeypatch.setattr(swe_bench_mod, "_NO_REDIRECT_OPENER", opener)

        SWEBenchScorer._download_artifact(
            "http://service-host:18080/",
            {
                "name": "preds.json",
                "url": "/v1/runs/run-1/artifacts/preds.json",
            },
            report_dir,
            "run-1",
        )

        read_calls = response.__enter__.return_value.read.call_args_list
        assert [call.args for call in read_calls] == [(1024 * 1024,)] * 3
        assert target.read_bytes() == b'{"prediction":"patch"}'
        assert not target.with_suffix(".json.tmp").exists()
        assert not (report_dir / "preds.json").exists()

    def test_interrupted_artifact_download_preserves_existing_target(
        self, report_dir, monkeypatch
    ):
        target = report_dir / "swe_bench_runs" / "run-1" / "preds.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"stale")
        response = MagicMock()
        response.__enter__.return_value.read.side_effect = [
            b"partial",
            OSError("download interrupted"),
        ]
        opener = MagicMock()
        opener.open.return_value = response
        monkeypatch.setattr(swe_bench_mod, "_NO_REDIRECT_OPENER", opener)

        SWEBenchScorer._download_artifact(
            "http://service-host:18080/",
            {
                "name": "preds.json",
                "url": "/v1/runs/run-1/artifacts/preds.json",
            },
            report_dir,
            "run-1",
        )

        assert target.read_bytes() == b"stale"
        assert not target.with_suffix(".json.tmp").exists()

    def test_artifact_result_fallback(self, report_dir, monkeypatch):
        result_path = report_dir / "swe_bench_runs" / "run-1" / "swe_bench_results.json"

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            return {
                "run_id": "run-1",
                "status": "succeeded",
                "artifacts": [
                    {
                        "name": "swe_bench_results.json",
                        "url": "/v1/runs/run-1/artifacts/swe_bench_results.json",
                    }
                ],
            }

        def fake_download(service_url, status, output_dir, auth_token=None):
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({"resolved_instances": 1, "submitted_instances": 3})
            )

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        monkeypatch.setattr(SWEBenchScorer, "_download_artifacts", fake_download)

        score, _ = _make_scorer(report_dir).score()

        assert score == pytest.approx(1 / 3)
        assert json.loads((report_dir / "swe_bench_results.json").read_text()) == {
            "resolved_instances": 1,
            "submitted_instances": 3,
        }

    def test_stale_results_outside_current_run_are_not_used(
        self, report_dir, monkeypatch
    ):
        canonical = report_dir / "swe_bench_results.json"
        canonical.write_text(
            json.dumps({"resolved_instances": 3, "submitted_instances": 3})
        )
        stale_run_result = (
            report_dir / "swe_bench_runs" / "run-old" / "swe_bench_results.json"
        )
        stale_run_result.parent.mkdir(parents=True)
        stale_run_result.write_text(
            json.dumps({"resolved_instances": 3, "submitted_instances": 3})
        )
        monkeypatch.setattr(
            SWEBenchScorer,
            "_http_json",
            classmethod(
                lambda cls, url, **kwargs: {
                    "run_id": "run-current",
                    "status": "succeeded",
                    "artifacts": [],
                }
            ),
        )

        score, _ = _make_scorer(report_dir).score()

        assert score is None
        assert not canonical.exists()
        assert stale_run_result.exists()

    def test_inline_result_atomically_overwrites_downloaded_result(
        self, report_dir, monkeypatch
    ):
        atomic_writes: list[tuple[Path, dict]] = []
        real_atomic_write = swe_bench_mod.atomic_write_bytes

        def spy_atomic_write(path, data):
            atomic_writes.append((path, msgspec.json.decode(data, type=dict)))
            real_atomic_write(path, data)

        def fake_http_json(url, *, method="GET", payload=None, **kwargs):
            return {
                "run_id": "run-1",
                "status": "succeeded",
                "result": {"resolved_instances": 2, "submitted_instances": 3},
                "artifacts": [],
            }

        run_result = report_dir / "swe_bench_runs" / "run-1" / "swe_bench_results.json"

        def fake_download(service_url, status, output_dir, auth_token=None):
            run_result.parent.mkdir(parents=True, exist_ok=True)
            run_result.write_text(
                json.dumps({"resolved_instances": 0, "submitted_instances": 3})
            )

        monkeypatch.setattr(SWEBenchScorer, "_http_json", fake_http_json)
        monkeypatch.setattr(SWEBenchScorer, "_download_artifacts", fake_download)
        monkeypatch.setattr(swe_bench_mod, "atomic_write_bytes", spy_atomic_write)

        score, _ = _make_scorer(report_dir).score()

        canonical = report_dir / "swe_bench_results.json"
        assert score == pytest.approx(2 / 3)
        assert atomic_writes == [
            (
                canonical,
                {"resolved_instances": 2, "submitted_instances": 3},
            )
        ]
        assert not canonical.with_suffix(".json.tmp").exists()
