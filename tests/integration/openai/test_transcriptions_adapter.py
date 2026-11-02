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

"""End-to-end transcription run: dataset -> worker -> multipart upload -> WER.

Drives the real HTTP worker pool against a mock /v1/audio/transcriptions server,
so the multipart body, the adapter-declared Content-Type, and the WER scorer are
all exercised on the same path a real Whisper run takes.
"""

import asyncio
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

import msgspec
import pytest
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    LoadPattern,
    LoadPatternType,
    ModelParams,
)
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import (
    APIType,
    Query,
    QueryResult,
    TextModelOutput,
)
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined.whisper_librispeech import (
    WhisperLibriSpeech,
)
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.evaluation.scoring import WERScorer
from inference_endpoint.load_generator.session import BenchmarkSession, PhaseConfig
from inference_endpoint.openai.transcriptions_adapter import (
    OpenAITranscriptionsAdapter,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from .conftest import MockTranscriptionServer, write_wav

MODEL = "openai/whisper-large-v3"
DATASET_NAME = "whisper_librispeech"

# LibriSpeech transcripts are uppercase; the mock returns Whisper-style cased,
# punctuated text so the scorer's normalization is actually exercised.
CORPUS = [
    (
        "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES",
        "Mister Quilter is the apostle of the middle classes.",
    ),
    ("AND WE ARE GLAD TO WELCOME HIS GOSPEL", "And we are glad to welcome his gospel."),
    (
        "HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK IS A GREAT PAINTER",
        "He has grave doubts whether Sir Frederick is a great painter.",
    ),
    ("THE RAINBOW COMES AND GOES", "The rainbow comes and goes."),
]


class _NoOpPublisher:
    def publish(self, event_record: EventRecord) -> None:
        pass

    def flush(self) -> None:
        pass


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """Write an MLCommons-shaped manifest plus WAVs; return (manifest, bytes)."""
    audio_dir = tmp_path / "dev-all-repack"
    entries = []
    audio: dict[str, bytes] = {}
    for i, (transcript, _) in enumerate(CORPUS):
        name = f"{i}.wav"
        # Distinct payload per clip so a mixed-up upload is detectable.
        audio[name] = write_wav(audio_dir / name, bytes([i]) * 512)
        entries.append(
            {
                "transcript": transcript,
                "files": [
                    {"fname": f"dev-all-repack/{name}", "speed": 1, "duration": 29.0}
                ],
                "original_duration": 29.0,
            }
        )
    manifest = tmp_path / "dev-all-repack.json"
    manifest.write_text(json.dumps(entries))
    return manifest, audio


@pytest.fixture
def dataset(tmp_path: Path, corpus) -> Dataset:
    manifest, _ = corpus
    ds = WhisperLibriSpeech.get_dataloader(
        datasets_dir=tmp_path / "cache", manifest=manifest
    )
    ds.load(
        api_type=APIType.OPENAI_TRANSCRIPTIONS,
        model_params=ModelParams(name=MODEL, temperature=0.0),
    )
    return ds


# Mock returns the Whisper-style transcript keyed by uploaded filename.
TRANSCRIPTS = {f"{i}.wav": hyp for i, (_, hyp) in enumerate(CORPUS)}


async def _run(url: str, dataset) -> tuple[dict[str, str], dict[str, int]]:
    loop = asyncio.get_running_loop()
    n_samples = dataset.num_samples()
    responses: dict[str, str] = {}

    def on_complete(result: QueryResult) -> None:
        responses[result.id] = result.get_response_output_string()

    http_config = HTTPClientConfig(
        api_type=APIType.OPENAI_TRANSCRIPTIONS,
        endpoint_urls=[
            urljoin(
                url.rstrip("/") + "/", APIType.OPENAI_TRANSCRIPTIONS.default_route()
            )
        ],
        warmup_connections=0,
        num_workers=2,
    )
    client = await HTTPEndpointClient.create(http_config, loop)
    try:
        session = BenchmarkSession(
            issuer=HttpClientSampleIssuer(client),
            event_publisher=_NoOpPublisher(),
            loop=loop,
            on_sample_complete=on_complete,
        )
        rt_settings = RuntimeSettings(
            metrics.Throughput(100),
            [metrics.Throughput(100)],
            min_issue_duration_ms=0,
            max_issue_duration_ms=60_000,
            n_samples_from_dataset=n_samples,
            n_samples_to_issue=n_samples,
            min_sample_count=1,
            rng_sched=random.Random(1234),
            rng_sample_index=random.Random(1234),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )
        result = await asyncio.wait_for(
            session.run([PhaseConfig("performance", rt_settings, dataset)]),
            timeout=120.0,
        )
    finally:
        await client.shutdown_async()

    assert result.perf_results[0].issued_count == n_samples
    return responses, result.perf_results[0].uuid_to_index


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("mock_transcription_server", [TRANSCRIPTS], indirect=True)
async def test_full_run_uploads_audio_and_scores_wer(
    mock_transcription_server: MockTranscriptionServer,
    dataset,
    corpus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, audio = corpus
    responses, uuid_to_index = await _run(mock_transcription_server.url, dataset)

    # --- every sample completed and carried the right transcript back ---
    assert len(responses) == len(CORPUS)
    for sample_uuid, text in responses.items():
        _, expected_hypothesis = CORPUS[uuid_to_index[sample_uuid]]
        assert text == expected_hypothesis

    # --- the server saw a well-formed upload per sample ---
    assert len(mock_transcription_server.requests) == len(CORPUS)
    seen = {}
    for req in mock_transcription_server.requests:
        assert req["fields"]["model"] == MODEL
        assert req["fields"]["response_format"] == "json"
        assert req["fields"]["temperature"] == "0.0"
        assert "stream" not in req["fields"]
        seen[req["filename"]] = req["audio"]
    # Each clip's bytes arrived intact and with its own filename - a shared
    # multipart boundary must not bleed one upload into another.
    assert seen == audio

    # --- the real scorer over the real run artifacts ---
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "sample_idx_map.json").write_bytes(
        msgspec.json.encode({DATASET_NAME: uuid_to_index})
    )
    encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
    with (report_dir / "events.jsonl").open("wb") as f:
        for sample_uuid, text in responses.items():
            f.write(
                encoder.encode(
                    EventRecord(
                        event_type=SampleEventType.COMPLETE,
                        sample_uuid=sample_uuid,
                        data=TextModelOutput(output=text),
                    )
                )
                + b"\n"
            )

    monkeypatch.setattr(
        WERScorer, "_normalizer", lambda self: EnglishTextNormalizer({})
    )
    scorer = WERScorer(
        DATASET_NAME, dataset, report_dir, ground_truth_column="transcript"
    )
    score, n_repeats = scorer.score()
    breakdown = scorer.score_breakdown()
    assert breakdown is not None

    # Hypotheses differ from the references only in case and punctuation.
    assert score == pytest.approx(100.0)
    assert n_repeats == 1
    assert breakdown["edit_distance"] == 0
    assert breakdown["missing_samples"] == 0
    assert breakdown["reference_words"] == sum(len(r.split()) for r, _ in CORPUS)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_rejects_a_non_multipart_body(
    mock_transcription_server: MockTranscriptionServer, tmp_path: Path
) -> None:
    """Guards the Content-Type wiring: JSON to this route is a 415."""
    url = f"{mock_transcription_server.url}/v1/audio/transcriptions"
    req = urllib.request.Request(
        url,
        data=b'{"model": "m"}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 415


@pytest.mark.integration
def test_dataset_rows_are_request_ready(dataset) -> None:
    """The loaded row is used verbatim as Query.data, so it must encode."""
    row = dataset.load_sample(0)
    assert set(row) == {
        "audio_file",
        "model",
        "language",
        "response_format",
        "temperature",
    }
    assert row["language"] == "en"
    body = OpenAITranscriptionsAdapter.encode_query(Query(id="q", data=row))
    assert Path(row["audio_file"]).read_bytes() in body
