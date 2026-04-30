# WAN 2.2 Endpoint Client — Design Summary

This document describes the changes made to the `inference-endpoint` library to support the MLPerf WAN 2.2 text-to-video benchmark workload.

---

## Overview

WAN 2.2 (T2V-A14B) generates 720×1280 portrait videos at 81 frames / 5 s using 20 denoising steps. The inference server is **trtllm-serve**, which exposes a video generation endpoint at `POST /v1/videos/generations`.

The client-side changes add a new `videogen` module that plugs into the existing `HTTPEndpointClient` pipeline without touching any hot-path code.

---

## Architecture

```
MLPerf Harness / Benchmark CLI
          │
          │  YAML config  (api_type: videogen)
          ▼
  HTTPEndpointClient
          │
          │  selects adapter via APIType.WAN22
          ▼
     VideoGenAdapter  ──────────────────────────────────────────────┐
          │  encode_query()                                       │
          │  VideoPathRequest JSON                                │
          ▼                                                       │
  HTTP Worker (ZMQ)                                              │
          │                                                       │
          │  POST /v1/videos/generations                          │
          ▼                                                       │
     trtllm-serve                                                 │
          │  perf mode:     saves video to Lustre, returns path   │
          │  accuracy mode: returns base64 video bytes inline     │
          ▼                                                       │
     VideoGenAdapter  ◄──────────────────────────────────────────────┘
          │  decode_response()  (dispatches on response shape)
          │  QueryResult(metadata={video_path | video_bytes: ...})
          ▼
  MetricsReporter / MLPerf harness
```

**Key design decision:** the adapter supports two `response_format` values selected per-request via `query.data["response_format"]`:

- **`video_path` (default, perf mode)** — server writes the encoded video to Lustre and returns only the file path. Avoids 3–5 MB payloads over HTTP + ZMQ per request.
- **`video_bytes` (accuracy mode)** — server returns the base64-encoded H.264/MJPEG payload inline so the accuracy evaluator can score the video content directly.

`VideoGenAdapter.decode_response` dispatches on the response shape: if `video_bytes` is present the body is parsed as `VideoPayloadResponse`; otherwise it is parsed as `VideoPathResponse`.

---

## New Files

```
videogen/
├── __init__.py     Public exports
├── types.py        Pydantic wire models: VideoPathRequest,
│                   VideoPathResponse, VideoPayloadResponse, HealthResponse
├── adapter.py      VideoGenAdapter (HttpRequestAdapter) + VideoGenAccumulator (no-op SSE)
└── dataset.py      VideoGenDataset — loads MLPerf prompt JSONL, injects negative
                    prompt and optional latent path
```

---

## Components

### `types.py` — Wire Models

`VideoPathRequest` mirrors trtllm-serve's `VideoGenerationRequest`. All fields carry MLPerf defaults so only `prompt` is required from the dataset.

| Field                 | MLPerf value   | Notes                                                                  |
| --------------------- | -------------- | ---------------------------------------------------------------------- |
| `prompt`              | from dataset   | Required                                                               |
| `negative_prompt`     | `None`         | Omitted from JSON when absent; server uses its own default             |
| `size`                | `"720x1280"`   | Portrait orientation                                                   |
| `seconds`             | `5.0`          | 81 frames ÷ ~16.2 fps                                                  |
| `fps`                 | `16`           |                                                                        |
| `num_inference_steps` | `20`           |                                                                        |
| `guidance_scale`      | `4.0`          | Primary CFG scale                                                      |
| `guidance_scale_2`    | `3.0`          | Null-text secondary CFG (two-stage denoising)                          |
| `seed`                | `42`           | Fixed for reproducibility                                              |
| `latent_path`         | `None`         | Optional path to a fixed latent tensor on shared storage               |
| `output_format`       | `"auto"`       | H.264 if ffmpeg available, MJPEG otherwise                             |
| `response_format`     | `"video_path"` | Default = perf mode (Lustre path); set to `"video_bytes"` for accuracy |

Two response models cover the two `response_format` values:

- `VideoPathResponse` — `video_id`, `video_path` (perf mode).
- `VideoPayloadResponse` — `video_id`, `video_bytes` (accuracy mode; base64-encoded).

Serialization uses `model_dump_json(exclude_none=True)` so `None` fields are omitted from the request body.

### `adapter.py` — Request/Response Adapter

`VideoGenAdapter` implements `HttpRequestAdapter`:

```
encode_query(query)         →  VideoPathRequest JSON bytes
                                (response_format defaults to "video_path";
                                 override via query.data["response_format"])
decode_response(bytes, id)  →  QueryResult with metadata={video_path: ...}
                                or metadata={video_bytes: ...} depending on
                                the response shape
decode_sse_message(bytes)   →  NotImplementedError (WAN 2.2 is non-streaming)
dataset_transforms(params)  →  []  (no token-level transforms needed)
```

`VideoGenAccumulator` is a no-op `SSEAccumulatorProtocol` implementation required by `HTTPClientConfig`. WAN 2.2 uses synchronous HTTP POST/response — there is no SSE stream to accumulate.

`APIType.WAN22` is registered in `core/types.py` with `default_route() = "/v1/videos/generations"`.

### `dataset.py` — VideoGenDataset

Loads a prompt text file (one prompt per non-blank line; the MLPerf dataset bundled at `examples/09_Wan22_VideoGen_Example/wan22_prompts.jsonl` has 248 prompts). Injects up to two extra fields into every sample:

- `prompt` — from file
- `negative_prompt` — MLPerf canonical string by default; pass `negative_prompt=None` to omit
- `latent_path` — optional absolute path to a pre-computed latent tensor (`fixed_latent.pt`); omitted when not configured

```
JSONL file  ──►  VideoGenDataset.load()  ──►  sample dict
                                              ├── prompt
                                              ├── negative_prompt  (MLPerf canonical, optional)
                                              └── latent_path      (optional)
                                                        │
                                                        ▼
                                             VideoGenAdapter.encode_query()
                                                        │
                                                        ▼
                                             VideoPathRequest JSON
                                             (response_format=video_path by default)
```

The MLPerf canonical negative prompt:

```
vivid colors, overexposed, static, blurry details, subtitles, style,
work of art, painting, picture, still, overall grayish, worst quality,
low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn
hands, poorly drawn face, deformed, disfigured, deformed limbs, fused
fingers, static image, cluttered background, three legs, many people in
the background, walking backwards
```

---

## Integration Point

The videogen module plugs into the existing pipeline at the `api_type` config field. No hot-path code was modified.

```yaml
# offline_wan22.yaml
endpoint_config:
  api_type: videogen # → selects VideoGenAdapter
  endpoints:
    - http://localhost:8000
```

`HTTPEndpointClient` resolves `api_type: videogen` → `VideoGenAdapter` + `VideoGenAccumulator` via the existing adapter registry.

To run accuracy mode (server returns video bytes inline), set `response_format: video_bytes` in the dataset config so it is injected into every `query.data`.

---

## What Is NOT In This Module

| Concern                                  | Where it lives                                         |
| ---------------------------------------- | ------------------------------------------------------ |
| trtllm-serve startup / model loading     | trtllm-serve (separate process)                        |
| Fixed latent loading (`fixed_latent.pt`) | trtllm-serve, optionally per-request via `latent_path` |
| Video storage path on Lustre             | trtllm-serve config                                    |
| `guidance_scale_2` wiring on server side | Pending — see "Pending" below                          |

---

## Pending

- **`guidance_scale_2` server-side wiring** — the client already serializes `guidance_scale_2` (MLPerf default 3.0); trtllm-serve currently ignores the field and uses a single CFG scale. Tracking on the server side.
