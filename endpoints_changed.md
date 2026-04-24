# WAN 2.2 Endpoint Client — Design Summary

This document describes the changes made to the `inference-endpoint` library to support the MLPerf WAN 2.2 text-to-video benchmark workload.

---

## Overview

WAN 2.2 (T2V-A14B) generates 720×1280 portrait videos at 81 frames / 5 s using 20 denoising steps. The inference server is **trtllm-serve**, which exposes a video generation endpoint at `POST /v1/videos/generations`.

The client-side changes add a new `wan22` module that plugs into the existing `HTTPEndpointClient` pipeline without touching any hot-path code.

---

## Architecture

```
MLPerf Harness / Benchmark CLI
          │
          │  YAML config  (api_type: wan22)
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
          │  saves video to Lustre                                │
          │  returns VideoPathResponse JSON                       │
          ▼                                                       │
     VideoGenAdapter  ◄──────────────────────────────────────────────┘
          │  decode_response()
          │  QueryResult(metadata={video_path: ...})
          ▼
  MetricsReporter / MLPerf harness
```

**Key design decision:** `response_format=video_path` — the server writes the encoded video to Lustre and returns only the file path. This avoids transferring ~300 MB video bytes over HTTP and ZMQ per request.

---

## New Files

```
wan22/
├── __init__.py     Public exports
├── types.py        Pydantic wire models: VideoPathRequest, VideoPathResponse, HealthResponse
├── adapter.py      VideoGenAdapter (HttpRequestAdapter) + VideoGenAccumulator (no-op SSE)
└── dataset.py      VideoGenDataset — loads MLPerf prompt JSONL, injects fixed latent path
```

---

## Components

### `types.py` — Wire Models

`VideoPathRequest` mirrors trtllm-serve's `VideoGenerationRequest`. All fields carry MLPerf defaults so only `prompt` is required from the dataset.

| Field | MLPerf value | Notes |
|---|---|---|
| `prompt` | from dataset | Required |
| `negative_prompt` | `None` | Omitted from JSON when absent; server uses its own default |
| `size` | `"720x1280"` | Portrait orientation |
| `seconds` | `5.0` | 81 frames ÷ ~16.2 fps |
| `fps` | `16` | |
| `num_inference_steps` | `20` | |
| `guidance_scale` | `4.0` | Primary CFG scale |
| `guidance_scale_2` | `3.0` | Null-text secondary CFG (two-stage denoising) |
| `seed` | `42` | Fixed for reproducibility |
| `output_format` | `"auto"` | H.264 if ffmpeg available, MJPEG otherwise |
| `response_format` | `"video_path"` | Always; avoid byte payload |
| `latent_path` | `None` | Optional path to `fixed_latent.pt` on shared storage |

`VideoPathResponse` carries `video_id` and `video_path` returned by the server.

Serialization uses `model_dump_json(exclude_none=True)` so `None` fields are omitted from the request body.

### `adapter.py` — Request/Response Adapter

`VideoGenAdapter` implements `HttpRequestAdapter`:

```
encode_query(query)  →  VideoPathRequest JSON bytes
decode_response(bytes, id)  →  QueryResult(metadata={video_path: ...})
decode_sse_message(bytes)   →  NotImplementedError  (WAN 2.2 is non-streaming)
dataset_transforms(params)  →  []  (no token-level transforms needed)
```

`VideoGenAccumulator` is a no-op `SSEAccumulatorProtocol` implementation required by `HTTPClientConfig`. WAN 2.2 uses synchronous HTTP POST/response — there is no SSE stream to accumulate.

`APIType.WAN22` is registered in `core/types.py` with `default_route() = "/v1/videos/generations"`.

### `dataset.py` — VideoGenDataset

Loads a JSONL prompt file (248 MLPerf prompts). Injects three fields into every sample:

- `prompt` — from file
- `negative_prompt` — MLPerf canonical string (default); overridable
- `latent_path` — absolute path to `fixed_latent.pt` (optional); passed through to `VideoPathRequest`

```
JSONL file  ──►  VideoGenDataset.load()  ──►  sample dict
                                              ├── prompt
                                              ├── negative_prompt  (MLPerf canonical)
                                              └── latent_path      (if configured)
                                                        │
                                                        ▼
                                             VideoGenAdapter.encode_query()
                                                        │
                                                        ▼
                                             VideoPathRequest JSON
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

The wan22 module plugs into the existing pipeline at the `api_type` config field. No hot-path code was modified.

```yaml
# offline_wan22.yaml
endpoint_config:
  api_type: wan22          # → selects VideoGenAdapter
  endpoints:
    - http://localhost:8000
```

`HTTPEndpointClient` resolves `api_type: wan22` → `VideoGenAdapter` + `VideoGenAccumulator` via the existing adapter registry.

---

## What Is NOT In This Module

| Concern | Where it lives |
|---|---|
| trtllm-serve startup / model loading | trtllm-serve (separate process) |
| Fixed latent loading (`fixed_latent.pt`) | trtllm-serve `pipeline_wan.py` at startup, or per-request via `latent_path` (pending Task 2 in plan) |
| Video storage path on Lustre | trtllm-serve config |
| `guidance_scale_2` wiring on server side | Pending — see `wan22-trtllm-plan.md` Task 1 |

---

## Pending

See `docs/superpowers/plans/wan22-trtllm-plan.md` for the remaining trtllm-serve changes:

- **Task 1** — Wire `guidance_scale_2` through trtllm HTTP API (client already sends it; server ignores it today)
- **Task 2** — Wire per-request `latent_path` through trtllm HTTP API
- **Task 3** — Fix `negative_prompt` default (`""` → `None`) in `VideoPathRequest` and `VideoGenDataset`
