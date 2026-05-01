# WAN 2.2 Endpoint Client — Design Summary

Client-side changes that add an `inference-endpoint` adapter for the MLPerf WAN 2.2 text-to-video benchmark.

## Overview

WAN 2.2 (T2V-A14B) generates 720×1280 portrait videos at 81 frames / 5 s using 20 denoising steps. The inference server is **trtllm-serve**, exposing `POST /v1/videos/generations`. The new `videogen` module plugs into the existing `HTTPEndpointClient` pipeline via the `api_type` config field — no hot-path code is touched.

```
YAML  ──►  HTTPEndpointClient  ──►  VideoGenAdapter ──►  HTTP Worker (ZMQ)
                                          │                    │
                                          │       POST /v1/videos/generations
                                          ▼                    ▼
                                   QueryResult       trtllm-serve
                                   (metadata holds   (perf: saves to Lustre, returns path
                                    video_path or     accuracy: returns base64 bytes inline)
                                    video_bytes)
```

## `videogen/` module layout

```
videogen/
├── __init__.py     Public exports
├── types.py        Pydantic wire models (VideoPathRequest, VideoPathResponse, VideoPayloadResponse)
└── adapter.py      VideoGenAdapter + VideoGenAccumulator (no-op SSE)
```

The MLPerf prompts dataset ships as plain JSONL at `examples/09_Wan22_VideoGen_Example/wan22_prompts.jsonl` (248 rows; each row carries `prompt`, `negative_prompt` (MLPerf canonical), `sample_id`, `sample_index`). The generic `JsonlLoader` ingests it directly — no workload-specific dataset class is needed.

## Two response formats

The adapter supports both server response formats, selected per-request via `query.data["response_format"]`:

- **`video_path`** (default, perf mode) — server saves the encoded video to Lustre and returns only the path. Avoids 3–5 MB payloads over HTTP + ZMQ per request.
- **`video_bytes`** (accuracy mode) — server returns the base64-encoded H.264/MJPEG payload inline so the accuracy evaluator can score directly.

`decode_response` dispatches on the response body shape: `video_bytes` key present → `VideoPayloadResponse`; otherwise → `VideoPathResponse`. To run accuracy mode, set `response_format: video_bytes` in the per-row dataset data so it is injected into every `query.data`.

## Wire model defaults (`types.py`)

`VideoPathRequest` mirrors trtllm-serve's `VideoGenerationRequest`. All fields carry MLPerf defaults so only `prompt` is required from the dataset.

| Field                 | MLPerf default | Notes                                                                   |
| --------------------- | -------------- | ----------------------------------------------------------------------- |
| `prompt`              | from dataset   | Required                                                                |
| `negative_prompt`     | `None`         | Omitted from JSON when `None`; bundled JSONL carries the canonical text |
| `size`                | `"720x1280"`   | Portrait                                                                |
| `seconds`             | `5.0`          | 81 frames ÷ ~16.2 fps                                                   |
| `fps`                 | `16`           |                                                                         |
| `num_inference_steps` | `20`           |                                                                         |
| `guidance_scale`      | `4.0`          | Primary CFG                                                             |
| `guidance_scale_2`    | `3.0`          | Null-text secondary CFG (two-stage denoising)                           |
| `seed`                | `42`           | Fixed for reproducibility                                               |
| `latent_path`         | `None`         | Optional path to a fixed latent tensor on shared storage                |
| `output_format`       | `"auto"`       | H.264 if ffmpeg available, MJPEG otherwise                              |
| `response_format`     | `"video_path"` | See "Two response formats" above                                        |

Serialization uses `model_dump_json(exclude_none=True)` so `None` fields fall back to server-side defaults.

## Adapter contract (`adapter.py`)

`VideoGenAdapter` implements `HttpRequestAdapter`:

```
encode_query(query)         →  VideoPathRequest JSON bytes
decode_response(bytes, id)  →  QueryResult with metadata={video_path: ...} or {video_bytes: ...}
decode_sse_message(bytes)   →  NotImplementedError (WAN 2.2 is non-streaming)
dataset_transforms(params)  →  []  (no token-level transforms needed)
```

`VideoGenAccumulator` is a no-op `SSEAccumulatorProtocol` to satisfy `HTTPClientConfig`'s type contract. `APIType.VIDEOGEN` is registered in `core/types.py` with `default_route() = "/v1/videos/generations"`.

## Out of scope

| Concern                                  | Where it lives                                         |
| ---------------------------------------- | ------------------------------------------------------ |
| trtllm-serve startup / model loading     | trtllm-serve (separate process)                        |
| Fixed latent loading (`fixed_latent.pt`) | trtllm-serve, optionally per-request via `latent_path` |
| Video storage path on Lustre             | trtllm-serve config                                    |

## Pending

- **`guidance_scale_2` server-side wiring** — the client serializes the field (MLPerf default 3.0); trtllm-serve currently ignores it and uses a single CFG scale.
