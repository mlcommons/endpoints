# trtllm-serve Design Docs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the WAN2.2 MLPerf text-to-video benchmark stack: the endpoints client module is done; this plan covers the remaining trtllm-serve API changes needed to pass MLPerf-required inference parameters end-to-end.

**Architecture:** The endpoints client (`wan22/`) POSTs `VideoPathRequest` JSON directly to trtllm-serve's `/v1/videos/generations` with `response_format=video_path`. The server saves the video to Lustre and returns the path — avoiding large byte payloads. The missing piece is that trtllm-serve's `VideoGenerationRequest` does not yet accept `guidance_scale_2` or `latent_path`, so those fields are silently dropped at the HTTP layer.

### Request Flow

```
endpoints client                         trtllm-serve (inference server)
─────────────────                        ────────────────────────────────────────────
VideoGenDataset
  └─ prompt                ──────────►  VideoGenerationRequest   (openai_protocol.py)
  └─ negative_prompt                          │
  └─ guidance_scale_2  ✗ (dropped today)      │  parse_visual_gen_params()
  └─ latent_path       ✗ (dropped today)      │  (visual_gen_utils.py)
                                              ▼
                                        VisualGenParams          (visual_gen.py)
                                              │
                                              │  WanPipeline.infer(req)
                                              ▼
                                        pipeline_wan.forward()   (pipeline_wan.py)
                                              │
                                   ┌──────────┴──────────┐
                                   │                     │
                             fixed_latent           guidance_scale_2
                          (startup config)          (falls back to
                                                     guidance_scale today)
                                   │
                                   ▼
                              video saved to Lustre
                                   │
POST response  ◄─────────────────  │  {"video_id": ..., "video_path": ...}
```

### After This Plan

```
endpoints client                         trtllm-serve (inference server)
─────────────────                        ────────────────────────────────────────────
  └─ guidance_scale_2=3.0  ──────────►  VideoGenerationRequest  +guidance_scale_2
  └─ latent_path=<pt file> ──────────►                          +latent_path
                                              │
                                              │  parse_visual_gen_params()
                                              ▼
                                        VisualGenParams          +latent_path
                                              │
                                              ▼
                                        pipeline_wan.forward()   +latent_path param
                                              │
                                   ┌──────────┴──────────┐
                                   │                     │
                          per-request latent        guidance_scale_2=3.0
                          (overrides fixed_latent)   (two-stage denoising)
```

### trtllm-serve Internal Layers

```
  HTTP layer          Mapper                  Params            Pipeline
  ──────────          ──────                  ──────            ────────
  openai_             visual_gen_             visual_           pipeline_
  protocol.py         utils.py               gen.py            wan.py
  ┌────────────┐      ┌──────────────────┐   ┌─────────────┐   ┌────────────────┐
  │ Video      │      │ parse_visual_    │   │ VisualGen   │   │ WanPipeline    │
  │ Generation │ ───► │ gen_params()     │──►│ Params      │──►│ .infer(req)    │
  │ Request    │      │                  │   │             │   │ .forward(...)  │
  └────────────┘      └──────────────────┘   └─────────────┘   └────────────────┘
  Pydantic model      maps HTTP fields        plain dataclass   runs the model
  (what we add to)    (what we wire in)       (what we extend)  (what we update)
```

**Tech Stack:** Python 3.12, Pydantic v2 (trtllm `OpenAIBaseModel`), PyTorch (`torch.load` for latent tensor), FastAPI (trtllm-serve), pytest (endpoints unit tests).

---

## Key Invariants

All trtllm files live on the inference server. `trtllm/` refers to the `tensorrt_llm/` package directory inside the TensorRT-LLM checkout.

- `VideoGenerationRequest` inherits `OpenAIBaseModel` (Pydantic). Use `Optional[T] = Field(default=None, ...)` for new fields.
- `VisualGenParams` is a plain Python dataclass (not Pydantic). Just add `field: Optional[type] = None`.
- `guidance_scale` and `guidance_rescale` mappings in `parse_visual_gen_params` are in the **outer** block (applies to all request types). Use `hasattr` guard when adding to this block — `ImageGenerationRequest` and `ImageEditRequest` don't have video-only fields.
- `pipeline_wan.forward()` already has `guidance_scale_2: Optional[float] = None` in its signature. It does NOT have `latent_path` yet.
- `self.fixed_latent` (loaded at startup from model config) stays valid. Per-request `latent_path` overrides it but does not replace it.

---

## Task 1: Wire `guidance_scale_2` through trtllm HTTP API

`VisualGenParams` already has `guidance_scale_2` and `pipeline_wan.forward()` already uses it. Only the HTTP model and the parameter mapper are missing.

**Files:**
- Modify: `openai_protocol.py` (after `guidance_scale` field, ~line 1385)
- Modify: `visual_gen_utils.py` (after `guidance_scale` mapping, ~line 27)

- [ ] **Step 1: Verify current state**

```bash
# Run on inference server:
grep -n "guidance_scale" trtllm/serve/openai_protocol.py | grep -v rescale | head -5
```

Expected: `guidance_scale` field present, no `guidance_scale_2`.

- [ ] **Step 2: Add `guidance_scale_2` to `VideoGenerationRequest`**

Find the `guidance_scale` / `guidance_rescale` block in `openai_protocol.py` (~line 1380):

```python
    guidance_scale: Optional[float] = Field(
        default=None, description="Classifier-free guidance scale.")
    guidance_scale_2: Optional[float] = Field(
        default=None,
        description=(
            "Secondary guidance scale for two-stage denoising (WAN 2.2 only). "
            "Applied during the second denoising stage when boundary_ratio is set. "
            "MLPerf standard: 3.0."
        ),
    )
    guidance_rescale: Optional[float] = Field(
        default=None, description="Classifier-free guidance rescale.")
```

- [ ] **Step 3: Wire `guidance_scale_2` in `parse_visual_gen_params`**

In `visual_gen_utils.py`, after the existing `guidance_scale` mapping:

```python
    if request.guidance_scale is not None:
        params.guidance_scale = request.guidance_scale
    if hasattr(request, "guidance_scale_2") and request.guidance_scale_2 is not None:
        params.guidance_scale_2 = request.guidance_scale_2
    if request.guidance_rescale is not None:
        params.guidance_rescale = request.guidance_rescale
```

- [ ] **Step 4: Verify with import test**

```bash
# Run on inference server:
cd trtllm && python3 -c '
from tensorrt_llm.serve.openai_protocol import VideoGenerationRequest
from tensorrt_llm.serve.visual_gen_utils import parse_visual_gen_params
req = VideoGenerationRequest(prompt="test", guidance_scale=4.0, guidance_scale_2=3.0, fps=16, seconds=5.0)
params = parse_visual_gen_params(req, id="test")
assert params.guidance_scale == 4.0
assert params.guidance_scale_2 == 3.0
print("OK: guidance_scale_2 wired correctly")
'
```

Expected: `OK: guidance_scale_2 wired correctly`

---

## Task 2: Add per-request `latent_path` through full trtllm stack

Four files: HTTP model → param mapper → `VisualGenParams` → `pipeline_wan.forward()`.

**Files:**
- Modify: `openai_protocol.py` (add field to `VideoGenerationRequest`)
- Modify: `visual_gen_utils.py` (map in `VideoGenerationRequest` branch)
- Modify: `visual_gen.py` (add `latent_path` to `VisualGenParams`)
- Modify: `pipeline_wan.py` (param in `forward`/`infer`, latent prep block)

- [ ] **Step 1: Add `latent_path` to `VisualGenParams`**

In `visual_gen.py`, add to the Wan-specific section (~line 466):

```python
    # Wan-specific parameters
    guidance_scale_2: Optional[float] = None
    boundary_ratio: Optional[float] = None
    last_image: Optional[str] = None
    latent_path: Optional[str] = None  # Per-request fixed latent tensor path (MLPerf)
```

- [ ] **Step 2: Add `latent_path` field to `VideoGenerationRequest`**

In `openai_protocol.py`, add after the `response_format` field in `VideoGenerationRequest`:

```python
    latent_path: Optional[str] = Field(
        default=None,
        description=(
            "Absolute path to a pre-computed latent tensor (.pt file) on shared "
            "storage accessible to the server (e.g. Lustre). When provided, the "
            "server uses this tensor as the initial denoising noise instead of "
            "sampling random noise. MLPerf uses a fixed latent for reproducibility."
        ),
    )
```

- [ ] **Step 3: Map `latent_path` in `parse_visual_gen_params`**

In `visual_gen_utils.py`, inside the `isinstance(request, VideoGenerationRequest)` branch, after the `seed` mapping:

```python
        if request.seed is not None:
            params.seed = int(request.seed)

        if request.latent_path is not None:
            params.latent_path = request.latent_path
```

- [ ] **Step 4: Verify param mapper**

```bash
# Run on inference server:
cd trtllm && python3 -c '
from tensorrt_llm.serve.openai_protocol import VideoGenerationRequest
from tensorrt_llm.serve.visual_gen_utils import parse_visual_gen_params
req = VideoGenerationRequest(prompt="test", fps=16, seconds=5.0, latent_path="/tmp/latent.pt")
params = parse_visual_gen_params(req, id="test")
assert params.latent_path == "/tmp/latent.pt", f"got {params.latent_path}"
print("OK: latent_path mapped to VisualGenParams")
'
```

Expected: `OK: latent_path mapped to VisualGenParams`

- [ ] **Step 5: Add `latent_path` to `pipeline_wan.forward()` signature**

Find the `forward()` method (~line 345). Add `latent_path: Optional[str] = None` after `max_sequence_length`:

```python
    def forward(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_scale_2: Optional[float] = None,
        boundary_ratio: Optional[float] = None,
        seed: int = 42,
        max_sequence_length: int = 512,
        latent_path: Optional[str] = None,
    ):
```

- [ ] **Step 6: Update latent preparation block in `forward()`**

Find the "Prepare Latents" block (~line 409):

```python
        # Prepare Latents
        if self.fixed_latent is not None:
            latents = self.fixed_latent.to(device=self.device, dtype=self.dtype)
        else:
            latents = self._prepare_latents(batch_size, height, width, num_frames, generator)
```

Replace with:

```python
        # Prepare Latents
        # Priority: per-request latent_path > startup fixed_latent > random noise
        if latent_path is not None:
            per_req_latent = torch.load(
                latent_path, map_location=self.device, weights_only=True
            )
            latents = per_req_latent.to(device=self.device, dtype=self.dtype)
        elif self.fixed_latent is not None:
            latents = self.fixed_latent.to(device=self.device, dtype=self.dtype)
        else:
            latents = self._prepare_latents(batch_size, height, width, num_frames, generator)
```

- [ ] **Step 7: Update `infer()` to pass `latent_path`**

Find `infer(self, req)` (~line 319). Add `latent_path=req.latent_path`:

```python
    def infer(self, req):
        return self.forward(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            guidance_scale_2=req.guidance_scale_2,
            boundary_ratio=req.boundary_ratio,
            seed=req.seed,
            max_sequence_length=req.max_sequence_length,
            latent_path=req.latent_path,
        )
```

- [ ] **Step 8: Verify pipeline signature**

```bash
# Run on inference server:
cd trtllm && python3 -c '
import inspect
from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline
sig = inspect.signature(WanPipeline.forward)
assert "latent_path" in sig.parameters
print("OK: forward() has latent_path")
'
```

Expected: `OK: forward() has latent_path`


---

## Task 3: Fix `negative_prompt` in endpoints client

`VideoPathRequest.negative_prompt` defaults to `""`, which trtllm-serve sets unconditionally, overriding any model default. Fix: use `None` as default (omitted from JSON via `exclude_none=True`), and inject the MLPerf canonical negative prompt via `VideoGenDataset`.

**Files:**
- Modify: `wan22/types.py`
- Modify: `wan22/adapter.py`
- Modify: `wan22/dataset.py`
- Modify: `test_adapter.py`

- [ ] **Step 1: Write failing tests**

In `test_adapter.py`, add to `TestVideoGenAdapter`:

```python
    def test_encode_query_omits_negative_prompt_when_absent(self):
        query = Query(id="q1", data={"prompt": "test"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload.get("negative_prompt") is None

    def test_encode_query_sends_negative_prompt_when_set(self):
        query = Query(id="q1", data={"prompt": "test", "negative_prompt": "blurry"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["negative_prompt"] == "blurry"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd <endpoints-repo>
pytest tests/unit/wan22/test_adapter.py::TestVideoGenAdapter::test_encode_query_omits_negative_prompt_when_absent -xvs
```

Expected: FAIL — payload currently has `negative_prompt: ""`.

- [ ] **Step 3: Fix `VideoPathRequest.negative_prompt` in `types.py`**

Change `negative_prompt: str = ""` (line 31) to:

```python
    negative_prompt: str | None = Field(
        default=None,
        description="Text describing what to avoid. None = let server default.",
    )
```

- [ ] **Step 4: Fix `encode_query` in `adapter.py`**

```python
        req = VideoPathRequest(
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt"),
            size=data.get("size", "720x1280"),
            seconds=data.get("seconds", 5.0),
            fps=data.get("fps", 16),
            num_inference_steps=data.get("num_inference_steps", 20),
            guidance_scale=data.get("guidance_scale", 4.0),
            guidance_scale_2=data.get("guidance_scale_2", 3.0),
            seed=data.get("seed", 42),
            output_format=data.get("output_format", "auto"),
            response_format="video_path",
            latent_path=data.get("latent_path"),
        )
        return req.model_dump_json(exclude_none=True).encode()
```

- [ ] **Step 5: Inject MLPerf canonical negative prompt in `dataset.py`**

```python
_MLPERF_NEGATIVE_PROMPT = (
    "vivid colors, overexposed, static, blurry details, subtitles, style, "
    "work of art, painting, picture, still, overall grayish, worst quality, "
    "low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
    "poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
    "static image, cluttered background, three legs, many people in the background, "
    "walking backwards"
)


class VideoGenDataset(Dataset, dataset_id="wan22_mlperf"):
    def __init__(
        self,
        prompts_path: Path | str,
        negative_prompt: str | None = _MLPERF_NEGATIVE_PROMPT,
        latent_path: Path | str | None = None,
    ) -> None:
```

Also update `load()` to omit the key when `negative_prompt is None`:

```python
    def load(self, **kwargs: Any) -> None:
        assert self.dataframe is not None
        self.data = [
            {
                "prompt": row["prompt"],
                **({"negative_prompt": self.negative_prompt} if self.negative_prompt is not None else {}),
                "sample_id": str(i),
                "sample_index": i,
            }
            for i, row in self.dataframe.iterrows()
        ]
```

- [ ] **Step 6: Run all wan22 unit tests**

```bash
pytest -m unit tests/unit/wan22/ -v
```

Expected: All pass.

- [ ] **Step 7: Run pre-commit**

```bash
pre-commit run --all-files
```

Expected: All hooks pass.

---

## Task 4: E2E smoke test

Verify all new fields flow end-to-end through a running trtllm-serve instance.

- [ ] **Step 1: Create smoke test script on the inference server**

```bash
# Run on inference server:
cat > ~/test_trtllm_wan22_mlperf.py << 'PYEOF'
#!/usr/bin/env python3
import argparse, json, urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
parser.add_argument("--latent", default=None, help="Path to fixed_latent.pt")
args = parser.parse_args()

LATENT = args.latent
NEG = (
    "vivid colors, overexposed, static, blurry details, subtitles, style, "
    "work of art, painting, picture, still, overall grayish, worst quality, "
    "low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
    "poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
    "static image, cluttered background, three legs, many people in the background, "
    "walking backwards"
)

payload = {
    "prompt": "a golden retriever running on a beach",
    "negative_prompt": NEG,
    "size": "720x1280",
    "seconds": 5.0,
    "fps": 16,
    "num_inference_steps": 20,
    "guidance_scale": 4.0,
    "guidance_scale_2": 3.0,
    "seed": 42,
    "output_format": "mp4",
    "response_format": "video_path",
    "latent_path": LATENT,
}

url = f"{args.url}/v1/videos/generations"
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
print(f"POST {url}")
with urllib.request.urlopen(req, timeout=600) as resp:
    data = json.loads(resp.read())
    print(f"Response: {json.dumps(data, indent=2)}")
    assert "video_path" in data
    print(f"OK: video saved to {data['video_path']}")
PYEOF
```

- [ ] **Step 2: Start trtllm-serve and run the test**

Start the server in a separate terminal, then:

```bash
# Run on inference server:
python ~/test_trtllm_wan22_mlperf.py --url http://localhost:8000 --latent /path/to/fixed_latent.pt
```

Expected output ends with `OK: video saved to <video_path>`.

If you get `422 Unprocessable Entity`, the server doesn't recognize the new fields — restart after code changes.
