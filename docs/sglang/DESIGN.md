# SGLang Adapter — Design Spec

> SGLang-specific adapter implementing the same `HttpRequestAdapter` and `SSEAccumulatorProtocol` contracts as the OpenAI path, with SGLang wire format for requests and responses.

**Component specs:** [async_utils](../async_utils/DESIGN.md) · [commands](../commands/DESIGN.md) · [config](../config/DESIGN.md) · [core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) · [endpoint_client](../endpoint_client/DESIGN.md) · [evaluation](../evaluation/DESIGN.md) · [load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md) · [openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) · [profiling](../profiling/DESIGN.md) · **sglang** · [testing](../testing/DESIGN.md) · [utils](../utils/DESIGN.md)

---

## Overview

`sglang/` is a thin adapter implementing the same `HttpRequestAdapter` and
`SSEAccumulatorProtocol` contracts as `openai/`, targeting the SGLang-specific API format.
It is structurally parallel to `openai/` and follows the same patterns.

## Responsibilities

- Format `Query` dicts into SGLang-compatible HTTP request bodies
- Parse SGLang streaming and non-streaming responses into `QueryResult` / `StreamChunk`

## Files

| File             | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `adapter.py`     | `HttpRequestAdapter` implementation for SGLang      |
| `accumulator.py` | `SSEAccumulatorProtocol` for SGLang response format |
| `types.py`       | Python type annotations for SGLang response objects |

## Public Interface

Identical protocols to `openai/` — the adapter implements `dataset_transforms()`,
`encode_query()`, `decode_response()`, and `decode_sse_message()`, while the accumulator
implements `add_chunk()` and `get_final_output()`. The only difference is the wire format of
requests and responses.

## Design Decisions

**Shared protocol, separate implementation**

The `HttpRequestAdapter` and `SSEAccumulatorProtocol` protocols are defined in
`endpoint_client/adapter_protocol.py` and `endpoint_client/accumulator_protocol.py` respectively.
Both `openai/` and `sglang/` implement these protocols independently. `endpoint_client/config.py`
selects the appropriate implementation at construction time based on `api_type`.

This means adding a new API format (e.g. TGI, vLLM native) requires only implementing the two
protocols and registering the implementation in `config.py` — no changes to worker or client code.

## Integration Points

| Component                   | Role                                          |
| --------------------------- | --------------------------------------------- |
| `endpoint_client/config.py` | Selects SGLang adapter when `api_type=SGLANG` |
| `endpoint_client/worker.py` | Same call sites as OpenAI adapter             |
| `core/types.py`             | Output types are identical to OpenAI path     |
