# OpenAI Adapter — Design Spec

> Translates internal `Query` objects into OpenAI Chat Completions HTTP requests and parses streaming SSE and non-streaming JSON responses back into `QueryResult`/`StreamChunk`.

**Component specs:** [async_utils](../async_utils/DESIGN.md) · [commands](../commands/DESIGN.md) · [config](../config/DESIGN.md) · [core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) · [endpoint_client](../endpoint_client/DESIGN.md) · [evaluation](../evaluation/DESIGN.md) · [load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md) · **openai** · [plugins](../plugins/DESIGN.md) · [profiling](../profiling/DESIGN.md) · [sglang](../sglang/DESIGN.md) · [testing](../testing/DESIGN.md) · [utils](../utils/DESIGN.md)

---

## Overview

`openai/` adapts the system's internal `Query` type to OpenAI-compatible HTTP requests and
converts OpenAI API responses back into `QueryResult` and `StreamChunk`. It also provides
adapter-specific dataset transforms through the `HttpRequestAdapter` interface.

## Responsibilities

- Format `Query.data` into OpenAI Chat Completions request bodies
- Parse streaming SSE responses (chunked) into `StreamChunk` events
- Parse non-streaming JSON responses into `QueryResult`
- Provide a msgspec-optimised fast path for the hot path

## Component Map

```
Query
        |
        v
HttpRequestAdapter (ABC)
        |
  openai_msgspec_adapter.py   <-- hot path: msgspec encoding, no json.dumps
  openai_adapter.py           <-- general path
        |
        v
raw HTTP request bytes
        |
        v
HTTP response bytes
        |
        v
SSEAccumulatorProtocol (Protocol)
        |
  accumulator.py              <-- assembles StreamChunk stream into QueryResult
        |
        v
QueryResult / StreamChunk
```

## Public Interface

### `HttpRequestAdapter` (ABC, defined in `endpoint_client/adapter_protocol.py`)

```python
class HttpRequestAdapter(ABC):
    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]: ...

    @classmethod
    def encode_query(cls, query: Query) -> bytes: ...

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult: ...

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> str: ...
```

`dataset_transforms()` returns adapter-specific transforms that shape dataset rows into the
expected `Query.data` schema. `encode_query()` serialises a `Query` to HTTP request bytes.
`decode_response()` parses a non-streaming response. `decode_sse_message()` extracts the content
string from a single SSE JSON payload; `parse_sse_chunk()` (concrete, on the base class) iterates
the SSE buffer and calls it repeatedly.

### `SSEAccumulatorProtocol` (protocol, defined in `endpoint_client/accumulator_protocol.py`)

```python
class SSEAccumulatorProtocol(Protocol):
    def __init__(self, query_id: str, stream_all_chunks: bool) -> None: ...
    def add_chunk(self, delta: Any) -> StreamChunk | None: ...
    def get_final_output(self) -> QueryResult: ...
```

Workers construct a fresh accumulator for each streaming request by passing the request ID and
the `stream_all_chunks` mode. `add_chunk()` processes one API-specific SSE delta and returns a
`StreamChunk` when content should be emitted (None otherwise). `get_final_output()` returns the
assembled `QueryResult` after the stream is complete, so state is isolated per request rather
than shared across a connection.

## Key Files

| File                        | Purpose                                                      |
| --------------------------- | ------------------------------------------------------------ |
| `openai_msgspec_adapter.py` | Hot-path adapter; uses msgspec for request encoding          |
| `openai_adapter.py`         | Standard adapter; uses stdlib json                           |
| `accumulator.py`            | Per-request streaming accumulator for OpenAI SSE deltas      |
| `types.py`                  | Python type annotations for OpenAI response objects          |
| `openai_types_gen.py`       | Auto-generated from `openapi.yaml`; do not edit manually     |
| `harmony.py`                | Optional `openai-harmony` integration for compatibility shim |
| `openapi.yaml`              | OpenAI API spec snapshot; excluded from pre-commit           |

## Design Decisions

**msgspec adapter as the default hot path**

`openai_msgspec_adapter.py` encodes requests using `msgspec.json.encode()` rather than
`json.dumps()`. At 50k+ QPS with small request bodies, the encoding time is measurable.
msgspec is 2-5x faster than stdlib json for typical Chat Completions request shapes.

**Fresh accumulator per request**

Workers construct a new accumulator for each streaming request. This keeps the accumulator
interface small (`add_chunk()` / `get_final_output()`) and avoids having to manage explicit
reset semantics across reused connections.

**`openai_types_gen.py` is auto-generated**

OpenAI type definitions are generated from the official OpenAPI spec. Manual edits would be
overwritten on regeneration. The file is excluded from ruff and pre-commit.

## Integration Points

| Component                   | Role                                                    |
| --------------------------- | ------------------------------------------------------- |
| `endpoint_client/worker.py` | Calls `encode_query()` and `accumulator.add_chunk()`    |
| `endpoint_client/config.py` | Selects `openai_msgspec_adapter` when `api_type=OPENAI` |
| `core/types.py`             | `StreamChunk`, `QueryResult` are the output types       |
