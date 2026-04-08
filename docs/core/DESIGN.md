# Core Types — Design Spec

> Shared `msgspec.Struct` data structures used across transport, endpoint adapters, and metrics, with small helper methods and auto-managed timing fields.

**Component specs:** [async_utils](../async_utils/DESIGN.md) · [commands](../commands/DESIGN.md) · [config](../config/DESIGN.md) · **core** · [dataset_manager](../dataset_manager/DESIGN.md) · [endpoint_client](../endpoint_client/DESIGN.md) · [evaluation](../evaluation/DESIGN.md) · [load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md) · [openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) · [profiling](../profiling/DESIGN.md) · [sglang](../sglang/DESIGN.md) · [testing](../testing/DESIGN.md) · [utils](../utils/DESIGN.md)

---

## Overview

`core/` defines the fundamental data structures passed between all system components. Every other package
depends on these types; they depend on nothing else in the project.

## Responsibilities

- Define the wire format for queries, results, and streaming chunks
- Provide a single source of truth for status and output representation
- Minimize serialization overhead on the hot path

## Key Types

### `Query`

Represents a single inference request issued to an endpoint.

| Field        | Type             | Description                                       |
| ------------ | ---------------- | ------------------------------------------------- |
| `id`         | `str`            | UUID string for result correlation (auto-set)     |
| `data`       | `dict[str, Any]` | Request payload (prompt, model, params, etc.)     |
| `headers`    | `dict[str, str]` | HTTP headers (e.g., authorization)                |
| `created_at` | `float`          | Epoch timestamp when query was created (auto-set) |

The adapter layer (`openai/`, `sglang/`) is responsible for structuring `data` — `Query` itself is format-agnostic.

### `QueryResult`

Represents a completed (success or failure) inference response.

| Field             | Type                             | Description                                                     |
| ----------------- | -------------------------------- | --------------------------------------------------------------- |
| `id`              | `str`                            | Matches originating `Query.id`                                  |
| `response_output` | `TextModelOutput \| str \| None` | Response content (None on error; plain `str` remains supported) |
| `metadata`        | `dict[str, Any]`                 | Additional response metadata (token counts, etc.)               |
| `error`           | `ErrorData \| None`              | Structured error if query failed                                |
| `completed_at`    | `int`                            | Monotonic timestamp in nanoseconds (auto-set)                   |

### `StreamChunk`

Represents one SSE delta from a streaming response.

| Field            | Type             | Description                    |
| ---------------- | ---------------- | ------------------------------ |
| `id`             | `str`            | Matches originating `Query.id` |
| `response_chunk` | `str`            | Incremental token text         |
| `is_complete`    | `bool`           | True for the final chunk       |
| `metadata`       | `dict[str, Any]` | Per-chunk metadata             |

### `TextModelOutput`

Holds the final model response text and optional reasoning trace.

| Field       | Type                             | Description                                                            |
| ----------- | -------------------------------- | ---------------------------------------------------------------------- |
| `output`    | `str \| tuple[str, ...]`         | Decoded text; tuple for streaming accumulation                         |
| `reasoning` | `str \| tuple[str, ...] \| None` | Optional reasoning trace; tuple when accumulated from streaming chunks |

### Supporting Types

`core/types.py` also defines `PromptData` (attached to issued events for token metrics) and
`ErrorData` (structured error payloads used on `QueryResult.error`).

### `QueryStatus`

Enum: `PENDING` → `RUNNING` → `COMPLETED` / `FAILED` / `CANCELLED`

## Design Decisions

**`msgspec.Struct` with `frozen=True`, `array_like=True`, `gc=False`, `omit_defaults=True`**

All four flags are deliberate hot-path optimisations:

- `frozen=True` prevents accidental mutation after creation.
- `array_like=True` serialises to a JSON array (positional fields) rather than a dict, cutting wire size.
- `gc=False` removes the type from GC tracking; structs with no cyclic references don't need it.
- `omit_defaults=True` reduces serialised size for optional fields.

Field mutation is prohibited. Use `msgspec.structs.force_setattr()` only in controlled accumulator code.

**Minimal helper logic on otherwise transport-oriented types**

The core structs are primarily data containers, but they do include small helper behaviors where
the implementation needs them: `QueryResult.completed_at` is auto-set in `__post_init__`,
`TextModelOutput.__str__()` flattens output for reporting, and `TextModelOutput.text_after_first_chunk()`
supports TPOT calculation.

## Serialisation Contract

Types are serialised with `msgspec.json.encode()` and decoded with `msgspec.json.decode()`.
Because `array_like=True`, the wire format is positional:

```
Query  →  ["<id>", {<data>}, {<headers>}, <created_at>]
```

Field order is determined by struct definition order and must not be changed without a migration.

## Integration Points

| Consumer                 | Usage                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `endpoint_client/`       | Creates `Query`; receives `QueryResult` and `StreamChunk`                           |
| `load_generator/`        | Passes `Query` to `SampleIssuer`; routes `QueryResult`/`StreamChunk` to event hooks |
| `async_utils/transport/` | Serialises/deserialises these types over ZMQ IPC                                    |
| `metrics/recorder.py`    | Reads `id` and timing fields for event recording                                    |
| `openai/`, `sglang/`     | Constructs `QueryResult` and `StreamChunk` from API responses                       |
