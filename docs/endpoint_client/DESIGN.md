# Endpoint Client — Design Spec

> Multi-process HTTP worker pool that sends queries to inference endpoints over persistent connections and delivers responses back to the load generator with minimal latency overhead.

**Component specs:** [async_utils](../async_utils/DESIGN.md) · [commands](../commands/DESIGN.md) · [config](../config/DESIGN.md) · [core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) · **endpoint_client** · [evaluation](../evaluation/DESIGN.md) · [load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md) · [openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) · [profiling](../profiling/DESIGN.md) · [sglang](../sglang/DESIGN.md) · [testing](../testing/DESIGN.md) · [utils](../utils/DESIGN.md)

---

## Overview

`endpoint_client/` sends queries to remote inference endpoints and delivers responses back to the
load generator. It uses a **multi-process worker pool** communicating over ZMQ IPC to bypass the
GIL and sustain high request rates.

This file is the primary component-level design spec for the endpoint client. For deeper
implementation detail on the connection pool, worker internals, SSE handling, and performance
analysis, see the companion deep-dive document
[ENDPOINT_CLIENT.md](../ENDPOINT_CLIENT.md).

For detailed CPU affinity configuration and tuning parameters, see
[CLIENT_PERFORMANCE_TUNING.md](../CLIENT_PERFORMANCE_TUNING.md) and
[PERF_ARCHITECTURE.md](../PERF_ARCHITECTURE.md).

## Responsibilities

- Spawn and manage a pool of worker processes
- Route outbound queries to workers via round-robin
- Deliver inbound responses (`QueryResult`, `StreamChunk`) to callers
- Manage persistent TCP connections per worker
- Apply CPU affinity for NUMA-aware placement

## Architecture

```
Main Process
┌─────────────────────────────────────────┐
│  HTTPEndpointClient                     │
│    ├── uvloop event loop                │
│    └── WorkerManager                    │
│          └── WorkerPoolTransport (ZMQ) │
└──────────────┬──────────────────────────┘
               │ ZMQ IPC (inproc/ipc)
    ┌──────────┴──────────┐
    │                     │
Worker 0              Worker N
┌──────────┐        ┌──────────┐
│ uvloop   │        │ uvloop   │
│ Worker   │        │ Worker   │
│ ConnPool │        │ ConnPool │
└──────┬───┘        └──────┬───┘
       │ HTTP/1.1           │
       └─────────┬──────────┘
             Endpoint
```

## Public Interface

### `HTTPEndpointClient`

```python
class HTTPEndpointClient:
    def __init__(self, config: HTTPClientConfig, ...) -> None

    # Sample issuer interface
    def issue(self, query: Query) -> None          # non-blocking, round-robin
    def shutdown(self) -> None                     # synchronous shutdown

    # Response retrieval (use one pattern per call-site)
    def poll(self) -> QueryResult | StreamChunk | None   # non-blocking
    async def recv(self) -> QueryResult | StreamChunk | None  # blocking async
    def drain(self) -> list[QueryResult | StreamChunk]        # batch
```

### `HTTPClientConfig`

```python
class HTTPClientConfig(WithUpdatesMixin, BaseModel):
    endpoint_urls: list[str]
    api_type: APIType = APIType.OPENAI
    api_key: str | None = None
    num_workers: int = -1               # -1 = auto (NUMA-aware)
    worker_gc_mode: Literal["disabled", "relaxed", "system"] = "relaxed"
    max_idle_time: float = 4.0
    warmup_connections: int = -1        # -1 = auto (50% of max_connections)
    max_connections: int = -1           # -1 = bound by ephemeral port limit
    stream_all_chunks: bool = False     # expose every SSE chunk (perf cost)
    cpu_affinity: AffinityPlan | None = None
```

## Data Flow

**Outbound (issue → endpoint):**

```
HTTPEndpointClient.issue(query)
  → select next worker (round-robin index)
  → serialize Query with msgspec.json
  → ZMQ PUSH to worker socket
  → Worker receives query
  → HttpRequestAdapter formats HTTP request
  → ConnectionPool acquires connection
  → HTTP/1.1 request sent
```

**Inbound (endpoint → caller):**

```
HTTP response received by Worker
  → HttpResponseProtocol (httptools parser)
  → Accumulator builds QueryResult / StreamChunk
  → ZMQ PUSH result back to main process
  → WorkerPoolTransport routes to response queue
  → HTTPEndpointClient.recv() / poll() / drain()
```

## Key Components

### `ConnectionPool`

Maintains a persistent TCP connection pool per worker. Connections can be warmed up before the
benchmark starts to reduce cold-start latency. Idle connections are evicted after
`max_idle_time` seconds.

### `HttpResponseProtocol`

`asyncio.Protocol` implementation using `httptools.HttpResponseParser` (llhttp C parser). Handles
both streaming (SSE) and non-streaming responses. Connections are reused between requests via
`reset()` without re-establishing TCP.

### `WorkerManager`

Spawns worker processes via `multiprocessing.Process`. Monitors liveness with periodic checks
during startup. Applies CPU affinity via `AffinityPlan` after all workers are alive.

### `HttpRequestAdapter` (abstract base class)

Converts a `Query` to a raw HTTP request bytes. Implementations:

- `openai/openai_msgspec_adapter.py` — fast path using msgspec
- `sglang/adapter.py` — SGLang-specific format

## Design Decisions

**Multi-process over multi-thread**

Python's GIL prevents true parallelism in a threaded HTTP client at high QPS. Worker processes
each own a uvloop event loop and a connection pool, achieving genuine concurrency. ZMQ IPC
has lower overhead than inter-process queues or sockets for this pattern.

**Round-robin dispatch (not work-stealing)**

Round-robin is O(1) and avoids contention on a shared queue. Workers have equal capacity, so
skewed distribution is not a concern in practice.

**`httptools` over `aiohttp`/`httpx`**

`httptools` is the same C parser used by Node.js (llhttp). It exposes a callback API that feeds
directly into the asyncio protocol, eliminating intermediate buffering. `aiohttp` and `httpx` add
abstraction layers that increase latency variance.

**`stream_all_chunks=False` by default**

Exposing every SSE chunk requires passing each through the ZMQ transport, adding per-chunk
serialisation cost. By default the client still forwards the first chunk for TTFT measurement,
suppresses intermediate chunks, and then returns the final assembled `QueryResult` at end of
stream. Enable `stream_all_chunks` only when callers need every chunk, not just TTFT and the
final response.

## Integration Points

| Dependency               | Role                                                   |
| ------------------------ | ------------------------------------------------------ |
| `core/types.py`          | `Query` in, `QueryResult`/`StreamChunk` out            |
| `async_utils/transport/` | ZMQ IPC between main process and workers               |
| `openai/`, `sglang/`     | `HttpRequestAdapter` and accumulator implementations   |
| `load_generator/`        | Provides the `SampleIssuer` ABC consumed by the client |
| `config/`                | `HTTPClientConfig` derived from `RuntimeSettings`      |
