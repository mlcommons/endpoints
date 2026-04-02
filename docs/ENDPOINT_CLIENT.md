# MLPerf Inference Endpoints: HttpClient Design Document

> **Note:** The canonical design reference is now [docs/endpoint_client/Design.md](endpoint_client/Design.md). This document contains deeper implementation detail on connection pool architecture and functional requirements.

Detailed design for the `HTTPEndpointClient`: functional requirements, performance constraints, connection pool architecture, and worker process integration.

## Table of Contents

- [1. Introduction \& Constraints](#1-introduction--constraints)
  - [1.1 Functional Requirements](#11-functional-requirements)
  - [1.2 Performance Requirements](#12-performance-requirements)
  - [1.3 Constraints](#13-constraints)
  - [1.4 Non-Goals](#14-non-goals)
- [2. System Overview](#2-system-overview)
  - [2.1 Usage](#21-usage)
  - [2.2 Inference-Endpoints Integration](#22-inference-endpoints-integration)
- [3. Types](#3-types)
- [4. HTTPClientConfig](#4-httpclientconfig)
- [5. HTTPEndpointClient](#5-httpendpointclient)
  - [5.1 Architecture](#51-architecture)
  - [5.2 Design Choices](#52-design-choices)
- [6. Worker](#6-worker)
  - [6.1 Request Lifecycle](#61-request-lifecycle)
  - [6.2 Call Chain](#62-call-chain)
  - [6.3 Design Choices](#63-design-choices)
- [7. Transport](#7-transport)
  - [7.1 ZMQ Implementation](#71-zmq-implementation)
  - [7.2 Benchmarks](#72-benchmarks)
- [8. HTTP Engine](#8-http-engine)
  - [8.1 HttpResponseProtocol](#81-httpresponseprotocol)
  - [8.2 HttpRequestTemplate](#82-httprequesttemplate)
  - [8.3 Connection Pool](#83-connection-pool)
  - [8.4 Socket Config](#84-socket-config)
  - [8.5 Design Choices](#85-design-choices)
  - [8.6 Benchmark Results vs aiohttp](#86-benchmark-results-vs-aiohttp)
- [9. Adapters](#9-adapters)
  - [9.1 HttpRequestAdapter](#91-httprequestadapter)
  - [9.2 SSEAccumulatorProtocol](#92-sseaccumulatorprotocol)
  - [9.3 SSE Stream Parsing](#93-sse-stream-parsing)
  - [9.4 Implementations](#94-implementations)
- [10. Initialization \& Shutdown](#10-initialization--shutdown)
  - [10.1 WorkerManager](#101-workermanager)
  - [10.2 Startup](#102-startup)
  - [10.3 Shutdown](#103-shutdown)
- [11. Performance Analysis](#11-performance-analysis)
  - [11.1 Worker Scaling](#111-worker-scaling)
  - [11.2 Stream Interval Sensitivity](#112-stream-interval-sensitivity)
  - [11.3 Worker Thread Profile](#113-worker-thread-profile-pidstat--t)
  - [11.4 Context Switches](#114-context-switches-pidstat--w)
  - [11.5 CPU Symbol Profile](#115-cpu-symbol-profile-perf-top)
  - [11.6 Syscall Profile](#116-syscall-profile-strace--c)
  - [11.7 Run Queue Latency](#117-run-queue-latency-runqlat-bpfcc)
  - [11.8 Hardware Performance Counters](#118-hardware-performance-counters-tiptop)
- [Appendix A: Concepts](#appendix-a-concepts)
- [Appendix B: Work in Progress (POR)](#appendix-b-work-in-progress-por)
- [Appendix C: Future Optimizations](#appendix-c-future-optimizations)
- [Appendix D: Performance Changelog](#appendix-d-performance-changelog)
- [Bibliography](#bibliography)

---

## Terminology & Acronyms

| Term                  | Definition                                                                                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Body/stream chunk** | A fragment of an HTTP response body. In streaming mode, each chunk carries one or more SSE events; the terms "body chunk" and "stream chunk" are used interchangeably |
| **GC**                | Garbage Collector - Python's automatic memory management                                                                                                              |
| **GIL**               | Global Interpreter Lock - Python mutex preventing true thread parallelism                                                                                             |
| **IPC**               | Inter-Process Communication - data exchange between OS processes                                                                                                      |
| **LLM**               | Large Language Model                                                                                                                                                  |
| **LoadGen**           | Load Generator - the component that schedules and issues queries - user of http client                                                                                |
| **NUMA**              | Non-Uniform Memory Access - memory architecture where access time depends on memory location relative to CPU                                                          |
| **Query**             | Internal IPC message carrying one inference task from main process to worker                                                                                          |
| **QPS**               | Queries Per Second                                                                                                                                                    |
| **Request**           | HTTP/1.1 POST sent by a worker to the LLM endpoint                                                                                                                    |
| **Sample**            | LoadGen's unit of work — one inference task (prompt + parameters) issued to the SUT                                                                                   |
| **Sequence**          | Ordered series of tokens — input sequence (prompt) or output sequence (generation)                                                                                    |
| **SMT**               | Simultaneous Multi-Threading - hyperthreading; multiple logical CPUs per physical core                                                                                |
| **SSE**               | Server-Sent Events - HTTP streaming protocol for server-to-client push                                                                                                |
| **SUT**               | System Under Test - the LLM endpoint being benchmarked                                                                                                                |
| **TFB**               | Time to First Byte - latency until first HTTP response byte received                                                                                                  |
| **TPOT**              | Time Per Output Token - average latency between consecutive tokens                                                                                                    |
| **TPS**               | Tokens Per Second                                                                                                                                                     |
| **TTFT**              | Time To First Token - latency until first generated token received                                                                                                    |
| **Worker**            | HTTP request engine running in a separate OS process                                                                                                                  |
| **ZMQ**               | ZeroMQ - high-performance asynchronous messaging library                                                                                                              |

---

## 1. Introduction & Constraints

This document defines the architecture of the HTTP client used for benchmarking LLM Servers by the MLPerf Inference-Endpoints LoadGen.

#### 1.1 Functional Requirements

| #   | Requirement                                                                 | Status |
| --- | --------------------------------------------------------------------------- | ------ |
| 1   | Run multiple HTTP/1.1 POST requests concurrently                            | ✅     |
| 2   | Batch mode requests (final response returned when ready)                    | ✅     |
| 3   | Streaming mode requests (HTTP-SSE chunks with tokens as they are generated) | ✅     |
| 4   | Multiple API types (OpenAI, SGLANG, TRTLLM)                                 | ✅     |
| 5   | Configurable retry logic                                                    | TODO   |

#### 1.2 Performance Requirements

| #   | Requirement                 | Target  | Achieved (x86)                             | Achieved (ARM)                            | Notes                                                                  |
| --- | --------------------------- | ------- | ------------------------------------------ | ----------------------------------------- | ---------------------------------------------------------------------- |
| 1   | QPS (offline)               | 100k    | ~300k QPS @ 14 workers                     | ~300k QPS @ 14 workers                    | Roofline; 1 query ≈ 1000 tokens. No streaming overhead                 |
| 2   | QPS (streaming, worst-case) | 70k     | ~90.6k QPS, ~79.7M SSE-pkts/s @ 96 workers | ~133k QPS, ~121M SSE-pkts/s @ 132 workers | `stream_interval=1` (1 char per SSE chunk → 1000 chunks/response)      |
| 3   | Per-request overhead        | O(µs)   | O(µs)                                      | O(µs)                                     | 300k QPS / 14 workers ≈ 21.4k req/s/worker; ~47µs pure client overhead |
| 4   | Run-to-run jitter           | Minimal |                                            |                                           |                                                                        |

**Test environments:** x86 = Intel Xeon Platinum 8570 × 2 (112 cores / 224 threads, HT); ARM = NVIDIA Grace × 2 (144 cores). Measured using `benchmark_httpclient.py` (`src/inference_endpoint/utils/`). See [§11](#11-performance-analysis) for full results.

#### 1.3 Constraints

The design operates within these constraints, which shape all subsequent architectural decisions.

| #   | Constraint                | Detail                                                                                                                                                                                                                                                                                      | Implication                                                                                                                                                               |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Pure Python 3.12+         | Python-native project. Performance-critical paths rely on existing C-backed libraries (`httptools`, `msgspec`, `uvloop`, `pyzmq`) for parsing, serialization, event-loop, and IPC respectively                                                                                              | Performance ceiling bounded by Python call overhead; hot path must minimize Python-level work and maximize time spent in C library code                                   |
| 2   | GIL bypass via processes  | CPython's GIL prevents true thread parallelism. To saturate high-throughput endpoints we need many concurrent HTTP connections, each independently issuing requests and processing responses                                                                                                | Multi-process architecture via `multiprocessing.Process` with `spawn` start method. Each worker is an independent OS process with its own interpreter, GC, and event loop |
| 3   | Async I/O with event loop | Each worker must drive a many concurrent HTTP connections — writing requests to network buffers, multiplexing reads across many open TCP sockets, and forwarding responses back to the main process via IPC. An event loop minimizes per-operation overhead for both send and receive paths | We use `uvloop` [4], a Cython/libuv-based drop-in `asyncio` replacement that uses `epoll`/`kqueue` for O(1) readiness notification across thousands of file descriptors   |

#### 1.4 Non-Goals

| #   | Non-Goal                               | Rationale                                                                                                                         |
| --- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Client-side rate limiting / throttling | LoadGen controls request pacing; client is a transparent pipe                                                                     |
| 2   | Windows / macOS production support     | CPU affinity, NUMA, `TCP_QUICKACK`, and several socket options are Linux-specific; cross-platform runs with reduced functionality |
| 3   | HTTP/2 or HTTP/3                       | Not in POR since trtllm-server does not support this yet.                                                                         |

---

## 2. System Overview

![High Level System Architecture](res/endpoint_client/01_high_level_architecture.png)

The HTTP client (`HTTPEndpointClient`) is a multi-process, async HTTP engine that the LoadGen uses to send requests to and receive responses from the target LLM endpoint (e.g. vLLM, SGLang, TRT-LLM). It exposes a `SampleIssuer` interface (`HttpClientSampleIssuer`) so the LoadGen can call `issue(sample)` without knowledge of the underlying transport or HTTP details.

On initialization, the client spawns N worker processes and sets up IPC channels between the main process event loop and each worker. The **main process** runs two threads: the LoadGen thread (orchestrating the test and scheduling requests) and the event loop thread (`uvloop`-based async dispatcher). The event loop accepts queries from the LoadGen thread via `issue()`, dispatches them to workers in round-robin order, and collects responses back from all workers on the return path. Incoming responses (`StreamChunk` and `QueryResult` messages) are routed to `SampleEventHandler` callbacks.

Each **worker process** runs its own async event loop in a separate OS process (avoiding GIL contention). A worker receives queries from its IPC channel, builds HTTP requests using the configured API adapter, sends them over its connection pool, and returns responses back via IPC. Each worker handles multiple in-flight requests concurrently.

### 2.1 Usage

The client can be used directly:

```python
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.core.types import Query, QueryResult, StreamChunk

config = HTTPClientConfig(endpoint_urls=["http://localhost:8000"])

# Transport context is managed internally — no external context needed
client = HTTPEndpointClient(config)

# Issue a query
client.issue(Query(data={
    "prompt": "What is machine learning?",
    "model": "Qwen/Qwen3-8B",
    "max_completion_tokens": 100,
    "stream": True,
}))

# Collect responses (sync — for callers on a non-async thread)
response = client.poll()              # Non-blocking: StreamChunk | QueryResult | None
responses = client.drain()            # Non-blocking: returns all available responses

# Collect responses (async — for callers already on an event loop)
# response = await client.recv()      # Blocking: waits for next response; None when closed

client.shutdown()
```

### 2.2 Inference-Endpoints Integration

In benchmarking mode, the `HttpClientSampleIssuer` bridges the LoadGen thread and the async client. `HttpClientSampleIssuer` implements the `SampleIssuer` interface from the `inference-endpoints` LoadGen framework, converting `Sample` objects to `Query` and routing responses back to `SampleEventHandler` callbacks.

```python
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer

client = HTTPEndpointClient(config)
issuer = HttpClientSampleIssuer(client)
```

---

## 3. Types

The endpoint client uses three core message types defined in the parent project (`core/types.py`) for IPC communication. All types are `msgspec.Struct` [5] with performance-oriented options (`frozen`, `array_like`, `gc=False`, `omit_defaults` — see [A.5](#a5-msgspec-serialization) for the full convention table). The `tag` field on `QueryResult` and `StreamChunk` enables union type discrimination during MessagePack deserialization on the fan-in path.

| Type          | Direction     | Purpose                                                                 |
| ------------- | ------------- | ----------------------------------------------------------------------- |
| `Query`       | Main → Worker | Request payload with `id`, `data` (prompt/params), `headers`            |
| `StreamChunk` | Worker → Main | Intermediate streaming token with `response_chunk`, `is_complete` flag  |
| `QueryResult` | Worker → Main | Final response with `response_output`, `error`, auto-set `completed_at` |
| `QueryStatus` | Internal      | Enum: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`          |

```python
class Query(msgspec.Struct, frozen=True, kw_only=True, array_like=True, omit_defaults=True, gc=False):
    id: str
    data: dict[str, Any]
    headers: dict[str, str]
    created_at: float

class QueryResult(msgspec.Struct, tag="query_result", kw_only=True, frozen=True, array_like=True, omit_defaults=True, gc=False):
    id: str
    response_output: str | tuple[str, ...] | dict[str, str | list[str]] | None
    metadata: dict[str, Any]
    error: str | None
    completed_at: int | msgspec.UnsetType  # auto-set via __post_init__

class StreamChunk(msgspec.Struct, tag="stream_chunk", frozen=True, kw_only=True, array_like=True, omit_defaults=True, gc=False):
    id: str
    response_chunk: str
    is_complete: bool
    metadata: dict[str, Any]

class QueryStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

## 4. HTTPClientConfig

`HTTPClientConfig` (`config.py`) is a Pydantic `BaseModel` that configures the client, worker pool, and connection management. Several fields support auto-detection via sentinel defaults (`-1`), resolved in the `_resolve_defaults` model validator.

**Classes:**

| Class              | Source      | Description                                              |
| ------------------ | ----------- | -------------------------------------------------------- |
| `HTTPClientConfig` | `config.py` | `@dataclass`: client, worker pool, and connection config |

```python
class APIType(str, Enum):
    OPENAI = "openai"
    SGLANG = "sglang"

class HTTPClientConfig(BaseModel):
    # Target endpoint URLs; workers assigned round-robin at spawn time
    endpoint_urls: list[str]
    # Selects adapter + accumulator pair (see §9)
    api_type: APIType = APIType.OPENAI
    # Bearer token for endpoint authentication
    api_key: str | None = None

    # Worker process count
    # -1 = auto: min(max(8, numa_domain_size), 24)
    num_workers: int = -1

    record_worker_events: bool = False
    event_logs_dir: Path | None = None
    log_level: str = "INFO"

    # When True, all SSE chunks emitted via IPC (high main-thread overhead).
    # When False (default), only first chunk per response (for TTFT measurement).
    stream_all_chunks: bool = False

    # CPU pinning plan for worker processes; None = no pinning
    cpu_affinity: AffinityPlan | None = None

    # Worker lifecycle timeouts (seconds)
    worker_initialization_timeout: float = 60.0
    worker_graceful_shutdown_wait: float = 0.5
    worker_force_kill_timeout: float = 0.5

    # Discard connections idle longer than this (seconds).
    # Prevents keep-alive race where server closes idle connection
    # at the exact moment client sends a new request (half-closed TCP).
    max_idle_time: float = 4.0

    # Pre-establish TCP connections during init for reuse at runtime.
    # -1 = auto (50% of pool); 0 = disabled; >0 = explicit count
    warmup_connections: int = -1

    # Maximum concurrent TCP connections per worker.
    # -1 = unlimited (bound by system ephemeral port limit)
    max_connections: int = -1

    # Minimum required connections for init; warns if not enough ports.
    # -1 = auto (90% of system ephemeral port range); 0 = disable check
    min_required_connections: int = -1

    # GC strategy for worker processes to reduce latency spikes
    # "disabled" = GC off; "relaxed" = 100x higher threshold; "system" = defaults
    worker_gc_mode: Literal["disabled", "relaxed", "system"] = "relaxed"

    # Transport config — owns transport class, context, and socket options
    transport: ZMQTransportConfig | None = None

    # Pluggable components (None = auto-resolved from api_type in model_validator)
    adapter: Annotated[Any, cyclopts.Parameter(parse=False)] = None
    accumulator: Annotated[Any, cyclopts.Parameter(parse=False)] = None
```

#### Auto-configuration (`_resolve_defaults` model validator)

Three fields resolve `-1` sentinels by probing the host at construction time:

**`num_workers=-1`:** Detects the NUMA node of the current process, counts physical CPUs in that NUMA domain, and clamps to `min(max(8, numa_cpu_count), 24)`. Falls back to 8 if NUMA info is unavailable. The intent is to keep all workers local to the same NUMA node for memory locality; users can override to use more cores (workers will be pinned to additional cores outside the NUMA domain if an `AffinityPlan` is provided).

**`max_connections=-1`:** Reads the system ephemeral port range from `/proc/sys/net/ipv4/ip_local_port_range`, counts currently used TCP sockets in that range from `/proc/net/tcp` and `/proc/net/tcp6`, and sets `max_connections` to the remaining available ports. Also sets `min_required_connections` to 90% of the total system port range (used to warn during warmup if too few ports are available). If an explicit `max_connections` value exceeds available ports, raises `RuntimeError`.

**`cpu_affinity`:** When an `AffinityPlan` is provided (or computed via `compute_affinity_plan()`), cores are auto-detected from sysfs topology and ranked by performance. Ranking sources, checked in order: ACPI CPPC `highest_perf` (Intel P-core vs E-core), ARM `cpu_capacity` (big.LITTLE), `cpuinfo_max_freq` (fallback). The fastest cores are reserved for the main process (LoadGen thread + event loop daemon + transport I/O threads); remaining physical cores are assigned 1:1 to workers.

---

## 5. HTTPEndpointClient

Unified multi-process HTTP client for LLM inference. Manages a pool of worker processes behind a simple issue/poll/drain interface — actual HTTP request dispatch and response processing run in background worker processes. Exposes both synchronous methods (for callers on non-async threads) and an async `recv()` (for callers already on an event loop).

**Classes:**

| Class                | Source           | Description                                                                     |
| -------------------- | ---------------- | ------------------------------------------------------------------------------- |
| `HTTPEndpointClient` | `http_client.py` | HTTP client: owns event loop daemon thread, `WorkerManager`, and pool transport |

**Public API:**

| Method         | Async   | Description                                                                |
| -------------- | ------- | -------------------------------------------------------------------------- |
| `issue(query)` | No      | Dispatch query to next worker (round-robin via `call_soon_threadsafe`)     |
| `poll()`       | No      | Return one response if available, else `None`                              |
| `recv()`       | **Yes** | `await` for next response; `None` when closed. Use from async callers only |
| `drain()`      | No      | Return all available responses (`list(iter(self.poll, None))`)             |
| `shutdown()`   | No      | Synchronous graceful shutdown of workers and loop; blocks until complete   |

### 5.1 Architecture

On construction, the client creates a `uvloop` event loop via `LoopManager` (or accepts an external one) and initializes a `WorkerManager` ([§10](#10-initialization--shutdown)) that spawns N worker processes connected via IPC ([§7](#7-transport)). The **main process** dispatches queries round-robin to workers and collects responses from all workers into a single fan-in queue. Each **worker** (×N) is a separate OS process with its own event loop, executing HTTP requests against the target endpoint and returning results via IPC ([§6](#6-worker)).

**Main process:**

| Component             | Responsibility                                                   | Deep Dive                           |
| --------------------- | ---------------------------------------------------------------- | ----------------------------------- |
| `WorkerPoolTransport` | Fan-out queries to workers, fan-in responses into a single queue | [§7](#7-transport)                  |
| `WorkerManager`       | Spawn, monitor, and shut down worker processes                   | [§10](#10-initialization--shutdown) |

**Worker process** (×N, [§6](#6-worker)):

| Component              | Responsibility                                                                                         | Deep Dive                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| `Adapter`              | Encode queries into HTTP request bodies; decode HTTP responses and SSE messages into typed results     | [§9.1](#91-httprequestadapter)     |
| `Accumulator`          | Collect streaming SSE chunks into a final result; track tokens, emit per-chunk events, assemble output | [§9.2](#92-sseaccumulatorprotocol) |
| `RequestTemplate`      | Pre-build static HTTP/1.1 request bytes (method, path, headers); per-request cost: Content-Length only | [§8.2](#82-httprequesttemplate)    |
| `ConnectionPool`       | Pool of persistent TCP connections to one endpoint; LIFO reuse, stale detection, warmup at init        | [§8.3](#83-connection-pool)        |
| `HttpResponseProtocol` | Parse HTTP responses from raw bytes via C-level callbacks; expose async read and streaming interfaces  | [§8.1](#81-httpresponseprotocol)   |

<img src="res/endpoint_client/02_client_architecture.png" alt="Client Architecture" width="655">

### 5.2 Design Choices

| Choice              | Implementation                                       | Alternative               | Rationale                                                                                                                                                                                                                                                      |
| ------------------- | ---------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Worker dispatch     | Round-robin                                          | Load-aware, work-stealing | Predictable and zero-overhead at dispatch time. Load-aware routing adds per-request decision cost. Work-stealing is a future consideration — see [C.3](#c3-work-stealing-dispatch)                                                                             |
| Endpoint assignment | Single URL per worker, assigned round-robin at spawn | Per-request routing       | Each worker maintains its own connection pool to one endpoint, avoiding cross-worker coordination. Per-request routing is a future consideration tied to work-stealing ([C.3](#c3-work-stealing-dispatch)) — both require workers to handle multiple endpoints |
| Event Loop library  | `uvloop` [4]                                         | `asyncio`                 | Lower per-iteration overhead                                                                                                                                                                                                                                   |

---

## 6. Worker

Each worker is a separate OS process running its own uvloop event loop ([A.2](#a2-event-loops-and-eager-task-factory)). It receives queries via IPC, executes HTTP requests against its assigned endpoint, and returns responses via IPC. The worker's operation decomposes into two concurrent components on that single event loop:

- **Main loop (`_run_main_loop`)** — a tight recv → prepare → acquire → write → create_task cycle. Each iteration receives one query from IPC, encodes it to HTTP bytes, acquires a pooled connection, writes the request to the socket, and creates a response task. The loop never blocks on a response — it immediately loops back to receive the next query. All of its await points (`recv`, `pool.acquire`) are designed to return synchronously in the common case.

- **Concurrent response tasks (`_process_response`)** — each spawned via `create_task()`. A single worker may have hundreds of response tasks alive simultaneously, each waiting independently on network I/O. The event loop multiplexes between them using `HttpResponseProtocol` callbacks ([§8.1](#81-httpresponseprotocol)) that resume suspended tasks when data arrives.

**Classes:**

| Class    | Source      | Description                                                             |
| -------- | ----------- | ----------------------------------------------------------------------- |
| `Worker` | `worker.py` | Runs the main loop and concurrent response tasks on a single event loop |

**Public API:**

| Function / Method                    | Async   | Description                                                       |
| ------------------------------------ | ------- | ----------------------------------------------------------------- |
| `worker_main(id, connector, config)` | No      | Process entry point: GC config, uvloop install, `Worker.run()`    |
| `Worker.run()`                       | **Yes** | Top-level: init HTTP components, signal handlers, enter main loop |
| `Worker.shutdown(signum, frame)`     | No      | SIGTERM handler: sets `_shutdown`, closes request transport       |

### 6.1 Request Lifecycle

Each request flows through the main loop and a spawned response task — the two concurrent components from [§6](#6-worker). The main loop handles request dispatch; the response task handles response processing independently. The hot-path diagram below traces this lifecycle visually.

Every async step _can_ resolve synchronously in the common case — suspending only when data is not yet available.

**Main loop — request dispatch:**

| Step                 | Async   | Description                                                                                   |
| -------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `recv()`             | **Yes** | Receive next query from IPC; suspends only if queue empty                                     |
| `_prepare_request()` | No      | Encode query → JSON → HTTP bytes via adapter ([§9.1](#91-httprequestadapter))                 |
| `pool.acquire()`     | **Yes** | Acquire connection from pool ([§8.3](#83-connection-pool), LIFO); suspends only if pool empty |
| `protocol.write()`   | No      | Write HTTP request bytes to kernel socket buffer                                              |
| `create_task()`      | No      | Spawn response task                                                                           |

The loop returns to `recv()` immediately after `create_task()` — it never waits for a response.

**Response task** (`_process_response`, ×N concurrent):

| Step             | Async   | Description                                                                                             |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `read_headers()` | **Yes** | Wait for HTTP response headers from network ([§8.1](#81-httpresponseprotocol))                          |
| `iter_body()`    | **Yes** | Yield buffered body chunks; suspends only if buffer empty ([§8.1](#81-httpresponseprotocol))            |
| parse + send     | No      | Regex extract + msgspec decode ([§9.3](#93-sse-stream-parsing)); IPC send `StreamChunk` / `QueryResult` |
| `pool.release()` | No      | Return connection to idle stack; always synchronous                                                     |

For streaming requests, `iter_body()` → parse + send repeats in a loop until the stream completes, then a final `QueryResult` is sent before `pool.release()`. Non-streaming requests read the full body in a single `read_body()` call, decode, and send one `QueryResult` directly.

Response processing is spawned as an asyncio Task rather than awaited directly:

| Approach                          | Behavior                                 | Throughput                  |
| --------------------------------- | ---------------------------------------- | --------------------------- |
| `await process_response()`        | Main loop blocks until response complete | 1 request at a time         |
| `create_task(process_response())` | Main loop continues immediately          | 100s of concurrent requests |

The main loop must keep dispatching new requests while previous responses are still streaming. Without tasks, each request would wait for its full response (potentially seconds for LLM generation) before the next could start.

**Hot path:**

The diagram below traces the full lifecycle. The left column shows the main loop's dispatch cycle; the right column shows the concurrent response task with its streaming/non-streaming branch. Red nodes are AWAIT points (potential suspend); green nodes execute synchronously.

<img src="res/endpoint_client/12_worker_lifecycle.png" alt="Worker Request Lifecycle" width="624">

### 6.2 Call Chain

The lifecycle above identifies _what_ each step does; the call chain below shows _when_ each step actually suspends. Every `await` in the request path is a potential context switch: the coroutine yields control to the event loop, the event loop selects the next ready task, and that task resumes. The system is designed so that every `await` _can_ resolve synchronously — meaning the entire dispatch-to-send sequence can complete without ever yielding to the event loop:

| Await Point        | Completes Synchronously When                                                                            | Suspends When                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `recv()`           | Query already queued in IPC buffer                                                                      | No queries available — waits for main process to dispatch |
| `pool.acquire()`   | Idle connection available in pool                                                                       | Pool exhausted — waits for a connection to be released    |
| `read_headers()`   | Response headers already arrived                                                                        | Server hasn't responded yet — waits for network I/O       |
| `iter_body()`      | Response data buffered faster than consumed — drains all available chunks synchronously before yielding | Buffer empty — waits for next network read                |
| `protocol.write()` | **Always** — kernel socket buffer accepts bytes immediately                                             | Never                                                     |
| `pool.release()`   | **Always** — returns connection to idle stack                                                           | Never                                                     |

In the common case, the request path executes with zero event loop round-trips between `recv()` returning and `read_headers()` suspending on network I/O.

<img src="res/endpoint_client/03_request_response_flow.png" alt="Request → Response Flow" width="990">

### 6.3 Design Choices

A standard implementation would use `aiohttp` for HTTP and `zmq.asyncio` for IPC. The worker instead uses a custom HTTP stack ([§8](#8-http-engine)) and a custom loopless transport ([§7](#7-transport)); per-component optimizations are documented in those sections. Worker-specific design choices:

| Choice             | Implementation                                                                                    | Alternative          | Rationale                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Main loop split    | Separate sync prepare, async fire, and task spawn into three explicit steps                       | Single combined task | Main loop has only 2 await points and never blocks on a response                                                                 |
| Eager task factory | `asyncio.eager_task_factory` on worker event loop ([A.2](#a2-event-loops-and-eager-task-factory)) | Default task factory | New tasks execute synchronously until their first true `await`, so `create_task()` runs the response task without yielding first |
| GC tuning          | gen0=70000, gen1=10, gen2=100 ("relaxed" mode); full `gc.collect(2)` before entering main loop    | Default thresholds   | Reduces GC pause frequency by 100× during steady state; pre-loop collection ensures a clean heap before the hot path begins      |
| Early conn release | Release connection to pool before sending the final result via IPC                                | Release after send   | Connection becomes available for reuse by other tasks while the IPC send completes, reducing pool contention under load          |
| No-retry on error  | Wrap error in `QueryResult(error=...)`, send via IPC, continue loop                               | Retry with backoff   | Caller (LoadGen) decides retry policy; worker stays simple and never stalls on transient failures                                |

---

## 7. Transport

The transport layer handles all IPC between the main process and worker processes. It defines abstract protocols (`async_utils/transport/protocol.py`) with a concrete ZMQ implementation (`async_utils/transport/zmq/`), keeping the worker and client code decoupled from the underlying messaging library. The protocols are serialization-agnostic — implementations bring their own serialization/deserialization. By convention, all serialization throughout the client uses `msgspec` [5] (MessagePack for IPC, JSON for HTTP); see [A.5](#a5-msgspec-serialization) for the shared usage patterns. `WorkerPoolTransport` owns the full topology; workers only see their own `ReceiverTransport` + `SenderTransport` pair, with no knowledge of other workers or the pool structure.

**Classes:**

| Class                 | Source                              | Description                                                    |
| --------------------- | ----------------------------------- | -------------------------------------------------------------- |
| `ReceiverTransport`   | `async_utils/transport/protocol.py` | Protocol: async message receipt (`recv`, `poll`, `close`)      |
| `SenderTransport`     | `async_utils/transport/protocol.py` | Protocol: non-blocking message send (`send`, `close`)          |
| `WorkerConnector`     | `async_utils/transport/protocol.py` | Protocol: picklable factory yielding per-worker transport pair |
| `WorkerPoolTransport` | `async_utils/transport/protocol.py` | Protocol: main-process fan-out/fan-in across all workers       |

<img src="res/endpoint_client/06b_transport_overview.png" alt="Transport Protocol Hierarchy" width="700">

**Public API — `ReceiverTransport`:**

| Method    | Async   | Description                                                                       |
| --------- | ------- | --------------------------------------------------------------------------------- |
| `recv()`  | **Yes** | Blocking receive — returns next deserialized message, or `None` when closed       |
| `poll()`  | No      | Non-blocking receive — returns message if immediately available, `None` otherwise |
| `close()` | No      | Close transport and release resources; subsequent `recv()` returns `None`         |

**Public API — `SenderTransport`:**

| Method       | Async | Description                                     |
| ------------ | ----- | ----------------------------------------------- |
| `send(data)` | No    | Send a serialized message through the transport |
| `close()`    | No    | Close transport and release resources           |

**Public API — `WorkerConnector`:**

| Method               | Async   | Description                                                                                                                   |
| -------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `connect(worker_id)` | **Yes** | Async context manager yielding `(ReceiverTransport, SenderTransport)` pair for a worker; signals readiness, cleans up on exit |

**Public API — `WorkerPoolTransport`:**

| Method / Property                        | Async   | Description                                                                             |
| ---------------------------------------- | ------- | --------------------------------------------------------------------------------------- |
| `create(loop, num_workers, config=None)` | No      | Factory classmethod — creates configured pool transport bound to the given event loop   |
| `worker_connector`                       | No      | Property returning the picklable `WorkerConnector` to pass to spawned worker processes  |
| `send(worker_id, query)`                 | No      | Fan-out: dispatch a `Query` to a specific worker by ID                                  |
| `poll()`                                 | No      | Fan-in: non-blocking poll for a `QueryResult` or `StreamChunk` from any worker          |
| `recv()`                                 | **Yes** | Fan-in: blocking receive — awaits next response from any worker, `None` when closed     |
| `wait_for_workers_ready(timeout)`        | **Yes** | Block until all workers signal readiness; raises `TimeoutError` if timeout exceeded     |
| `cleanup()`                              | No      | Close all transports and release resources (including IPC socket temp dirs). Idempotent |

**Usage (main process):**

```python
# Create pool transport bound to the event loop
pool = ZmqWorkerPoolTransport.create(loop, num_workers=4)
# Spawn worker processes, passing the picklable connector
for i in range(4):
    Process(target=worker_main, args=(i, pool.worker_connector, config)).start()
# Block until all workers have connected and signaled readiness
await pool.wait_for_workers_ready(timeout=30)

# Fan-out: dispatch a query to a specific worker
pool.send(worker_id=0, query=query)
# Fan-in (non-blocking): returns result immediately or None
result = pool.poll()
# Fan-in (blocking): awaits next result from any worker
result = await pool.recv()

# Tear down transports and clean up
pool.cleanup()
```

**Usage (worker process):**

```python
# Connect to IPC and signal readiness to the main process
async with connector.connect(worker_id=0) as (receiver, sender):
    # Block until the next query arrives
    query = await receiver.recv()
    # Send the result back via fan-in transport
    sender.send(result)
```

### 7.1 ZMQ Implementation

The ZMQ implementation (`ZmqWorkerPoolTransport`) uses direct event loop integration via `add_reader`/`add_writer` on ZMQ file descriptors, rather than pyzmq's async APIs or aiozmq. This eliminates extra abstraction layers and keeps all I/O on the uvloop event loop.

**Classes:**

| Class                    | Source                                   | Description                                                                          |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `ZmqWorkerPoolTransport` | `async_utils/transport/zmq/transport.py` | Concrete ZMQ pool transport                                                          |
| `_ZmqReceiverTransport`  | `async_utils/transport/zmq/transport.py` | ZMQ PULL receiver with edge-triggered FD handling                                    |
| `_ZmqSenderTransport`    | `async_utils/transport/zmq/transport.py` | ZMQ PUSH sender with buffered writes                                                 |
| `_ZmqWorkerConnector`    | `async_utils/transport/zmq/transport.py` | Picklable ZMQ connector (`@dataclass`, `slots=True`)                                 |
| `ZMQTransportConfig`     | `async_utils/transport/zmq/transport.py` | ZMQ transport config (Pydantic) — socket tuning, transport class, context management |
| `ManagedZMQContext`      | `async_utils/transport/zmq/context.py`   | Singleton ZMQ context wrapper — lifecycle managed by transport implementation        |

#### 7.1.1 Serialization

The transport uses `msgspec.msgpack` [5] for all IPC serialization. Each transport instance holds a pre-constructed `Encoder` or `Decoder` reused across every message, amortizing the construction cost over the transport lifetime rather than paying it per-message. The decoder is instantiated with a target type (e.g. `msgspec.msgpack.Decoder(type=QueryResult | StreamChunk)`), enabling schema-aware deserialization that allocates the result Struct directly without an intermediate `dict`.

On the request path, each worker's `ReceiverTransport` holds a `Decoder(type=Query)`. On the response fan-in path, the main process holds a single `Decoder(type=QueryResult | StreamChunk)` — the `tag` field on each Struct (see [§3](#3-types)) tells msgpack which union variant to instantiate without a type-discriminator wrapper or try/except decode fallback.

The same `msgspec.Struct` type definitions also drive JSON serialization on the HTTP path via `msgspec.json` in the adapter layer (see [§9.1](#91-httprequestadapter)), so there is one schema per message type shared across both IPC and HTTP — no separate serialization models to maintain. See [A.5](#a5-msgspec-serialization) for the full set of Struct conventions, encoder/decoder patterns, and a cross-reference of where msgspec is used across layers.

#### 7.1.2 Topology

The main process maintains one dedicated PUSH socket per worker for request fan-out (explicit targeting, no load-balancer overhead) and a single shared PULL socket for response fan-in from all workers. Each worker connects a PULL socket for incoming queries and a PUSH socket for outgoing responses. All sockets use `ipc://` (Unix domain sockets) for zero-copy kernel transport.

<img src="res/endpoint_client/07_socket_topology.png" alt="Socket Topology" width="685">

#### 7.1.3 Message Flow

Each IPC message traverses the following path through pyzmq and the kernel (see [A.3](#a3-zeromq-zmq) for pyzmq source-level detail):

| Step | Thread     | Operation                                                                                                                            |
| ---- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | Sender     | Serialize the message to msgpack bytes using `msgspec.encode()`                                                                      |
| 2    | Sender     | pyzmq allocates a ZMQ message (`zmq_msg_init_size`), copies bytes into it (`memcpy`), and enqueues it to the ZMQ mailbox (`NOBLOCK`) |
| 3    | ZMQbg/IO/0 | ZMQ I/O thread dequeues the message from the mailbox and writes it to the Unix domain socket via `write()` syscall                   |
| 4    | Kernel     | Kernel copies data from sender's socket buffer to receiver's socket buffer (`unix_stream_sendmsg` to `unix_stream_recvmsg`)          |
| 5    | ZMQbg/IO/0 | ZMQ I/O thread reads from the Unix domain socket via `recv()` syscall and enqueues the message to the receiver's mailbox             |
| 6    | Receiver   | pyzmq receives the ZMQ frame (`zmq_msg_recv`, `NOBLOCK`) and deserializes the msgpack bytes using `msgspec.decode()`                 |

Per message: 4 context switches (sender↔IO thread↔kernel↔IO thread↔receiver), 2 kernel syscalls (UDS write + recv). A WIP shared-memory transport eliminates steps 3–5 entirely, achieving ~400k QPS vs ZMQ's ~300k (1.33×) — see [Appendix B.1](#b1-shared-memory-transport-wip).

#### 7.1.4 Edge-Triggered FD Handling

ZMQ exposes a single file descriptor per socket for event loop integration (`zmq.FD`). Unlike regular sockets, this FD is **edge-triggered** — it signals state _change_, not data presence. A single edge fires when the socket transitions from "no messages" to "has messages", but does _not_ fire again for subsequent messages that arrive while existing ones are still buffered. If the handler reads only one message per callback (level-triggered style), remaining messages sit unprocessed until an unrelated state change re-triggers the FD. Both `_ZmqReceiverTransport` and `_ZmqSenderTransport` handle this with a drain-and-reschedule pattern (adapted from aiozmq's `_ZmqLooplessTransportImpl` [2]):

**Key Pattern:**

```python
def _on_readable(self) -> None:
    # 1. Drain ALL available messages synchronously
    while True:
        try:
            nbytes = self._sock.recv_into(self._recv_buf, zmq.NOBLOCK)
            self._deque.append(self._decoder.decode(self._recv_view[:nbytes]))
        except zmq.Again:
            break

    # 2. Wake waiter ONCE after draining
    if self._waiter:
        self._waiter.set_result(None)

    # 3. Reschedule to catch racing messages
    self._soon_call = self._loop.call_soon(self._on_readable)
```

| Step                   | Purpose                                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Drain loop             | Consume all buffered messages synchronously, since the edge notification will not re-trigger for data already present |
| Single wake            | Wake the waiting coroutine once after the entire drain completes, rather than once per message                        |
| `call_soon` reschedule | Schedule another drain to catch messages that arrived during the current drain, since no new edge fires for those     |

Step 3 addresses a race condition: if a message arrives while draining, there is no new edge notification. The `call_soon` reschedule catches these racing messages.

**Sender Fast/Slow Path:**

`_ZmqSenderTransport.send()` uses a two-tier strategy to avoid buffer allocation on the common case:

```python
def send(self, data: Any) -> None:
    serialized = self._encoder.encode(data)

    # Fast path: direct send when buffer is empty
    if not self._buffer:
        try:
            self._sock.send(serialized, zmq.NOBLOCK, copy=False)
            return
        except zmq.Again:
            pass

    # Slow path: buffer and register writer
    self._buffer.append(serialized)
    if not self._writing:
        self._writing = True
        self._loop.add_writer(self._fd, self._on_writable)

def _on_writable(self) -> None:
    # Drain buffer (same edge-triggered pattern as receiver)
    while self._buffer:
        try:
            self._sock.send(self._buffer[0], zmq.NOBLOCK, copy=False)
            self._buffer.popleft()
        except zmq.Again:
            break

    if not self._buffer:
        self._loop.remove_writer(self._fd)
        self._writing = False
    else:
        # Reschedule to catch racing writability
        self._soon_call = self._loop.call_soon(self._on_writable)
```

| Path  | Trigger                          | Behavior                                                               |
| ----- | -------------------------------- | ---------------------------------------------------------------------- |
| Fast  | Buffer empty and socket ready    | Direct `send(NOBLOCK)` — zero buffer overhead                          |
| Slow  | Socket would block (`zmq.Again`) | Append to `deque`, register `add_writer` callback                      |
| Drain | `_on_writable` fires             | Same edge-triggered drain + `call_soon` reschedule pattern as receiver |

#### 7.1.5 Design Choices

| Choice                 | Implementation                                                            | Alternative                                                      | Rationale                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fan-out topology       | N PUSH sockets (one per worker)                                           | Single shared PUSH/DEALER                                        | Explicit worker targeting; prerequisite for future work-stealing or load-aware dispatch ([§7.1.2](#712-topology))                                                                      |
| Event loop integration | `add_reader`/`add_writer` on ZMQ FD                                       | pyzmq async, aiozmq                                              | Direct control over hot path; avoids extra abstraction layers (adapted from aiozmq's `_ZmqLooplessTransportImpl` [2])                                                                  |
| FD handling            | Edge-triggered drain + `call_soon` reschedule                             | Level-triggered (one-msg-per-callback)                           | Must drain all messages per callback; reschedule catches racing arrivals (see [§7.1.4](#714-edge-triggered-fd-handling))                                                               |
| Receive queue          | `deque` + `Future`                                                        | `asyncio.Queue`                                                  | No lock overhead; batched waiter notification (wake once after drain, not per-message)                                                                                                 |
| Send strategy          | Direct `NOBLOCK` fast path, `add_writer` + buffer slow path               | Always buffer                                                    | Fast path avoids buffer allocation when socket ready; slow path reuses edge-triggered drain (see [§7.1.4](#714-edge-triggered-fd-handling))                                            |
| Serialization          | `msgspec.msgpack` [5] with reusable typed `Encoder`/`Decoder`             | pickle, JSON                                                     | Schema-aware decode direct to Struct; tagged union discrimination without try/except (see [§7.1.1](#711-serialization))                                                                |
| Recv buffer            | Pre-allocated `bytearray` + `recv_into()`, decode from `memoryview` slice | `recv(copy=False)` (Frame alloc + `getsockopt(RCVMORE)` per msg) | Zero per-message allocation; avoids Frame object + wasted RCVMORE syscall. +10-17% msg/s for small messages (see [§7.2](#72-benchmarks)). `send(copy=False, track=False)` on send path |
| I/O threads            | 4 C++ background threads                                                  | 1, or scale with workers                                         | Tested for 100 workers on 224-core x86; each thread needs a physical core; throughput scales with message rate, not worker count                                                       |

### 7.2 Benchmarks

The custom transport's primary advantage is eliminating pyzmq's async abstraction layer — all I/O runs directly on uvloop via `add_reader`/`add_writer` callbacks, avoiding the overhead of pyzmq's `ZMQEventLoop` and its per-message future allocation. The benchmarks below measure single-process round-trip (send + recv) throughput across message sizes representative of LLM inference traffic.

**Test Configuration:** Single-process round-trip (send + recv), varying message sizes.

| Message Type | Size                  | Custom (msg/s) | Custom (MB/s) | pyzmq async (msg/s) | pyzmq async (MB/s) | Speedup |
| ------------ | --------------------- | -------------- | ------------- | ------------------- | ------------------ | ------- |
| Query        | 32 chars (101 B)      | 512,400        | 51.8          | 64,552              | 8.3                | 7.9x    |
| StreamChunk  | 32 chars (52 B)       | 633,400        | 32.9          | 66,800              | 6.4                | 9.5x    |
| Query        | 512 chars (582 B)     | 507,800        | 295.5         | 64,400              | 39.2               | 7.9x    |
| StreamChunk  | 512 chars (533 B)     | 535,800        | 285.6         | 63,424              | 36.6               | 8.4x    |
| Query        | 4096 chars (4166 B)   | 329,200        | 1371.4        | 60,200              | 252.4              | 5.5x    |
| StreamChunk  | 4096 chars (4117 B)   | 358,200        | 1474.7        | 61,553              | 256.1              | 5.8x    |
| Query        | 16384 chars (16454 B) | 158,200        | 2603.0        | 53,316              | 878.7              | 3.0x    |
| StreamChunk  | 16384 chars (16405 B) | 169,200        | 2775.7        | 53,787              | 884.7              | 3.1x    |
| Query        | 32768 chars (32838 B) | 67,600         | 2219.8        | 44,089              | 1449.0             | 1.5x    |
| StreamChunk  | 32768 chars (32789 B) | 78,600         | 2577.2        | 43,651              | 1433.2             | 1.8x    |

**Observations:**

- 6-9x msg/s over pyzmq async for typical LLM response sizes (52B - 4KB); narrows with larger messages as memory bandwidth dominates

---

## 8. HTTP Engine

The HTTP engine provides the low-level TCP connection management and HTTP/1.1 request/response handling that each worker uses to communicate with its assigned endpoint.

**Classes:**

| Class                  | Source    | Description                                            |
| ---------------------- | --------- | ------------------------------------------------------ |
| `HttpResponseProtocol` | `http.py` | `asyncio.Protocol` with `httptools` (llhttp) parser    |
| `HttpRequestTemplate`  | `http.py` | Pre-built HTTP headers, minimal per-request allocation |
| `ConnectionPool`       | `http.py` | TCP connection lifecycle, reuse, limiting, and warmup  |
| `PooledConnection`     | `http.py` | Connection wrapper with staleness detection            |

### 8.1 HttpResponseProtocol

`HttpResponseProtocol` bridges two programming models: asyncio's callback-based **Protocol** interface (see [§8.1](#81-httpresponseprotocol)) and the async/await world that the worker's response tasks live in. It subclasses `asyncio.Protocol` and wraps `httptools.HttpResponseParser` — a Python binding to Node.js's llhttp, the same C HTTP parser used in production by Node.js and other high-performance servers.

**Why this architecture:** asyncio's transport/protocol layer operates at the callback level — the event loop calls `data_received(data)` whenever TCP bytes arrive, with no coroutine suspension involved. This is the fastest path for I/O in Python's async ecosystem, but it means the protocol cannot `await` anything. Meanwhile, the worker's response tasks need to `await read_headers()` and `async for chunks in iter_body()`. The protocol bridges this gap using Futures and Events as synchronization primitives: callbacks _set_ them, async methods _await_ them.

**How callbacks become awaitable results:** When TCP bytes arrive, the C parser fires synchronous callbacks (`on_headers_complete`, `on_body`, `on_message_complete`). Each callback sets a Future or Event — a zero-cost bridge primitive. On the other side, async worker code awaits those same primitives. The diagram below shows the three parallel lanes: each callback (green, left) sets a bridge primitive (amber diamond, center), which an async method (blue, right) awaits.

<img src="res/endpoint_client/17c_http_response_protocol.png" alt="HttpResponseProtocol Data Flow" width="775">

**FD-based event loop handling:** The diagram below shows the full lifecycle — callbacks on the left write to shared state (Futures, Events, chunk lists) in the center, which the async API on the right awaits. The `iter_body()` sync-drain loop is the key optimization: it drains all buffered chunks synchronously before yielding to the event loop, reducing context switches when data arrives faster than processing.

<img src="res/endpoint_client/18_fd_event_loop_protocol.png" alt="HttpResponseProtocol FD Event Loop" width="850">

**Public API:**

| Method           | Async | Description                                                                                                                                       |
| ---------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `read_headers()` | Yes   | Returns `(status_code, headers)`. Fast path: returns immediately if `_headers_complete` already set.                                              |
| `read_body()`    | Yes   | Returns full body bytes (`b"".join(_body_chunks)`). Used for non-streaming responses.                                                             |
| `iter_body()`    | Yes   | Async generator yielding chunk batches. Drains `_stream_chunks` synchronously, then `await _stream_event.wait()`. See [§8.5](#85-design-choices). |
| `write(data)`    | No    | Delegates to `transport.write()` (kernel-buffered, non-blocking).                                                                                 |
| `reset()`        | No    | Clear all state for connection reuse. Lazy parser creation — `_parser = None` until first `data_received()`, amortizing reset cost.               |

**Connection reuse:** Each `PooledConnection` holds one `HttpResponseProtocol` instance for the lifetime of the TCP connection. Between requests, `reset()` clears response state without closing the socket. The parser is set to `None` and lazily re-created on next `data_received()` — this avoids allocating the C parser object during reset when the connection may sit idle.

**TCP half-close handling:** `eof_received()` marks `_connection_lost = True` to prevent reuse of a connection where the server sent FIN. Without this, a reused connection would accept writes (TCP half-close allows it) but reads would hang forever — a known asyncio footgun [9]. The `should_close` property combines three conditions: `_should_close` (server sent `Connection: close`), `_connection_lost` (EOF/error), and `_exc is not None` (parse error). The pool checks this after each response to decide whether to release or discard the connection.

### 8.2 HttpRequestTemplate

HTTP libraries like `aiohttp` and `httpx` build request bytes from scratch on every call — assembling the request line, encoding headers into a dict, serializing the body, and concatenating everything. For a benchmarking client that sends thousands of structurally identical requests per second (same endpoint, same path, same auth headers), this per-request work is pure waste. `HttpRequestTemplate` eliminates it by splitting the HTTP request into **static** parts (built once) and **dynamic** parts (built per-request), then concatenating them with a single `b"".join()`.

**Public API:**

| Method                                           | Async | Description                                                                |
| ------------------------------------------------ | ----- | -------------------------------------------------------------------------- |
| `from_url(host, port, path)`                     | No    | Classmethod: create template with pre-encoded request line + Host header   |
| `cache_headers(headers)`                         | No    | Pre-encode headers (e.g. Authorization) into bytes; call once during setup |
| `build_request(body, streaming, extra_headers?)` | No    | Build complete HTTP/1.1 request bytes; fast path when no extra headers     |

**Request byte segments:**

| Segment        | When Built   | Description                                                                                                                                                                      |
| -------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Static prefix  | Once (init)  | `POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:8000\r\n` — pre-encoded at construction via `from_url()`                                                                  |
| Cached headers | Once (setup) | `Authorization: Bearer sk-...\r\n` — pre-encoded via `cache_headers()`; reused across all requests                                                                               |
| Content-Type   | Once (class) | Two class-level `bytes` constants: `HEADERS_STREAMING` and `HEADERS_NON_STREAMING`. The streaming/non-streaming branch selects between them — no string encoding at request time |
| Content-Length | Per-request  | `Content-Length: 1234\r\n\r\n` — the only header that changes per request; built from a single f-string and ASCII-encoded                                                        |
| Body           | Per-request  | JSON bytes produced by the adapter's `encode_query()` ([§9.1](#91-httprequestadapter))                                                                                           |

![build_request() fast/slow path](res/endpoint_client/11_build_request_flow.png)

The fast path (no extra headers) joins the 5 segments above in a single `b"".join()` — no allocation beyond the final buffer. The slow path (extra headers present) adds a `frozenset` cache key lookup; the first call per unique header set encodes the headers (~1us), subsequent calls hit the cache (~50ns) and join 6 segments.

### 8.3 Connection Pool

Each worker maintains its own `ConnectionPool` to its assigned endpoint. The pool manages the full TCP connection lifecycle — creation, reuse, limiting, staleness detection, warmup, and shutdown. It uses a **LIFO stack** for idle connections (recently-used connections are reused first, keeping them "hot" in kernel buffers and reducing staleness) and a **FIFO waiter queue** (`OrderedDict`) for fairness when all connections are in use.
**Public API:**

| Method          | Async   | Description                                                       |
| --------------- | ------- | ----------------------------------------------------------------- |
| `acquire()`     | **Yes** | Get connection: idle stack (LIFO) → create new → wait for release |
| `release(conn)` | No      | Return connection to idle stack; notify waiters                   |
| `warmup(count)` | **Yes** | Pre-establish TCP connections via concurrent `gather()`           |
| `close()`       | **Yes** | Close all connections and cancel pending waiters                  |

<img src="res/endpoint_client/09_connection_pool.png" alt="Connection Pool Acquire" width="800">

**Idle Connection Validation (`is_stale`):**

When `acquire()` pops a connection from the idle stack, it must verify the server hasn't closed it. `PooledConnection.is_stale()` combines a fast-path skip with a zero-cost kernel probe:

```python
def is_stale(self) -> bool:
    # Fast path: skip select() for recently-used connections.
    # A server won't close a connection within 1s of last use,
    # so the syscall is unnecessary. Saves ~1µs per acquire.
    if time.monotonic() - self.last_used < 1.0:
        return False

    # Zero-timeout select(): asks the kernel if any data or
    # errors are pending. A healthy idle socket has neither.
    readable, _, exceptional = select.select([fd], [], [fd], 0)
    return bool(readable or exceptional)
```

| `select()` Result  | Kernel State                                             | Pool Action                       |
| ------------------ | -------------------------------------------------------- | --------------------------------- |
| `readable=True`    | Server sent FIN, indicating it is closing the connection | Discard connection, try next idle |
| `exceptional=True` | Socket error such as a TCP reset from the server         | Discard connection, try next idle |
| Both empty         | No pending data or errors on the socket                  | Connection is healthy, use it     |

The fast-path skip (`< 1.0s`) avoids the `select()` syscall entirely for connections in active rotation — reducing validation from ~1.2µs to ~161ns (see latencies below).

**Operation Latencies:**

Per-operation latency measurements for `ConnectionPool` (localhost TCP, uvloop):

| Operation              | Median | p99   | Description                                          |
| ---------------------- | ------ | ----- | ---------------------------------------------------- |
| `_create_connection()` | 92µs   | 204µs | New TCP connection (3-way handshake)                 |
| `_try_get_idle()`      | 564ns  | 610ns | Pop connection from idle stack                       |
| `acquire()`            | 665ns  | 760ns | Get connection (with idle hit)                       |
| `release()`            | 452ns  | 543ns | Return connection to pool                            |
| `is_stale()`           | 1.2µs  | 2.5µs | Check if server closed connection (`select` syscall) |
| `is_stale()` [skip]    | 161ns  | 304ns | Skip path (recently used < 1s)                       |
| `is_alive()`           | 130ns  | 149ns | Check socket state (flag check only)                 |

**Hot Path vs Cold Path:**

| Scenario                 | Median Latency | Notes                                          |
| ------------------------ | -------------- | ---------------------------------------------- |
| Pool has idle connection | ~1µs           | `acquire()` + `release()` overhead per request |
| Pool empty, must create  | ~93µs          | ~100x slower; TCP handshake on critical path   |

### 8.4 Socket Config

The `_SocketConfig` class (`http.py`) defines socket options applied to all TCP connections created by the connection pool. These options are tuned for low-latency streaming workloads where individual request latency directly impacts benchmark measurements.

| Option             | Value | Effect                                                                                                            | Interaction                                                                                                                                                                          |
| ------------------ | ----- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TCP_NODELAY`      | 1     | Disables Nagle's algorithm, allowing small packets to be sent immediately rather than being buffered for batching | With `TCP_QUICKACK`: eliminates both send batching (Nagle) and receive-side delayed ACK, removing the two primary sources of TCP-induced latency                                     |
| `TCP_QUICKACK`     | 1     | Immediately acknowledge received packets instead of delaying acknowledgments (Linux-specific)                     | Not sticky — kernel may revert to delayed ACK mode; re-applied per connection. Together with `TCP_NODELAY`, ensures neither side introduces artificial delays                        |
| `SO_KEEPALIVE`     | 1     | Enable TCP keepalive probes at the socket level                                                                   | Activates the kernel keepalive mechanism; actual probe timing controlled by `TCP_KEEPIDLE`, `TCP_KEEPCNT`, `TCP_KEEPINTVL`                                                           |
| `TCP_KEEPIDLE`     | 1s    | Start probing after 1 second idle (Linux-specific)                                                                | With `TCP_KEEPCNT=1` + `TCP_KEEPINTVL=1s`: total dead-connection detection time is 2 seconds (1s idle + 1 probe × 1s interval)                                                       |
| `TCP_KEEPCNT`      | 1     | 1 failed probe = connection declared dead                                                                         | Aggressive: default is 9. Appropriate here because a failed probe to a local/VPC endpoint indicates genuine failure, not transient packet loss                                       |
| `TCP_KEEPINTVL`    | 1s    | 1 second between probes                                                                                           | With only 1 probe needed (`TCP_KEEPCNT=1`), a single missed probe triggers connection close                                                                                          |
| `SO_RCVBUF`        | 4 MB  | Receive buffer size                                                                                               | 4 MB buffer accommodates approximately 1M tokens (at 4 bytes per token) for offline-mode full responses. Prevents the kernel from dropping data when the application is briefly busy |
| `SO_SNDBUF`        | 4 MB  | Send buffer size                                                                                                  | Sized for large request payloads (long prompts). Allows `write()` to complete without blocking even for large HTTP bodies                                                            |
| `TCP_USER_TIMEOUT` | 0     | Disabled — no timeout on unacknowledged sent data (Linux-specific)                                                | Keepalive handles dead-connection detection; setting this to 0 avoids interfering with long-running SSE streams where the server may take seconds between chunks                     |

**Cross-platform compatibility:** Applied via `_SocketConfig.apply(sock)` with `hasattr()` checks for Linux-specific options (`TCP_KEEPIDLE`, `TCP_QUICKACK`, `TCP_USER_TIMEOUT`). On non-Linux platforms, these options are silently skipped — the system runs with reduced tuning but remains functional.

### 8.5 Design Choices

| Choice                   | Implementation                                                                                                                             | Alternative                            | Rationale                                                                                                                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HTTP parser              | `httptools` [3] (llhttp, C); lazy parser creation (`_parser = None` in `reset()`, constructed on first `data_received()`)                  | `httpx`, `aiohttp`                     | Same parser as Node.js; zero-copy callbacks. Lazy creation amortizes reset cost — parser object only allocated when response data actually arrives                                                                                                                               |
| Idle connection strategy | LIFO stack                                                                                                                                 | Random from idle list                  | Lower error rate: high load reuses hot connections; low load (long idle) stale ones sink to bottom                                                                                                                                                                               |
| Waiter queue             | FIFO via `OrderedDict`                                                                                                                     | List, deque                            | Fair scheduling; O(1) insert/remove                                                                                                                                                                                                                                              |
| Connection limiting      | `max_connections` cap; tracks `len(all_connections) + _creating`                                                                           | Unlimited                              | Prevents ephemeral port exhaustion — a real production failure mode at high concurrency (see [§8.6](#86-benchmark-results-vs-aiohttp)). Counting in-progress connections (`_creating`) prevents race where concurrent `acquire()` calls overshoot the limit during TCP handshake |
| Staleness detection      | `select()` probe on FD with zero timeout                                                                                                   | Timeout-based, `poll()`                | Detects server FIN without I/O. `select()` over `poll()` avoids poll object creation overhead                                                                                                                                                                                    |
| Preclose skip            | `time.monotonic() - last_used < 1.0` → return not-stale immediately                                                                        | Always probe                           | Server unlikely to close within 1s of last use; skips `select()` syscall entirely on hot connections under load                                                                                                                                                                  |
| Socket tuning            | `TCP_NODELAY` + `TCP_QUICKACK` [7], 4MB buffers                                                                                            | Defaults                               | `TCP_NODELAY` disables Nagle batching; `TCP_QUICKACK` disables delayed ACK; together they eliminate both send and receive latency sources. 4MB buffers sized for offline-mode full responses. See [§8.4](#84-socket-config) for full socket option table                         |
| Ephemeral port detection | Read `/proc/sys/net/ipv4/ip_local_port_range` + `/proc/net/tcp{,6}`                                                                        | Manual configuration                   | Auto-sizes `max_connections` to available ports; raises `RuntimeError` if explicit value exceeds limit                                                                                                                                                                           |
| Connection warmup        | Default 50% of pool; `gather(return_exceptions=True)` for concurrent establishment                                                         | 0% or 100%                             | 100% = SYN flood risk to server; 0% = ~100x cold-start penalty per [§8.3](#83-connection-pool); 50% balances startup load vs latency. `return_exceptions=True` ensures individual failures don't abort the warmup batch                                                          |
| Idle connection discard  | `max_idle_time=4.0s` proactive close                                                                                                       | Rely on staleness only                 | Proactive discard avoids keepalive race with server timeout; 4s chosen to be shorter than typical server keepalive (5-60s)                                                                                                                                                       |
| `iter_body` sync drain   | `while _stream_chunks: yield chunks` before `await _stream_event.wait()`                                                                   | Await after every chunk                | Drains all available data synchronously before yielding to event loop; reduces context switches when data arrives faster than processing                                                                                                                                         |
| Request build fast path  | No extra headers: single `b"".join(...)` skipping cache; with extra headers: `frozenset(headers.items())` keyed cache of pre-encoded bytes | Single code path, encode every request | Common case (no per-request headers) skips dict ops and cache lookup entirely. When headers present, cache hits cost ~50ns vs ~1µs to re-encode; same Authorization header repeated across all requests                                                                          |
| Body await primitive     | `asyncio.Event.wait()` in `iter_body()`                                                                                                    | `asyncio.Queue`                        | No lock overhead; only suspends when buffer empty. `Event.set()` from `on_body` callback is a single pointer write vs Queue's internal locking                                                                                                                                   |

### 8.6 Benchmark Results vs aiohttp

Comparison against aiohttp's connection handling.

**Microbenchmarks:**

| Benchmark            | Throughput Speedup | p99 Improvement |
| -------------------- | ------------------ | --------------- |
| Request Building     | 2.20x              | 2.73x           |
| Pool Acquire/Release | 5.11x              | 5.79x           |
| Full Request Cycle   | 7.81x              | 7.43x           |
| Streaming Response   | 3.19x              | 4.94x           |

**End-to-End Benchmark:** Offline mode, 60k queries, vLLM backend (Qwen/Qwen2.5-0.5B-Instruct):

| Implementation                | QPS    | TPS     | Errors |
| ----------------------------- | ------ | ------- | ------ |
| aiohttp                       | 563.80 | 733.76  | 20,956 |
| Custom (max_connections=1024) | 721.62 | 1443.23 | 0      |
| Custom (max_connections=22k)  | 595.75 | 1170.80 | 1,042  |

**End-to-End Benchmark:** Offline mode, 20k queries (within ephemeral port limit):

| Implementation                | QPS    | TPS     | Errors |
| ----------------------------- | ------ | ------- | ------ |
| aiohttp                       | 532.04 | 959.06  | 1,974  |
| Custom (max_connections=1024) | 696.28 | 1392.56 | 0      |
| Custom (max_connections=22k)  | 648.97 | 1297.94 | 0      |

**Observations:**

- Bounded connection pool eliminates ephemeral port exhaustion errors
- Lower max_connections (1024) achieves higher throughput than unlimited (22k) due to reduced connection churn
- Custom implementation eliminates "Cannot Assign Given Address" and connection timeout errors common with aiohttp under high load

---

## 9. Adapters

The adapter and accumulator layers convert between the endpoint client's internal types (`Query`, `QueryResult`, `StreamChunk`) and endpoint-specific wire formats. Each API backend (OpenAI, SGLang) provides an adapter and accumulator pair. Adding a new backend requires implementing these two interfaces — no changes to the HTTP engine, transport, or worker code.

**Classes:**

| Class                    | Source                    | Description                                        |
| ------------------------ | ------------------------- | -------------------------------------------------- |
| `HttpRequestAdapter`     | `adapter_protocol.py`     | ABC: encode queries, decode responses, parse SSE   |
| `SSEAccumulatorProtocol` | `accumulator_protocol.py` | Protocol: per-request streaming token accumulation |

### 9.1 HttpRequestAdapter

Abstract base class (`endpoint_client/adapter_protocol.py`) for HTTP request/response encoding. All methods are `@classmethod` — adapters carry no per-instance state.

**Public API:**

| Method                                      | Async | Description                                                                                                                                             |
| ------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_transforms(model_params)`          | No    | Returns `list[Transform]` to shape dataset rows into `Query.data` dicts for this API format. Must include a `ColumnFilter` to strip extraneous columns. |
| `encode_query(query)`                       | No    | Serialize `Query` to HTTP request body bytes (JSON via `msgspec.json.Encoder`)                                                                          |
| `decode_response(response_bytes, query_id)` | No    | Deserialize HTTP response body to `QueryResult` (JSON via `msgspec.json.Decoder` with typed schema)                                                     |
| `decode_sse_message(json_bytes)`            | No    | Extract content string from a single SSE JSON message                                                                                                   |
| `parse_sse_chunk(buffer, end_pos)`          | No    | Extract all content strings from a buffer region. Default: regex `findall` → loop of `decode_sse_message` calls                                         |

The base class defines `SSE_DATA_PATTERN: re.Pattern[bytes] = re.compile(rb"data:\s*(\{[^\n]+\})")` — a pre-compiled regex shared by all adapters for extracting JSON documents from SSE `data:` lines.

All adapters use `msgspec` [5] for serialization — class-level `Encoder`/`Decoder` instances are reused across all requests, and typed decoders write directly into Struct fields with no intermediate `dict` allocation. See [A.5](#a5-msgspec-serialization) for the full set of msgspec usage patterns shared across adapters and transports.

### 9.2 SSEAccumulatorProtocol

Protocol class (`endpoint_client/accumulator_protocol.py`) for collecting streaming SSE deltas into final results. Unlike adapters, accumulators are per-request instances (they track state across chunks).

**Public API:**

| Method                                  | Async | Description                                                                                  |
| --------------------------------------- | ----- | -------------------------------------------------------------------------------------------- |
| `__init__(query_id, stream_all_chunks)` | No    | Initialize with request ID and chunk emission mode                                           |
| `add_chunk(delta)`                      | No    | Process one SSE delta. Returns `StreamChunk` if content should be emitted, `None` otherwise. |
| `get_final_output()`                    | No    | Return complete accumulated result after stream ends                                         |

**Chunk emission modes:**

| `stream_all_chunks` | Behavior                                                                                            | Use Case                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `False` (default)   | `add_chunk` returns `StreamChunk` only for the first content delta; subsequent deltas return `None` | Time-to-first-token (TTFT) measurement only; minimizes IPC traffic by sending a single chunk |
| `True`              | `add_chunk` returns `StreamChunk` for every content delta                                           | Full token-by-token streaming to main process for per-token latency measurement              |

The first emitted chunk carries `metadata={"first_chunk": True}` for TTFT measurement. The final `QueryResult` from `get_final_output()` carries `metadata={"final_chunk": True}`.

### 9.3 SSE Stream Parsing

SSE streams deliver multiple JSON messages per network read. The parsing strategy combines regex extraction with batched exception handling to minimize per-message overhead.

`TODO: Populate with ablation study results (regex vs line-by-line, try-per-iteration vs try-outside-loop, msgspec vs stdlib json)`

**Pattern:**

```python
# Single-pass regex extraction over the raw SSE buffer — C-level findall,
# avoids line-by-line splitting and per-line prefix checks.
json_docs = SSE_DATA_PATTERN.findall(buffer[:end_pos])
parsed_contents = []

# try/except wraps entire loop rather than per-iteration: exception frame
# setup has measurable overhead, so we amortize it across the batch.
# Non-content SSE messages (role, finish_reason) raise on decode — expected.
try:
    for json_doc in json_docs:
        content = decode_sse_message(json_doc)
        parsed_contents.append(content)
except Exception:
    pass

return parsed_contents
```

**Design Choices:**

| Choice                      | Implementation                             | Rationale                                                                                                                     |
| --------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Regex extraction            | Pre-compiled pattern with `findall()`      | Extracts all JSON documents in a single pass over the buffer using C-level regex iteration, avoiding line-by-line splitting   |
| Exception handler placement | `try` wraps entire loop, not per-iteration | Python exception frame setup has measurable overhead; wrapping the entire batch shares that cost across all messages          |
| Typed decoding              | `msgspec.json.Decoder` with schema         | Decodes JSON directly into typed Struct fields, skipping the intermediate `dict` allocation that stdlib `json.loads` produces |
| Batch yield                 | Yields `list[str]` per network read        | Returns all parsed content from one network read as a single batch, reducing async suspend/resume cycles in the caller        |

### 9.4 Implementations

| Class                   | Source                             | Description                                      |
| ----------------------- | ---------------------------------- | ------------------------------------------------ |
| `OpenAIMsgspecAdapter`  | `openai/openai_msgspec_adapter.py` | OpenAI Chat Completions adapter (msgspec codecs) |
| `OpenAISSEAccumulator`  | `openai/accumulator.py`            | OpenAI streaming delta accumulator               |
| `SGLangGenerateAdapter` | `sglang/adapter.py`                | SGLang generate endpoint adapter                 |
| `SGLangSSEAccumulator`  | `sglang/accumulator.py`            | SGLang streaming delta accumulator               |

All OpenAI and SGLang request/response types follow the Struct conventions from [A.5](#a5-msgspec-serialization) (`frozen`, `kw_only`, `omit_defaults`, `gc=False`). Notable exceptions: `ChatCompletionResponse` uses `omit_defaults=False` (must encode all fields for downstream consumers). OpenAI `SSEDelta` includes a `reasoning: str` field for reasoning model outputs. `SSEMessage.choices` is typed as `tuple[SSEChoice, ...]` (not `list`) for immutability.

---

## 10. Initialization & Shutdown

The initialization and shutdown subsystem manages worker process lifecycle: spawning, CPU pinning, readiness barrier, and graceful termination.

**Classes:**

| Class           | Source              | Description                                                         |
| --------------- | ------------------- | ------------------------------------------------------------------- |
| `WorkerManager` | `worker_manager.py` | Orchestrates worker lifecycle: spawn, pin, liveness check, shutdown |

### 10.1 WorkerManager

The `WorkerManager` (main process) orchestrates the worker lifecycle: spawn, CPU pinning, liveness-check, and shutdown. Each worker process goes through a deterministic startup sequence before entering the request-processing main loop (see [§6](#6-worker)).

**Public API:**

| Method         | Async   | Description                                                  |
| -------------- | ------- | ------------------------------------------------------------ |
| `initialize()` | **Yes** | Spawn workers, pin CPUs, wait for readiness signals          |
| `shutdown()`   | **Yes** | Terminate → wait → kill remaining → join → cleanup transport |

### 10.2 Startup

<img src="res/endpoint_client/17_startup_sequence.png" alt="Startup Sequence" width="750">

### 10.3 Shutdown

<img src="res/endpoint_client/18_shutdown_sequence.png" alt="Shutdown Sequence" width="740">

---

## 11. Performance Analysis

Empirical measurements of the endpoint client under sustained load. Benchmarks use `benchmark_httpclient.py` (`src/inference_endpoint/utils/`). The profiling approach starts with macro-level benchmarks (§11.1–§11.2) to establish throughput ceilings, then drills into per-worker behavior (§11.3–§11.8) using progressively finer-grained tools to identify where CPU time is actually spent.

**Key findings:**

| Section | Tool                  | What It Measures                           | Finding                                                                                                                                                                                                                  |
| ------- | --------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 11.1    | benchmark (offline)   | End-to-end throughput vs worker count      | ~300k QPS @ 14 workers on both x86 and ARM Grace                                                                                                                                                                         |
| 11.2    | benchmark (streaming) | End-to-end throughput with SSE streaming   | x86: ~90.6k QPS @ 96 workers; ARM Grace: ~133k QPS @ 132 workers                                                                                                                                                         |
| 11.3    | `pidstat -t`          | Per-thread CPU split (user vs kernel)      | The worker runs two active threads: the event loop (78% CPU, 55% user / 23% kernel) and the ZMQ I/O thread (29% CPU, 8% user / 20% kernel). The ZMQ thread spends more than twice as much time in kernel as in userspace |
| 11.4    | `pidstat -w`          | Context switch frequency                   | 3k voluntary and 1.3k involuntary context switches per second; the worker yields on epoll more often than it is preempted by the scheduler                                                                               |
| 11.5    | `perf top`            | CPU symbol-level hotspots per thread       | ~45% of total CPU is spent inside kernel syscalls (22.73% python3 thread + 22.67% ZMQ thread). The CPython interpreter (`_PyEval_EvalFrameDefault`) is the single hottest userspace function at 7.25% exclusive CPU      |
| 11.6    | `strace -c`           | Syscall frequency and cumulative time      | 131k syscalls per second dominated by `write` (57%) and `read` (23%). `getpid` accounts for 53k redundant calls from libzmq's fork-safety check, wasting ~5% of total worker CPU                                         |
| 11.7    | `runqlat`             | Kernel scheduling delay histogram          | Most thread wake-ups are scheduled within single-digit microseconds; the worker rarely waits for a CPU core                                                                                                              |
| 11.8    | `tiptop`              | Hardware performance counters (IPC, cache) | Instructions per cycle is approximately 1.0, consistent with CPython interpreter workloads                                                                                                                               |

**IPC overhead:** ~25% of worker CPU (reverse Amdahl's Law: WIP [shared-memory transport](#b1-shared-memory-transport-wip) at ~400k QPS vs ZMQ's ~300k QPS → `1 - 1/1.33 ≈ 0.25`). Down from ~56% before `recv_into` and `array_like` optimizations. Pre-optimization profiling detail in §11.3–§11.8.

**Test Environments:**

| Role            | CPU                               | Arch    | Cores / Threads              | Sections    |
| --------------- | --------------------------------- | ------- | ---------------------------- | ----------- |
| Benchmark (x86) | Intel Xeon Platinum 8570 × 2      | x86_64  | 112 cores / 224 threads (HT) | §11.1–§11.2 |
| Benchmark (ARM) | NVIDIA Grace × 2                  | aarch64 | 144 cores                    | §11.1–§11.2 |
| Profiling       | AMD Ryzen Threadripper PRO 7965WX | x86_64  | 24 cores / 48 threads (SMT)  | §11.3–§11.8 |

### 11.1 Offline Roofline

**x86 (Intel Xeon Platinum 8570 × 2, 112 cores):**

![Offline Benchmark Sweep — x86](res/endpoint_client/19_bench_offline_scaling.png)

**ARM (NVIDIA Grace × 2, 144 cores):**

![Offline Benchmark Sweep — ARM](res/endpoint_client/19b_bench_offline_scaling_arm.png)

Offline (non-streaming) scaling sweep — 1 query = 1000 characters/tokens, `max_concurrency=100000`. The benchmark server (`MaxThroughputServer`) returns pre-built responses with no compute, so all measured overhead is purely client-side. This measures raw request dispatch and response collection throughput without per-token streaming overhead:

- **Send Rate** reaches ~300k QPS at ~14 workers on both x86 and ARM Grace. Beyond the plateau, send throughput is flat.
- **Recv Rate** tracks send rate closely in offline mode since each response is returned as a single body read — no per-chunk event loop pressure.
- **Stall%** measures the fraction of send time the benchmark spent blocked on back-pressure (in-flight requests hit `max_concurrency`).

### 11.2 Streaming Worst-Case

**x86 (Intel Xeon Platinum 8570 × 2, 112 cores):**

![Streaming Benchmark Sweep — x86](res/endpoint_client/20_bench_streaming_scaling.png)

**ARM (NVIDIA Grace × 2, 144 cores):**

![Streaming Benchmark Sweep — ARM](res/endpoint_client/20b_bench_streaming_scaling_arm.png)

Streaming scaling sweep (`stream_interval=1` — server emits 1 character per SSE chunk, so a 1000-char response produces ~1000 SSE events; worst-case for event loop and parsing pressure), 4–128 workers, `duration=10.0`, `max_concurrency=100000`, with per-second variation bands:

- **Send Rate** peaks at ~90.6k QPS at ~96 workers on x86; ~133k QPS at ~132 workers on ARM Grace.
- **Recv Rate** peaks at ~79.4k resp/s on x86; ~121.3k resp/s on ARM Grace.
- **SSE Rate** scales near-linearly to ~79.7M SSE-pkts/s on x86; ~121M SSE-pkts/s on ARM Grace. Per-worker streaming throughput is independent — each worker's `iter_body()` drain loop ([§6.2](#62-call-chain)) processes chunks without contention.
- **Stall%** remains nonzero even at high worker counts — streaming is recv-limited (main process fan-in bottleneck).

The remaining profiles (§11.3–§11.8) were captured on the profiling machine — AMD Ryzen Threadripper PRO 7965WX (24 cores / 48 threads) — to isolate per-worker behavior under streaming load:

| Tool            | Command                       | What It Measures                                                  |
| --------------- | ----------------------------- | ----------------------------------------------------------------- |
| `pidstat`       | `pidstat -p <pid> -t 5`       | Per-thread CPU breakdown (usr/sys/total) at 5-second intervals    |
| `pidstat`       | `pidstat -p <pid> -w 5`       | Voluntary and involuntary context switches per second             |
| `perf top`      | `sudo perf top -p <pid>`      | Live CPU sampling — hottest functions across kernel and userspace |
| `strace`        | `sudo strace -c -p <pid>`     | Syscall frequency, cumulative time, and error counts              |
| `runqlat-bpfcc` | `sudo runqlat-bpfcc -p <pid>` | BPF-traced kernel run queue latency histogram (scheduling delay)  |
| `tiptop`        | `tiptop -p <pid>`             | Hardware performance counters — IPC, cache misses, branch misses  |

### 11.3 Worker Thread Profile (`pidstat -t`)

![pidstat -t — per-thread CPU breakdown](res/endpoint_client/21_pidstat_threads.png)

`pidstat -p <pid> -t 5` output for a single worker process during a streaming run. Two 5-second sample intervals are shown, followed by averages. In `pidstat -t` output, the first `python3` row (TID = PID) is the **process total** — the aggregate of all threads. Subsequent rows are individual threads:

| Row                     | %usr | %sys | %wait | %CPU | Role                                                                                       |
| ----------------------- | ---- | ---- | ----- | ---- | ------------------------------------------------------------------------------------------ |
| python3 (process total) | 65   | 42   | 2.80  | 107  | Aggregate across all threads — not a thread itself                                         |
| python3 (worker thread) | 55   | 23   | 2.80  | 78   | Main worker thread — uvloop event loop running `_run_main_loop`, protocol callbacks, tasks |
| python3 (idle thread)   | 0    | 0    | 0.0   | 0    | Idle thread — no activity during profiling                                                 |
| `ZMQbg/IO/0`            | 8    | 20   | 9.30  | 29   | ZMQ background I/O thread — handles IPC socket reads/writes at the C++ level               |
| `_jemalloc_bg_thd`      | 0    | 0    | 0.0   | 0    | jemalloc background purge thread — negligible overhead                                     |
| `iou-sqp-*`             | 0    | 0    | 0.0   | 0    | io_uring submission queue polling thread — present but idle (not used by this workload)    |
| `ZMQbg/Reaper`          | 0    | 0    | 0.0   | 0    | ZMQ socket cleanup thread — negligible                                                     |

The worker thread consumes ~78% CPU (55% usr + 23% sys), with sys time reflecting kernel socket operations (`sendmsg`/`recvmsg`, `epoll_wait`). The ZMQ I/O thread adds ~29% CPU independently but with an inverted profile: it spends more than 2× as much time in kernel (20% sys) as in userspace (8% usr), confirming that ZMQ's C++ layer is a thin dispatch wrapper with the real work happening in kernel I/O (Unix domain socket reads/writes, polling). Combined, the process total reaches ~107% (more than one core) because the two threads run on separate cores concurrently.

**`%wait` — scheduling delay:** This column measures the percentage of time a runnable thread spent waiting in the kernel run queue for a CPU core. It is not I/O wait — all I/O is non-blocking via epoll. The ZMQ I/O thread has the highest wait at 9.3%, meaning it frequently wakes to drain IPC messages but must wait for a core. The worker thread's 2.8% is consistent with ~1,295 involuntary context switches/s ([§11.4](#114-context-switches-pidstat--w)).

**Core pinning does not eliminate this contention.** Experimentally, increasing the number of physical cores pinned to workers does not reduce `%wait` — the bottleneck is internal to the ZMQ transport layer, not core availability.

### 11.4 Context Switches (`pidstat -w`)

![pidstat -w — context switch rates](res/endpoint_client/27_pidstat_context_switches.png)

`pidstat -p <pid> -w 5` on the same worker process (Threadripper PRO 7965WX):

| Metric    | Average | Analysis                                                                                                              |
| --------- | ------- | --------------------------------------------------------------------------------------------------------------------- |
| cswch/s   | 3,034   | Voluntary context switches — corresponds to `epoll_pwait` calls when no events are ready                              |
| nvcswch/s | 1,295   | Involuntary preemptions — OS scheduler displacing the worker; CPU pinning ([§4](#4-httpclientconfig)) minimizes these |

These numbers were captured under sustained streaming load. The ~3:1 voluntary-to-involuntary ratio shows that even under full pressure, the worker still yields voluntarily (via `epoll_pwait`) more often than it is preempted — meaning the event loop occasionally drains all ready events and briefly blocks for new ones.

### 11.5 CPU Symbol Profile (`perf top`)

![perf top — worker process hottest symbols](res/endpoint_client/22_perf_top_worker.png)

`sudo perf top -p <worker_pid>` during a streaming run. Two threads: `python3` (worker event loop) and `ZMQbg/IO/0` (ZMQ I/O thread). Collapsed from 65+ symbols into functional domains:

- **Self%** = exclusive CPU time in that function only (no callees). Can be summed across rows without double-counting.
- **Children%** = time in that function + everything it calls. Cannot be summed (overlaps across rows).
- **~45% of total CPU** is spent inside kernel syscalls (22.73% python3 + 22.67% ZMQbg).

**python3 thread** (worker event loop):

| Functional Domain      | Self%     | Children% | Call Chain                                                                                                            | Analysis                                                                                                                                         |
| ---------------------- | --------- | --------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Python interpreter** | **7.25%** | 21.15%    | `_PyEval_EvalFrameDefault` + `_PyObject_MakeTpCall`                                                                   | Highest exclusive CPU consumer. CPython bytecode interpretation for msgpack serialization/deserialization, JSON encoding, and protocol callbacks |
| **Kernel syscalls**    | 0.16%     | 22.73%    | `entry_SYSCALL_64` → `do_syscall_64`                                                                                  | 22.73% of CPU spent in kernel work for this thread, primarily TCP stack operations and epoll                                                     |
| ↳ TCP TX               | 1.16%     | 13.02%    | `sock_write_iter` → `tcp_sendmsg_locked` → `tcp_write_xmit` → `__tcp_transmit_skb` → `ip_output` → `__dev_queue_xmit` | HTTP request and response data written through the kernel TCP stack                                                                              |
| ↳ epoll                | —         | —         | `epoll_wait` → `do_epoll_wait` → `ep_poll`                                                                            | Event loop polling for readiness on HTTP and IPC file descriptors                                                                                |
| **uvloop**             | 0.39%     | 15.68%    | `Handle__run` → `UVStream.__try_write` → `__libc_write`                                                               | Minimal exclusive CPU time; uvloop is an optimized C/Cython event loop that passes through to kernel I/O                                         |

**ZMQbg/IO/0 thread** (ZMQ I/O):

| Functional Domain   | Self% | Children% | Call Chain                                                                        | Analysis                                                                                                                                          |
| ------------------- | ----- | --------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Kernel syscalls** | 0.15% | 22.67%    | `entry_SYSCALL_64` → `do_syscall_64`                                              | 22.67% of CPU spent in kernel work for this thread, primarily IPC I/O and scheduling                                                              |
| ↳ Scheduler         | 0.74% | 10.30%    | `schedule` → `__schedule` → `pick_next_task` → `pick_next_task_fair`              | Kernel context-switching as the ZMQ I/O thread yields on `epoll_wait` and resumes when data arrives                                               |
| ↳ epoll             | —     | —         | `epoll_wait` → `do_epoll_wait` → `ep_poll`                                        | Event loop polling for readiness on ZMQ IPC file descriptors                                                                                      |
| **ZMQ internals**   | 0.01% | 29.42%    | `clone3` → `start_thread` → ZMQ internals + `__libc_recv` + `unix_stream_recvmsg` | ZMQ userspace code is a thin dispatch layer; the 29.42% Children is dominated by kernel I/O underneath (Unix domain socket reads/writes, polling) |

**Shared (softirq context)** — softirqs are deferred interrupt handlers that run in kernel context after a hardware interrupt (e.g., NIC signals packet arrival). They execute outside any thread, borrowing whatever CPU was interrupted, so their cost is not attributed to either thread above:

| Functional Domain | Self% | Children% | Call Chain                                                                                         | Analysis                                                                                           |
| ----------------- | ----- | --------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Network RX**    | 0.94% | 5.83%     | `__do_softirq` → `net_rx_action` → `__napi_poll` → `__netif_receive_skb` → `ip_rcv` → `tcp_v4_rcv` | Kernel receives incoming TCP packets, reassembles segments, and wakes the relevant `epoll` waiters |

### 11.6 Syscall Profile (`strace -c`)

![strace -c — syscall summary](res/endpoint_client/23_strace_syscall_summary.png)

`sudo strace -c -p <worker_pid>` — syscall summary for a single worker during a streaming run:

| Syscall          | % Time | Calls   | Errors | Path      | Analysis                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------- | ------ | ------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `write`          | 56.95% | 108,778 |        | IPC, HTTP | uvloop `UVStream.__try_write` (HTTP/TCP) + libzmq `zmq_sendiov` (IPC/UDS)                                                                                                                                                                                                                                                                                                            |
| `read`           | 22.87% | 69,854  |        | IPC, HTTP | uvloop `uv__read` (HTTP responses) + libzmq (8-byte IPC signaling reads)                                                                                                                                                                                                                                                                                                             |
| `getpid`         | 11.65% | 53,086  |        | IPC       | 53k calls (~27k/s) from libzmq's fork-safety check (`ctx_t::check_tag`) on every `zmq_msg_send`/`zmq_msg_recv`. glibc ≥2.25 removed its PID cache, so each call is a real syscall. With kernel syscalls at ~45% of total CPU ([§11.5](#115-cpu-symbol-profile-perf-top)), this wastes ~5% of the worker's CPU on a redundant check. See [`getpid` cache shim](#c4-getpid-cache-shim) |
| `poll`           | 2.43%  | 10,953  |        | IPC       | libzmq internal signaling during `zmq_sendiov` on the ZMQ signaling socket                                                                                                                                                                                                                                                                                                           |
| `epoll_pwait`    | 1.42%  | 695     |        | IPC, HTTP | uvloop event loop — monitors both IPC and HTTP FDs; low call count = rarely blocks under load                                                                                                                                                                                                                                                                                        |
| `futex`          | 0.98%  | 1,591   | 232    | Ser/des   | malloc lock contention from msgspec `Encoder.encode` → `_PyBytes_Resize` → `PyObject_Realloc`                                                                                                                                                                                                                                                                                        |
| `getsockname`    | 0.94%  | 3,806   |        | HTTP      | uvloop `TCPTransport._call_connection_made` — retrieves local address on new connections                                                                                                                                                                                                                                                                                             |
| `setsockopt`     | 0.74%  | 3,461   |        | HTTP      | uvloop `PseudoSocket.setsockopt` — TCP_NODELAY etc. per connection ([§8.4](#84-socket-config))                                                                                                                                                                                                                                                                                       |
| `connect`        | 0.40%  | 347     | 347    | HTTP      | uvloop `uv__tcp_connect` — all return `EINPROGRESS` (async TCP handshake)                                                                                                                                                                                                                                                                                                            |
| `io_uring_enter` | 0.39%  | 1,039   |        | IPC, HTTP | uvloop `uv__epoll_ctl_flush` — io_uring used to batch epoll_ctl modifications                                                                                                                                                                                                                                                                                                        |
| `socket`         | 0.19%  | 347     |        | HTTP      | TCP socket creation for new connections                                                                                                                                                                                                                                                                                                                                              |

**With [`getpid` cache shim](#c4-getpid-cache-shim) applied:**

![strace with getpid cache shim](res/endpoint_client/29_strace_with_getpid_shim.png)

`getpid` drops to **0 calls**. The syscall profile is now dominated by `write` (69.23%) and `read` (22.46%) — the actual I/O work, no longer diluted by getpid overhead.

### 11.7 Run Queue Latency (`runqlat-bpfcc`)

![runqlat-bpfcc — scheduler latency histogram](res/endpoint_client/24_runqlat_histogram.png)

`sudo runqlat-bpfcc -p <worker_pid>` — BPF-traced histogram of kernel run queue latency (time between a thread becoming runnable and actually getting scheduled onto a CPU). The vast majority of wake-ups complete in single-digit microseconds, but a small percentage experience longer scheduling delays.

### 11.8 Hardware Performance Counters (`tiptop`)

![tiptop — IPC and cache metrics](res/endpoint_client/25_tiptop_ipc.png)

![tiptop — branch prediction metrics](res/endpoint_client/26_tiptop_branch_miss.png)

`tiptop` captures two views of hardware performance counters for the worker process:

**View 1 — IPC and Cache:**

| Metric | Value  | Analysis                                                                                                                                                                 |
| ------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| %CPU   | 105.8% | Process total exceeds 100% — worker thread and ZMQ I/O thread run truly parallel on separate cores                                                                       |
| %SYS   | 22.9%  | ~23% of cycles in kernel mode — consistent with `perf top` syscall overhead                                                                                              |
| Mcycle | 4,960M | ~5 billion cycles in sample period                                                                                                                                       |
| Minstr | 4,851M | ~5 billion instructions executed                                                                                                                                         |
| IPC    | 0.98   | ~1 instruction per cycle — below the ~2–4 IPC of optimized native code. Reflects Python interpreter's unpredictable branching and pointer-chasing memory access patterns |
| %MISS  | 2.69%  | Last-level cache miss rate — moderate; working set fits mostly in L2/L3                                                                                                  |
| %BMIS  | 0.91%  | Branch misprediction rate — low; hot loops (`_PyEval_EvalFrameDefault`, `epoll_wait`) are predictable                                                                    |

**View 2 — Branch and Instruction Cache:**

| Metric | Value  | Analysis                                                                                                    |
| ------ | ------ | ----------------------------------------------------------------------------------------------------------- |
| %CPU   | 108.4% | Consistent with view 1 — two threads running in parallel                                                    |
| %MIS/I | 0.88%  | Instruction cache misses per instruction — Python's large interpreter loop causes moderate I-cache pressure |
| %MISP  | 4.19%  | Mispredicted branches as % of all branches — indirect call dispatch (Python vtable, protocol callbacks)     |
| %BR/I  | 21.0%  | ~1 in 5 instructions is a branch — high branch density typical of interpreter loops and event dispatch      |

---

## Appendix A: Concepts

This appendix provides background on key concepts referenced throughout the design document.

---

### A.2 Event Loops and Eager Task Factory

The event loop is the core scheduler for async code. It monitors I/O readiness and dispatches callbacks.

**Event Loop Execution Model:**

![Event Loop Iteration](res/endpoint_client/20_event_loop_iteration.png)

**Performance considerations:**

| Factor              | Impact                                           | Mitigation                                                   |
| ------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| Loop implementation | Default asyncio has Python overhead              | Use `uvloop` (Cython, libuv-based)                           |
| Task creation       | `create_task()` schedules for next iteration     | Use `eager_task_factory` for immediate execution (see below) |
| Callback overhead   | Each callback has dispatch cost                  | Batch operations; drain patterns                             |
| I/O polling         | `select`/`poll` don't scale; `epoll` is O(ready) | uvloop uses epoll/kqueue automatically                       |

**Eager Task Factory (Python 3.12):** `asyncio.eager_task_factory` changes `create_task()` to execute the coroutine synchronously until its first true `await`, rather than scheduling it for the next loop iteration. This saves one full event loop round-trip per task and prevents task starvation under load.

```python
loop.set_task_factory(asyncio.eager_task_factory)

task = loop.create_task(self._worker_cycle_send(query))
# With eager: _worker_cycle_send runs NOW until first await
# The IPC send completes synchronously before create_task returns
```

| Aspect                   | Default                        | Eager                            |
| ------------------------ | ------------------------------ | -------------------------------- |
| `create_task()` returns  | Immediately (coro not started) | After coro runs to first `await` |
| Synchronous code in coro | Runs later                     | Runs immediately                 |

---

### A.3 ZeroMQ (ZMQ)

ZeroMQ is a high-performance asynchronous messaging library that provides socket-like abstractions for IPC, TCP, and multicast communication [1].

**Socket Types Used:**

| Pattern  | Socket Pair | Behavior                                        |
| -------- | ----------- | ----------------------------------------------- |
| Pipeline | PUSH / PULL | Unidirectional; PUSH distributes, PULL collects |

ZMQ contexts can spawn background I/O threads to handle socket operations asynchronously:

```python
# Default in this client: 4 background I/O threads
context = zmq.Context(io_threads=4)
```

| Parameter      | Effect                                 |
| -------------- | -------------------------------------- |
| `io_threads=0` | All I/O on calling thread (blocks)     |
| `io_threads=N` | N background threads handle socket I/O |

These threads are created in the main process only (workers use `io_threads=1`). The LoadGen requires physical cores for these threads to achieve consistent throughput (see A.4).

**Edge-Triggered FD Semantics:**

ZMQ exposes a file descriptor via `getsockopt(ZMQ_FD)` for integration with event loops [2]:

- Signals when internal state _changes_, not when data is present
- Requires draining all messages on each callback
- Requires reschedule via `call_soon` to catch racing messages

pyzmq's `zmq.asyncio.Socket` uses the same pattern as our transport — `add_reader(zmq_fd)` → drain loop → `_schedule_remaining_events` via `call_later(0, ...)`. Our `_on_readable` + `call_soon` reschedule is equivalent.

**pyzmq Send/Recv Internals:**

pyzmq has a `copy_threshold` (default 65536). Messages smaller than 64KB **silently ignore `copy=False`** on send and take the `_send_copy` path (malloc + memcpy). On recv, `copy=False` gives true zero-copy (Frame wraps zmq_msg_t buffer), but pyzmq always calls `zmq_getsockopt(ZMQ_RCVMORE)` — unnecessary for PUSH/PULL sockets.

Per-message round-trip: **4 context switches** (sender↔IO thread↔kernel↔IO thread↔receiver), **2 kernel syscalls** (UDS write + recv), ~5 `_check_rc` calls (each invokes `zmq_errno()` + `PyErr_CheckSignals()`). See [§7.1.3](#713-message-flow) for the full per-message flow table.

---

### A.4 CPU Affinity and NUMA

**CPU Affinity** restricts a process to run only on specified CPU cores, preventing the OS scheduler from migrating it.

**NUMA (Non-Uniform Memory Access)** is a memory architecture where each CPU socket has "local" memory (fast) and can access other sockets' memory (slow, ~100ns penalty).

**Physical Cores vs Logical CPUs (SMT/Hyperthreading):**

- Physical core: Actual execution unit
- Logical CPU: OS-visible CPU (2 per physical core with SMT)
- Hyperthreads share execution resources; pinning to both ensures full core utilization

**Affinity Strategy in this client (from `cpu_affinity.py`):**

| Component              | CPU Assignment                          | Rationale                           |
| ---------------------- | --------------------------------------- | ----------------------------------- |
| LoadGen (main process) | First N fastest physical cores          | Hosts multiple threads (see below)  |
| Workers                | Remaining physical cores (1 per worker) | Isolation prevents context switches |

**LoadGen Thread Breakdown:**

| Thread                 | Count                    | Notes                          |
| ---------------------- | ------------------------ | ------------------------------ |
| Main (LoadGen/Session) | 1                        | Python main thread             |
| Event loop daemon      | 1                        | uvloop in `HTTPEndpointClient` |
| Transport I/O          | `io_threads` (default 4) | Background threads for IPC     |

Default `loadgen_cores=2` accommodates the 2 Python threads (Session + Event loop).

**Performance ranking sources (checked in order):**

1. ACPI CPPC `highest_perf` - Intel P-cores vs E-cores
2. ARM `cpu_capacity` - big.LITTLE architectures
3. `cpuinfo_max_freq` - Fallback to frequency

---

### A.5 msgspec Serialization

`msgspec` [5] is used throughout the client for both JSON (HTTP bodies) and MessagePack (IPC transport) serialization. The same `msgspec.Struct` type definitions serve both paths — one schema per message type, no separate serialization models to maintain.

**Struct conventions:**

| Convention           | Example                                             | Effect                                                                                                                                                                               |
| -------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `kw_only=True`       | `class ChatCompletionRequest(Struct, kw_only=True)` | Named-field initialization maps naturally to JSON key-value pairs; prevents positional argument errors on schema evolution                                                           |
| `omit_defaults=True` | `class Query(Struct, omit_defaults=True)`           | Fields at default value omitted from encoded output; reduces message size for messages with empty optional fields                                                                    |
| `tag="..."`          | `class QueryResult(Struct, tag="query_result")`     | Enables union type discrimination during MessagePack deserialization on the IPC fan-in path                                                                                          |
| `frozen=True`        | `class Query(Struct, frozen=True)`                  | All core IPC types and adapter types are frozen for immutability; also enables faster struct decoding via fixed memory offset attribute access                                       |
| `gc=False`           | `class Query(Struct, gc=False)`                     | Exempts instances from garbage collector tracking; safe for Structs with only scalar/dict fields and no reference cycles. Reduces GC pause overhead                                  |
| `array_like=True`    | `class Query(Struct, array_like=True)`              | Encodes as positional array instead of keyed object (e.g., `["id", {...}, {...}, 0.0]` vs `{"id": ..., "data": ...}`). ~6-50% size reduction and ~6-29% faster encode/decode for IPC |

**Encoder / Decoder patterns:**

Each component holds class-level or instance-level `Encoder` and `Decoder` objects that are reused across all messages:

- **Class-level on adapters:** `_request_encoder = msgspec.json.Encoder()` and `_response_decoder = msgspec.json.Decoder(ChatCompletionResponse)` as class attributes. Construction cost amortized; decoder builds an internal parse plan on first use and reuses it.
- **Instance-level on transports:** `msgspec.msgpack.Encoder()` and `msgspec.msgpack.Decoder(type=QueryResult | StreamChunk)` per transport instance. Tagged union discrimination without try/except.
- **Typed decoders:** `Decoder(ResponseType)` writes directly into Struct fields during parsing — no intermediate `dict` allocation. Unknown JSON keys silently skipped, so adapters tolerate server-side schema additions.
- **Zero-copy encoding:** `Encoder.encode(Struct)` serializes directly from Struct field slots to bytes in a single C-level pass. Avoids the `Struct → dict → json.dumps → bytes` pipeline.
- **Buffer-reuse on receive:** `_ZmqReceiverTransport` uses `sock.recv_into(bytearray)` with a pre-allocated buffer, then decodes from a `memoryview` slice (`decoder.decode(view[:nbytes])`). Avoids per-message `bytes` allocation on the receive path.

**Where used:**

| Layer      | Format            | Encoder/Decoder                                           | Section                        |
| ---------- | ----------------- | --------------------------------------------------------- | ------------------------------ |
| Adapter    | `msgspec.json`    | Class-level on `HttpRequestAdapter` subclasses            | [§9.1](#91-httprequestadapter) |
| Transport  | `msgspec.msgpack` | Instance-level on `SenderTransport` / `ReceiverTransport` | [§7.1.1](#711-serialization)   |
| SSE        | `msgspec.json`    | Class-level `_sse_decoder = Decoder(SSEMessage)`          | [§9.3](#93-sse-stream-parsing) |
| Core types | Both              | `Query`, `QueryResult`, `StreamChunk` Struct definitions  | [§3](#3-types)                 |

**Struct option benchmarks (core IPC types):**

Combined effect of the Struct options above on core IPC types (`core/types.py`), measured via `msgspec.msgpack` encode/decode:

| Type        | Payload | Encode (old → new)  | Decode (old → new)  | Wire Size (old → new) |
| ----------- | ------- | ------------------- | ------------------- | --------------------- |
| Query       | 32 ch   | 249 → 182 ns (-27%) | 398 → 372 ns (-7%)  | 128 → 101 B (-21%)    |
| QueryResult | 32 ch   | 202 → 134 ns (-34%) | 519 → 441 ns (-15%) | 113 → 61 B (-46%)     |
| StreamChunk | 32 ch   | 159 → 90 ns (-44%)  | 204 → 155 ns (-24%) | 96 → 52 B (-46%)      |
| Query       | 512 ch  | 287 → 233 ns (-19%) | 559 → 507 ns (-9%)  | 609 → 582 B (-4%)     |
| StreamChunk | 512 ch  | 233 → 182 ns (-22%) | 362 → 319 ns (-12%) | 577 → 533 B (-8%)     |
| Query       | 4096 ch | 337 → 289 ns (-14%) | 920 → 888 ns (-4%)  | 4193 → 4166 B (-1%)   |
| StreamChunk | 4096 ch | 309 → 231 ns (-25%) | 783 → 753 ns (-4%)  | 4161 → 4117 B (-1%)   |

Size reduction is largest for small messages where key names dominate the payload. E2E transport impact in [§7.2](#72-benchmarks).

**Adapter type benchmarks (OpenAI):**

| Operation       | Payload | Old Mean (ns) | New Mean (ns) | Change |
| --------------- | ------- | ------------- | ------------- | ------ |
| Request Encode  | empty   | 1,155         | 585           | -49%   |
| Request Encode  | 1k      | 1,925         | 730           | -62%   |
| Request Decode  | 1k      | 2,639         | 1,145         | -57%   |
| Response Decode | 1k      | 1,242         | 1,069         | -14%   |
| SSE Decode      | empty   | 905           | 350           | -61%   |
| SSE Decode      | 1k      | 1,626         | 793           | -51%   |
| SSE Encode      | 1k      | 1,338         | 611           | -54%   |

---

## Appendix B: Work in Progress (POR)

### B.1 Shared-Memory Transport (WIP)

A WIP shared-memory transport replaces ZMQ IPC with direct inter-process memory access, eliminating the ZMQ I/O thread, Unix domain socket syscalls, and kernel buffer copies.

**Measured improvement:** ~400k QPS vs ZMQ's ~300k QPS (1.33×). With `recv_into` and `array_like` optimizations, IPC overhead has dropped from ~56% to ~25% of worker CPU, narrowing the gap.

**What it eliminates** (per message):

| ZMQ path (current)                    | Shared-memory path                        |
| ------------------------------------- | ----------------------------------------- |
| zmq mailbox lock + enqueue            | Lock-free ring buffer write               |
| ZMQbg/IO/0 thread wakeup + dequeue    | Eliminated — no I/O thread                |
| `write()` syscall on UDS              | Eliminated — no syscall                   |
| Kernel `unix_stream_sendmsg` (copy)   | Eliminated — no kernel involvement        |
| Receiver IO thread `recv()` + enqueue | Eliminated — no I/O thread                |
| `zmq_msg_recv` + RCVMORE getsockopt   | Direct read from shared ring buffer       |
| `getpid()` × N per round-trip         | Eliminated — no libzmq fork-safety checks |

The remaining cost after shared-memory is HTTP networking (~44% of original CPU): TCP send/recv, IP stack, and the Python interpreter for request encoding/response decoding.

---

## Appendix C: Future Optimizations

### C.1 Nginx Reverse Proxy for Multi-Endpoint Load Balancing

When the SUT exposes multiple backend endpoints (e.g., multiple vLLM instances behind separate ports), the HTTP client currently handles multi-endpoint distribution at the worker level via round-robin URL assignment at construction time. An alternative approach is to front all backends with an nginx reverse proxy, presenting a single endpoint URL to the client.

**Architecture:** All workers connect to a single nginx endpoint, which load-balances across the backend fleet. This simplifies the client's URL assignment (one URL for all workers) and delegates backend health-checking and failover to nginx.

### C.2 TCP Fast Open (TFO)

TCP Fast Open allows the client to send data (the HTTP request) inside the initial SYN packet, eliminating the TCP handshake latency penalty for new connections.

**Standard TCP vs TFO:**

```
Standard TCP (3-way handshake):          TCP Fast Open:

Client         Server                    Client         Server
   │                │                       │                │
   │──── SYN ──────►│                       │── SYN+DATA ───►│  ← Request sent immediately
   │                │                       │                │
   │◄─── SYN-ACK ───│                       │◄── SYN-ACK ────│
   │                │                       │                │
   │──── ACK ──────►│                       │──── ACK ──────►│
   │                │                       │                │
   │──── DATA ─────►│  ← Request sent       │                │
   │                │                       │                │

Latency: 1.5 RTT before request           Latency: 0.5 RTT before request
```

**Impact on Connection Pool Strategy:**

| Metric                  | Standard TCP | With TFO                       |
| ----------------------- | ------------ | ------------------------------ |
| Cold connection latency | ~150µs       | ~50µs (SYN+DATA in one packet) |
| Warm connection latency | ~1µs         | ~1µs (unchanged)               |
| Cold/Warm ratio         | 150x         | ~50x                           |

With TFO enabled, the cold-start penalty shrinks significantly, potentially making reactive connection creation viable without background refresh overhead.

**System Configuration (Linux):**

```bash
# Check current setting
cat /proc/sys/net/ipv4/tcp_fastopen
# 1 = client only, 2 = server only, 3 = both (recommended)

# Enable both client and server
echo 3 | sudo tee /proc/sys/net/ipv4/tcp_fastopen
```

**Server-Side Implementation:**

```python
import socket
import asyncio

# Linux constant
TCP_FASTOPEN = 23

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Enable TFO on server (queue length = 5)
try:
    sock.setsockopt(socket.SOL_TCP, TCP_FASTOPEN, 5)
except OSError:
    pass

sock.bind(('0.0.0.0', 8080))
sock.listen(128)
sock.setblocking(False)

async def main():
    server = await asyncio.start_server(
        handle_request,
        sock=sock
    )
```

**Client-Side Implementation:**

```python
import socket

# Linux 4.11+ constant
TCP_FASTOPEN_CONNECT = 30

def create_tfo_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Enable TFO Connect - kernel uses cached cookie if available
        sock.setsockopt(socket.SOL_TCP, TCP_FASTOPEN_CONNECT, 1)
    except OSError:
        pass

    return sock

# On first connect: kernel requests TFO cookie (standard handshake)
# On subsequent connects: kernel sends data in SYN if cookie cached
```

**Implementation Status:** Not yet implemented. Requires kernel support verification and benchmark validation before adoption.

### C.3 Work-Stealing Dispatch

The current dispatch model assigns queries to workers round-robin ([§5.1](#51-architecture)). Under skewed response times, some workers sit idle while others accumulate a backlog.

**Goal:** Balance SSE chunks/s evenly across workers. Possible load signals include active chunk rate per worker (observable via fan-in PULL socket), event loop stall rate (a periodic sleep task measures scheduling delay), or in-flight request count. The right metric is TBD.

**Two levels:**

| Level              | Scope                              | Metric                          | What It Solves                                                         |
| ------------------ | ---------------------------------- | ------------------------------- | ---------------------------------------------------------------------- |
| **Intra-endpoint** | Workers sharing the same endpoint  | SSE chunks/s per worker         | Uneven response times — slow generation, GC pauses, scheduling         |
| **Inter-endpoint** | Workers across different endpoints | Aggregate chunks/s per endpoint | Uneven endpoint speeds — different GPUs, model sizes, partial failures |

Intra-endpoint is simpler: the main process routes new queries to the least-loaded worker for that endpoint. Inter-endpoint requires workers to hold connection pools to multiple endpoints and the main process to track per-endpoint load.

**Implementation Status:** Not yet implemented.

### C.4 `getpid` Cache Shim

libzmq calls `getpid()` on every `zmq_msg_send`/`zmq_msg_recv` for fork-safety (`ctx_t::check_tag`). Since glibc 2.25 [10], the C library no longer caches PID — each `getpid()` is a real syscall. The glibc PID cache was removed because it was not 100% reliable in certain scenarios (e.g., applications bypassing glibc's `fork()` wrapper via raw `syscall(SYS_clone)`). At high message rates this adds ~6–12% overhead to the syscall profile ([§11.6](#116-syscall-profile-strace--c): 53k `getpid` calls out of 256k total).

```c
// getpid_cache.c
// Build:  cc -shared -fPIC -O2 -o getpid_cache.so getpid_cache.c
// Usage:  LD_PRELOAD=./getpid_cache.so python worker.py ...
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__GNUC__) || defined(__clang__)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define UNLIKELY(x) (x)
#endif

static pid_t cached_pid = 0;

pid_t getpid(void) {
    if (UNLIKELY(cached_pid == 0))
        cached_pid = (pid_t)syscall(SYS_getpid);
    return cached_pid;
}
```

**Implementation Status:** Tested locally.

---

## Appendix D: Performance Changelog

Record of performance-impacting changes and their measured E2E effect.

| PR       | Change                                        | Offline QPS (x86) | IPC Overhead |
| -------- | --------------------------------------------- | ----------------- | ------------ |
| Baseline | Pre-optimization                              | ~175k             | ~56%         |
| #131     | `recv_into` buffer reuse                      | —                 | —            |
| #74      | `array_like`, `frozen`, `gc=False` on Structs | —                 | —            |
| Combined | #131 + #74                                    | ~300k (+71%)      | ~25%         |

Profiling data in §11.3–§11.8 was captured at the pre-optimization baseline.

**Baseline E2E sweeps (x86, pre-#131/#74):**

![Baseline Offline Sweep](res/endpoint_client/19_bench_offline_scaling_baseline.png)

![Baseline Streaming Sweep](res/endpoint_client/20_bench_streaming_scaling_baseline.png)

---

## Bibliography

| Ref  | Description                                                 | URL                                              |
| ---- | ----------------------------------------------------------- | ------------------------------------------------ |
| [1]  | ZeroMQ - An open-source universal messaging library         | https://zeromq.org/                              |
| [2]  | ZMQ socket integration with event loops (aiozmq source)     | https://github.com/aio-libs/aiozmq               |
| [3]  | httptools - Python binding for llhttp (Node.js HTTP parser) | https://github.com/MagicStack/httptools          |
| [4]  | uvloop - Fast drop-in asyncio event loop (libuv-based)      | https://github.com/MagicStack/uvloop             |
| [5]  | msgspec - Fast serialization library with struct support    | https://github.com/jcrist/msgspec                |
| [6]  | Python GC documentation                                     | https://docs.python.org/3/library/gc.html        |
| [7]  | Linux TCP socket options                                    | https://man7.org/linux/man-pages/man7/tcp.7.html |
| [8]  | TCP Fast Open (RFC 7413)                                    | https://datatracker.ietf.org/doc/html/rfc7413    |
| [9]  | asyncio TCP half-close bug (bpo-44805)                      | https://bugs.python.org/issue44805               |
| [10] | glibc 2.25 release — `getpid` cache removal                 | https://sourceware.org/glibc/wiki/Release/2.25   |

---
