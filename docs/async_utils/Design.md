# Async Utils — Design Spec

> Async infrastructure shared across the system: uvloop event loop lifecycle management, ZMQ-based IPC transport between processes, and a pub/sub event bus for real-time metric streaming.

**Component specs:** **async_utils** · [commands](../commands/Design.md) · [config](../config/Design.md) · [core](../core/Design.md) · [dataset_manager](../dataset_manager/Design.md) · [endpoint_client](../endpoint_client/Design.md) · [evaluation](../evaluation/Design.md) · [load_generator](../load_generator/Design.md) · [metrics](../metrics/Design.md) · [openai](../openai/Design.md) · [plugins](../plugins/Design.md) · [profiling](../profiling/Design.md) · [sglang](../sglang/Design.md) · [testing](../testing/Design.md) · [utils](../utils/Design.md)

---

## Overview

`async_utils/` provides the async infrastructure shared across the system: event loop lifecycle
management, ZMQ-based IPC transport, event pub/sub, and background services. All other components
depend on this package for their async primitives.

## Responsibilities

- Create and manage uvloop event loops with `eager_task_factory`
- Provide ZMQ IPC transport between the main process and worker processes
- Provide a pub/sub event bus for real-time metric streaming
- Host background services (event logger, metrics aggregator) as independent processes

## Component Map

```
async_utils/
├── loop_manager.py          ← creates/tracks named uvloop event loops
├── event_publisher.py       ← ZMQ-backed pub/sub for event records
├── runner.py                ← async runner utilities
├── transport/               ← ZMQ IPC between processes
│   ├── protocol.py          ← message framing definitions
│   └── zmq/
│       ├── context.py       ← managed ZMQ context lifecycle
│       ├── pubsub.py        ← PUB/SUB socket pair
│       └── transport.py     ← PUSH/PULL worker pool transport
└── services/
    ├── event_logger/        ← writes events to JSONL or SQLite (see Design.md)
    └── metrics_aggregator/  ← real-time metric computation (see Design.md)
```

Sub-service specs:

- [Event Logger](services/event_logger/Design.md)
- [Metrics Aggregator](services/metrics_aggregator/Design.md)

## Public Interface

### `LoopManager`

Singleton via `SingletonMixin` — `LoopManager()` always returns the same instance. All event
loops in the process are created and tracked here.

```python
class LoopManager(SingletonMixin):
    def create_loop(
        self,
        name: str,
        backend: Literal["uvloop", "asyncio"] = "uvloop",
        task_factory_mode: Literal["eager", "lazy"] = "eager",
    ) -> asyncio.AbstractEventLoop

    @property
    def default_loop(self) -> asyncio.AbstractEventLoop
    # The loop running on the main thread
```

The `task_factory_mode="eager"` setting installs Python 3.12's `eager_task_factory`, which runs
new coroutines synchronously until their first `await`. This eliminates a scheduling round-trip for
short-lived coroutines on the hot path.

### `EventPublisherService`

Singleton via `SingletonMixin` — after the first construction, subsequent calls return the
cached instance. The first construction requires a `ManagedZMQContext`. Subscribers receive
`EventRecord` messages over a ZMQ SUB socket.

```python
class EventPublisherService(SingletonMixin, ZmqEventRecordPublisher):
    def __init__(
        self,
        managed_zmq_context: ManagedZMQContext,
        extra_eager: bool = False,
        isolated_event_loop: bool = False,
    ) -> None

    def publish(self, record: EventRecord) -> None
```

### ZMQ Transport

The transport layer is not called directly by application code. `HTTPEndpointClient` and
`WorkerManager` construct `WorkerPoolTransport` via the factory in `transport/zmq/transport.py`.

```python
# Protocol (async_utils/transport/protocol.py)
class WorkerPoolTransport(Protocol):
    def send(self, worker_id: int, query: Query) -> None
    def poll(self) -> QueryResult | StreamChunk | None
    async def recv(self) -> QueryResult | StreamChunk | None
```

## Design Decisions

**uvloop everywhere**

uvloop replaces the default asyncio event loop with a libuv-backed implementation that reduces
per-event overhead. All event loops in the system — main process and workers — use uvloop unless
explicitly overridden for tests.

**`eager_task_factory` for minimal await overhead**

Python 3.12 introduced `eager_task_factory`, which runs a coroutine synchronously until its first
suspension point before scheduling it. On the hot path, many coroutines (e.g. `recv()` from an
already-full buffer) complete without ever suspending, eliminating a full scheduler round-trip.

**ZMQ PUSH/PULL for worker IPC**

PUSH/PULL sockets provide load-balanced, message-framed IPC without any acknowledgement overhead.
Messages are framed at the ZMQ layer, so the application never needs to handle partial reads or
message boundaries. The alternative (TCP + asyncio streams) requires manual framing and is slower
for small messages.

**Inproc vs IPC socket selection**

Workers on the same machine use ZMQ transports backed by a managed context. Depending on how that
context is created, the implementation may use `ipc://` or other ZMQ transport details internally.
Callers do not select this directly; `zmq/context.py` encapsulates it.

## Integration Points

| Consumer                         | Usage                                               |
| -------------------------------- | --------------------------------------------------- |
| `endpoint_client/http_client.py` | Uses `WorkerPoolTransport` for worker communication |
| `endpoint_client/worker.py`      | Runs its own uvloop via `LoopManager`               |
| `async_utils/services/`          | Background service processes subscribe via ZMQ SUB  |
