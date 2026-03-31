# ReadyCheck Design

## Problem

Subprocess startup is asynchronous. The main process spawns workers or service
subprocesses, but cannot use them until they have completed initialization
(bound sockets, subscribed to topics, loaded resources). Without synchronization,
the main process may send messages that are dropped because the subprocess isn't
listening yet.

## Solution

A generic PUSH/PULL readiness protocol that works for any subprocess type:

```
Main Process                         Subprocess (worker or service)
┌───────────────────┐               ┌───────────────────────────┐
│ ReadyCheckReceiver│               │                           │
│   (PULL, bind)    │               │  1. Initialize transports │
│                   │               │  2. Subscribe / connect   │
│   await wait(N)   │◄─── READY ───│  3. send_ready_signal()   │
│   blocks until N  │    (PUSH)     │     (ctx, path, id)       │
│   signals arrive  │               │  4. Start processing      │
└───────────────────┘               └───────────────────────────┘
```

## Why PUSH/PULL

PUB/SUB has a "slow joiner" problem — the subscriber may miss messages
published before it connects. PUSH/PULL guarantees delivery: if the PULL
socket is bound before the PUSH connects, no messages are lost.

Multiple PUSH sockets can connect to a single PULL socket (ZMQ fan-in).
This means one receiver socket handles readiness from all subprocesses.

## Components

### ReadyCheckReceiver (host side)

- Binds a ZMQ PULL socket on an IPC path
- `wait(timeout)` blocks until `count` signals arrive
- Returns list of identities in arrival order
- Closes socket after all signals received (or on timeout)
- Timeout is a total deadline, not per-message

### `send_ready_signal()` (subprocess side)

- Free async function: `send_ready_signal(zmq_context, path, identity)`
- Uses the subprocess's **existing** ZMQ context — no new context created
- Opens one PUSH socket, sends one msgpack-encoded int, closes the socket
- Bounded LINGER (5s) to avoid hanging if receiver is gone

## Usage Patterns

### Workers (PUSH/PULL primary transport)

The `_ZmqWorkerConnector` calls `send_ready_signal()` with the worker's
existing ZMQ context after connecting its request/response transports:

```python
requests = _create_receiver(loop, request_path, zmq_context, ...)
responses = _create_sender(loop, response_path, zmq_context, ...)

await send_ready_signal(zmq_context, self.readiness_path, worker_id)

yield requests, responses
```

The `ZmqWorkerPoolTransport` creates a `ReadyCheckReceiver` and delegates
`wait_for_workers_ready()` to it.

### Services (PUB/SUB primary transport)

Services (EventLoggerService, MetricsAggregatorService) accept
`--readiness-path` and `--readiness-id` CLI arguments. After calling
`service.start()`, they signal readiness using the same ZMQ context:

```python
service.start()

if args.readiness_path:
    await send_ready_signal(zmq_ctx, args.readiness_path, args.readiness_id)

await shutdown_event.wait()
```

### ServiceLauncher

```python
launcher = ServiceLauncher(zmq_context)
procs = await launcher.launch([
    ServiceConfig(module="...event_logger", args=["--socket-dir", d, ...]),
    ServiceConfig(module="...metrics_aggregator", args=["--socket-dir", d, ...]),
], timeout=30.0)

# ... run benchmark, publish ENDED ...

ServiceLauncher.wait_for_exit(procs, timeout=60.0)
```

The launcher:

1. Creates a `ReadyCheckReceiver` bound to a unique IPC path
2. Spawns each service as `python -m <module> ... --readiness-path <path> --readiness-id <i>`
3. Awaits all readiness signals (total deadline timeout)
4. Returns subprocess handles for later `wait_for_exit()`
5. On failure, checks for subprocess crashes and kills remaining processes

## Ordering Guarantee

The ready signal is sent **after** the subprocess has completed its
initialization (transport connect, topic subscribe, reader registration).
This guarantees that when the main process's `wait()` returns, all
subprocesses are ready to process messages.
