# ZMQ Transport Design Document

## Overview

IPC between main process and worker processes using ZeroMQ PUSH/PULL sockets over Unix domain sockets, integrated with Python's asyncio event loop.

## Architecture

### System Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            MAIN PROCESS                                 │
│                                                                         │
│  send_request(worker_id, query)          poll() / recv()                │
│         │                                      ▲                        │
│         ▼                                      │                        │
│  ┌─────────────────────────────┐        ┌────────────────────────────┐  │
│  │   Request Sockets (PUSH)    │        │   Response Socket (PULL)   │  │
│  │  [sock0] [sock1] ... [sockN]│        │      [Single Socket]       │  │
│  │   BIND    BIND        BIND  │        │           BIND             │  │
│  └──────┬───────┬──────────┬───┘        └──────────────▲─────────────┘  │
└─────────┼───────┼──────────┼───────────────────────────┼────────────────┘
          │       │          │                           │
          ▼       ▼          ▼                           │
      ┌─────────────────┐  ┌─────────────────┐           │
      │    WORKER 0     │  │    WORKER N     │           │
      │                 │  │                 │           │
      │  PULL(connect)  │  │  PULL(connect)  │           │
      │       ↓         │  │       ↓         │           │
      │  HTTP Req/Res   │  │  HTTP Req/Res   │           │
      │       ↓         │  │       ↓         │           │
      │  PUSH(connect)  │  │  PUSH(connect)  │           │
      └───────┬─────────┘  └───────┬─────────┘           │
              │                    │                     │
              └────────────────────┴─────────────────────┘
```

### Data Flow

1. **Request**: Main selects specific worker PUSH socket -> IPC -> Worker PULL socket.
2. **Processing**: Worker executes HTTP request.
3. **Response**: Worker sends to PUSH socket -> IPC -> Main PULL socket (multiplexed).

### Lifecycle

```
1. Main creates ZmqWorkerPoolTransport (binds all sockets)
2. Main spawns workers with connector
3. Workers connect and signal readiness
4. Main calls wait_for_workers_ready()
5. Ready for send_request / recv
```

### Bind vs Connect Rules

| Component    | Socket        | Role     | Pattern        |
| ------------ | ------------- | -------- | -------------- |
| **Request**  | PUSH (Main)   | **BIND** | 1:1 per worker |
|              | PULL (Worker) | Connect  |                |
| **Response** | PULL (Main)   | **BIND** | N:1 aggregated |
|              | PUSH (Worker) | Connect  |                |

### Low-Level Transports

| Class                   | Socket | Operations                       |
| ----------------------- | ------ | -------------------------------- |
| `_ZmqReceiverTransport` | PULL   | `receive()`, `poll()`, `close()` |
| `_ZmqSenderTransport`   | PUSH   | `send()`, `close()`              |

Both integrate with asyncio via `add_reader`/`add_writer` on ZMQ's FD.

## File Structure

```
transport/
├── protocol.py           # Protocol definitions
└── zmq/                  # ZMQ Implementation
    ├── __init__.py       # Exports ZmqWorkerPoolTransport
    └── transport.py      # Main implementation
        ├── ZMQTransportConfig
        ├── _ZmqReceiverTransport
        ├── _ZmqSenderTransport
        ├── _ZmqWorkerConnector
        └── ZmqWorkerPoolTransport
```
