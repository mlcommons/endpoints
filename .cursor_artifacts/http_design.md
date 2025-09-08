# HTTP Endpoint Client Design

## Overview

This document describes the design of the HTTP endpoint client for the MLPerf Inference Endpoint Benchmarking System. The client leverages:

- **aiohttp** for async HTTP requests
- **ZMQ Push/Pull** sockets for inter-process communication
- **uvloop** for high-performance async event loops in workers
- **Multiprocessing** for true parallelism

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTPEndpointClient                           │
│              (implements EndpointClient ABC)                    │
│  ┌─────────────────┐                                           │
│  │  issue_query    │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ├─────ZMQ PUSH (Query)────▶ Worker 1 Queue           │
│           ├─────ZMQ PUSH (Query)────▶ Worker 2 Queue           │
│           └─────ZMQ PUSH (Query)────▶ Worker N Queue           │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                 ZMQ PULL (per worker)
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        WorkerManager                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Worker 1   │  │   Worker 2   │  │   Worker N   │  ...     │
│  │   (uvloop)   │  │   (uvloop)   │  │   (uvloop)   │         │
│  │   aiohttp    │  │   aiohttp    │  │   aiohttp    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
│              ZMQ PUSH (QueryResult)                              │
│                            ▼                                     │
│                    ┌────────────────┐                           │
│                    │ Response Queue  │ (Shared)                 │
│                    └────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                   ZMQ PULL (blocking)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Handler                              │
│                 (calls complete_callback)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. HTTPEndpointClient

**Purpose**: HTTP implementation with integrated future-based and callback-based interfaces.

**Key Attributes**:

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class HTTPClientConfig:
    """Configuration for the HTTP endpoint client."""
    endpoint_url: str
    num_workers: int = 4
    max_concurrency: int = -1  # -1 means unlimited, otherwise limits concurrent requests via semaphore

@dataclass
class AioHttpConfig:
    """Configuration for aiohttp client session and connectors."""
   ...

@dataclass
class ZMQConfig:
    """Configuration for ZMQ sockets and communication."""
    ...
```

**Key API Methods**:

```python
from typing import Optional, Callable
import asyncio
from inference_endpoint.core.types import Query, QueryResult

class HTTPEndpointClient:
    """
    HTTP endpoint client with multiprocessing workers and ZMQ communication.

    Provides both future-based and callback-based response handling in a single class.
    """

    def __init__(
        self,
        config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        complete_callback: Optional[Callable] = None
    ):
        """
        Initialize HTTP endpoint client.

        Args:
            config: HTTP client configuration
            aiohttp_config: aiohttp configuration
            zmq_config: ZMQ configuration
            complete_callback: Optional callback for completed requests
        """

    def issue_query(self, query: Query) -> asyncio.Future[QueryResult]:
        """
        Send a query to the endpoint and return a future for the response.

        The returned future can be:
        - Awaited directly: `result = await client.issue_query(query)`
        - Checked for completion: `if future.done(): result = future.result()`
        - Used with asyncio utilities: `done, pending = await asyncio.wait([future])`

        Args:
            query: Query object containing request details

        Returns:
            asyncio.Future that will contain the QueryResult
        """

    async def start(self) -> None:
        """Initialize client and start worker manager."""

    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
```

### 2. WorkerManager

**Purpose**: Manages the lifecycle of worker processes and coordinates IPC.

**Key Responsibilities**:

- Initialize and manage worker processes
- Set up ZMQ Push/Pull sockets for request distribution
- Set up ZMQ Push/Pull sockets for response collection
- Handle worker health monitoring and restart
- Manage graceful shutdown

**Key API**:

```python
class WorkerManager:
    def __init__(
        self,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        zmq_context: zmq.asyncio.Context
    ):
        """Initialize worker manager with configurations."""

    async def initialize(self) -> None:
        """Initialize workers and ZMQ infrastructure."""

    async def shutdown(self) -> None:
        """Graceful shutdown of all workers."""
```

### 3. Worker

**Purpose**: Process that performs actual HTTP requests using aiohttp on uvloop.

**Key Features**:

- Runs on uvloop for maximum performance
- Pulls requests from shared ZMQ socket
- Supports both streaming and non-streaming requests
- Pushes responses back via ZMQ

**Streaming Behavior**:

- For streaming requests, the worker sends only two QueryResult messages:
  1. First chunk - `QueryResult` with `metadata["first_chunk"]: True` containing the first token
  2. Final response - `QueryResult` with `metadata["final_chunk"]: True` containing the complete accumulated output
- This minimizes inter-process communication while still providing streaming feedback

```python
import aiohttp
import json
from typing import Optional

class Worker:
    def __init__(
        self,
        worker_id: int,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        request_socket_addr: str,
        response_socket_addr: str
    ):
       ...

    async def run(self) -> None:
        """Main worker loop - pull requests, execute, push responses"""
        ...
```

### 4. ZMQ Abstraction Layer

**Purpose**: Simplify ZMQ socket operations and provide clean APIs.

**Key Classes**:

```python
import zmq
import zmq.asyncio
import pickle
from typing import Any, Optional

class ZMQPushSocket:
    """Async wrapper for ZMQ PUSH socket"""

    def __init__(self, context: zmq.asyncio.Context, address: str, config: ZMQConfig):
        ...

    async def send(self, data: Any) -> None:
        """Serialize and send data through push socket (non-blocking)"""
       ...

    def close(self) -> None:
        """Close socket cleanly"""
        ...

class ZMQPullSocket:
    """Async wrapper for ZMQ PULL socket"""

    def __init__(self, context: zmq.asyncio.Context, address: str, config: ZMQConfig, bind: bool = False):
        ...

    async def receive(self) -> Any:
        """Receive and deserialize data from pull socket (blocking)"""
        ...

    def close(self) -> None:
        """Close socket cleanly"""
        ...
```

## Data Models

The client uses data models defined in `core/types.py`:

- `Query`: Base class for requests (e.g., `ChatCompletionQuery`)
- `QueryResult`: For all responses (both streaming and non-streaming)
  - Uses `metadata["first_chunk"]: True` to indicate first streaming chunk
  - Uses `metadata["final_chunk"]: True` to indicate final complete response
  - Non-streaming responses have neither flag set

Query objects are passed directly through ZMQ to workers, and QueryResult objects are sent back to the main process.

## Usage Examples

### Basic Usage with HTTPEndpointClient

```python
import asyncio
from typing import Union
from inference_endpoint.core.types import ChatCompletionQuery, QueryResult

# Initialize configurations
http_config = HTTPClientConfig(
    endpoint_url="https://api.openai.com/v1",
    num_workers=4
)

aiohttp_config = AioHttpConfig()  # Use defaults
zmq_config = ZMQConfig()  # Use defaults

# Example 1: Future-based usage (recommended)
client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
await client.start()

# Create a query
query = ChatCompletionQuery(
    id="req-001",
    model="gpt-4",
    prompt="What is the capital of France?",
    stream=False
)

# Send request and get future immediately
future = client.issue_query(query)

# Can await the future when needed
try:
    result = await future
    print(f"Response: {result.response_output}")
except Exception as e:
    print(f"Error: {e}")

await client.shutdown()

# Example 2: Callback-based usage (for event-driven patterns)
async def handle_response(response: QueryResult):
    if response.error:
        print(f"Error: {response.error}")
    else:
        print(f"Response: {response.response_output}")

client = HTTPEndpointClient(
    http_config,
    aiohttp_config,
    zmq_config,
    complete_callback=handle_response
)
await client.start()

# Send request - both callback and future work
future = client.issue_query(query)
# Can still await even with callback
result = await future

await client.shutdown()

# Example 3: Multiple concurrent requests
client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
await client.start()

# Send multiple queries
queries = [
    ChatCompletionQuery(prompt=f"Question {i}", model="gpt-4")
    for i in range(5)
]

# Collect futures
futures = []
for query in queries:
    future = client.issue_query(query)
    futures.append(future)

# Wait for all to complete with timeout
done, pending = await asyncio.wait(futures, timeout=10.0)

# Process completed
for future in done:
    result = future.result()
    print(f"Completed: {result.response_output}")

# Cancel pending
for future in pending:
    future.cancel()

await client.shutdown()

# Example 4: Mixed async patterns with asyncio utilities
client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
await client.start()

# Send multiple requests and wait for first completion
futures = [client.issue_query(ChatCompletionQuery(prompt=f"Q{i}")) for i in range(5)]

# Wait for first to complete
done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
first_result = done.pop().result()

# Cancel remaining
for future in pending:
    future.cancel()

await client.shutdown()
```
