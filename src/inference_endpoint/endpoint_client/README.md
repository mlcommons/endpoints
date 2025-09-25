# HTTP Endpoint Client

A high-performance HTTP client for the MLPerf Inference Endpoint Benchmarking System that leverages multiprocessing, async I/O, and ZMQ for efficient request handling.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTPEndpointClient                           │
│              (implements EndpointClient ABC)                    │
│  ┌─────────────────┐                                            │
│  │  issue_query    │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ├─────ZMQ PUSH (Query)────▶ Worker 1 Queue            │
│           ├─────ZMQ PUSH (Query)────▶ Worker 2 Queue            │
│           └─────ZMQ PUSH (Query)────▶ Worker N Queue            │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                 ZMQ PULL (per worker)
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        WorkerManager                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Worker 1   │  │   Worker 2   │  │   Worker N   │  ...      │
│  │   (uvloop)   │  │   (uvloop)   │  │   (uvloop)   │           │
│  │   aiohttp    │  │   aiohttp    │  │   aiohttp    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                           │                                     │
│                    ZMQ PUSH (QueryResult)                       │
│                           ▼                                     │
│                    ┌────────────────┐                           │
│                    │ Response Queue │ (Shared)                  │
│                    └────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                    ZMQ PULL (blocking)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Handler                             │
│                 (calls complete_callback)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Required dependencies
pip install aiohttp zmq orjson

# Optional for better performance
pip install uvloop
```

## Configuration

Main configuration for the HTTP client:

```python
from inference_endpoint.endpoint_client import HTTPClientConfig

config = HTTPClientConfig(
    endpoint_url="https://api.openai.com/v1/chat/completions",
    num_workers=4,  # Number of worker processes
    max_concurrency=-1  # -1 for unlimited, or positive int to limit concurrent requests
)
```

Recommended to use defaults for the remaining:

```python
from inference_endpoint.endpoint_client import AioHttpConfig
from inference_endpoint.endpoint_client import ZMQConfig

aiohttp_config = AioHttpConfig() # Socket, TCP Connection, HTTP configs
zmq_config = ZMQConfig() # IPC, worker configs
```

## Usage Examples

### Basic Usage - Direct API

The `HTTPEndpointClient` manages its own event loop in a background thread. Use this when you don't need futures or callbacks.

```python
import asyncio
from inference_endpoint.endpoint_client import (
    HTTPEndpointClient,
    HTTPClientConfig,
    AioHttpConfig,
    ZMQConfig
)
from inference_endpoint.core.types import Query

# Create client (manages its own event loop)
http_config = HTTPClientConfig(
    endpoint_url="https://api.openai.com/v1/chat/completions",
    num_workers=4,
    max_concurrency=-1  # unlimited
)
client = HTTPEndpointClient(http_config, AioHttpConfig(), ZMQConfig())
client.start()

# Issue queries (synchronous calls)
queries = [
    Query(id=i, data={"prompt": f"Request {i}", "model": "gpt-4"})
    for i in range(10)
]

for query in queries:
    client.issue_query(query)

# Poll for responses
responses_received = 0
while responses_received < len(queries):
    response = client.get_ready_responses()

    if response:
        print(f"Response {response.id}: {response.response_output}")
        responses_received += 1

client.shutdown()
```

### Advanced Usage - Futures API

The `FuturesHttpClient` integrates with your existing event loop and provides futures for easier async handling.

- `FuturesHttpClient` uses your current event loop (no separate thread)
- Returns `asyncio.Future` objects for each query
- Supports optional callbacks for response handling

```python
import asyncio
from inference_endpoint.endpoint_client import (
    FuturesHttpClient,
    HTTPClientConfig,
    AioHttpConfig,
    ZMQConfig
)
from inference_endpoint.core.types import Query

async def main():
    # Create futures client (integrates with current async context)
    http_config = HTTPClientConfig(
        endpoint_url="https://api.openai.com/v1/chat/completions",
        num_workers=4
    )

    # Optional: Define callback for responses
    def handle_response(response):
        print(f"Callback received: {response.id}")

    client = FuturesHttpClient(
        http_config,
        AioHttpConfig(),
        ZMQConfig(),
        complete_callback=handle_response  # optional
    )

    # IMPORTANT: Use async_start(), not start()
    await client.async_start()

    try:
        # Issue queries and collect futures
        futures = []
        for i in range(10):
            query = Query(
                id=f"req-{i}",
                data={
                    "prompt": f"Request {i}",
                    "model": "gpt-4",
                    "stream": False
                }
            )
            # issue_query() returns a future
            future = await client.issue_query(query)
            futures.append(future)

        # Wait for all responses
        results = await asyncio.gather(*futures)

        for result in results:
            print(f"Result {result.id}: {result.response_output}")

    finally:
        # IMPORTANT: Use async_shutdown(), not shutdown()
        await client.async_shutdown()

asyncio.run(main())
```
