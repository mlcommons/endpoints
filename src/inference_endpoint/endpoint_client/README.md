# HTTP Endpoint Client

HTTP client for LLM inference with multiprocessing workers and ZMQ communication.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   HTTPEndpointClient                                    │
│  ┌─────────────────┐                                                                    │
│  │     issue       │                                                                    │
│  └────────┬────────┘                                                                    │
│           │                                                         ┌────────────────┐  │
│           ├─────ZMQ PUSH────▶ Worker 1 ───HTTP POST───────────────▶ │                │  │
│           ├─────ZMQ PUSH────▶ Worker 2 ───HTTP POST───────────────▶ │    Endpoint    │  │
│           └─────ZMQ PUSH────▶ Worker N ───HTTP POST───────────────▶ │                │  │
│                                    │                                └────────────────┘  │
│                                    │ ◀─────────HTTP Response─────────────────┘          │
│                                    │                                                    │
│  ┌───────────────────────┐         │                                                    │
│  │   poll() / recv()     │◀────────┴ ZMQ PULL                                           │
│  │      drain()          │                                                              │
│  └───────────────────────┘                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.core.types import Query

client = HTTPEndpointClient(
    HTTPClientConfig(endpoint_urls=["http://localhost:8000/v1/completions"])
)

# Sync issue (fire-and-forget)
client.issue(Query(
    id="q-1",
    data={"prompt": "Hello", "stream": False},
    headers={"Content-Type": "application/json"},
))

# Non-blocking poll (returns None if no response available)
response = client.poll()

# Blocking receive (waits for next response)
response = await client.recv()

# Drain all available responses
responses = client.drain()

if response:
    print(f"Response for {response.id}: {response}")
```

## CPU Affinity

For optimal performance, compute an `AffinityPlan` and pass it to `HTTPClientConfig`.
The plan partitions physical cores between the main process (LoadGen) and workers,
assigning all hyperthreads of each core together.

```python
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.cpu_affinity import pin_loadgen

# 1. Compute plan and pin LoadGen (main process)
plan = pin_loadgen(num_workers=8)

# 2. Pass plan to client (workers get pinned automatically)
client = HTTPEndpointClient(
    HTTPClientConfig(
        endpoint_urls=["http://localhost:8000/v1/completions"],
        num_workers=8,
        cpu_affinity=plan,
    )
)
```

## Shutdown

Shutdown is optional. Workers and event loop thread are daemons - they terminate automatically with the main process.

```python
# Optional: graceful shutdown for early exit
client.shutdown()
```
