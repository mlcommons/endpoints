# HTTP Endpoint Client

HTTP client for LLM inference with multiprocessing workers and ZMQ communication.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   HTTPEndpointClient                                    │
│  ┌─────────────────┐                                                                    │
│  │  issue_query    │                                                                    │
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
from inference_endpoint.endpoint_client import (
    HTTPEndpointClient,
    HTTPClientConfig,
    AioHttpConfig,
    ZMQConfig,
)
from inference_endpoint.core.types import Query

client = HTTPEndpointClient(
    HTTPClientConfig(endpoint_url="http://localhost:8000/v1/completions", num_workers=2),
    AioHttpConfig(),
    ZMQConfig(),
)

# Sync issue (fire-and-forget)
client.issue_query(Query(
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

### With HttpClientSampleIssuer

```python
from inference_endpoint.endpoint_client import (
    HTTPEndpointClient,
    HTTPClientConfig,
    AioHttpConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator.sample import Sample

client = HTTPEndpointClient(
    HTTPClientConfig(endpoint_url="http://localhost:8000/v1/completions", num_workers=4),
    AioHttpConfig(),
    ZMQConfig(),
)
issuer = HttpClientSampleIssuer(client)

issuer.issue(Sample(
    uuid="req-1",
    data={"prompt": "Hello", "stream": False},
))
```

## Configuration

```python
HTTPClientConfig(
    endpoint_url="http://localhost:8000/v1/completions",
    num_workers=4,  # Number of worker processes
)

AioHttpConfig()  # Socket, TCP, HTTP configs (use defaults)
ZMQConfig()      # IPC configs (use defaults)
```

## Shutdown

Shutdown is optional. Workers and event loop thread are daemons - they terminate automatically with the main process.

```python
# Optional: graceful shutdown for early exit
client.shutdown()
```
