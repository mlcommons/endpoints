# Testing Utilities — Design Spec

> Standalone local server implementations — echo, max-throughput, and variable-throughput — that substitute for real inference endpoints during local development and CI.

**Component specs:** [async_utils](../async_utils/Design.md) · [commands](../commands/Design.md) · [config](../config/Design.md) · [core](../core/Design.md) · [dataset_manager](../dataset_manager/Design.md) · [endpoint_client](../endpoint_client/Design.md) · [evaluation](../evaluation/Design.md) · [load_generator](../load_generator/Design.md) · [metrics](../metrics/Design.md) · [openai](../openai/Design.md) · [plugins](../plugins/Design.md) · [profiling](../profiling/Design.md) · [sglang](../sglang/Design.md) · **testing** · [utils](../utils/Design.md)

---

## Overview

`testing/` provides local server implementations that mimic inference endpoints. They allow
the full benchmark stack to be exercised without a real GPU or remote service.

## Responsibilities

- Provide an OpenAI-compatible echo server for local functional testing
- Provide a configurable throughput server for performance testing
- Provide a Docker-based server launcher for integration test environments

## Servers

### `echo_server.py`

Mirrors the request prompt back as the response. Used for:

- Verifying end-to-end benchmark plumbing (CLI → client → server → metrics)
- Testing streaming and non-streaming response paths
- CI integration tests

```bash
python -m inference_endpoint.testing.echo_server --port 8765
python -m inference_endpoint.testing.echo_server --host 0.0.0.0 --port 9000
```

The server implements the OpenAI Chat Completions API and accepts the standard `messages` request
shape. Responses are returned with a configurable delay to simulate latency.

### `max_throughput_server.py`

Returns minimal valid responses as fast as possible. Used for:

- Measuring the upper bound of client throughput (removes server as a bottleneck)
- Performance regression testing of the HTTP client and transport layer

### `variable_throughput_server.py`

Returns responses at a configurable rate. Used for:

- Testing scheduler behaviour under varying server latency
- Validating Poisson and concurrency scheduler correctness

### `docker_server.py`

Manages a Docker container running a real or simulated inference server. Used for:

- Integration tests that require a more realistic server environment
- Automated test setup without manual container management

## Design Decisions

**Echo server uses the real OpenAI API format**

The echo server accepts the full Chat Completions request shape, not a simplified subset. This
ensures that integration tests exercise the actual adapter code path (prompt formatting, header
generation) rather than a shortcut.

**Servers are not test fixtures**

These servers are standalone Python modules, not pytest fixtures. They can be run from the
command line independently of any test framework. Pytest fixtures in `tests/conftest.py`
(`mock_http_echo_server`, `mock_http_oracle_server`) wrap them for test use.

## Integration Points

| Consumer                | Usage                                                  |
| ----------------------- | ------------------------------------------------------ |
| `tests/conftest.py`     | Wraps `echo_server` as `mock_http_echo_server` fixture |
| `docs/LOCAL_TESTING.md` | Step-by-step guide for manual testing with echo server |
| `docs/DEVELOPMENT.md`   | References echo server for local development workflow  |
