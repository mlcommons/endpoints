# Testing Utilities — Design Spec

> Standalone local server implementations — echo, max-throughput, and
> variable-throughput — that substitute for real inference endpoints during
> local development and CI.

**Component specs:** [async_utils](../async_utils/DESIGN.md) ·
[commands](../commands/DESIGN.md) · [config](../config/DESIGN.md) ·
[core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) ·
[endpoint_client](../endpoint_client/DESIGN.md) ·
[evaluation](../evaluation/DESIGN.md) ·
[load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md)
· [openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) ·
[profiling](../profiling/DESIGN.md) · [sglang](../sglang/DESIGN.md) ·
**testing** · [utils](../utils/DESIGN.md)

---

## Overview

`testing/` provides local server implementations that mimic inference endpoints.
They allow the full benchmark stack to be exercised without a real GPU or remote
service.

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
uv run python -m inference_endpoint.testing.echo_server --port 8765
uv run python -m inference_endpoint.testing.echo_server --host 0.0.0.0 --port 9000
```

The server implements the OpenAI Chat Completions API and accepts the standard
`messages` request shape. It is intended for functional testing of the
request/response path, not for configurable latency simulation.

Routes:

| Route                       | Behavior                                                                               |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `POST /v1/chat/completions` | Echoes the first user message, as JSON or SSE                                          |
| `POST /v1/completions`      | Echoes `prompt` (a token-ID list is joined with spaces), as `text_completion` JSON/SSE |
| `POST /echo`                | Returns the full request details as JSON                                               |

Like vLLM, it reports token usage as
`{"prompt_tokens": P, "completion_tokens": C, "total_tokens": P + C}`:

- `P` is the length of the first user message content (chat) or of `prompt`
  (completions; the token-ID count for a list prompt).
- `C` is the number of output characters actually sent. A streamed response is
  `" ".join(text.split())`, so whitespace runs collapse and are not counted.
- Non-streaming responses always carry the `usage` key.
- A stream sends a usage frame (`"choices": []`) after the finish frame and
  before `[DONE]`, only when the request sets `stream_options.include_usage`.
- `stream_options` without `stream=True` returns HTTP 400
  (``Stream options can only be defined when `stream=True`.``). The check is on
  truthiness, so `stream_options: {}` is accepted.

`EchoServer(report_usage=False)` (CLI `--no-usage`) models a server that never
reports usage: non-streaming responses send `"usage": null` and streams carry no
usage frame. `set_max_osl(0)` makes every output empty, for failure-path tests.

### `max_throughput_server.py`

Returns minimal valid responses as fast as possible. Used for:

- Measuring the upper bound of client throughput (removes server as a
  bottleneck)
- Performance regression testing of the HTTP client and transport layer

Streams end with the finish frame, then a usage frame (`prompt_tokens` fixed at
10, one completion token per output character) only when the request sets
`stream_options.include_usage`, then `[DONE]`. The responses with and without
the usage frame are precompiled, so picking one costs a byte search of the
request body and one branch.

### `variable_throughput_server.py`

Returns responses at a configurable rate. Used for:

- Testing scheduler behaviour under varying server latency
- Validating Poisson and concurrency scheduler correctness

Streams end like the max-throughput server's: finish frame, usage frame only
when the request sets `stream_options.include_usage`, then `[DONE]`.

### `docker_server.py`

Manages a Docker container running a real or simulated inference server. Used
for:

- Integration tests that require a more realistic server environment
- Automated test setup without manual container management

## Design Decisions

**Echo server uses the real OpenAI API format**

The echo server accepts the full Chat Completions request shape, not a
simplified subset. This ensures that integration tests exercise the actual
adapter code path (prompt formatting, header generation) rather than a shortcut.

**Servers are not test fixtures**

These servers are standalone Python modules, not pytest fixtures. They can be
run from the command line independently of any test framework. Pytest fixtures
in `tests/conftest.py` (`mock_http_echo_server`, `mock_http_oracle_server`) wrap
them for test use.

## Integration Points

| Consumer                | Usage                                                  |
| ----------------------- | ------------------------------------------------------ |
| `tests/conftest.py`     | Wraps `echo_server` as `mock_http_echo_server` fixture |
| `docs/LOCAL_TESTING.md` | Step-by-step guide for manual testing with echo server |
| `docs/DEVELOPMENT.md`   | References echo server for local development workflow  |
