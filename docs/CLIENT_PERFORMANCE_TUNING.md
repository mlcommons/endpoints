# Client Performance Tuning

CPU affinity settings to reduce latency jitter in benchmark measurements.

---

## Overview

The CPU affinity system partitions physical cores between LoadGen (main process) and Workers. Each process gets all hyperthreads (SMT siblings) of its assigned physical cores to prevent cross-process cache thrashing.

**Key concepts:**

- **Physical core isolation**: LoadGen and workers never share physical cores
- **Hyperthread grouping**: Each process gets all logical CPUs of its physical cores
- **Performance-based ranking**: Fastest cores assigned to LoadGen first

---

## Configuration

| Setting        | Location  | Default | Purpose                                       |
| -------------- | --------- | ------- | --------------------------------------------- |
| `cpu_affinity` | Top-level | `-1`    | Pin loadgen and worker processes to CPU cores |

**Values:**

- `-1` (auto): Physical core isolation with SMT siblings, fastest cores to loadgen
- `list[int]`: Use specific cores (shared by loadgen and workers)
- `null`: Disabled

```yaml
cpu_affinity: -1 # Auto: physical core isolation with SMT siblings
# cpu_affinity: [4, 5, 6, 7, 8, 9, 10, 11]  # Explicit cores
# cpu_affinity: null  # Disabled
```

**Auto mode allocation** (default 6 physical cores for loadgen):

- 1 core: Session thread (scheduler, busy-wait timing)
- 1 core: Event loop thread (uvloop, response handling)
- 4 cores: ZMQ I/O threads
- Remaining physical cores: Workers (one per core with all SMT siblings)

## Platform Notes

- **Linux only**: Uses `os.sched_setaffinity()` and sysfs for topology detection
- **Non-Linux**: Affinity settings are skipped with a warning
- **Performance ranking**: Uses ACPI CPPC `highest_perf`, ARM `cpu_capacity`, or `cpuinfo_max_freq` (in order of preference)

## Finding Optimal Worker Count

Optimal worker count depends on your workload — prompt size, streaming mode, and connection count all affect throughput. Use the benchmark script to sweep worker counts against your expected prompt lengths and pick the configuration that maximizes recv rate.

### Full sweep

```bash
python -m inference_endpoint.utils.benchmark_httpclient --full -d 5
python -m inference_endpoint.utils.benchmark_httpclient --full -d 5 --stream
```

Runs all common worker counts against a range of prompt lengths (CPU pinning is on by default). Produces a plot at `/tmp/sweep_*.png` showing send/recv rate per configuration, with shaded variation bands and a stall% overlay.

With `--stream`, the full sweep also varies stream interval (0%, 50%, 100% of prompt length) and adds an SSE-pkts/s subplot. Streaming typically requires more workers to sustain the same recv rate because each response involves many SSE events that must be parsed individually.

### Targeted sweeps

```bash
# Sweep workers for a specific prompt length
python -m inference_endpoint.utils.benchmark_httpclient -w 1:16 -l 4096 -d 10

# Sweep workers with explicit values
python -m inference_endpoint.utils.benchmark_httpclient -w 1,2,4,8,12,16 -l 4096 -d 10

# Cartesian product: workers x prompt lengths
python -m inference_endpoint.utils.benchmark_httpclient -w 1:16::8 -l 128,1024,8192 -d 5

# Streaming: sweep workers with a fixed stream interval (chars per SSE event)
python -m inference_endpoint.utils.benchmark_httpclient -w 1:16 -l 4096 --stream --stream-interval 100 -d 5

# Streaming: sweep stream intervals (total events = ceil(output_length / interval))
python -m inference_endpoint.utils.benchmark_httpclient -w 8 --stream --stream-interval 1,50,500 -d 5
```

### Reading the results

- **Send Rate**: requests/s the client can issue. Higher is better.
- **Recv Rate**: responses/s received. This is the effective throughput.
- **SSE-pkts/s**: SSE events received per second (streaming mode only). Derived from `recv_rate * events_per_response`. Use this to gauge how the client handles high packet rates at different stream intervals.
- **Stall%**: fraction of send time spent blocked on back-pressure (inflight limit). High stall% indicates client-side overhead — the client can't process responses fast enough to make room for new sends. The target server (MaxThroughputServer) returns pre-built responses with no compute, so stall is purely client overhead.
- **Variation bands**: shaded region shows min/max per-second rate during each run. Wide bands indicate instability.

Pick the worker count where recv rate peaks and stall% is low.

For streaming workloads, also watch **SSE-pkts/s** — a small stream interval (fine-grained events) dramatically increases packet rate and may require more workers to keep up. If SSE-pkts/s plateaus while recv rate drops, the client is bottlenecked on SSE parsing overhead.

---

## Test Servers

Two built-in servers for benchmarking without a real GPU endpoint.

### MaxThroughputServer

Returns identical pre-compiled responses instantly — zero compute, pure client roofline.

```bash
python -m inference_endpoint.testing.max_throughput_server --port 12345 --stats
python -m inference_endpoint.testing.max_throughput_server --stream --stream-interval 50 --stats
```

| Flag                | Default | Description              |
| ------------------- | ------- | ------------------------ |
| `--output-length`   | 4000    | Characters in response   |
| `--stream`          | off     | SSE streaming mode       |
| `--stream-interval` | 1       | Characters per SSE event |
| `--num-workers`     | 4       | Server worker processes  |

### VariableResponseServer

Realistic LLM simulation with per-request variable output lengths, TTFT, and TPOT.

Two mutually exclusive timing modes:

- **Response-rate mode** (`--response-rate-mean`): per-worker token bucket controls global throughput
- **Inter-token mode** (`--inter-token-latency`): per-token generation time (TPOT) in ms. Inter-SSE-event delay = TPOT × stream_interval

```bash
# Non-streaming with response-rate control
python -m inference_endpoint.testing.variable_throughput_server --stats \
    --response-rate-mean 1000

# Streaming with TPOT + TTFT
python -m inference_endpoint.testing.variable_throughput_server --stream --stats \
    --inter-token-latency 15 --first-chunk-latency 1.5 --stream-interval 10

# With jitter
python -m inference_endpoint.testing.variable_throughput_server --stream --stats \
    --response-rate-mean 50 --response-rate-spread 0.2 \
    --first-chunk-latency 0.5 --first-chunk-spread 0.2
```

| Flag                    | Default | Description                                                                   |
| ----------------------- | ------- | ----------------------------------------------------------------------------- |
| `--output-len-mean`     | 1000    | Mean output length (chars)                                                    |
| `--output-len-spread`   | 0.3     | CoV for output length (lognormal)                                             |
| `--response-rate-mean`  | 0       | Global throughput (resp/sec). Mutually exclusive with `--inter-token-latency` |
| `--inter-token-latency` | 0       | Per-token delay in ms (TPOT). Mutually exclusive with `--response-rate-mean`  |
| `--first-chunk-latency` | 0       | Mean TTFT in seconds                                                          |
| `--first-chunk-spread`  | 0.2     | CoV for TTFT                                                                  |
| `--stream-interval`     | 1       | Chars per SSE event                                                           |
| `--max-concurrency`     | 0       | Max concurrent requests (0 = unlimited)                                       |
| `--num-workers`         | 10      | Server worker processes                                                       |
