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
```

Runs all common worker counts against a range of prompt lengths (CPU pinning is on by default). Produces a plot at `/tmp/sweep_*.png` showing send/recv rate per configuration, with shaded variation bands and a stall% overlay.

### Targeted sweeps

```bash
# Sweep workers for a specific prompt length
python -m inference_endpoint.utils.benchmark_httpclient -w 1:16 -l 4096 -d 10

# Sweep workers with explicit values
python -m inference_endpoint.utils.benchmark_httpclient -w 1,2,4,8,12,16 -l 4096 -d 10

# Cartesian product: workers x prompt lengths
python -m inference_endpoint.utils.benchmark_httpclient -w 1:16::8 -l 128,1024,8192 -d 5
```

### Reading the results

- **Send Rate**: requests/s the client can issue. Higher is better.
- **Recv Rate**: responses/s received. This is the effective throughput.
- **Stall%**: fraction of send time spent blocked on back-pressure (inflight limit). High stall% means the client is sending faster than responses return — adding workers won't help, the bottleneck is downstream.
- **Variation bands**: shaded region shows min/max per-second rate during each run. Wide bands indicate instability.

Pick the worker count where recv rate peaks and stall% is low. Beyond that point, adding workers adds overhead without throughput gain.
