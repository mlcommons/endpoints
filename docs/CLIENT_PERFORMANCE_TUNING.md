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
