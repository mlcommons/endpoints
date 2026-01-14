# Client Performance Tuning

CPU affinity settings to reduce latency jitter in benchmark measurements.

---

## Settings Overview

| Setting                | Location          | Default  | Purpose                           |
| ---------------------- | ----------------- | -------- | --------------------------------- |
| `cpu_affinity`         | `settings.client` | `"auto"` | Pin worker processes to CPU cores |
| `loadgen_cpu_affinity` | Top-level         | `null`   | Pin load generator to a CPU core  |

---

## Worker CPU Affinity (`cpu_affinity`)

```yaml
settings:
  client:
    workers: 4
    cpu_affinity: "auto" # NUMA-aware auto-assignment, excludes loadgen CPU
    # cpu_affinity: [4, 5, 6, 7]  # Explicit cores
    # cpu_affinity: null          # Disabled
```

---

## Loadgen CPU Affinity (`loadgen_cpu_affinity`)

| Value          | Behavior                                   |
| -------------- | ------------------------------------------ |
| `null`         | Auto-detect CPU with highest max frequency |
| `0`, `1`, etc. | Pin to specific core                       |

```yaml
loadgen_cpu_affinity: null # Default: Auto-detect CPU with highest max frequency
# loadgen_cpu_affinity: 4   # Pin to specific core
```

---

## Platform Notes

- **Linux only**: Uses `os.sched_setaffinity()` and sysfs for topology detection
- **Non-Linux**: Affinity settings are skipped with a warning
