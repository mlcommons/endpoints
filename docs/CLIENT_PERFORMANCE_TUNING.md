# Client Performance Tuning

CPU affinity settings to reduce latency jitter in benchmark measurements.

---

## Settings Overview

| Setting                | Location          | Default  | Purpose                           |
| ---------------------- | ----------------- | -------- | --------------------------------- |
| `cpu_affinity`         | `settings.client` | `"auto"` | Pin worker processes to CPU cores |
| `loadgen_cpu_affinity` | Top-level         | `"auto"` | Pin loadgen to NUMA domain        |

---

---

## Worker CPU Affinity (`cpu_affinity`)

```yaml
settings:
  client:
    workers: 4
    cpu_affinity: "auto" # NUMA-aware auto-assignment
    # cpu_affinity: [4, 5, 6, 7]  # Explicit cores
    # cpu_affinity: null          # Disabled
```

---

## Loadgen CPU Affinity (`loadgen_cpu_affinity`)

```yaml
loadgen_cpu_affinity: "auto" # Auto-assign to NUMA domain of fastest CPU
# loadgen_cpu_affinity: null  # Disabled
```

## Platform Notes

- **Linux only**: Uses `os.sched_setaffinity()` and sysfs for topology detection
- **Non-Linux**: Affinity settings are skipped with a warning
