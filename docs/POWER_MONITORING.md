# Power Monitoring

First-class, **vendor-neutral** power/energy telemetry collected _during_ a
benchmark run, windowed to the performance phase, and written to a sibling
`power.json` (plus a section in `report.txt`). The `Report` /
`result_summary.json` are never mutated — power is a separate artifact, exactly
like profiling.

## Design: a general framework + pluggable sources

The core (`settings.power`) knows nothing about GPUs, Prometheus, or any vendor.
A **source** is a small registered builder that turns the config into a sidecar
command + a parse spec. `nvidia_smi` is just the **first reference source**;
everything vendor-specific lives inside its builder, not in the schema.

```
setup → [start power sidecar] → warmup → PERFORMANCE phase → teardown → [stop sidecar]
                                              └── trace sliced to this window ──┘
```

The sidecar is best-effort: a broken collector writes `status:"failed"` and the
benchmark completes normally. It can never fail or perturb the run.

## Config

The vendor-neutral core is tiny; per-source settings go in `options` (each
source documents its keys):

```yaml
settings:
  power:
    source: nvidia_smi # nvidia_smi | prometheus | command | <custom plugin>
    interval_s: 1.0 # sampling interval (default 1.0)
    options: {} # source-specific settings
    # advanced (hidden, sane defaults): value_kind, env
```

When `source` is unset the whole block is omitted from the serialized config —
disabling power leaves zero footprint.

### Built-in sources

```yaml
# NVIDIA GPUs on the load-gen host (the reference source)
power:
  source: nvidia_smi
  options: { gpu_indices: [0, 1, 2, 3] }   # optional; default = all GPUs

# Remote GPUs / cluster: scrape a server-side exporter (DCGM, etc.)
power:
  source: prometheus
  options:
    url: "http://gpu-node:9400"
    query: "DCGM_FI_DEV_POWER_USAGE"       # watts; one series per GPU/instance

# Anything else, zero Python: a program that prints canonical JSONL
power:
  source: command
  options:
    argv: ["ssh", "gpu-node", "my-power-reader", "--interval", "1"]
```

The `command` argv must print **one JSONL sample per line**:
`{"ts": <epoch_s>, "value": <float>, "label": "<series>"}`.

## Adding your own source (plugin)

No core edits, no entry-points — register a builder and pass it settings via
`options`:

```python
from inference_endpoint.power import power_source, ResolvedSource

@power_source("redfish")
def _build(cfg):
    return ResolvedSource(
        argv=["redfish-power", cfg.options["bmc"], "--interval", str(cfg.interval_s)],
        fmt="jsonl", value_kind="power_w",
        ts_field="ts", value_field="value", label_field="label", csv_header=False,
    )
```

```yaml
power: { source: redfish, options: { bmc: "10.0.0.5" } }
```

(For non-Python sources, prefer the built-in `command` source — the process
boundary is the plugin API.)

## Output: `power.json`

```json
{
  "schema_version": "1.0",
  "status": "ok",
  "window": {"start_epoch_s": ..., "end_epoch_s": ..., "duration_s": 305.0, "basis": "performance_phase"},
  "totals": {
    "energy_j": 1045212.2,
    "mean_power_w": 3426.9,
    "output_tokens": 4800915,
    "energy_per_output_token_j": 0.2177,
    "consistent_with_window": true
  },
  "sources": [
    {"label": "gpu0", "value_kind": "power_w", "energy_j": 130651.5,
     "power_w": {"mean": 428.4, "p50": 410.0, "p95": 658.0, "max": 879.1}, "sample_count": 305}
  ],
  "provenance": {"command": [...], "interval_s": 1.0, "samples_in_window": 305, "samples_dropped": 0}
}
```

### Energy-per-output-token consistency

`energy_per_output_token_j` is emitted **only** when the token count and the
energy share the same window (single tracked performance phase). Otherwise it is
`null` with an explanatory note — preventing a windowed-energy ÷ global-tokens
mismatch.

## Remote GPUs / multi-node

The sidecar runs **where its command runs** (the load-gen host). For remote
GPUs, either point `prometheus` at a server-side exporter (recommended) or use a
`command` source. For an ssh-wrapped command, use `ssh -tt` so the remote
sampler receives the stop signal. Clocks are assumed NTP-synced.

## Caveats / scope

- `power_w` energy uses a trapezoid integral with no edge interpolation
  (≤ one sample-interval bias at each window edge — negligible at 1 s / 300 s+).
- The window is the full performance phase (includes drain).
- v1 collects a single source; multi-source totals / scope-aware double-count
  guards are future work.
