# Compliance Audit Module — Design Plan

Status: **Proposed (redesign)** · First test: **TEST04**, with TEST01/06/07/09 as planned extensions.

This document plans a modular compliance/audit framework for the endpoint benchmarking
tool that re-implements the _intent_ of the MLPerf Inference compliance ("audit") tests.
The reference implementation lives in the MLCommons inference repo
(`compliance/nvidia/TESTxx`).

This is a ground-up redesign. The driving requirements come from two sources: the
maintainer's workflow constraints (a single command that runs both phases back-to-back
against the same endpoint) and a first-principles design review (TEST04 must not be bolted onto
the benchmark via per-phase config surgery — it needs a first-class, extensible
abstraction). Section 8 maps each requirement to where the design satisfies it.

---

## 1. Background: what MLPerf audit tests do

MLPerf compliance tests detect that a submitter is not gaming the benchmark (caching,
truncating outputs, running a different/cheaper model in the perf run, EOS exploits).
They are built on three LoadGen-specific pieces:

1. **`audit.config`** — a file LoadGen reads at `StartTest()` that overrides run settings
   to enable the test (e.g. issue duplicate samples, log a sample of outputs, fix seeds).
2. **`mlperf_log_accuracy.json`** — the SUT logs raw **output token IDs** during the run.
3. **`run_verification.py`** — a post-run script that consumes the logs and emits
   `verify_*.txt` with a `Performance check pass: True/False` / `TEST PASS` line.

### Test matrix (LLM-relevant subset)

| Test   | Detects                                             | Category      | Required for                         |
| ------ | --------------------------------------------------- | ------------- | ------------------------------------ |
| TEST01 | Different model in perf vs accuracy run             | orchestrator  | ResNet50, BERT, SDXL, RetinaNet, …   |
| TEST04 | Caching of duplicate queries (throughput inflation) | orchestrator  | ResNet50, SDXL (LLMs largely exempt) |
| TEST06 | LLM output consistency (EOS / first-token / length) | analyzer      | llama2/3.1, mixtral, deepseek        |
| TEST07 | Accuracy ≥ threshold in perf mode                   | analyzer      | gpt-oss-120b                         |
| TEST09 | Mean output token length within ±10% of reference   | analyzer      | gpt-oss-120b                         |
| TEST08 | DLRM-v3 streaming accuracy                          | n/a (not LLM) | DLRM-v3 — **out of scope**           |

**TEST04 (mechanism).** `audit.config` sets `performance_issue_same=1` /
`performance_issue_same_index=3` so LoadGen issues the **same sample repeatedly** for the
**same number of queries** as the standard run, then the verification compares throughput.
Pass if the audit run is **not more than 10% faster** than the reference (20% for
low-throughput streams). If the SUT caches responses for duplicate queries, throughput
inflates → FAIL.

> **LLM nuance.** MLPerf exempts variable-length-input LLMs from TEST04 because prefix
> caching legitimately speeds up identical prompts. On an LLM endpoint, TEST04 will see
> real prefix-cache gains; the tolerance (and whether the audit run disables prefix cache)
> is a deliberate knob. We build it faithfully to the reference (±10% / ±20%) and expose
> the tolerance.

---

## 2. Conceptual mapping: MLPerf → this repo

This tool is its own HTTP load generator (no LoadGen). The audit module re-implements the
_intent_ over this repo's own artifacts.

| MLPerf                                 | This repo                                                   |
| -------------------------------------- | ----------------------------------------------------------- |
| `audit.config` (run-setting override)  | a typed **`SampleOrderSpec`** carried on a **`RunSpec`**    |
| `mlperf_log_accuracy.json` (token IDs) | `events.jsonl` (must carry token IDs for token-level tests) |
| `run_verification.py` → `verify_*.txt` | an **`AuditTest.verify()`** → `verify_<TEST>.txt` + JSON    |
| LoadGen runs both phases of a test     | a **generic orchestrator** runs `plan_runs()` back-to-back  |
| compliance submission dir layout       | mirrored under the run's report dir                         |

### Feasibility note for the token-level tests (TEST06/09)

A finished run captures decoded **text** + `finish_reason` for all adapters, but **raw
output token IDs only for the SGLang adapter** (`QueryResult.metadata["token_ids"]`).
OpenAI/completions runs lose the token-ID stream. TEST06's EOS/first-token checks and exact
TEST09 need token IDs, so faithful TEST06/09 will require a small, localized data-path
addition (capture token IDs under an audit-capture flag when the server can return them —
logprobs / SGLang `token_delta`). **This only matters when TEST06/09 are implemented;**
TEST04 (throughput-only) and TEST01 need none of it.

---

## 3. Two axes (the core principle)

Every audit test decomposes into two independent concerns. Keeping them separate is what
prevents test-specific knowledge from leaking into general-purpose code.

- **Axis A — run modification** (the `audit.config` analogue): _how_ a test alters the
  benchmark run(s). For TEST04 it is "issue one fixed sample repeatedly for the audit
  phase." This is expressed as a generic, typed **`SampleOrderSpec`**, not a per-test
  boolean. The load generator never learns the string "test04".
- **Axis B — verification**: a pure post-run check comparing run artifacts → a verdict.
  Per-test, registered.

---

## 4. Architecture

### Component map

```
benchmark from-config
   │
   ├─ run main benchmark: perf  [+ accuracy when accuracy datasets present]   (existing path)
   │
   └─ if config.audit is set ▼   (additive post-step, same report_dir)
   run_audit(config)                         commands/audit.py  ── the generic loop
            │
            │ 1. get_audit_test(config.audit.test)
            ▼
   AuditTest  ──────────────────────────────  compliance/tests/test04.py
     ├─ plan_runs(cfg) -> list[RunSpec]        (declarative: what phases to run)
     └─ verify(runs)   -> AuditVerdict         (pure: read artifacts → verdict)
            │
            │ 2. for each RunSpec
            ▼
   setup_benchmark(config, run_spec)           commands/benchmark/execute.py  (reused)
            │   run_spec.sample_order
            ▼
   create_sample_order(settings)               load_generator/sample_order.py
     └─ switch on SampleOrderSpec              (WITHOUT_REPLACEMENT | SINGLE(index))
            │   no "test04" knowledge here
            ▼
   run_benchmark_async(ctx) ─► RunArtifacts    (final_snapshot.json, events.jsonl)
            │
            │ 3. verify(runs) ; 4. write_verdict (atomic)
            ▼
   verify_TEST04.txt  +  audit_verdict.json
```

### Program flow (TEST04, two phases)

```
config.audit = {test: test04, samples: 64, audit_samples: 64, sample_index: 0, threshold: 0.10}

 run_audit
    │
    ├─ specs = Test04Audit.plan_runs(cfg)
    │     → [ RunSpec("reference", n_samples=64, WITHOUT_REPLACEMENT),
    │         RunSpec("test04",    n_samples=64, SINGLE(index=0))     ]   ← equal here; audit_samples may be lowered
    │
    ├─ setup phase 1 ─► validate ALL specs vs loaded dataset size      ← before any run
    │                   (sample_index in range?) — fail fast
    │
    ├─ Phase 1  "reference"  ─ 64 distinct samples ───► RunArtifacts[0]   (qps_ref)
    │                                                   back-to-back, same endpoint
    ├─ Phase 2  "test04"     ─ 64 × sample[0] ────────► RunArtifacts[1]   (qps_audit)
    │
    ├─ verdict = Test04Audit.verify([ref, audit])
    │     guard: each phase completed >= requested × (1 - 0.10)   (else FAIL: phase failed)
    │     rule:  PASS  iff  qps_audit < qps_ref × (1 + 0.10)
    │           FAIL  → SUT served duplicates from cache
    │
    └─ write_verdict(verdict, report_dir)   →  exit 0 (PASS) / 1 (FAIL) / 2 (error)
```

Analyzer tests (TEST06/07/09) take the same path with a single-element `plan_runs`, so
phase 2 simply doesn't exist and `verify` reads the one run's artifacts.

In a `type: submission` config (see §5) this whole `run_audit` block runs **after** the
main perf [+ accuracy] run, under the same `report_dir`.

### The `AuditTest` abstraction

A single protocol covers **both** categories — orchestrators (must execute a
specially-configured run) and analyzers (pure post-run). An analyzer is just an audit whose
plan is a single normal run, so the orchestration loop never special-cases a category.

```python
class AuditTest(Protocol):
    test_id: ClassVar[AuditTestId]                          # AuditTestId.TEST04
    def plan_runs(self, cfg: AuditConfig) -> list[RunSpec]: ...
    def verify(self, runs: list[RunArtifacts]) -> AuditVerdict: ...
```

- **Orchestrator (TEST04, TEST01):** `plan_runs` returns ≥2 specs.
- **Analyzer (TEST06, TEST07, TEST09):** `plan_runs` returns 1 normal-run spec; all logic
  lives in `verify`.

### `RunSpec` — declarative and typed

Replaces ad-hoc per-phase `model_copy` surgery and stringly-typed override kwargs.

```python
@dataclass(frozen=True, slots=True)
class RunSpec:
    label: str                    # "reference" / "test04" → report subdir
    n_samples: int                # this phase's query count (may differ per phase)
    sample_order: SampleOrderSpec # WITHOUT_REPLACEMENT | SINGLE(index)
```

### `SampleOrderSpec` — the one generic load-gen seam

```python
# load_generator/sample_order.py
class SampleOrderSpec:   # WITHOUT_REPLACEMENT | SINGLE(index=...)
    ...

def create_sample_order(settings: RuntimeSettings) -> SampleOrder:
    spec = settings.sample_order            # generic; default WITHOUT_REPLACEMENT
    ...                                      # switch on spec, no "test04" knowledge
```

### `AuditConfig` — per-test discriminated union on `BenchmarkConfig`

Each test carries **only its own knobs** in a per-test config model, discriminated on
`test`. This avoids a flat model where one `threshold` field means different things per
test (caching tolerance vs OSL band vs accuracy floor) and is meaningless for the
equality-based tests (TEST01/06). No `DatasetType.AUDIT`, no audit fields on the shared
`Dataset` model.

```python
class AuditTestId(str, Enum):
    TEST04 = "test04"

class Test04Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    test: Literal[AuditTestId.TEST04]
    samples: int | None = None        # reference phase count; None = full dataset
    audit_samples: int | None = None  # audit phase count; None = equals `samples`
    sample_index: int = 0             # MLPerf performance_issue_same_index
    threshold: float = 0.10           # caching tolerance (TEST04-specific)

# One member today; becomes a discriminated union as tests are added:
#   AuditConfig = Annotated[Test04Config | Test01Config | ..., Field(discriminator="test")]
AuditConfig = Test04Config
```

On `BenchmarkConfig`: `audit: AuditConfig | None = None`. With a single member the alias is
just `Test04Config`; the `test: Literal[...]` discriminator field is already in place, so
adding the second test only assembles the `Annotated[... , Field(discriminator="test")]`
union — no change to existing tests.

**Independent counts (deliberate — see the decision record in §5).** `samples` sizes the
reference phase, `audit_samples` the fixed-sample phase; `audit_samples=None` falls back to
`samples` (equal counts — the shipped examples). They **may** differ (set `audit_samples`
lower, e.g. 64 / 32, to shorten the audit phase — upstream TEST04 does this; see §5). The
verdict relies on `qps` being rate-normalized plus a per-phase completion guard, so it does
not require equal counts.

### Generic orchestrator

`run_benchmark` first executes the main benchmark (performance, plus accuracy scoring when
the config carries accuracy datasets) exactly as today. Then, when `config.audit is not
None`, it runs `run_audit(config)` (in `commands/audit.py`) as an **additive
post-step** under the same `report_dir`. The two stages are independent, self-contained
operations sequenced at the top level — not per-phase config surgery — so one
`type: submission` YAML can produce the full set: perf [+ accuracy] + the audit's reference
and test04 phases + verdict (§5). The audit runs its **own** reference phase at
`samples`; it does not reuse the (typically larger, full-dataset) submission perf run. The
generic loop never names a specific test:

1. `test = get_audit_test(config.audit.test)`
2. `specs = test.plan_runs(config.audit)`
3. **Validate before any run executes.** The load pattern must be unpaced —
   `max_throughput` (offline) or `concurrency` (single-stream); a paced `poisson` load caps
   throughput and would mask caching, so it is rejected up front. Every spec's
   `sample_index` must be in range for the **loaded** dataset (set up phase 1 to learn the
   row count, then bounds-check all specs before phase 1 runs — never discover an
   out-of-range index after a full reference run has already executed).
4. Execute each spec back-to-back via the existing `setup_benchmark` /
   `run_benchmark_async` path (no duplicated report-dir or `config.yaml` logic). Each phase
   config has `audit=None` to prevent re-entry into `run_audit`. If any phase raises
   (`SetupError` / `ExecutionError`), `run_audit` aborts **without verifying** — a crashed
   phase must never produce a verdict — and surfaces the error as exit `2`.
5. `verdict = test.verify(runs)`
6. Atomically write the verdict (`tmp → fsync → rename → fsync(parent)`).
7. Return the typed `AuditVerdict`. Because `run_benchmark` currently returns `None` and
   `cli.py` ignores its return, the audit path must **propagate** the verdict: `run_audit`
   returns it, `run_benchmark` returns it for an audit config, and `cli.py` maps it to
   `sys.exit` — `0` (PASS) / `1` (FAIL) / `2` (setup/IO/phase error). The on-disk
   `audit_verdict.json` is the durable record; the exit code is the automation signal.

### Verifier — one core + thin adapters

```python
@dataclass(frozen=True, slots=True)
class RunStats:          # .from_report(Report, n_requested) | .from_dir(Path, n_requested)
    qps: float
    n_completed: int
    n_requested: int

def verify_test04(ref: RunStats, audit: RunStats, threshold: float = 0.10) -> AuditVerdict:
    # qps > 0 for both phases
    # per-phase completion guard: each phase completed >= requested * (1 - threshold)
    #   (catches a phase that mostly failed — bogus low qps — without assuming ref == audit)
    # caching rule:               audit.qps < ref.qps * (1 + threshold)
```

The phases may issue **different** counts (`samples` vs `audit_samples`), so the verdict does
**not** require `ref.n_completed == audit.n_completed`. Validity comes from `qps` being a
rate (caching still shows up as a throughput spike) plus the per-phase completion guard,
which rejects a run that crashed partway and would otherwise post a misleadingly low qps.

`from_report` is the in-process adapter the orchestrator uses; `from_dir` is the re-check
adapter (verify a finished pair of run dirs without re-running). `from_dir` treats every
malformed-artifact case as a clean error — caller exits `2`, never an uncaught traceback:
missing or permission-denied snapshot, non-`dict` or truncated JSON, and a
`Report.from_snapshot` that raises `KeyError`/`TypeError` are all caught and reported.

---

## 5. Module layout

```
src/inference_endpoint/compliance/
├── __init__.py        # AuditTest protocol, RunSpec, RunStats, AuditVerdict, get_audit_test()
├── verdict.py         # AuditVerdict + atomic write → verify_<TEST>.txt + audit_verdict.json
└── tests/
    ├── __init__.py    # imports submodules so registration fires
    └── test04.py      # Test04Audit: plan_runs (reference + audit specs) + verify_test04 core
```

CLI surface: an `audit:` block in the benchmark YAML, picked up by `benchmark from-config`.

```yaml
# bench.yaml
audit:
  test: test04
  samples: 64
  threshold: 0.10
```

```
inference-endpoint benchmark from-config --config bench.yaml
# detects audit:, runs reference (64 distinct) + test04 (64 × fixed index) back-to-back,
# writes verify_TEST04.txt + audit_verdict.json, exits 0/1/2
```

### Unified submission (perf + accuracy + audit in one file)

`audit:` is additive, so a single `type: submission` config drives the whole submission:
`run_benchmark` does the performance run, scores the accuracy datasets, then runs the audit
— one command, one `report_dir`. Each piece is optional: drop `audit:` for perf+acc, or
omit accuracy datasets for perf+audit.

The committed example is `examples/09_Wan22_VideoGen_Example/offline_wan22_submission.yaml`:

```yaml
# Full WAN 2.2 Offline submission: performance + VBench accuracy + TEST04 audit.
# One command runs all three under a single report_dir:
#   inference-endpoint benchmark from-config \
#       examples/09_Wan22_VideoGen_Example/offline_wan22_submission.yaml
#
# Execution order (run_benchmark):
#   1. performance run  — full 248-prompt dataset (the submission perf result)
#   2. accuracy scoring — VBench over the produced videos
#   3. audit (TEST04)   — reference + fixed-sample phases (equal counts here), then verdict
#
# NOTE: the `audit:` block reflects the PROPOSED schema in
# docs/compliance_audit_plan.md and is not yet implemented on main. The
# performance + accuracy portion mirrors offline_wan22_accuracy.yaml.

name: "submission-wan22-video-generation"
version: "1.0"
type: "submission"
benchmark_mode: "offline" # required for type: submission

model_params:
  name: "wan22"
  max_new_tokens: 1 # ignored by VideoGenAdapter; kept >0 for api_type debug swaps
  streaming: "off" # WAN 2.2 uses non-streaming HTTP POST/response

datasets:
  # Performance dataset drives request issuance (the submission perf run).
  - name: wan22_perf
    path: examples/09_Wan22_VideoGen_Example/wan22_prompts.jsonl
    type: "performance"
    samples: 248

  # Accuracy dataset reuses the same prompts; videos are scored VBench-style.
  - name: wan22_vbench
    path: examples/09_Wan22_VideoGen_Example/wan22_prompts.jsonl
    type: "accuracy"
    samples: 248
    accuracy_config:
      eval_method: "vbench"
      ground_truth: "prompt" # VBench input is (prompt, video), not a GT comparison
      num_repeats: 1

# TEST04 caching audit — additive post-step. Runs its OWN reference + fixed-sample
# phases at equal counts (the audit count may be lowered to shorten the phase).
audit:
  test: "test04"
  samples: 64 # reference phase count (subset of the 248 prompts)
  audit_samples: 64 # audit (fixed-sample) phase count; lower (e.g. 32) to shorten the audit phase
  sample_index: 3 # MLCommons audit.config performance_issue_same_index=3
  threshold: 0.10 # audit qps must stay < reference qps * (1 + threshold)

settings:
  runtime:
    # NOTE: runs are count-driven (n_samples_to_issue / audit.samples). min_duration_ms is
    # NOT enforced as a duration floor by the current stop logic (counts take priority);
    # MLCommons' 10-min minimum / AND-semantics is future work. Only max_duration_ms caps.
    max_duration_ms: 14400000 # 4-hour ceiling
    scheduler_random_seed: 42
    dataloader_random_seed: 42
    n_samples_to_issue: 248 # applies to the perf/accuracy run; audit uses audit.samples

  load_pattern:
    type: "max_throughput"

endpoint_config:
  endpoints:
    - "http://localhost:8000"
  api_type: "videogen"
  api_key: null

report_dir: logs/wan22_submission
```

Resulting `report_dir/` (main perf/accuracy artifacts keep their current layout; the audit
adds its labelled phase subdirs + verdict):

```
report_dir/
├── final_snapshot.json   # submission perf run (existing top-level layout)
├── events.jsonl
├── …                     # accuracy scoring outputs (existing)
├── reference/            # audit reference phase    (samples=64)
├── test04/               # audit fixed-sample phase (samples=64)
├── verify_TEST04.txt
└── audit_verdict.json
```

### WAN2.2-T2V — the first target

The first workload to exercise TEST04 is **WAN2.2-T2V-A14B** (MLPerf text-to-video), served
through the `videogen` adapter (`api_type: videogen`, model `wan22`, non-streaming HTTP).
Prompts come from the 248-row `examples/09_Wan22_VideoGen_Example/wan22_prompts.jsonl`.
Two scenarios must be covered: **Offline** (`max_throughput`) and **SingleStream**
(`concurrency`, one request in-flight).

**MLCommons knobs and how they map to `AuditConfig`:**

| MLCommons (WAN2.2 `audit.config` / `mlperf.conf`) | `AuditConfig`                 | Notes                                               |
| ------------------------------------------------- | ----------------------------- | --------------------------------------------------- |
| `performance_issue_same=1`                        | (implied by TEST04)           | audit phase issues one fixed prompt for every query |
| `performance_issue_same_index=3`                  | `sample_index: 3`             | which prompt is repeated                            |
| TEST04 throughput tolerance                       | `threshold: 0.10`             | `0.20` for the low-throughput SingleStream scenario |
| `min_query_count` (reference / audit)             | `samples` / `audit_samples`   | independent per-phase counts (§4)                   |
| `min_duration` (compliance ≥ 10 min)              | _not yet enforced_ (see note) | counts take priority in current stop logic          |

> **Design decision — equal counts in the shipped examples; independent counts supported.** > `samples` sizes the reference phase and `audit_samples` the fixed-sample phase
> (`audit_samples=None` falls back to `samples`). The **shipped examples use equal counts** —
> Offline `samples: 64` / `audit_samples: 64`, SingleStream `samples: 20` — which addresses
> the maintainer's fairness concern ("comparing QPS of 50 distinct vs 20 repeated … doesn't
> seem fair", PR #332) by comparing like-for-like.
>
> The schema still **supports** independent counts because upstream MLPerf TEST04 itself uses
> them: the MLCommons `compliance/nvidia/TEST04/audit.config` overrides
> `stable-diffusion-xl.Offline.min_query_count = 500` against a `mlperf.conf` reference of
> `5000` — i.e. a **5000 reference / 500 audit** split, compared as samples-per-second. So
> `audit_samples < samples` is a valid, upstream-faithful way to shorten the (expensive) audit
> phase. The verdict does **not** require equal counts — `qps` is rate-normalized and a
> **per-phase completion guard** (each phase must complete ≥ `requested × (1 − threshold)`)
> catches a crashed run — but the examples default to equal for the clearest, least-contentious
> comparison.

> **`min_duration` is not a duration floor (current limitation).** The load-generator stop
> check (`session.py`) halts a phase on **sample count** or **`max_duration_ms`** only;
> `min_duration_ms` merely _derives_ a count when no explicit count is set. Because TEST04
> drives an explicit `samples` count, each phase stops at `samples` and `min_duration_ms` is
> **not** honored as a "run for at least 10 minutes" floor. MLCommons' 10-minute compliance
> minimum therefore is **not** enforced today; combining a count floor with a duration floor
> ("AND-semantics") is future work. Set `samples` large enough that each phase reaches a
> stable throughput on its own.

**Offline (`max_throughput`):**

```yaml
# Illustrative: Offline TEST04 audit-only (perf + audit, no accuracy datasets).
# The committed file is offline_wan22_submission.yaml (perf + accuracy + audit).
type: offline
model_params: { name: wan22, streaming: off }
audit:
  test: test04
  samples: 64 # reference phase count (tunable subset of the 248-prompt dataset)
  audit_samples: 64 # audit (fixed-sample) phase count; lower (e.g. 32) to shorten the audit phase
  sample_index: 3 # MLCommons performance_issue_same_index
  threshold: 0.10
datasets:
  - {
      name: wan22_prompts,
      path: examples/09_Wan22_VideoGen_Example/wan22_prompts.jsonl,
    }
settings:
  runtime: {} # count-driven (samples / audit_samples); min_duration_ms is not a floor (see note)
  load_pattern: { type: max_throughput }
endpoint_config: { api_type: videogen, endpoints: ["http://localhost:8000"] }
```

**SingleStream (`concurrency` = 1):**

```yaml
# Illustrative: SingleStream TEST04 audit-only (perf + audit, no accuracy datasets).
# The committed file is single_stream_wan22_submission.yaml (perf + accuracy + audit).
type: online
model_params: { name: wan22, streaming: off }
audit:
  test: test04
  samples: 20 # MLCommons SingleStream min_query_count (audit_samples omitted → also 20)
  sample_index: 3
  threshold: 0.20 # low-throughput stream tolerance
datasets:
  - {
      name: wan22_prompts,
      path: examples/09_Wan22_VideoGen_Example/wan22_prompts.jsonl,
    }
settings:
  runtime: {} # count-driven (see note); min_duration_ms is not a floor
  load_pattern: { type: concurrency, target_concurrency: 1 }
endpoint_config: { api_type: videogen, endpoints: ["http://localhost:8000"] }
```

---

## 6. File-by-file changes (against `main`)

| File                                                                     | Change                                                                                             |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `compliance/__init__.py`                                                 | **new** — `AuditTest` protocol, `RunSpec`, `RunStats`, `AuditVerdict`, `get_audit_test()` registry |
| `compliance/verdict.py`                                                  | **new** — `AuditVerdict` + atomic `write_verdict` (reference `verify_TEST04.txt` wording + JSON)   |
| `compliance/tests/__init__.py`                                           | **new** — imports submodules so registration fires                                                 |
| `compliance/tests/test04.py`                                             | **new** — `Test04Audit.plan_runs` (reference + audit specs) + `verify_test04` core                 |
| `commands/audit.py`                                                      | **new** — generic `run_audit` loop (plan → validate-all → execute → verify → write)                |
| `config/schema.py`                                                       | **+** `AuditTestId`, `AuditConfig`, `audit: AuditConfig \| None` on `BenchmarkConfig`              |
| `load_generator/sample_order.py`                                         | **+** `SampleOrderSpec` + `SingleSampleOrder`; `create_sample_order` switches on the spec          |
| `config/runtime_settings.py`                                             | **+** `sample_order: SampleOrderSpec` (generic; default `WITHOUT_REPLACEMENT`)                     |
| `commands/benchmark/execute.py`                                          | **+** typed `run_spec` seam in `setup_benchmark`; `run_benchmark` dispatches to `run_audit`        |
| `examples/09_Wan22_VideoGen_Example/offline_wan22_submission.yaml`       | **new** — WAN2.2 Offline submission (perf + accuracy + TEST04)                                     |
| `examples/09_Wan22_VideoGen_Example/single_stream_wan22_submission.yaml` | **new** — WAN2.2 SingleStream submission (perf + accuracy + TEST04)                                |

---

## 7. Extending to other audit tests

Adding a test whose run behavior is already expressible touches **four things**: a new file
under `compliance/tests/`, one `AuditTestId` enum value, a per-test config model added to
the `AuditConfig` discriminated union (`Annotated[Test04Config | TestNNConfig, Field(discriminator="test")]`),
and one import line in `compliance/tests/__init__.py`. The orchestrator, load generator,
verdict writer, and CLI are untouched.

**Orchestrator example (TEST01 — same-model check):**

```python
# compliance/tests/test01.py
class Test01Audit:
    test_id = AuditTestId.TEST01

    def plan_runs(self, cfg: AuditConfig) -> list[RunSpec]:
        return [
            RunSpec("performance", cfg.samples, SampleOrderSpec.without_replacement()),
            RunSpec("accuracy",    cfg.samples, SampleOrderSpec.without_replacement()),
        ]

    def verify(self, runs: list[RunArtifacts]) -> AuditVerdict:
        perf, acc = runs
        return AuditVerdict("TEST01", perf.model_outputs_match(acc), {...})

register(Test01Audit())
```

**Analyzer example (TEST09 — output-length check):** `plan_runs` returns a single normal
run; `verify` reads `events.jsonl` and checks mean OSL within `[ref × 0.9, ref × 1.1]`.

**What costs more than one file (honest limits):**

1. A test needing run behavior `SampleOrderSpec` cannot express → add **one variant** to
   `SampleOrderSpec` + its branch in `create_sample_order`. A typed extension of the single
   generic seam, not leakage.
2. TEST06/09 need raw output token IDs (see §2) → one isolated, audit-capture data-path
   addition shared by all token-level tests. TEST04 and TEST01 need none of it.

---

## 8. Requirements traceability

Covers **every** comment thread on PR #332 — the maintainer workflow threads
(@nvzhihanj, @viraatc), both "Review Council" passes, and the Gemini robustness comments.

### Maintainer workflow & example-config threads

| Comment                                                            | Resolution                                                                                                                                                                                                                 |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Run **one command**, not two/three; phases back-to-back            | `run_audit` generic loop; `audit:` block on `benchmark from-config`                                                                                                                                                        |
| Perf + accuracy + audit from a single config                       | `type: submission` YAML; `run_benchmark` runs perf [+acc], then `run_audit` additively (§5)                                                                                                                                |
| Comparing 50 distinct vs 20/25 repeated "doesn't seem fair"        | **resolved** — shipped examples use **equal** counts (Offline 64/64, SingleStream 20). `audit_samples` allows independent counts (upstream TEST04 uses 5000/500 for SDXL) as an opt-in to shorten the audit phase; see §5. |
| "Forced to run 248 in audit … too long"                            | `samples` (reference) and `audit_samples` (audit) are independent subsets; no full-dataset requirement                                                                                                                     |
| Audit sample "shuffled or fixed?"                                  | fixed — reference = `WITHOUT_REPLACEMENT`, audit = `SINGLE(sample_index)` (MLPerf `issue_same`)                                                                                                                            |
| Need an audit config for single-stream too                         | load-pattern validation admits `concurrency` (single-stream) and `max_throughput` (offline)                                                                                                                                |
| Paced loads should not silently pass                               | `poisson` rejected up front (§4 step 3) — pacing caps throughput and masks caching                                                                                                                                         |
| Inconsistent / context-free example file names                     | the shipped example YAMLs use context-rich sibling names (`offline_wan22_submission.yaml` / `single_stream_wan22_submission.yaml`); verdict artifacts use fixed `verify_TEST04.txt` + JSON                                 |
| `num_workers` hard-coded in example YAMLs; use default             | omitted from the shipped examples — they carry only what TEST04 requires (endpoint defaults otherwise)                                                                                                                     |
| README / unrelated dependency churn (`pip`, `aiohttp`) in the diff | this PR contains only the design doc + the two WAN2.2 example configs — no README or dependency changes bundled                                                                                                            |

### Design-review findings (both Review Council passes)

| Finding (severity)                                           | Resolution                                                                                                                                           |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ref_samples` dead write / mismatched counts (high)          | each phase's count is an explicit `RunSpec.n_samples` honored via `n_samples_to_issue` (the bug was the reference count being silently dropped)      |
| No `AuditTest` abstraction; TEST04 hardcoded (high)          | `AuditTest` protocol + `get_audit_test` registry; generic loop                                                                                       |
| `DatasetType.AUDIT` abstraction leak (high)                  | dropped; phases derive from a normal PERFORMANCE dataset                                                                                             |
| `test04` boolean in `RuntimeSettings`/load-gen (high)        | generic `SampleOrderSpec`; load-gen has no test knowledge                                                                                            |
| `_OVERRIDE_TEST04_SAMPLE_INDEX` stringly-typed kwarg (med)   | typed `run_spec` seam                                                                                                                                |
| Two-phase `model_copy` surgery; ref skips validation (med)   | declarative `RunSpec`; validate all specs before any run                                                                                             |
| Orchestrator untested (med)                                  | unit tests assert per-phase counts + early-return paths                                                                                              |
| Scattered params / hardcoded threshold (med)                 | per-test config model (`Test04Config`), discriminated on `test` — each test carries only its own knobs                                               |
| Unfair QPS comparison across counts/contents (med)           | examples use **equal** counts; per-phase completion guard + qps rate-normalization keep unequal counts sound when opted into (upstream-faithful, §5) |
| Audit params belong in `AuditConfig`, not `Dataset` (med)    | `AuditConfig` sub-model on `BenchmarkConfig`; `Dataset` untouched                                                                                    |
| Two parallel verifier entry points (low)                     | one `verify_test04(RunStats, RunStats)` core + `from_*` adapters                                                                                     |
| `sample_index` bound-checked late (low)                      | validated vs loaded dataset size before any run                                                                                                      |
| `audit_config` re-entrancy trap (critical)                   | every phase config sets `audit=None`; cannot re-enter `run_audit`                                                                                    |
| Orchestrator returns `None`; PASS/FAIL indistinguishable     | `run_audit` returns a typed `AuditVerdict`; CLI exits `0`/`1`/`2`                                                                                    |
| Non-atomic verdict write (high)                              | `write_verdict` uses `tmp → fsync → rename → fsync(parent)`                                                                                          |
| Duplicates `setup_benchmark` dir / `config.yaml` logic (med) | phases reuse `setup_benchmark`; no recomputed report-dir                                                                                             |
| `_audit_marker` parsed twice in error path (low)             | n/a — orchestrator owns phase labels, so no directory-swap guard                                                                                     |

### Robustness & API hygiene (Gemini + Review Council)

| Comment                                                              | Resolution                                                                       |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| CLI catches only `FileNotFoundError`/`ValueError`; write outside try | `from_dir` adapter + CLI catch all `OSError`; verdict write inside the guard     |
| `_audit_marker` `AttributeError` on non-`dict` JSON                  | adapters `isinstance`-guard parsed JSON; malformed input → clean error, exit `2` |
| `Report.from_snapshot` `KeyError`/`TypeError` uncaught               | wrapped in the `from_dir` adapter and re-raised as a clean error (exit `2`)      |
| Public entry points missing from `__all__`                           | `compliance/__init__.py` `__all__` exports the full public surface               |

---

## 9. Success criteria (goal-driven; verify before done)

1. **Integration** — `benchmark from-config` with an `audit:` block runs both phases
   back-to-back and writes `verify_TEST04.txt` + `audit_verdict.json`; PASS against a
   no-caching `mock_http_echo_server`, FAIL against a caching mock.
2. **Completion guard** — a phase that completes far fewer than its _requested_ count fails
   the verdict (`completed < requested × (1 − threshold)` → FAIL), independent of the other
   phase's count.
3. **Unit** — `SingleSampleOrder` always yields the configured index (bounds-checked);
   `verify_test04` PASS within threshold, FAIL above, boundary at the strict `<` line,
   slower-passes, custom threshold, zero/negative inputs raise, and the completion guard
   trips; `Test04Audit.plan_runs` emits a reference spec at `samples` and an audit spec at
   `audit_samples` (which may differ).
4. **Unit (orchestrator)** — assert the reference phase issues `samples` and the audit phase
   issues `audit_samples` (defaulting to `samples` when omitted), validation fires before any
   run, the typed verdict propagates (PASS/FAIL distinguishable), and a phase config never
   carries `audit` (no re-entry).
5. **Validation** — a paced (`poisson`) load and an out-of-range `sample_index` are both
   rejected before any phase runs.
6. **Robustness** — `from_dir` on a missing / non-`dict` / truncated / structurally-invalid
   snapshot exits `2` with a clear message, never a traceback.
7. **No leakage** — `grep -r test04 src/inference_endpoint/{load_generator,config/runtime_settings.py}`
   returns nothing.
8. `pre-commit run --all-files` clean (ruff / mypy / license headers).
