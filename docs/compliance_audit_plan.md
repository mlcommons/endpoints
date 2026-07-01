# Compliance Audit Module — Design Plan

Status: **Implemented** (TEST04) · TEST01/06/07/09 as planned extensions.

This document plans a modular compliance audit framework for the endpoint benchmarking
tool that re-implements the _intent_ of the MLPerf Inference compliance ("audit") tests.
The reference implementation lives in the MLCommons inference repo
(`compliance/nvidia/TESTxx`).

This is a ground-up redesign. The driving requirements come from two sources: the
maintainer's workflow constraints (a single command that runs both phases back-to-back
against the same endpoint) and a first-principles design review (TEST04 must not be bolted onto
the benchmark via per-phase config surgery — it needs a first-class, extensible
abstraction).

---

## The `AuditTest` abstraction

A single protocol covers both kinds of test — those that must execute one or more
specially-configured runs, and those that only analyze an ordinary run's artifacts after
the fact. The difference is just how many specs `plan_runs` returns, so the orchestration
loop never special-cases a test.

```python
class AuditTest(Protocol):
    test_id: ClassVar[AuditTestId]                          # AuditTestId.OUTPUT_CACHING_TEST
    def plan_runs(self, cfg: AuditConfig) -> list[AuditRunSpec]: ...
    def verify(self, runs: list[AuditRunArtifacts], cfg: AuditConfig) -> AuditResult: ...
```

- **Multi-phase (TEST04, TEST01):** `plan_runs` returns ≥2 specs.
- **Post-run only (TEST06, TEST07, TEST09):** `plan_runs` returns 1 normal-run spec; all
  logic lives in `verify`.

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

| Test   | Detects                                             | Required for                                 |
| ------ | --------------------------------------------------- | -------------------------------------------- |
| TEST01 | Different model in perf vs accuracy run             | ResNet50, BERT, SDXL, RetinaNet, …           |
| TEST04 | Caching of duplicate queries (throughput inflation) | ResNet50, SDXL, WAN2.2 (LLMs largely exempt) |
| TEST06 | LLM output consistency (EOS / first-token / length) | llama2/3.1, mixtral, deepseek                |
| TEST07 | Accuracy ≥ threshold in perf mode                   | gpt-oss-120b                                 |
| TEST09 | Mean output token length within ±10% of reference   | gpt-oss-120b                                 |
| TEST08 | DLRM-v3 streaming accuracy                          | DLRM-v3 — **out of scope** (not LLM)         |

**Category** is this design's own grouping (not an MLPerf term): _orchestrator_ tests
drive extra benchmark phases and compare them (TEST01, TEST04 — what this module runs);
_analyzer_ tests post-process a single run's logs (TEST06/07/09). It determines whether a
test reuses the `setup/run/finalize` engine (orchestrator) or only reads artifacts (analyzer).

**TEST04 (mechanism).** `audit.config` sets `performance_issue_same=1` /
`performance_issue_same_index=3` so LoadGen issues the **same sample repeatedly** for the
**same number of queries** as the standard run, then the verification compares throughput.
Pass if the audit run is **not more than 10% faster** than the reference (20% for
low-throughput streams). If the SUT caches responses for duplicate queries, throughput
inflates → FAIL.

---

## 2. Conceptual mapping: MLPerf → this repo

This tool is its own HTTP load generator (no LoadGen). The audit module re-implements the
_intent_ over this repo's own artifacts.

| MLPerf                                 | This repo                                                     |
| -------------------------------------- | ------------------------------------------------------------- |
| `audit.config` (run-setting override)  | a typed **`SampleOrderSpec`** carried on a **`AuditRunSpec`** |
| `mlperf_log_accuracy.json` (token IDs) | `events.jsonl` (must carry token IDs for token-level tests)   |
| `run_verification.py` → `verify_*.txt` | an **`AuditTest.verify()`** → `verify_<TEST>.txt` + JSON      |
| LoadGen runs both phases of a test     | a **generic orchestrator** runs `plan_runs()` back-to-back    |
| compliance submission dir layout       | mirrored under the run's report dir                           |

This repo names MLPerf **TEST04** the **output-caching test**: id `output_caching_test`
(`AuditTestId.OUTPUT_CACHING_TEST`), config class `OutputCachingTestConfig`, audit
`OutputCachingAudit`, and artifacts (under `<report_dir>/audit/`)
`verify_OUTPUT_CACHING_TEST.txt` + `audit_result.json`. Where this doc writes
"TEST04" it means the upstream MLPerf test the output-caching audit re-implements.

---

## 3. Two axes (the core principle)

Every audit test decomposes into two independent concerns. Keeping them separate is what
prevents test-specific knowledge from leaking into general-purpose code.

- **Axis A — run modification** (the `audit.config` analogue): _how_ a test alters the
  benchmark run(s). For TEST04 it is "issue one fixed sample repeatedly for the audit
  phase." This is expressed as a generic, typed **`SampleOrderSpec`**, not a per-test
  boolean. The load generator never learns the string "output_caching_test".
- **Axis B — verification**: a pure post-run check comparing run artifacts → a result.
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
   AuditTest  ──────────────────────────────  compliance/audit_test/output_caching_test.py
     ├─ plan_runs(cfg) -> list[AuditRunSpec]        (declarative: what phases to run)
     └─ verify(runs,cfg)-> AuditResult         (pure: read artifacts → result)
            │
            │ 2. for each AuditRunSpec
            ▼
   setup_benchmark(config, run_spec)           commands/benchmark/execute.py  (reused)
            │   run_spec.sample_order
            ▼
   create_sample_order(settings)               load_generator/sample_order.py
     └─ switch on SampleOrderSpec              (WITHOUT_REPLACEMENT | SINGLE(index))
            │   no "output_caching_test" knowledge here
            ▼
   run_benchmark_async(ctx) ─► AuditRunArtifacts    (final_snapshot.json, events.jsonl)
            │
            │ 3. verify(runs, cfg) ; 4. write_result (atomic)
            ▼
   <report_dir>/audit/ : verify_OUTPUT_CACHING_TEST.txt  +  audit_result.json
```

### Program flow (output-caching audit / MLPerf TEST04, two phases)

Every decision gate is shown, with the exit code it produces. Exit codes:
`0` PASS · `1` FAIL · `3` SetupError · `4` ExecutionError · `130` interrupted
before the audit.

```
                         ┌─────────────────────────┐
                         │  benchmark from-config  │
                         └────────────┬────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │  run_benchmark(cfg,mode) │
                         │  → returns report_dir    │
                         └────────────┬────────────┘
                                      ▼
                  ┌──────────────────────────────────────┐
                  │ setup_benchmark + run main benchmark  │
                  │ (perf  [+ accuracy if acc datasets])  │
                  └────────────────┬─────────────────────┘
                                   ▼
                       ╱────────────────────────╲    yes   ┌─────────────────────────┐
                      ╱ KeyboardInterrupt during   ╲──────►│ salvage partial results │
                      ╲     the main run?           ╱       │ re-raise → exit 130     │
                       ╲────────────┬──────────────╱        │ (audit NOT started)     │
                                    │ no                    └─────────────────────────┘
                                    ▼
                          ╱──────────────────╲   no   ┌────────────────────────┐
                         ╱  config.audit set?  ╲─────►│ exit 0 (no audit)      │
                         ╲────────┬────────────╱       └────────────────────────┘
                                  │ yes  (cli._run dispatches)
                                  ▼
                   ┌────────────────────────────────────────┐
                   │ run_audit(cfg, report_dir / "audit")    │
                   │ test = get_audit_test(test_id)          │
                   └────────────────┬───────────────────────┘
                                    ▼
            ╱────────────────────────────────────────-╲   no   ┌───────────────────────┐
           ╱           ≥1 performance dataset?          ╲─────►│ raise SetupError      │
           ╲────────────────────┬───────────────────-──╱       │     → exit 3          │
                                 │ yes                                    ▲
                                 ▼                                        │
            ┌──────────────────────────────────────────┐                 │
            │ specs = test.plan_runs(cfg)               │                 │
            │   [ "reference"      (without_replacement),│                 │
            │     "output_caching" (single(index)) ]    │                 │
            └────────────────┬─────────────────────────┘                 │
                             ▼                                            │
        ╔═══════════ for each spec (back-to-back) ═════════════╗         │
        ║                    ▼                                  ║         │
        ║   ┌──────────────────────────────────────────┐       ║         │
        ║   │ phase_cfg = perf-only (accuracy datasets  │       ║         │
        ║   │   dropped; audit=None; report_dir=<label>)│       ║         │
        ║   │ ctx = setup_benchmark(phase_cfg, run_spec)│       ║         │
        ║   └──────────────┬───────────────────────────┘       ║         │
        ║                  ▼                                    ║         │
        ║          ╱──────────────╲ yes  ╱──────────────────────╲║         │
        ║         ╱  first phase?   ╲───►╱ test.validate(cfg, N): ╲╫─raises─┘
        ║         ╲────────┬────────╱    ╲  ref count ≤ N AND each ╱║(SetupError
        ║                  │ no           ╲ fixed index ∈ [0,N)?  ╱ ║ → exit 3)
        ║                  │               ╲──────┬─────────────-╱  ║
        ║                  │◄─────────────────────┘ ok             ║
        ║                  ▼                                       ║
        ║   ┌──────────────────────────────────────────┐          ║
        ║   │ run_benchmark_async(ctx) → finalize       │          ║
        ║   └──────────────┬───────────────────────────┘          ║
        ║                  ▼                                       ║
        ║       ╱───────────────────────╲ no   ┌─────────────────────────┐
        ║      ╱ report not None AND       ╲───►│ raise ExecutionError    │
        ║      ╲   report.complete?         ╱   │   → exit 4              │
        ║       ╲──────────┬──────────────╱     │ (no verdict on partial) │
        ║                  │ yes                 └─────────────────────────┘
        ║                  ▼                                       ║
        ║   ┌──────────────────────────────────────────┐          ║
        ║   │ append AuditRunArtifacts(label, report,   │          ║
        ║   │   n_requested = spec.n_samples)           │          ║
        ║   └──────────────────────────────────────────┘          ║
        ╚══════════════════╪═══════════════════════════════════════╝
                           ▼ (all phases done)
            ┌──────────────────────────────────────────┐
            │ result = test.verify([ref, audit])        │
            │  • completion guard:                      │
            │      completed ≥ requested × (1 − thr)    │
            │  • caching rule:                          │
            │      audit_qps < ref_qps × (1 + thr)      │
            └──────────────┬───────────────────────────┘
                           ▼
            ┌──────────────────────────────────────────┐
            │ write_result → <report_dir>/audit/ [atomic]│
            │   audit_result.json        (durable first) │
            │   verify_OUTPUT_CACHING_TEST.txt  (marker) │
            └──────────────┬───────────────────────────┘
                           ▼
            ┌──────────────────────────────────────────┐
            │ return AuditResult → cli._run             │
            │   exit 0 (PASS)  /  exit 1 (FAIL)         │
            └──────────────────────────────────────────┘
```

The first-phase gate calls `AuditTest.validate(cfg, N)` once `N` (the loaded dataset size)
is known: each audit owns its own preconditions there, so the generic loop never encodes a
single test's rules. For the output-caching test that means the distinct-sample reference
count must fit the dataset (else `WithoutReplacementSampleOrder` would wrap and re-issue,
making the baseline cacheable) and every fixed `sample_index` must be in range. A failure
raises `SetupError` (exit 3) before any phase issues load.

Analyzer tests (TEST06/07/09) take the same path with a single-element `plan_runs`, so
phase 2 simply doesn't exist and `verify` reads the one run's artifacts.

In a `type: submission` config (see §5) this whole `run_audit` block runs **after** the
main perf [+ accuracy] run, under the same `report_dir`.

### `AuditRunSpec` — declarative and typed

Replaces ad-hoc per-phase `model_copy` surgery and stringly-typed override kwargs.

```python
@dataclass(frozen=True, slots=True)
class AuditRunSpec:
    label: str                    # "reference" / "output_caching" → report subdir
    n_samples: int | None         # this phase's query count (may differ per phase; None = dataset default)
    sample_order: SampleOrderSpec # WITHOUT_REPLACEMENT | SINGLE(index)
```

### `SampleOrderSpec` — the one generic load-gen seam

```python
# load_generator/sample_order.py
class SampleOrderSpec:   # WITHOUT_REPLACEMENT | SINGLE(index=...)
    ...

def create_sample_order(settings: RuntimeSettings) -> SampleOrder:
    spec = settings.sample_order            # generic; default WITHOUT_REPLACEMENT
    ...                                      # switch on spec, no "output_caching_test" knowledge
```

### `AuditConfig` — per-test discriminated union on `BenchmarkConfig`

Each test carries **only its own knobs** in a per-test config model, discriminated on
`test`. This avoids a flat model where one `threshold` field means different things per
test (caching tolerance vs OSL band vs accuracy floor) and is meaningless for the
equality-based tests (TEST01/06). No `DatasetType.AUDIT`, no audit fields on the shared
`Dataset` model.

```python
class AuditTestId(str, Enum):
    OUTPUT_CACHING_TEST = "output_caching_test"   # MLPerf TEST04

class OutputCachingTestConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    test: Literal[AuditTestId.OUTPUT_CACHING_TEST]
    samples: int                      # reference phase count (required, ge=1)
    audit_samples: int | None = None  # audit phase count; None = equals `samples`
    sample_index: int = 0             # MLPerf performance_issue_same_index
    threshold: float = 0.10           # caching tolerance (MLPerf TEST04-specific)

# One member today; becomes a discriminated union as tests are added:
#   AuditConfig = Annotated[OutputCachingTestConfig | Test01Config | ..., Field(discriminator="test")]
AuditConfig = OutputCachingTestConfig
```

`audit:` holds a **single** `AuditConfig`, and the union is discriminated on `test`, so two
`OutputCachingTest` blocks can't coexist. To repeat the audit over several prompts, widen
`sample_index` to a list inside the one config and have `plan_runs` emit one audit phase per
index — not duplicate the block.

On `BenchmarkConfig`: `audit: AuditConfig | None = None`. With a single member the alias is
just `OutputCachingTestConfig`; the `test: Literal[...]` discriminator field is already in place, so
adding the second test only assembles the `Annotated[... , Field(discriminator="test")]`
union — no change to existing tests.

**One audit per run.** `audit` is a single object, not a list, and `sample_index` is a
single index — there is no way to configure two `OutputCachingTest` instances (e.g. two
different indices) in one config. This matches MLPerf TEST04, which fixes exactly one
`performance_issue_same_index`; to audit a second index, run the benchmark again with a
different `sample_index`. The union discriminates over _different test types_ (TEST04,
TEST01, …), not multiple instances of one test — that would be a deliberate future change
(`audit: list[AuditConfig]` with `run_audit` looping).

**`samples` is required (no full-dataset/`None` mode).** An audit needs an explicit
reference count so the per-phase completion guard has an independent target to validate
against (a duration-driven, countless phase would make the guard tautological). `samples`
sizes the reference phase; `audit_samples` the fixed-sample phase, falling back to `samples`
when omitted (equal counts — the shipped examples). The two counts **may** differ (set
`audit_samples` lower, e.g. 64 / 32, to shorten the audit phase — upstream TEST04 does this;
see §5): the result relies on `qps` being rate-normalized plus a per-phase completion guard,
so it does not require equal counts.

### Generic orchestrator

`run_benchmark` first executes the main benchmark (performance, plus accuracy scoring when
the config carries accuracy datasets) exactly as today and returns its `report_dir`. Then,
when `config.audit is not None`, `cli._run` runs `run_audit(config, report_dir / "audit")`
(in `commands/audit.py`) as an **additive post-step**. If the main run is interrupted
(`KeyboardInterrupt` / Ctrl-C), `run_benchmark` salvages partial results and re-raises
**without** starting the audit — an interrupted run must not silently roll into the (long)
compliance phases. The two stages are independent, self-contained operations sequenced at
the top level — not per-phase config surgery — so one `type: submission` YAML can produce
the full set: perf [+ accuracy] + the audit's reference and output_caching phases + result
(§5). The audit runs its **own** reference phase at `samples`; it does not reuse the
(typically larger, full-dataset) submission perf run. The generic loop never names a
specific test:

1. `test = get_audit_test(config.audit.test)`
2. `specs = test.plan_runs(config.audit)`
3. **Validate before any run executes.** The first phase's `setup_benchmark` loads the
   dataset; its row count `N` is then handed to `test.validate(config.audit, N)`, which owns
   every test-specific precondition (the orchestrator stays test-agnostic — it imposes no
   load-pattern restriction of its own). For the output-caching test that means the
   distinct-sample reference count must fit `N` (else the order wraps and re-issues samples,
   making the baseline cacheable) and every fixed `sample_index` must be in `[0, N)`. This
   reuses the first load — no separate probe — and surfaces a bad config as `SetupError`
   before any phase issues load.
4. Execute each spec back-to-back via the existing `setup_benchmark` /
   `run_benchmark_async` path (no duplicated report-dir or `config.yaml` logic). Each phase
   config is performance-only (accuracy datasets dropped so no phase re-issues or re-scores
   them) and has `audit=None` to prevent re-entry into `run_audit`. If any phase raises
   (`SetupError` / `ExecutionError`), `run_audit` aborts **without verifying** — a crashed
   phase must never produce a result. A phase that returns but whose `Report.complete` is
   `False` (metrics drain timed out, or the run was interrupted → partial stats) is likewise
   rejected with `ExecutionError` — a result is never certified on partial data. Errors
   propagate to the standard CLI handler (`main.py`), which maps `SetupError` → exit `3` and
   `ExecutionError` → exit `4`.
5. `result = test.verify(runs, cfg)`
6. Atomically write the result (`tmp → fsync → rename → fsync(parent)`).
7. Return the typed `AuditResult`. Because `run_benchmark` currently returns `None` and
   `cli.py` ignores its return, the audit path must **propagate** the result: `run_audit`
   returns it, `run_benchmark` returns it for an audit config, and `cli.py` maps it to
   `sys.exit` — `0` (PASS) / `1` (FAIL). Errors are not flattened to a single code:
   they propagate to `main.py`'s handler, which uses the repo-wide scheme
   (`InputValidationError` → `2`, `SetupError` → `3`, `ExecutionError` → `4`). The on-disk
   `audit_result.json` is the durable record; the exit code is the automation signal.

### Verifier — one core + in-process adapter

```python
@dataclass(frozen=True, slots=True)
class AuditRunStats:          # .from_report(Report, n_requested)
    qps: float
    n_completed: int
    n_requested: int
    # from_report raises ValueError when the report has no duration (qps is None)
    # or zero throughput (qps <= 0) — a degenerate run can't anchor the ratio.

def verify_output_caching(ref: AuditRunStats, audit: AuditRunStats, threshold: float = 0.10) -> AuditResult:
    # per-phase completion guard: each phase completed >= requested * (1 - threshold)
    #   (catches a phase that mostly failed — bogus low qps — without assuming ref == audit)
    # caching rule:               audit.qps < ref.qps * (1 + threshold)
```

The phases may issue **different** counts (`samples` vs `audit_samples`), so the result does
**not** require `ref.n_completed == audit.n_completed`. Validity comes from `qps` being a
rate (caching still shows up as a throughput spike) plus the per-phase completion guard,
which rejects a run that crashed partway and would otherwise post a misleadingly low qps.

`AuditRunStats.from_report(Report, n_requested)` is the sole adapter — the in-process path the
orchestrator uses, guarding `qps is None` (no duration) and `qps <= 0` (no completions) with
a clean `ValueError`. The redesign exposes no standalone verifier CLI and no offline
re-check-from-disk adapter — the audit runs only via `benchmark from-config`.

---

## 5. Module layout

```
src/inference_endpoint/compliance/
├── __init__.py        # AuditTest protocol, AuditRunSpec/Stats/Artifacts, get_audit_test(); imports audit_test to register
├── result.py          # AuditResult + atomic write → verify_<TEST>.txt + audit_result.json
└── audit_test/
    ├── __init__.py     # imports test modules so their register(...) calls fire
    ├── output_caching_test.py  # OutputCachingAudit: plan_runs (reference + audit specs) + verify_output_caching core
    └── README.md       # usage: config block, load patterns, output, pass criteria
```

CLI surface: an `audit:` block in the benchmark YAML, picked up by `benchmark from-config`.
For the config fields, supported load patterns, output files, and pass criteria, see
[`compliance/audit_test/README.md`](../src/inference_endpoint/compliance/audit_test/README.md).
This section covers only how the audit composes with the main run.

### Unified submission (perf + accuracy + audit in one file)

`audit:` is additive, so a single `type: submission` config drives the whole submission:
`run_benchmark` does the performance run, scores the accuracy datasets, then runs the audit
— one command, one `report_dir`. Each piece is optional: drop `audit:` for perf+acc, or
omit accuracy datasets for perf+audit.

The committed example is
[`examples/09_Wan22_VideoGen_Example/offline_wan22_submission.yaml`](../examples/09_Wan22_VideoGen_Example/offline_wan22_submission.yaml)
— one file, run in order: performance run (248-prompt dataset) → VBench accuracy scoring →
audit (reference + fixed-sample phases).

Resulting `report_dir/` (main perf/accuracy artifacts keep their current layout; the audit
nests all of its output under a dedicated `audit/` subfolder):

```
report_dir/
├── final_snapshot.json          # submission perf run (existing top-level layout)
├── events.jsonl
├── …                            # accuracy scoring outputs (existing)
└── audit/                       # all audit output lives here
    ├── reference/               # audit reference phase    (samples=64)
    ├── output_caching/          # audit fixed-sample phase (samples=64)
    ├── verify_OUTPUT_CACHING_TEST.txt
    └── audit_result.json
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
> phase. The result does **not** require equal counts — `qps` is rate-normalized and a
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

Both scenarios ship as committed configs (see also
[`compliance/audit_test/README.md`](../src/inference_endpoint/compliance/audit_test/README.md)):

- **Offline** — `load_pattern.type: max_throughput`, `samples: 64` / `audit_samples: 64`,
  `threshold: 0.10`:
  [`offline_wan22_submission.yaml`](../examples/09_Wan22_VideoGen_Example/offline_wan22_submission.yaml).
- **SingleStream** — `load_pattern.type: concurrency` (`target_concurrency: 1`), `samples: 20`
  (audit defaults to the same), `threshold: 0.20` (low-throughput tolerance):
  [`single_stream_wan22_submission.yaml`](../examples/09_Wan22_VideoGen_Example/single_stream_wan22_submission.yaml).

---

## 6. Extending to other audit tests (not yet implemented)

> Only `output_caching_test` (TEST04) exists today. This section sketches how the
> abstraction accommodates future tests; the examples below are illustrative, not shipped.

---

## 7. Extending to other audit tests

Adding a test whose run behavior is already expressible touches **four things**: a new file
under `compliance/audit_test/`, one `AuditTestId` enum value, a per-test config model added to
the `AuditConfig` discriminated union (`Annotated[OutputCachingTestConfig | TestNNConfig, Field(discriminator="test")]`),
and one entry in the `AUDIT_TESTS` map in `compliance/__init__.py`. The orchestrator, load
generator, result writer, and CLI are untouched.

**Orchestrator example (TEST01 — same-model check):**

```python
# compliance/audit_test/test01.py
class Test01Audit:
    test_id = AuditTestId.TEST01

    def plan_runs(self, cfg: AuditConfig) -> list[AuditRunSpec]:
        return [
            AuditRunSpec("performance", cfg.samples, SampleOrderSpec.without_replacement()),
            AuditRunSpec("accuracy",    cfg.samples, SampleOrderSpec.without_replacement()),
        ]

    def validate(self, cfg: AuditConfig, dataset_size: int) -> None:
        ...  # bounds-check this test's phases against the dataset

    def verify(self, runs: list[AuditRunArtifacts], cfg: AuditConfig) -> AuditResult:
        perf, acc = runs
        return AuditResult("TEST01", perf.model_outputs_match(acc), {...})

# wire into compliance/__init__.py:  AUDIT_TESTS[Test01Audit.test_id] = Test01Audit()
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

## 8. Success criteria (goal-driven; verify before done)

1. **Integration** — `benchmark from-config` with an `audit:` block runs both phases
   back-to-back and writes `verify_OUTPUT_CACHING_TEST.txt` + `audit_result.json`; PASS against a
   no-caching `mock_http_echo_server`, FAIL against a caching mock.
2. **Completion guard** — a phase that completes far fewer than its _requested_ count fails
   the result (`completed < requested × (1 − threshold)` → FAIL), independent of the other
   phase's count.
3. **Unit** — `SingleSampleOrder` always yields the configured index (bounds-checked);
   `verify_output_caching` PASS within threshold, FAIL above, FAIL at the exact boundary (`<`,
   matching upstream `verify_performance.py`), slower-passes, custom threshold, and the
   completion guard trips; `AuditRunStats.from_report`
   raises on a `None`-duration or non-positive `qps`; `OutputCachingAudit.plan_runs` emits a
   reference spec at `samples` and an audit spec at `audit_samples` (which may differ).
4. **Unit (orchestrator)** — assert the reference phase issues `samples` and the audit phase
   issues `audit_samples` (defaulting to `samples` when omitted), validation fires before any
   run, the typed result propagates (PASS/FAIL distinguishable), and a phase config never
   carries `audit` (no re-entry).
5. **Validation** — `AuditTest.validate` rejects an out-of-range `sample_index` and a
   reference count exceeding the loaded dataset, both before any phase runs.
6. **Robustness** — `AuditRunStats.from_report` raises a clean `ValueError` on a report with no
   duration (`qps is None`) or non-positive throughput (`qps <= 0`); a phase whose
   `Report.complete` is `False` (metrics drain timeout / interrupt) aborts the audit with
   `ExecutionError` rather than certifying a result on partial data.
7. **No leakage** — `grep -r test04 src/inference_endpoint/{load_generator,config/runtime_settings.py}`
   returns nothing.
8. `pre-commit run --all-files` clean (ruff / mypy / license headers).
