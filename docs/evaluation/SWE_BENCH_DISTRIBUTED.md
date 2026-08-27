# Distributed SWE-bench (`swe_bench_fleet`)

> Shards a SWE-bench accuracy run across several SWE-bench services, classifies
> infrastructure damage separately from genuine model failures, and refuses to
> emit an accuracy number unless every planned instance is accounted for exactly
> once.

`swe_bench_scorer` runs the whole instance list as one service run against one
endpoint. That is the right shape for a hundred instances on one Docker host. It
does not survive a 200-instance run spread over many hours and many hosts: a
client crash loses everything, a host that dies takes its instances with it, and
an evaluation container that wedges is booked as an ordinary `error` — accounted
for, never retried, and silently subtracted from the score.

`swe_bench_fleet` addresses those six gaps and nothing else. It reuses the
service HTTP protocol, the Docker/Pyxis runtimes, the exact instance-id binding,
the artifact allow-list and the secret redaction unchanged.

## Configuration

```yaml
datasets:
  - name: swe_bench
    accuracy_config:
      eval_method: swe_bench_fleet
      extras:
        swebench_service_urls:
          - http://swe-host-1:18080
          - http://swe-host-2:18080
        swebench_service_auth_token: ${SWEBENCH_TOKEN}
        num_instances: 200
        shard_size: 10 # instances per unit; 200 / 10 = 20 units
        max_attempts: 3
        expected_model: Org/Model-FP8 # optional; gates checkpoint identity
        min_prompt_tokens: 2000 # tool-call gate scale floor
        stall_timeout_s: 10800
```

| Extra                   | Default  | Meaning                                                                                                                                                     |
| ----------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `swebench_service_urls` | required | One URL per service host. Duplicates are refused: two entries for one host is not extra capacity, it is two runs contending for the same container runtime. |
| `shard_size`            | 10       | Instances per unit.                                                                                                                                         |
| `max_attempts`          | 3        | Counted attempts before a unit is abandoned. Environment faults are not counted.                                                                            |
| `expected_model`        | none     | When set, every endpoint must serve exactly this checkpoint.                                                                                                |
| `min_prompt_tokens`     | 2000     | Floor the tool-call gate must prove it reaches. `0` waives the proof.                                                                                       |
| `stall_timeout_s`       | 10800    | A service completing no unit in this long is quarantined even if healthy.                                                                                   |

## What runs, in order

1. **Preflight gates** (`preflight()`, before the benchmark starts) — every
   service's `/health`, plus the endpoint gates below. Any failure raises
   `SetupError` before a single instance is dispatched, and every gate runs so
   one preflight reports every problem.
2. **Plan** — the instance list is split into units, and the plan is written
   once to `units.json` with a digest over the run id and the ordered ids.
3. **Dispatch** — one in-flight service run per service; each service loop
   claims a unit, submits it, polls, downloads artifacts, classifies, and
   publishes or retries.
4. **Merge gate** — `merge_run(queue, run_id)`. All-or-nothing.

## The gates

| Gate                      | Refuses                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CheckpointIdentityGate`  | Any endpoint not serving _exactly_ `expected_model`. `/get_model_info` is preferred over `/v1/models`, because the latter echoes `--served-model-name`, which is routinely identical across checkpoints. Comparison is `==`: `Org/Model` is a strict prefix of `Org/Model-FP8`, so any `startswith`/`in` test accepts FP8 as BF16. Unidentifiable means fail. |
| `ToolCallGate`            | Any endpoint that does not return a well-formed tool call — right name, `arguments` parse as JSON, non-empty `command`.                                                                                                                                                                                                                                       |
| `EndpointFingerprintGate` | Any endpoint whose identity cannot be read at all. Records a fingerprint compared again at publish time.                                                                                                                                                                                                                                                      |

**The scale rule.** Every gate implements `assert_scale()`, it runs before the
gate's own check, and a scale failure is a _gate failure_, never a skip.
`ToolCallGate` measures its prompt with the server's own `/tokenize` and fails if
the prompt is below `min_prompt_tokens`. This exists because a tool-call gate
that exercised exactly the right operation with a 278-token prompt passed
cleanly while every prompt above 2000 tokens silently returned an empty
completion. SWE-bench prompts are all far above 2000 tokens; the gate was green
and the run scored zero. **A gate that cannot prove its scale is not a gate.**

## The work queue

Under `<report_dir>/swe_bench_wq/`:

```
units.json                     immutable plan + digest
claims/<unit>/owner            host, pid, boot id, plan digest, SLURM job/step
claims/<unit>/hb               heartbeat (mtime only)
results/<unit>.json            terminal record (succeeded OR abandoned)
failed/<unit>.<n>.json         one per counted attempt
failed/env/<unit>.*.json       environment faults (not counted)
failed/artifacts/<unit>.attemptN/   evidence snapshot taken before a retry
```

A unit is available when it is in the plan and has **neither** a claim **nor** a
result. Claiming is `os.mkdir` and nothing else — `makedirs(exist_ok=True)` would
hand the unit to every caller.

### Re-running a unit

```bash
python scripts/swe_bench_wq.py requeue REPORT_DIR run-a.s07
```

`requeue` is the **only** supported way. Deleting the result file does not
requeue anything: the claim tombstone still hides the unit. `requeue` removes the
result, the claim and the counted attempt records together, and prints exactly
what it removed.

### Reaping abandoned claims

```bash
python scripts/swe_bench_wq.py reap REPORT_DIR            # dry run
python scripts/swe_bench_wq.py reap REPORT_DIR --apply --slurm
```

A claim is released only when it has no result, its heartbeat is stale, **and**
its owner is provably gone. Uncertainty never escalates: if the liveness probe
fails, times out, or returns an implausible answer, nothing is released. A false
reap gives one unit two owners, duplicate results, and a wrong denominator, with
no error anywhere.

`SlurmStepLiveness` treats an owner as dead when its job is absent from `squeue`
**or** its step is absent from `scontrol show step` while the job lives — a step
can die inside a live job, and the job-level rule alone then blocks those units
for the whole allocation. Step liveness never uses `squeue -s`, which reports
only `.extern` on the clusters this targets and would mark every live step dead.

## Classification and retry

Every unit is classified after the service reports success. The rule list is
ordered and first-match-wins; the order is load-bearing.

| Kind                                                                                                                                                                     | Class             | Why                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `container_fork_eagain`, `container_exec_refused`, `runtime_read_timeout`, `image_build_timeout`, `image_build_error`, `step_infrastructure_failure`, `endpoint_changed` | infra → **retry** | Defects in infrastructure we provided.                                                                                                                                                                |
| `test_timeout`                                                                                                                                                           | genuine           | A patch that makes the suite loop is a failing patch.                                                                                                                                                 |
| `test_memory_exceeded`                                                                                                                                                   | genuine           | A patch that makes a graded test allocate without bound is a failing patch. The alternative to killing it was never "the test passes", it was "the host OOMs and the instance still never completes". |
| `patch_apply_failed`                                                                                                                                                     | genuine           | The model emitted a diff that does not apply. SWE-bench books it as `error`, but it is model behaviour.                                                                                               |
| `unknown`                                                                                                                                                                | genuine           | **The bias rule.**                                                                                                                                                                                    |

**The bias rule is deliberately asymmetric.** An error that cannot be classified
confidently is genuine, never infrastructure. A false bad-run costs one redo; a
false retry biases the measurement toward optimism, and an optimistic accuracy
number is worse than no number.

`endpoint_changed` deserves its own note: an engine restarted under a live client
yields a run that scores near zero and exits successfully, and nothing in the
result distinguishes it from a genuinely bad model. The endpoint fingerprint
recorded at claim time is re-read at publish time; a change requeues the unit.

### Attempt accounting

- **Environment fault** (service unreachable, submit failed) — recorded under
  `failed/env/`, does **not** consume the attempt budget, and counts toward
  quarantining that service. A broken host is a property of the host, not of the
  unit.
- **Infra / failed** — counted. After `max_attempts` the unit is published as
  `abandoned` and its claim released, so it stops burning capacity and shows up
  loudly in the merge gate instead of spinning forever.

Before every retry, the small files that explain the failure are snapshotted to
`failed/artifacts/<unit>.attemptN/`, because the unit's run directory is reused
and a unit that fails then succeeds would otherwise leave only the success's
artifacts behind.

## The merge gate

`merge_run(queue, run_id)` produces a number only when **all** hold:

1. every planned unit has a terminal result;
2. no result is abandoned;
3. every unit's accounted instance **ids** equal its planned ids exactly — a set
   comparison, never a count, because a shard with one duplicate and one missing
   id has the right count and the wrong content;
4. the union across units equals the plan, with no id claimed twice;
5. every result carries the plan's digest;
6. no unit lost instances to infrastructure.

Otherwise it raises `MergeRefusal` listing every reason. There is no force flag,
no partial-credit path, and **no `merge_all`**: `run_id` is required, and merging
"everything that looks finished" once combined hundreds of banked results from
unrelated configurations into one number.

### What a refusal still publishes

A refusal that carries no numbers is not the end of the story -- somebody still
has to report _something_, and with the gate silent they compute it by hand. That
is how a run which lost 106 of its 200 instances to infrastructure came to be
reported as **47.0%** and compared against a complete-run reference of 70.67%. It
was read as a model regression. It was attrition.

`assess_run(queue, run_id)` performs the same arithmetic without deciding
anything and returns a `CompletenessReport`, which is attached to both
`MergeResult` and `MergeRefusal` and written to the run's merge artifacts.

| Field                                         | Always present     | Meaning                                                                                                                             |
| --------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `resolved_rate`                               | yes, may be `null` | The headline. `null` unless the run is structurally complete **and** lost nothing to infrastructure.                                |
| `conditional_resolved_rate`                   | yes                | Resolved over the instances that actually completed. Honest about what it measures; **not** comparable to a complete-run reference. |
| `resolved_rate_lower_bound`                   | yes                | Resolved over everything planned. Infrastructure losses can only ever _add_ resolutions, so this bounds the truth from below.       |
| `incomplete_instance_ids`                     | yes                | Exactly which instances never reached a terminal state.                                                                             |
| `infra_lost_instances`, `infra_lost_unit_ids` | yes                | What the harness lost, as distinct from what the model failed.                                                                      |
| `resolved_rate_withheld_reason`               | yes, may be `null` | Which of the two conditions failed, and by how much.                                                                                |

The two conditions are deliberately separate and conflating them gets both
wrong. An instance the model attempted and failed is a legitimate score; one the
harness dropped never had the chance. A run short of instances has the wrong
denominator; a complete run that leaned on the infrastructure has the wrong
provenance. A model that resolved nothing at all is a score, not a casualty, and
publishes `0.0` rather than withholding.

Ported from `wq_merge.sh:7-9` in the banked campaign: "shard_merge.py refuses to
print an accuracy unless all 20 shards account for exactly their own 10 ids, and
that refusal is the single most important property in this campaign." What is
added here is that the refusal shows its working.

`verify_inventory()` cross-checks three independently produced views — the plan,
the claim directory and the result directory — and treats disagreement as an
error. Checking one view against itself is how a verification pass agrees with a
broken system.

## Infrastructure retry

`retry_on_provable_non_execution()` retries an operation **only** when the
failure proves the work never happened, and never merely because an error
occurred. Re-running something that may already have run can apply an edit
twice, delete twice, or double a test run, and none of those announce
themselves.

The evidence is the exception's `provable_non_execution` attribute, read as an
attribute rather than an isinstance check so producer and consumer stay
decoupled. For a Pyxis step it means the status file still read `pending` **and**
no in-band sentinel arrived -- the step script did not run even its first line.
Anything that does not make that claim is re-raised immediately and does not
consume the attempt budget, which makes every exception type this module has
never heard of safe by default.

Measured signature, from an isolated probe with no model and no GPU (20 nodes,
200 workers, 6273 ordinary shell steps): 63 steps failed and all 63 were still
`pending`.

Retries are bounded and, more importantly, counted. `InfraRetryLedger` appends
one JSON line per event so a run that dies still leaves its retry history, and
`summary()` publishes:

| Field                              | Meaning                                           |
| ---------------------------------- | ------------------------------------------------- |
| `infra_retries_total`              | Every retry event.                                |
| `instances_saved_by_retry`         | Targets that recovered and did not later exhaust. |
| `infra_retries_exhausted`          | Targets the budget could not rescue.              |
| `infra_retry_succeeded_on_attempt` | Which attempt each recovery landed on.            |
| `run_quality`                      | `CLEAN` / `OK_WITH_RETRIES` / `DEGRADED`.         |

`run_quality` is `DEGRADED` on any exhaustion or not-retryable failure, and also
when more than 2% of operations needed a retry **even if every one of them
eventually succeeded**. The banked campaign retried environment faults without
limit and without counting them (`wq_worker.sh:41` `WQ_MAX_ATTEMPTS=5`, `:256`
"ENVIRONMENT FAULTS DO NOT CONSUME THE UNIT'S ATTEMPT BUDGET"), which is exactly
why nobody knew how many there had been. Measured effect of adding this loop:
`RunnerError` 59 -> 7 and resolve 47.0% -> 70.0% against a banked 70.67% on the
identical 200 instances. A rescue on that scale is not a clean run, and
`run_quality` says so at 200/200.

Cluster guidance, deliberately not in the package: on a busy controller these
non-launches cluster around slurmctld RPC rate limiting (`Job credential
expired`). Pacing step creation below the controller's `rl_refill_rate` prevents
them; the retry loop only recovers from them.

## Resource guards

`MemoryGuard` kills a graded test only when its resident memory is at or above
`kill_bytes` (default 150 GiB) **and** it has a container-supervisor ancestor.
There is deliberately no working-directory term: an earlier version required a
cwd inside the testbed and skipped a runaway that had grown to 667 GiB because
its cwd was `/tmp`. Every extra conjunct is another way for the guard to miss
what it exists to catch.

Two rules are enforced by construction and by test:

- **Kill by pid, never by pattern.** There is no `pkill`/`pgrep` path in the
  module and it never shells out; `kill_by_pid` refuses this process and its
  ancestors. A pattern can match the guard's own command line, and a long-lived
  daemon can carry a dead process's argv for days.
- **A conjunctive guard must not degenerate.** `combine_terms()` returns
  `INDETERMINATE`, never `UNHEALTHY`, when any term has zero evidence. An
  AND-guard whose honest term loses its data source collapses into its weaker
  clauses and starts firing on healthy targets.

The kill marker is written _before_ the signal, and its phase is load-bearing:
only `eval.*` markers make an instance's error a genuine failure. An `agent`
kill merely makes one tool call return an error observation, so it is recorded
for audit and must not influence classification. An unresolvable container name
fails closed to `unknown`.

## Operational notes

- `--kill-on-bad-exit=0` does **not** prevent a scheduler force-terminating a
  whole step when one node OOMs; OOM escalation is separate from task exit codes.
  The defence is small, independently retryable units plus `MemoryGuard` acting
  before the host dies — not the flag.
- A service that answers `/health` while completing nothing is the silent
  failure. Verify the effect, never the status: the dispatcher quarantines a
  service that has completed no unit within `stall_timeout_s` and requeues its
  work.
- In shell tooling around this queue, remember `grep -c` exits 1 on zero
  matches; a pipeline under `set -e` will abort on an empty, correct answer.

## What is intentionally not here

Cluster-lifecycle machinery — allocation rotation, holder chaining, image-store
construction and distribution, node suspect lists, multi-configuration campaign
bookkeeping — is out of scope for a benchmark client. The Pyxis runtime pulls
per-instance images from a registry and never builds a store, so that whole
class of image-corruption failure is architecturally absent rather than worked
around.
