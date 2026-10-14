# Validation submodule - policy and rule checks for a clean submission

## 1. What the validation submodule is for

`inference_endpoint.validation` checks a completed MLPerf Endpoints submission
against the Endpoints Policy compliance rules and reports every violation as a
finding. It exists to answer one question per submission: _would this upload
pass the review checklist?_ Each check reads the submission's directory layout
(`system_desc.json`, per-point `r<N>/` folders holding `point.yaml`,
`result_summary.json`, `accuracy_results.json`, and the like) plus the
submission's own declared bounds (`max_supported_concurrency`, the Low Latency
point) and evaluates named rule rows against that evidence. A failure produces a
`Finding` whose `why` spells out the values that made the rule fail.

Policy and rule references live in
[`mlcommons/endpoints_policies`](https://github.com/mlcommons/endpoints_policies/blob/d062b470abd2/endpoints_rules.md)
(commit-pinned). Check descriptions cite the exact section they enforce.

## 2. How cohort submission checks work

The user-facing artifact is a directory of YAML files, which store submission
checks for a given cohort (used as the directory's name), i.e.:

```
my_cohort/
└── point.yaml
```

Filenames correspond to the scope of the checks:

- `system.yaml`
- `submission.yaml`
- `curve.yaml`
- `point.yaml`

Files are optional - a missing file indicates there are no checks to run for
that scope. Other files in the directory are refused at load.

Each file contains a list of checks, with the check's ID as its key. Example:

```yaml
your_check_id:
  description: What the rule holds to.
  rule: "Endpoints Policy Section X.Y, The name of the rule row."
  fix: How to resolve a failed report.
  severity: warn
  assert:
    - left: point.config.concurrency
      op: gt
      right: 0
```

A cohort directory's name states which cohort of submissions its checks are for:
one named directory per cohort.

Run a cohort against one submission with the CLI:

```bash
python -m inference_endpoint.validation my_cohort submission_result_directory/
```

The submission argument is the upload root (`system_desc.json` and
`results/<system>/<model>/r<N>/` below it); each point's `point.yaml` carries
its `concurrency`, its `offline` declaration and its `system_tps`. Pass
`--ruleset config/rulesets/<name>/` to make the `ruleset` fold (its
`models.yaml`/`datasets.yaml`) available to rows; the `regions` fold is derived
for you (Section 5.5).

The run prints one line per failing finding plus a summary count (a line is one
evaluated `assert` predicate); `--audit` prints the holds as well, so the
session doubles as the audit trail. Exit codes: `0` every applicable check held,
`1` at least one failing finding, `2` the cohort or the submission was refused
at load (a duplicate check id, a check naming a fact fold the submission cannot
provide, a point without `system_tps`, ...).

## 3. How the files are written

One YAML mapping per check block under its id in the matching scope file. The
block's fields:

| Field               | Required   | What it is                                                                                                                |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| _(the mapping key)_ | yes        | The check's `id`; the scope file it lives in (`point.yaml`, ...) is its `scope      `                                     |
| `description`       | yes        | The human sentence the report shows for the finding                                                                       |
| `rule`              | yes        | Where in the Endpoints Rules this comes from                                                                              |
| `severity`          | yes        | `fail` (a blocking finding) or `warn` (reported and passes through)                                                       |
| `fix`               | no         | How to resolve a failing report                                                                                           |
| `needs`             | no         | Upstream check ids that must pass before this check speaks                                                                |
| `when`              | predicates | Gate predicates: the check runs only where all hold; this list may be empty or absent when `assert` carries at least one  |
| `assert`            | predicates | Condition predicates; each is one finding per instance; this list may be empty or absent when `when` carries at least one |

`when` and `assert` take lists of predicates. A predicate is either a
**comparison** or a **measure** (an aggregate over a collection); the
subsections below spell out each form. (A check runs once per **instance** of
its scope: once per submission, system, curve, or measurement point.) A full
check block in `point.yaml` (both kinds in one check):

```yaml
offline_ordering:
  description: Dedicated Offline run - Section 5.7.2 clauses.
  rule:
    "Endpoints Policy Section 5.7.2, Relationship to Maximum Supported
    Concurrency."
  fix:
    Either raise the dedicated run's system_tps to 0.98x the C_max point's, or
    elect the C_max point as Offline.
  severity: warn
  when:
    - left: point.config.offline
      op: eq
      right: dedicated
  assert:
    - left: derived.offline_ordering
      op: eq
      right: true
```

### Comparisons

Every comparison takes `left` and `op`; the remaining keys follow the `op`:

1. `between` takes an `Interval` in the form of the `lo` and `hi` keys,
   inclusive. For instance, `lo: 1` and `hi: 10` denotes the interval `[1, 10]`,
   checking if `1 <= left <= 10`.
2. All other binary operators take in a single value `right`. The following
   binary ops are available:

   - `eq`: Equality check. By default, floating point values will be computed
     with strict equality. Accepts the following optional fields:
     - `abs_tol` - Absolute tolerance to be forwarded to `math.isclose`
     - `rel_tol` - Relative tolerance to be forwarded to `math.isclose`
   - `ne`: Negation of `eq` - see notes above for optional fields.
   - `gt`, `ge`, `lt`, and `le`: The `>`, `>=`, `<`, and `<=` operators
     respectively.
   - `in`: A `contains` / membership operator. True if `left` is contained in
     `right`. `right` is expected to be a list-like container. If `right` is a
     mapping (`dict`-like object), then `in` will use the _keys_ as the list to
     check against.
   - `matches`: A regex comparison. `right` must be a `{"literal": ...}` block
     (patterns are compile-checked at load). Ex.:
     ```yaml
     - left: point.config.target_cohort
       op: matches
       right: { "literal": "20[0-9]{2}-(0[1-9]|1[0-2])-C[01]" }
     ```
   - `same_multiset`: An un-ordered list comparison, respecting counts (aka 'bag
     equality). Ex.:
     - `[1, 2, 2]` matches `[2, 1, 2]`
     - `[1]` does _not_ match `[1, 1]`
     - `[[1], [2]]` matches `[[2], [1]]`

3. Unary operators do not accept a `right` parameter. They key on whether the
   `left` reference resolves to a fact in the parsed submission files: a dotted
   reference like `point.config.steady_state.status` addresses a key in the
   loaded artifact documents, and a whole-document reference like
   `point.accuracy_results` addresses its artifact file. (Only the artifact
   vocabulary in section 4 can be referenced this way - there is no form for
   arbitrary file names. `left` must name such a `Reference`: asking whether a
   literal constant is present is refused at load.)

   - `exists`: True if `left` resolves to a value. A key deliberately stored
     with value `null` still counts as present; only a missing key (or a missing
     artifact file) does not.
   - `absent`: True if `left` does not resolve to a value - the opposite of
     `exists`. A key stored with value `null` is present, so it is **not**
     absent.
   - `not_null`: True if `left` resolves to a value that is not `null` (both a
     missing key and a stored `null` fail).

   A present-but-unparsable artifact is refused loudly at `Finding.collect` (see
   "Refusals"), so in a checked run `absent` always means genuinely missing.

Example:

```yaml
assert:
  # a range comparison, inclusive at both ends
  - left: point.config.concurrency
    op: between
    lo: 1
    hi: 32

  # a unary presence check
  - left: point.accuracy_results
    op: exists

  # a float equality check under both tolerances
  - left: point.result_summary.system_tps
    op: eq
    right: 5400.0
    abs_tol: 5.0
    rel_tol: 0.02
```

### Measures

Measures, or aggregates, are similar to the comparison operators above, but
additionally take a `measure` key to indicate an aggregation method to perform
on `over` (functionally the same as `left` in comparisons) before performing the
specified `op`.

It is expected that `over`, unlike `left` is a list-like collection. If `over`
is a mapping (`dict`-like), then the `measure` aggregates over the _values_ of
the mapping. A bare string does not count as a collection.

Accepted `measure`s:

| `measure` | What it computes                    | Expected elements of `over`                       | Empty `over` |
| --------- | ----------------------------------- | ------------------------------------------------- | ------------ |
| `count`   | `len(over)`, the number of elements | any                                               | `0`          |
| `sum`     | `sum(over)`                         | numbers (`int`/`float`)                           | fails        |
| `mean`    | `sum(over) / len(over)`             | numbers (`int`/`float`)                           | fails        |
| `min`     | the smallest element                | mutually orderable values (numbers, also strings) | fails        |
| `max`     | the largest element                 | mutually orderable values (numbers, also strings) | fails        |
| `any`     | whether any element is truthy       | any (truthiness is tested)                        | `false`      |
| `all`     | whether every element is truthy     | any (truthiness is tested)                        | fails        |

Unlike normal comparisons, `measure`s only accept `op`s that are binary numeric
operations:

- `eq` (`==`)
- `ne` (`!=`)
- `gt` (`>`)
- `ge` (`>=`)
- `lt` (`<`)
- `le` (`<=`)

```yaml
# A check that uses `count` to check that there are at least 3 accuracy results
assert:
  - over: point.accuracy_results
    measure: count
    op: ge
    value: 3
```

### Gating with `needs`

A check can require other checks first by listing their `id`s in `needs`. Two
conditions describe one instance - one check run:

- **queued** - its `when` passes and the check is queued to run.
- **passed** - a queued check that _ran_ and returned a truthy value (every
  `assert` predicate held).

A check with `needs` (dependent check) produces findings only where every named
check **passed**:

- A needed check passed: the dependent runs.
- A needed check was queued but did not pass: the dependent is skipped and
  de-queued - its findings would possibly be corrupted by a failing
  prerequisite.
- A needed check was never queued (its `when` left that instance out): the
  dependent is skipped and de-queued.
- Any check may be listed in `needs`. If a listed check is at a lower scope,
  **all** of its instances must pass: a curve-level `pareto_clean` with
  `needs: [point_accuracy_pass]` runs only if `point_accuracy_pass` passes for
  every point on that curve.
- Unknown ids and `needs` cycles are refused at load.

Example of `point.yaml` check below, with both `needs` and a measure predicate
(the dependent runs only where `concurrency_in_range` was queued and passed.

```yaml
accuracy_sane:
  description: At least one accuracy result exists at this point.
  rule: "Endpoints Policy Section 4.3, Accuracy Metric."
  severity: fail
  needs: [concurrency_in_range]
  assert:
    - over: point.accuracy_results
      measure: count
      op: ge
      value: 1
```

This check will run per point in the submission: i.e. if an `r1024` point passes
`concurrency_in_range` and `r999999` does not, `accuracy_sane` will still run
for `r1024`, but is skipped for `r999999`.

### References

- Dotted strings (`Reference`s, e.g. `point.accuracy_results`) are resolved
  first to their values from the submission files passed. The first segment of
  the dotted string is its `scope` - e.g. `point.accuracy_results` has a scope
  of `point`. (Section 4's table lists every scope root.) **Every other dotted
  string is refused at load** - a one-edit near-miss reports "looks like a
  misspelled", anything else reports "is no scope root": a dotted _value_ must
  say so with `{"literal": ...}` (e.g. a dataset name like `squad.v2.0`).
  Scalars with no dot at all (`5`, `true`, `"dedicated"`) need no wrapper and
  stay literals.
- If a key must contain a period `.` character, a list can be used:
  `["ruleset", "datasets", "squad.v2.0"]` addresses the `squad.v2.0` dataset
  entry - each list element is one exact key, and the first must be a scope
  root.
- `{"literal": ...}` is the intentional literal: the value inside is used
  verbatim, even when it would otherwise read as a `Reference`. Lists stay
  literal unless they are non-empty all-strings and scope-root-headed (their
  near-miss heads are refused at load too).

All three forms in one stub (`point.yaml`):

```yaml
accuracy_sane:
  description: Some accuracy result exists at this point.
  rule: "Endpoints Policy Section 4.3, Accuracy Metric."
  severity: fail
  assert:
    # a dotted Reference (scope `point`)
    - over: point.accuracy_results
      measure: count
      op: ge
      value: 1

dataset_catalogued:
  description: The dataset's sample count is recorded in the ruleset catalog.
  rule: "Endpoints Policy Section 4.3, Accuracy Metric."
  severity: warn
  assert:
    # the key `squad.v2.0` contains a period, so the list form addresses it
    - left: ["ruleset", "datasets", "squad.v2.0", "num_samples"]
      op: gt
      right: 0

model_naming:
  description: The model name is one of the cohort's supported models.
  rule: "MLPerf Endpoints Rules §8.3, Measurement Point (point.yaml)."
  severity: warn
  assert:
    - left: point.config.model_name
      op: in
      right: ruleset.models
```

### Refusals

Untrusted uploads are refused loud rather than mis-read, never a silent pass. At
check load (each names its cohort file):

- a predicate whose shape or `matches` pattern does not validate (`matches`
  patterns require `{"literal": ...}` and are compile-checked at load)
- an empty cohort directory (rather than a clean pass)

At `Finding.collect` (per submitted run):

- a submission with a shape error
- a mirrored-fact disagreement (a carried fact vs its artifact)
- an unreadable artifact (a corrupt file is never read as an absent fact)
- a `when` predicate that cannot be judged

## 4. Possible keys

Every dotted operand resolves through the scope roots below. The keys below are
the full decoded vocabulary today.

- `submission` (one per upload)
  - `submission.config.<key>` - top-level submission-level settings
    (`config.yaml` at the submission root)
  - `submission.client_config.<key>` - client runtime settings (`config.yaml`)
  - `submission.system_desc.<key>` - system description fields
    (`max_supported_concurrency`, `model_name`, `division`, ...) from
    `system_desc.json`
  - `submission.system_power.<key>` - power declaration (`provisioned_power_kw`,
    ...) from `system_power.json`
  - `submission.results.<key>` - results summary rollup (`results.json`)
- `system` (one per system), `curve` (one per model), `point` (one per
  measurement point) - each exposes its directory's artifact documents the same
  way:
  - `<scope>.config.<key>` - `point.yaml`'s fields (`point.config.concurrency`)
  - `<scope>.client_config.<key>` - `config.yaml`'s fields
  - `<scope>.system_desc.<key>` - `system_desc.json` (fields as above)
  - `<scope>.system_power.<key>` - `system_power.json` (power declaration)
  - `<scope>.result_summary.<key>` - `result_summary.json` (raw metrics)
  - `<scope>.accuracy_results.<key>` - `accuracy_results.json`
  - `<scope>.results.<key>` - `results.json` (results summary rollup)
- `derived.<name>` - named derivations computed from the submission (see
  `DESIGN.md` for each source formula):
  - `derived.offline_ordering` - Section 5.7.2's pair-of-clauses verdict
  - `derived.point_cap` - Section 5.6's `<=32`
  - `derived.point_count_min` - Section 5.3's minimum count
  - `derived.accuracy_run_count_requirement` - Section 4.3's `N`
  - `derived.out_of_range_points` - Section 5.4's flagged points tuple
- `ruleset` - the ruleset directory passed with `--ruleset`:
  - `ruleset.models.<name>.<key>` - model definitions (`models.yaml`)
  - `ruleset.datasets.<name>.<key>` - dataset definitions (`datasets.yaml`)
- `regions` - computed per run from Section 5.5's reference algorithm:
  - `regions.coverage` - the `concurrency` interval a valid region covers
    (including the 10% margin)
  - `regions.by_kind.<kind>.<lo|hi>` - one of the five computed regions
    (`low_latency`, `low_concurrency`, `med_concurrency`, `high_concurrency`,
    `margin`) as an interval's bounds

The `<key>` slots (and multi-segment paths like `steady_state.status`) are open

- They address whatever keys the document carries:
  - `point.config.<key>` / `<scope>.system_desc.<key>` - field names follow the
    Endpoints Rules' own field tables (Section 8.3 for `point.yaml`, Section 8.2
    for `system_desc.json`): `concurrency`, `offline`, `dataset_name`,
    `model_name`, `steady_state.status`, `nodes_used`, `dp_shortfall`, and so on
    are addressed as `point.config.concurrency`, `point.config.offline`, ...
    (`point.yaml`'s mapping is the `config` key).
  - `<scope>.results.<key>` / `<scope>.result_summary.<key>` - keys follow their
    producer (the benchmark's report writer): `result_summary.json`'s raw
    metrics appear as `point.result_summary.<metric>`.
  - `ruleset.models.<name>.<key>` / `ruleset.datasets.<name>.<key>` - whatever
    the ruleset's `models.yaml` / `datasets.yaml` entry contains (`vocab_size`,
    `is_drafter`, ...).

Everything named here and above is the full decoded vocabulary; an unknown
`<key>` resolves as **absent** (what `exists`/`absent`/`not_null` are for), not
an error. Every dotted string must name a scope root or use `{"literal": ...}`
(`matches` patterns use that form always, dots or not).

- a stray dotted value is refused at load (near-miss heads likewise).

## Implementation details

The loader, tree walk, row operators, region derivation and the check runner are
documented in [`DESIGN.md`](DESIGN.md) - "what happens under the hood", kept
strictly separate from this how-to-use page.
