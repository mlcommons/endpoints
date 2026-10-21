# Validation submodule - policy and rule checks for a clean submission

## 1. What the validation submodule is for

`inference_endpoint.validation` checks a completed MLPerf Endpoints submission
against the Endpoints Policy compliance rules and reports every violation as a
finding. It exists to answer one question per submission: _would this upload
pass the review checklist?_ Each check reads the submission's directory layout
(per-point `r<N>/` folders holding `point.yaml`, `system_desc.json`,
`result_summary.json`, `accuracy_results.json`, and the like) plus the pareto's
own declared bounds (`max_supported_concurrency`, the Ultra Low Concurrency
point) and evaluates named rule rows against that evidence. A failure produces a
`Finding` whose `why` spells out the values that made the rule fail.

Policy and rule references live in
[`mlcommons/endpoints_policies`](https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/endpoints_rules.md)
(commit-pinned). Check descriptions cite the exact section they enforce.

## 2. How cohort submission checks work

The user-facing artifact is a directory of YAML files, which store submission
checks for a given cohort (used as the directory's name), i.e.:

```
my_cohort/
├── valid_refs.yaml
└── point.yaml
```

Filenames correspond to the scope of the checks:

- `system.yaml`
- `submission.yaml`
- `pareto.yaml`
- `point.yaml`

Scope files are optional - a missing file indicates there are no checks to run
for that scope. `valid_refs.yaml` is required: it is the cohort's ref table,
which every row's names are checked against (see "The cohort's
`valid_refs.yaml`" below). Only `valid_refs.yaml` and the four scope files
belong here; anything else in the directory is refused at load.

Each scope file contains a list of checks, with the check's ID as its key.
Example:

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

The submission argument is the upload root (`results/<system>/<model>/r<N>/`
below it). Each run directory `r<N>/` carries its own `system_desc.json`
(Section 8.2.1), its `point.yaml` (`concurrency`, `offline` declaration) and its
`result_summary.json`, whose Section 4.1 aggregate `system_tps` is the point's
throughput. Pass `--ruleset config/rulesets/<name>/` to make the `ruleset` fold
(its `models.yaml`/`datasets.yaml`) available to rows; the `regions` fold is
derived per pareto (Section 5.5).

The run prints one line per failing finding -- the line is one evaluated
`assert` predicate, labeled by the predicate itself (`1 le 128`) -- marked
`FAIL` or `WARN`, with the check's rules citation at the end; when the check
declares a `fix`, an indented `fix:` line follows the failing line. Also a
`<id> ran at N instance(s)` line per check and a closing
`N finding(s), F failing (B blocking)` summary: findings count holds and
failures alike, failing means a predicate did not hold, and blocking means a
failing `fail`. `--audit` prints the holds as well, marked `OK` (row label, no
rule/fix), so the session doubles as the audit trail. A check that ran at no
instance at all prints `NEVER-RAN <id>: ...` on stderr: a check its `when`
leaves out everywhere stays visible instead of passing by omission.

Exit codes, in plain words:

- `0` - every applicable check held, or the checks that did not hold failed only
  at `warn` severity (a warning is reported and passes through).
- `1` - at least one failing `fail` finding (`B blocking` nonzero): the
  submission is blocked.
- `2` - the cohort or the submission was refused before any check ran (a
  duplicate check id, a check naming a fact the cohort's `valid_refs.yaml` does
  not declare, a run directory without its `system_tps`, ...).

Severity shapes the submission's fate, not premise vouching: a failing `warn`
still withholds its `needs` dependents.

## 3. How the files are written

### The cohort's `valid_refs.yaml`

The valid refs a cohort's rows may name are declared per cohort, in
`valid_refs.yaml` beside the scope files (required - a directory without it is
refused at cohort load). The names and their policy notes live in one YAML per
cohort, so a cohort or policy version updates its fields and their absent-state
semantics by editing that file, and a typo never reaches a row: a name the table
does not declare dies at cohort load ("Possible keys" and "Refusals" spell out
the refusals).

The declaration nests exactly like a row's ref path (scope -> document ->
fields):

```yaml
point:
  config: # = the point.yaml document (wire: point.config.X)
    concurrency: { type: int, required: true }
    offline: { type: str, allowlist: [dedicated, elected, none], default: null }
    warmup: { duration_s: int }
    steady_state: { window: dict }
  system_desc:
    max_supported_concurrency: { type: int, required: true }
  result_summary: { system_tps: { type: float, required: true } }
  accuracy_results: dict
```

Each entry is one of four forms:

- a typed leaf: `field: bool|int|float|str|list` - one leaf fact, nothing deeper
  addressable below it.
- `field: dict` - an open subtree: any deeper name below it resolves.
- the mapping form
  `field: { type: ..., required: ..., default: ..., allowlist: [...] }` - a leaf
  with policy notes. Its reserved keys are `type`, `required`, `default` and
  `allowlist` (`type` names one of `bool`/`dict`/ `float`/`int`/`list`/`str`);
  `required:` and `default:` set the absent states under "Absent facts" below (a
  mapping with no `default`, e.g. `{ type: str }`, declares no absent-state
  value), and `allowlist:` fixes the values a present fact may carry (leaf types
  only: a `dict` subtree has nothing to enumerate). Three checks at cohort load
  guard the notes: an `allowlist` holds a non-empty list of scalar values and
  sits on a leaf type, and a non-`null` `default` must be one of its
  `allowlist`'s values (`null` exempt - it is the absent-state's own value).
- a mapping without `type` - a nested block of sub-entries (`config:`, `warmup:`
  above), as deep as the names go.

Because those four keys are reserved in the mapping form, a rule field literally
named `type` keeps its value a nested block by carrying any sibling key
(`field: { type: { ... } }` is a block).

Read against that table, `point.config.warmup.duration_s` is addressable and
nothing below it is (`is one leaf fact, nothing deeper`), anything below
`point.config.steady_state.window` resolves, and `accuracy_results` leaves the
whole `accuracy_results.json` document open (a whole artifact is addressable as
is, when declared `dict`). A row naming anything else is refused at cohort load
with `names no fact` (a one-edit near-miss adds `did you mean 'offline'?`). The
policy spells value domains out per field -- `offline` is Section 8.3's "One of:
dedicated, elected, absent or `none`", and `region` (five values,
`submitters_choice` among them) or `division` (`Standardized`/`Serviced`/`RDI`)
declare likewise where a cohort holds rows to them.

Two limits on what the file may declare:

- Only node scopes declare under (`submission`/`system`/`pareto`/`point`). A
  `derived:`, `regions:` or `ruleset:` top-level key is refused
  (`is not a node scope`): those roots are checker-owned (the derivation
  registry, the region kinds, the catalog's own data). Rows still name those
  folds as "Possible keys" describes; their names are the checker's, not the
  table's.
- The file parses with the same strict loader as the scope files: a duplicate
  key, a `<<` merge key or a YAML alias is a parse error, and an empty or
  non-mapping document is refused as
  `is not a ref table: expected scope names to declare under`.

### Check blocks

One YAML mapping per check block under its id in the matching scope file. The
block's fields:

| Field               | Required   | What it is                                                                                                                |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| _(the mapping key)_ | yes        | The check's `id`; the scope file it lives in (`point.yaml`, ...) is its `scope      `                                     |
| `description`       | yes        | The human sentence describing this check                                                                                  |
| `rule`              | yes        | Where in the Endpoints Rules this comes from (rendered at the end of each failing line)                                   |
| `severity`          | yes        | `fail` (a blocking finding) or `warn` (reported and passes through)                                                       |
| `fix`               | no         | How to resolve a failing row (rendered after each failing line)                                                           |
| `needs`             | no         | Upstream check ids that must pass before this check speaks                                                                |
| `when`              | predicates | Gate predicates: the check runs only where all hold; this list may be empty or absent when `assert` carries at least one  |
| `assert`            | predicates | Condition predicates; each is one finding per instance; this list may be empty or absent when `when` carries at least one |

`when` and `assert` take lists of predicates. A predicate is either a
**comparison** or a **measure** (an aggregate over a collection); the
subsections below spell out each form. (A check runs once per **instance** of
its scope: once per submission, system, pareto, or measurement point.) A full
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

1. All binary operators take a single value under `right`. The following binary
   ops are available:

   - `eq`: Equality check. By default, floating point values will be computed
     with strict equality. Accepts the following optional fields:
     - `abs_tol` - Absolute tolerance to be forwarded to `math.isclose`
     - `rel_tol` - Relative tolerance to be forwarded to `math.isclose`
   - `ne`: Negation of `eq` - see notes above for optional fields.
   - `gt`, `ge`, `lt`, and `le`: The `>`, `>=`, `<`, and `<=` operators
     respectively.
   - `in`: A `contains` / membership operator. True if `left` is contained in
     `right`. `right` is a list-like container or an `Interval` (e.g. a fold
     like `regions.coverage`). If `right` is a mapping, `in` uses its _keys_ -
     unless it is exactly a `{"lo": ..., "hi": ...}` range literal, an inclusive
     interval (`lo <= left <= hi`; both bounds required, `null` bounds refused).
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

2. Unary operators do not accept a `right` parameter. They key on whether the
   `left` reference resolves to a fact in the parsed submission files: a dotted
   reference like `point.config.steady_state.status` addresses a key in the
   loaded artifact documents, and a whole-document reference like
   `point.accuracy_results` addresses its artifact file. (Only the refs section
   4 lists can be referenced this way - there is no form for arbitrary file
   names. `left` must name such a `Ref`: asking whether a literal constant is
   present is refused at load.)

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
    op: in
    right: { lo: 1, hi: 32 }

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

### Absent facts

A row's operands know three kinds of absent fact -- the distinctions that keep a
`when` gate from switching a whole check off by omission (the cohort's
`valid_refs.yaml` refuses a row naming a fact it does not declare, so a typo
cannot land here):

- A **required** fact absent -- one the cohort's `valid_refs.yaml` declares
  `required: true`: `config.concurrency`,
  `system_desc.max_supported_concurrency`, `result_summary.system_tps`. A
  conforming upload always carries it, so seeing it absent is a schema breach
  and the row breaks loudly (a `when` row over one refuses the run before any
  check reports; an `assert` row reports
  `row failed to evaluate: required fact ... is absent`). For those three
  parser-enforced facts a walked upload is refused earlier still, at submission
  load.
- A fact whose declaration spells out its absent-state value -- a `default:` in
  the cohort's `valid_refs.yaml` (the worked example's `offline` leaf: Section
  8.3's "absent or `none`") -- lets a comparison judge as that value when the
  upload omits it. On a point without a declaration,
  `point.config.offline eq dedicated` is false and
  `point.config.offline ne dedicated` is true, exactly as on a point declaring
  `offline: none`. A leaf declared without `default` (e.g. `{ type: str }`)
  declares no absent-state value and falls to the case below.
- Every other omitted **optional** fact is its own not-declared state: the row
  does not hold, and a `when` row over it gates its check off at that instance
  (the check makes no claim there), instead of failing over evidence the rules
  never asked for.

Presence probes (`exists`/`absent`/`not_null`) always read the key's own raw
state, never the declared default: an omitted `config.offline` is absent
(`exists` false, `not_null` false), while `offline: none` is present, stored as
the string `none`. The two "no declaration" spellings (the omitted key under its
`default: null` declaration, and the stored `none`) read alike for `eq`/`ne`
against `dedicated` -- both `ne dedicated` hold, neither `eq dedicated` does --
and apart for `eq null` (true for the omitted key, false for the stored `none`).

Rows built in code against a hand-built `Cohort((check, ...))` carry no ref
table (`Cohort.refs` is `None`) and no annotations, so every absent fact of
theirs reads as the third case; `required`/`default` are filled only when
`Cohort.from_directory` loads the cohort's `valid_refs.yaml`.

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
- A needed check was silenced in turn (its own `needs` refused to vouch): the
  dependent is skipped and de-queued too - a failure one level down does not
  hide behind the check it silenced.
- Any check may be listed in `needs`. If a listed check is at a finer scope,
  **all** of its instances must pass: a pareto-level `pareto_clean` with
  `needs: [point_accuracy_pass]` runs only if `point_accuracy_pass` passes for
  every point in that pareto.
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

- Dotted strings (`Ref`s, e.g. `point.accuracy_results`) are resolved first to
  their values from the submission files passed. The first segment of the dotted
  string is its `scope` - e.g. `point.accuracy_results` has a scope of `point`.
  (Section 4's table lists every scope root.) **Every other dotted string is
  refused at load** - a one-edit near-miss reports "looks like a misspelled",
  anything else reports "is no scope root": a dotted _value_ must say so with
  `{"literal": ...}` (e.g. a dataset name like `squad.v2.0`). Scalars with no
  dot at all (`5`, `true`, `"dedicated"`) need no wrapper and stay literals.
- If a key must contain a period `.` character, a list can be used:
  `["ruleset", "datasets", "squad.v2.0"]` addresses the `squad.v2.0` dataset
  entry - each list element is one exact key, and the first must be a scope
  root.
- `{"literal": ...}` is the intentional literal: the value inside is used
  verbatim, even when it would otherwise read as a `Ref`. Lists stay literal
  unless they are non-empty all-strings and scope-root-headed (their near-miss
  heads are refused at load too).

All three forms in one stub (`point.yaml`):

```yaml
accuracy_sane:
  description: Some accuracy result exists at this point.
  rule: "Endpoints Policy Section 4.3, Accuracy Metric."
  severity: fail
  assert:
    # a dotted Ref (scope `point`)
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
- a row naming no fact: an unknown, undeclared or misspelled Ref (its cohort's
  `valid_refs.yaml` declares the valid names; `names no fact`; a one-edit
  near-miss adds "did you mean 'offline'?"), a `derived.<name>` outside the
  known derivation names, a `regions` path beyond
  `regions.coverage`/`regions.by_kind.<kind>`, a `derived`/`regions` fold named
  outside `pareto.yaml`/`point.yaml`, or a row naming a finer scope's facts than
  its own file holds
- an empty cohort directory (rather than a clean pass)

At submission load (`Submission.from_directory`, which parses every `Point` and
`Pareto` -- the `2` exit):

- a run directory whose `point.yaml`, `system_desc.json` or
  `result_summary.json` lacks or mistypes a parser-enforced fact (`concurrency`,
  `max_supported_concurrency`, `system_tps`), declares an `offline` value
  outside `dedicated`/`elected`/`none`, or fails to parse at all
- a pareto shape a derivation cannot judge, each naming its Endpoints Rules §9.1
  row: multiple `offline` declarations, an `elected` point not at C_max, a
  dedicated Offline run without exactly one C_max point, points disagreeing on
  `max_supported_concurrency`/`c_min_override`, or a declared `c_min_override`
  that Section 5.4 rejects (`null`, a bool, a non-integer, outside `[1, 32]`, or
  not equal to a submitted point's concurrency -- only a missing
  `c_min_override` key derives `C_min` from the points)

At `Finding.collect` (per submitted run):

- an unreadable artifact anywhere in the tree -- the same rule for the documents
  only rows read: a corrupt file is never read as an absent fact
- a present value its `valid_refs.yaml` declaration will not carry
  (`does not match its declared type`, `is not one of`, `is no block`)
- a `when` predicate that cannot be judged (an unjudgeable gate is refused, not
  read as false)
- a run directory with no carried measurement point, or one shared between two
  points (a hand-built `Submission`'s concern; `Submission.from_directory`
  cannot produce one)

## 4. Possible keys

Every dotted operand resolves through the scope roots below.

- `submission` (one per upload), `system` (one per `results/<system>/`),
  `pareto` (one per `results/<system>/<model>/`), `point` (one per `r<N>`
  measurement point) - each scope exposes its own directory's artifact
  documents, keyed by the artifact's field name:
  - `<scope>.config.<key>` - `point.yaml`'s fields (`point.config.concurrency`)
  - `<scope>.client_config.<key>` - `config.yaml`'s fields (client runtime
    settings)
  - `<scope>.system_desc.<key>` - `system_desc.json`'s system description fields
    (`point.system_desc.max_supported_concurrency`, `model_name`, `division`,
    ...; Section 8.2.1 puts a `system_desc.json` in every `r<N>/` run directory)
  - `<scope>.system_power.<key>` - `system_power.json`'s power declaration
    (`provisioned_power_kw`, ...)
  - `<scope>.result_summary.<key>` - `result_summary.json`'s raw metrics
    (`point.result_summary.system_tps`, ...)
  - `<scope>.accuracy_results.<key>` - `accuracy_results.json`
  - `<scope>.results.<key>` - `results.json` (the results summary rollup)
- `derived.<name>` - one pareto's named derivations (see `DESIGN.md` for each
  source formula). The fold is per-pareto, so only `pareto.yaml`/`point.yaml`
  rows may name it:
  - `derived.offline_ordering` - Section 5.7.2's pair-of-clauses verdict
  - `derived.point_cap` - Section 5.6's `<=32`
  - `derived.point_count_min` - Section 5.3's minimum count
  - `derived.accuracy_run_count_requirement` - Section 4.3's `N`
  - `derived.out_of_range_points` - Section 5.4's flagged points tuple
- `ruleset` - the ruleset directory passed with `--ruleset`:
  - `ruleset.models.<name>.<key>` - model definitions (`models.yaml`)
  - `ruleset.datasets.<name>.<key>` - dataset definitions (`datasets.yaml`)
- `regions` - one pareto's Section 5.5 reference-algorithm regions, likewise
  only from `pareto.yaml`/`point.yaml` rows:
  - `regions.coverage` - the `concurrency` interval any valid region covers
    (including the 10% margin), for `op: in`
  - `regions.by_kind.<kind>` - one named region as its interval, for `op: in`;
    `<kind>` is the reference algorithm's own key -- `low_latency` (the Ultra
    Low Concurrency region), `low_concurrency`, `med_concurrency`,
    `high_concurrency`, `margin`

Which names sit under the `<key>` slots is declared in the cohort's
`valid_refs.yaml` (the section above), scoped exactly like this list: scope ->
document -> fields (`point.config.concurrency` is the `config` document's
`concurrency` under `point`). What a cohort declares follows two rules:

- The name tables draw on the Endpoints Rules' own field tables (Section 8.3 for
  `point.yaml`, Section 8.2 for `system_desc.json`, Section 4.1/8.1 for
  `result_summary.json`) at policy `4ba8224dc746` -- `concurrency`, `offline`,
  `dataset_name`, `model_name`, `warmup.duration_s`, `steady_state.status`,
  `nodes_used`, `system_tps`, `qps`, and the rest of each table -- while the
  blocks the rules describe without listing their keys are declared `dict`
  (`config.runtime_settings`, `config.dataset`, `config.steady_state.window`/
  `state`/`total`, `result_summary`'s `percentiles`/`isl`/`osl`).
  `client_config`, `system_power`, `accuracy_results` and `results` are declared
  open the same way (the benchmark's report writer owns their spelling): under a
  `dict`, an unknown `<key>` resolves as **absent** (what
  `exists`/`absent`/`not_null` are for), not an error. `ruleset` keys are open
  data too -- whatever the `models.yaml`/`datasets.yaml` entry contains
  (`vocab_size`, `is_drafter`, ...).
- What a present value must carry is declared the same way and checked at
  collect time (`Finding.collect` refuses before any finding): match its leaf's
  type (`does not match its declared type`), sit in its `allowlist` where one is
  declared (`is not one of`), and a declared block must carry a block
  (`is no block`). The fence-posts: the parser-minimum trio (`concurrency`,
  `system_tps`, `max_supported_concurrency`) is code-side and refused at parse
  time by `Point.from_node` whatever the table says; a `dict` subtree is open
  and its values stay unchecked; an absent field carries no value to check (its
  absence reads per "Absent facts"); and a hand-built `Cohort((check, ...))`
  with no table checks nothing.

Rows name only declared facts: a row naming an undeclared or misspelled key is
refused at cohort load (`names no fact`; a one-edit near-miss adds "did you mean
'offline'?"), and so is any path below a leaf fact
(`system_desc.cooling.detail`: `is one leaf fact, nothing deeper`). An upload
may still carry keys its cohort does not declare -- they are simply not nameable
by a row. The strict names are what gives "Absent facts" above its meaning: a
misspelled key dies at cohort load instead of decoding to an absent fact and
switching a `when`-gated check off by omission. The price is upfront -- a rule
over a brand-new field first needs that field declared in the cohort's
`valid_refs.yaml`. Every dotted string must name a scope root or use
`{"literal": ...}` (`matches` patterns use that form always, dots or not); a
stray dotted value is refused at load (near-miss heads likewise).

## Implementation details

The loader, tree walk, row operators, region derivation and the check runner are
documented in [`DESIGN.md`](DESIGN.md) - "what happens under the hood", kept
strictly separate from this how-to-use page.
