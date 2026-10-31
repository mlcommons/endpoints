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
`valid_refs.yaml`" below). Only `valid_refs.yaml`, the four scope files, the
optional `cohort.yaml` manifest and the optional `ruleset/` catalog belong here;
anything else in the directory is refused at load.

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
throughput.

A submission may carry several paretos - one `<system>/<model>` pair each; each
check runs once per instance of its scope ("Check blocks" below), so a
`pareto.yaml` check judges every pareto in turn, and a `point.yaml` check every
`r<N>/`.

The cohort's optional `cohort.yaml` manifest names the packaged ruleset its
`ruleset.*` refs read (e.g. `mlperf-inference-v6.1`); without it, a `ruleset.*`
row is refused (`the tree was built without ruleset`).

The run prints one line per failing finding:

- The evaluated `assert` predicate (`1 le 128`), marked `FAIL` or `WARN`
- The check's rules citation (i.e. section in the Endpoints Policies document)
- When the check declares a `fix`, an indented `fix:` line follows the failing
  line.
- Additionally, each check prints `<id> ran at N instance(s)`

A summary (`N finding(s), F failing (B blocking)`) is shown at the end:

- `Findings` counts every evaluated `assert` predicate (holds included)
- `Failing` means a predicate did not hold (a row that failed to evaluate counts
  too)
  - `Blocking` is the subset whose check declares severity `fail`

Specifying `--audit` prints the holds as well, marked `OK` (row label, no
rule/fix), so the session doubles as the audit trail. Checks that did not run
print `NEVER-RAN <id>: ...` on stderr with their reason: the `when` conditions
did not hold at that instance, or a check in their `needs` chain did not vouch.

When a declared `int`/`float` value arrives as a numeric string, the run reads
the number it spells and prints
`coerced: <dir>: <path>: '<text>' read as <number>` on stderr: a shape note, not
a finding (see "The cohort's `valid_refs.yaml`").

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
- `field: dict` - an open subtree: any deeper name below it resolves; the leaf's
  own value must be a block, while its values stay unchecked.
- the mapping form
  `field: { type: ..., required: ..., default: ..., allowlist: [...] }`
  - `type`: one of `bool`/`dict`/`float`/`int`/`list`/`str`
  - `required`: denotes whether an entry must be present
  - `default:` sets the fact's absent-state value under "Absent facts" below (a
    mapping with no `default`, e.g. `{ type: str }`, declares no absent-state
    value)
  - `allowlist:` fixes the values a present fact may carry (leaf types only: a
    `dict` subtree has nothing to enumerate)
  - cohort load guards the notes: an `allowlist` holds a non-empty list of
    scalar values and sits on a leaf type, a non-`null` `default` must match its
    declared type and be one of its `allowlist`'s values where one is declared
    (`null` exempt - it is the absent-state's own value), every `allowlist`
    value must match the declared type, and `required: true` cannot carry a
    `default` (a default answers the fact's absence)
- a mapping without `type` - a nested block of sub-entries (`config:`, `warmup:`
  above), as deep as the names go.

A mapping matches the schema of a declared leaf only when its `type` value is a
type name (`int`, `bool`, `float`, `str`, `list`, `dict`) and every key is one
of that leaf's annotation keys (`type`, `required`, `default`, `allowlist`);
otherwise it is a nested block. So `field: { type: int }` is a leaf, while a
rule field literally named `type` keeps its value a block (its `type` value
there is a mapping, not a type name: `field: { type: { ... } }`).

Read against that table:

- `point.config.warmup.duration_s` is addressable; nothing below it is
  (`is one leaf fact, nothing deeper`)
- anything below `point.config.steady_state.window` resolves (declared `dict`)
- `accuracy_results` leaves the whole `accuracy_results.json` document open - a
  whole artifact is addressable as is, when declared `dict`

A row naming anything else is refused at cohort load with `names no fact`; basic
typo detection gives more meaningful error messages - `ofline` will produce
`did you mean 'offline'?`.

The allowed values for each field are explicitly defined - by the policy, or by
an allowlist in a cohort's `valid_refs.yaml`:

- `offline` is Section 8.3's "One of: dedicated, elected, absent or `none`"
- `region` (six values: `low_latency`, `low_concurrency`, `med_concurrency`,
  `high_concurrency`, `margin`, `submitters_choice`)
- `division` (`Standardized`/`Serviced`/`RDI`, the policy's three divisions)

A type name says what a value must carry, and `int`/`float` read a numeric
string: a leaf declared `int` accepts `512` and the string `"512"` (the integral
spelling `"512.0"` too), a `float` accepts `12.61` and `"12.61"`. Only exact
spellings decode - `"512 kw"`, `"1,000"`, `"inf"` and `"nan"` still refuse - and
only on numeric declarations: `"512"` on a `str` leaf stays a string. Each
decoding prints a `coerced:` line, so the leniency is visible, never silent. The
parser-enforced facts (`concurrency`, `max_supported_concurrency`, `system_tps`)
read the same way: a stringified number is not a mistyped one.

Two limits on what the file may declare:

- Only node scopes declare under (`submission`/`system`/`pareto`/`point`). A
  `derived:`, `regions:` or `ruleset:` top-level key is refused
  (`is not a node scope`): those namespaces are checker-owned (the derivation
  registry, the region kinds, the catalog's own data). Rows still name those
  namespaces as "Possible keys" describes; their names are the checker's, not
  the table's.
- The file parses with the same strict loader as the scope files: a duplicate
  key, a `<<` merge key or a YAML alias is a parse error, and an empty or
  non-mapping document is refused as
  `is not a ref table: expected scope names to declare under`.

### Check blocks

One YAML mapping per check block under its id in the matching scope file. The
block's fields:

| Field               | Required   | What it is                                                                                                                |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| _(the mapping key)_ | yes        | The check's `id`; the scope file it lives in (`point.yaml`, ...) is its `scope`                                           |
| `description`       | yes        | The human sentence describing this check                                                                                  |
| `rule`              | yes        | Where in the Endpoints Rules this comes from (rendered at the end of each failing line)                                   |
| `severity`          | yes        | `fail` (a blocking finding) or `warn` (reported and passes through)                                                       |
| `fix`               | no         | How to resolve a failing row (rendered after each failing line)                                                           |
| `needs`             | no         | Upstream check ids that must pass before this check speaks                                                                |
| `when`              | predicates | Gate predicates: the check runs only where all hold; this list may be empty or absent when `assert` carries at least one  |
| `assert`            | predicates | Condition predicates; each is one finding per instance; this list may be empty or absent when `when` carries at least one |

- `when` and `assert` each take a list of predicates; a predicate is either a
  **comparison** or an **aggregate** (a named computation over a collection) -
  the subsections below spell out each form
- a check runs once per **instance** of its scope: once per submission, system,
  pareto, or measurement point

A minimal block appears in "How the files are written" above. An additional
example is given below, showing a real rule from Section 5.7.2's
"pair-of-clauses" verdict.

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
     - `abs_tol` - Absolute tolerance (`eq`/`ne` only): number pairs compare
       under tolerance (integer pairs in integer arithmetic, float pairs via
       `math.isclose`); any other pair compares with `==`
     - `rel_tol` - Relative tolerance, same rules as `abs_tol`
   - `ne`: Negation of `eq` - see notes above for optional fields.
   - `gt`, `ge`, `lt`, and `le`: The `>`, `>=`, `<`, and `<=` operators
     respectively.
   - `in`: A `contains` / membership operator. True if `left` is contained in
     `right`. `right` is a list-like container or an `Interval` (e.g.
     `regions.coverage`). If `right` is a mapping, `in` uses its _keys_ - unless
     it is exactly a `{"lo": ..., "hi": ...}` range literal, an inclusive
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

It is possible a row's `Ref` operands are absent after loading. Depending on the
operand's specification, a `when` gate can switch a whole check off by omission
(the cohort's `valid_refs.yaml` refuses a row naming a `Ref` it does not
declare, so a typo cannot land here).

- If a **required** fact is absent, an error is raised.
  - `Ref`s in `valid_refs.yaml` that specify `required: true` must be present
    after loading
  - Examples include `config.concurrency`,
    `system_desc.max_supported_concurrency`, and `result_summary.system_tps`
  - If a `when` statement is processed that sees a missing "required" `Ref`, the
    run is aborted before any verdicts are reported, with
    `row failed to evaluate: required fact ... is absent`; an `assert` row over
    one fails with the same message.
  - Some required `Ref`s are enforced at submission upload, but no assumptions
    about this are made to support running the submission checks on local,
    unsubmitted files for testing purposes.
- A `Ref` in a cohort's `valid_refs.yaml` can specify a `default` to be used if
  a value is not detected at load time.
- If a `Ref` is not `required` and no `default` is specified, it is interpreted
  as a special "missing" state. If a `when` condition requires a `Ref` that is
  missing, then that `when` will evaluate to `false` and the check will be
  skipped.

Importantly, specifying `required: true` while also specifying a `default`
(including `default: null`) is invalid, as their behaviors contradict. An error
will be raised at load time. The legal shapes for a `Ref`'s definition are:

- `{ type: int, default: 1 }`
- `{ type: int, required: true }`

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

### Aggregates

An aggregate row is a comparison over a computed value: it names the computation
under `aggregate` and the collection it reads under `over` (functionally `left`
in comparisons), then applies the row's `op` to the computed value:

- `over` is expected to be a list-like collection - a mapping is read as its
  _values_, and a bare string does not count as a collection
- an empty collection passes only where the table below says so (`count` -> 0,
  `any` -> false); every other aggregate fails rather than pass vacuously

Accepted `aggregate` values:

| `aggregate` | What it computes                    | Expected elements of `over`                       | Empty `over` |
| ----------- | ----------------------------------- | ------------------------------------------------- | ------------ |
| `count`     | `len(over)`, the number of elements | any                                               | `0`          |
| `sum`       | `sum(over)`                         | numbers (`int`/`float`)                           | fails        |
| `mean`      | `sum(over) / len(over)`             | numbers (`int`/`float`)                           | fails        |
| `min`       | the smallest element                | mutually orderable values (numbers, also strings) | fails        |
| `max`       | the largest element                 | mutually orderable values (numbers, also strings) | fails        |
| `any`       | whether any element is truthy       | any (truthiness is tested)                        | `false`      |
| `all`       | whether every element is truthy     | any (truthiness is tested)                        | fails        |

Unlike normal comparisons, aggregate rows only accept `op`s that are binary
numeric operations:

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
    aggregate: count
    op: ge
    value: 3
```

An `aggregate` computes its value with the `aggregate` key (a computation from
the table above) and compares the result with `value` using one of the `op`s
listed above. `Op`s like `in`, `matches`, `exists`/`absent`/`not_null` have
semantics for collections, strings, and presence checks, which `aggregate`s do
not support.

`value` uses the same rules as comparison operands:

- a literal or a dotted `Ref`
- `abs_tol`/`rel_tol` are supported for `eq`/`ne`

This has a few limitations, including:

- A membership test or interval check requires multiple rows: `count in [4, 5]`
  would require both `ge 4` and `le 5`.
- The collection that `over` runs over must be an artifact of the row's own node
  or an ancestor. For instance, an `over` in a `point.yaml` cannot run over a
  list of values from across the entire pareto curve (i.e. a 'concurrency' list
  of all the concurrency values in the pareto).

### Gating with `needs`

A check can require other checks first by listing their `id`s in `needs`.
Results for checks are separated across scopes - for example:

- A `point.yaml` check can specify another `point.yaml` check as a `need`ed
  dependency.
- This check will have multiple **instances**, one per point on the curve.
- If the dependency fails in one point on the curve, it will only affect the
  check in that point, not the other instances for other points on the curve.

Two conditions describe a check's instance:

- **queued** - its `when` passes and the check is queued to run.
- **passed** - a queued check that _ran_ and returned a truthy value (every
  `assert` predicate held).

A check with `needs` (dependent check) produces findings only where every named
check **passed**:

- A needed check passed: the dependent runs.
- A needed check was queued but did not pass: the dependent is skipped and
  de-queued, since its findings would possibly be corrupted by a failing
  prerequisite.
  - If a `fail` check is skipped, this will de-queue all of its dependents down
    its chain(s); every `fail` check in that cascade is reported as
    `not judged: \`needs\` ... did not pass` at its own instance. A chain no
    upstream ever claimed at stays a no-finding skip.
- A needed check was never queued (its `when` left that instance out): the
  dependent is skipped and de-queued - no claim, so no finding.
- A needed check was de-queued (due to its own `needs` being de-queued or
  failing): the dependent is skipped and de-queued too (this will recurse down
  the dependency chain).
- Any check may be listed in `needs`. If a listed check is at a finer scope,
  **all** of its instances must pass: a pareto-level `pareto_clean` with
  `needs: [point_accuracy_pass]` runs only if `point_accuracy_pass` passes at
  every point in that pareto; a point its `when` skipped makes no claim either
  way.
- Unknown ids and `needs` cycles are refused at load.

A dependency illustration - not the accuracy-coverage rule: the dependent runs
only where `concurrency_in_range` was queued and passed.

```yaml
accuracy_sane:
  description: At least one accuracy result exists at this point.
  rule: "Endpoints Policy Section 4.3, Accuracy Metric."
  severity: fail
  needs: [concurrency_in_range]
  assert:
    - over: point.accuracy_results
      aggregate: count
      op: ge
      value: 1
```

This check will run per point in the submission: i.e. if an `r1024` point passes
`concurrency_in_range` and `r999999` does not, `accuracy_sane` will still run
for `r1024`, but is skipped for `r999999`.

The check above judges the document one point carries - it cannot enforce
coverage. Section 4.3/5.3 require exactly one accuracy validation run at the
designated point in each mandatory concurrency region (plus the non-agentic
Offline run), and none at submitter's-choice points; that is a pareto-level
count over points, from the cohort doc's "Not yet expressible" list.

Currently the shipped cohort's `point_accuracy_present` warn is a placeholder
for it: it warns wherever a point carries no `accuracy_results.json`, optional
points included.

### References

- Dotted strings (`Ref`s, e.g. `point.accuracy_results`) are resolved first to
  their values from the submission files passed. The first segment of the dotted
  string is its namespace - e.g. `point.accuracy_results`'s namespace is
  `point`. (Section 4's table lists every ref namespace.) **Every other dotted
  string is refused at load** - a one-edit near-miss reports "looks like a
  misspelled", anything else reports "is no ref namespace": a dotted _value_
  must say so with `{"literal": ...}` (e.g. a dataset name like `squad.v2.0`).
  Scalars with no dot at all (`5`, `true`, `"dedicated"`) need no wrapper and
  stay literals.
- If a key must contain a period `.` character, a list can be used:
  `["ruleset", "datasets", "squad.v2.0"]` addresses the `squad.v2.0` dataset
  entry - each list element is one exact key, and the first must be a ref
  namespace.
- `{"literal": ...}` is the intentional literal: the value inside is used
  verbatim, even when it would otherwise read as a `Ref`. Lists stay literal
  unless they are non-empty all-strings and namespace-headed (their near-miss
  heads are refused at load too).

- A component that spells a number reads a key spelling the same number:
  `point.result_summary.tpot.percentiles.90` finds a block keyed `"90.0"` (the
  canonicalisation the submission checker applies to percentile keys). An exact
  key always wins over an equivalent spelling.

All three forms in one stub (`point.yaml`):

```yaml
accuracy_sane:
  description: Some accuracy result exists at this point.
  rule: "Endpoints Policy Section 4.3, Accuracy Metric."
  severity: fail
  assert:
    # a dotted Ref (scope `point`)
    - over: point.accuracy_results
      aggregate: count
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
  `regions.coverage`/`regions.by_kind.<kind>`, a `derived`/`regions` ref named
  outside `pareto.yaml`/`point.yaml`, or a row naming a finer scope's facts than
  its own file holds
- an empty cohort directory (rather than a clean pass)

At submission load (`Submission.from_directory`, which parses every `Point` and
`Pareto` -- the `2` exit):

- a run directory whose `point.yaml`, `system_desc.json` or
  `result_summary.json` lacks or mistypes a parser-enforced fact (`concurrency`,
  `max_supported_concurrency`, `system_tps`), declares an `offline` value
  outside `dedicated`/`elected`/`none`, or fails to parse at all - a value no
  number can carry is the mistype: a numeric string reads as its number
- a pareto shape a derivation cannot judge, each naming its Endpoints Rules §9.1
  row: multiple `offline` declarations, an `elected` point not at C_max, a
  dedicated Offline run without exactly one C_max point, points disagreeing on
  `max_supported_concurrency`/`c_min_override`, or a declared `c_min_override`
  that Section 5.4 rejects (`null`, a bool, a non-integer, outside `[1, 32]`, or
  not equal to a submitted point's concurrency -- only a missing
  `c_min_override` key derives `C_min` from the points)
- a run directory whose numeric facts no arithmetic can carry: a non-finite
  `system_tps`, or a `system_tps`/`concurrency`/`max_supported_concurrency`
  outside `float` range
- a `<system>/` directory with no model directories (an empty system cannot hide
  from system-scoped checks)

At `Finding.collect` (per submitted run):

- an unreadable artifact anywhere in the tree -- the same rule for the documents
  only rows read: a corrupt file is never read as an absent fact
- a present value its `valid_refs.yaml` declaration will not carry
  (`does not match its declared type`, `is not one of`, `is no block`); a
  numeric string on a declared `int`/`float` leaf is not one of these - it reads
  as the number it spells, and the run reports `coerced:`
- a `when` predicate that cannot be judged (an unjudgeable gate is refused, not
  read as false)
- a run directory with no carried measurement point, or one shared between two
  points (a hand-built `Submission`'s concern; `Submission.from_directory`
  cannot produce one)

## 4. Possible keys

Every dotted operand resolves through the ref namespaces below: a ref's first
segment. `submission`, `system`, `pareto` and `point` are the four node scopes;
`derived`, `ruleset` and `regions` are checker-owned. A `dict` subtree accepts
any path below it (each level resolves as present or absent; its values stay
unchecked).

| Namespace                                 | Reads                                               | Rows may use it in          | Names come from                                                                                                                |
| ----------------------------------------- | --------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `submission`, `system`, `pareto`, `point` | that node's own artifact documents (the list below) | every scope file            | the cohort's [`valid_refs.yaml`](../../src/inference_endpoint/config/cohorts/2026-10-C1/valid_refs.yaml)                       |
| `derived`                                 | one pareto's named derivations                      | `pareto.yaml`, `point.yaml` | checker code (the fixed list below)                                                                                            |
| `ruleset`                                 | the packaged ruleset the cohort's manifest names    | every scope file            | the ruleset's data - open (the cohort's [`ruleset/` catalog](../../src/inference_endpoint/config/cohorts/2026-10-C1/ruleset/)) |
| `regions`                                 | the Section 5.5 reference-algorithm regions         | `pareto.yaml`, `point.yaml` | checker code (the fixed list below)                                                                                            |

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
  source formula). They are per-pareto, so only `pareto.yaml`/`point.yaml` rows
  may name it:
  - `derived.offline_ordering` - Section 5.7.2's pair-of-clauses verdict
  - `derived.point_cap` - Section 5.6's `<=32`
  - `derived.point_count_min` - Section 5.3's minimum count
  - `derived.accuracy_run_count_requirement` - Section 4.3's `N`
  - `derived.out_of_range_points` - Section 5.4's flagged points tuple
- `ruleset` - the packaged ruleset the cohort's `cohort.yaml` names (a
  registered round, e.g. `mlperf-inference-v6.1`):
  - `ruleset.models.<name>.<key>` - model definitions (the round's `models.py`)
  - `ruleset.datasets.<name>.<key>` - dataset definitions (its `datasets.py`)
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

- The name tables draw on the Endpoints Rules' own field tables at policy
  `4ba8224dc746` - Section 8.3 for `point.yaml`, Section 8.2 for
  `system_desc.json`, Section 4.1/8.1 for `result_summary.json` - `concurrency`,
  `offline`, `dataset_name`, `model_name`, `warmup.duration_s`,
  `steady_state.status`, `nodes_used`, `system_tps`, `qps`, and the rest of each
  table.
- Blocks the rules describe without listing their keys are declared `dict`:
  `config.runtime_settings`, `config.dataset`,
  `config.steady_state.window`/`state`/`total`, `result_summary`'s
  `percentiles`/`isl`/`osl`, plus `client_config`, `system_power`,
  `accuracy_results` and `results` (the benchmark's report writer owns their
  spelling). Under such a `dict`, an unknown `<key>` resolves as **absent**
  (what `exists`/`absent`/`not_null` are for), not an error, and `ruleset` keys
  are open data too - whatever the entry contains (`vocab_size`, `is_drafter`,
  ...).
- What a present value must carry is declared the same way and checked at
  collect time (`Finding.collect` refuses before any finding): match its leaf's
  type (`does not match its declared type`), sit in its `allowlist` where one is
  declared (`is not one of`), and a declared block must carry a block
  (`is no block`). The fence-posts:
  - the parser-minimum trio (`concurrency`, `system_tps`,
    `max_supported_concurrency`) is code-side: refused at parse time by
    `Point.from_node` whatever the table says
  - a `dict` subtree is open (its own value must be a block) and its values stay
    unchecked
  - an absent field carries no value to check (its absence reads per "Absent
    facts")
  - a hand-built `Cohort((check, ...))` with no table checks nothing

Rows name only declared facts:

- a row naming an undeclared or misspelled key is refused at cohort load
  (`names no fact`; a one-edit near-miss adds "did you mean 'offline'?"), and so
  is any path below a leaf fact (`system_desc.cooling.detail`:
  `is one leaf fact, nothing deeper`)
- an upload may still carry keys its cohort does not declare - they are simply
  not nameable by a row
- the strict names are what gives "Absent facts" above its meaning: a misspelled
  key dies at cohort load instead of decoding to an absent fact and switching a
  `when`-gated check off by omission
- the price is upfront: a rule over a brand-new field first needs that field
  declared in the cohort's `valid_refs.yaml`
- every dotted string must name a ref namespace or use `{"literal": ...}`
  (`matches` patterns use that form always, dots or not); a stray dotted value
  is refused at load (near-miss heads likewise)

## Implementation details

The loader, tree walk, row operators, region derivation and the check runner are
documented in [`DESIGN.md`](DESIGN.md) - "what happens under the hood", kept
strictly separate from this how-to-use page.

`compliance/checker.py` is a different checker in this repo: it validates one
completed Edge-Agentic (BFCL) run's report directory (the resolved `config.yaml`
plus scorer output) against the Edge-Agentic ruleset - config lock, the accuracy
gate, dropped-turn validity. This module checks an assembled Endpoints
submission against a cohort's declared checks. Both read the policy documents;
neither imports the other.
