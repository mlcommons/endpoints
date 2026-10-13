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

## 2. How it works from the user's perspective

The user-facing artifact is a **cohort** -- a directory of YAML files, one file
per check. Example layout (this is the shipped stub cohort under
`tests/assets/cohorts/stub/`):

```
my_cohort/
├── concurrency_in_range.yaml
├── offline_ordering.yaml
├── point_cap.yaml
└── accuracy_run_count.yaml
```

Every `*.yaml` in the directory is one check;
`Cohort.from_directory("my_cohort")` loads them sorted by filename and requires
unique `id`s across the directory (a duplicated `id` raises `ValueError` at
load). A cohort is ordinary config: it can live in version control next to the
submission template it belongs to.

Loading the cohort and running it against one upload is the two-line CLI entry
(see `DESIGN.md` for the underlying machinery):

```python
from pathlib import Path

from inference_endpoint.validation.check import Cohort
from inference_endpoint.validation.runner import Finding

cohort = Cohort.from_directory(Path("my_cohort"))
findings = Finding.collect(cohort, submission, regions)
```

`findings` is one record per **evaluated** `assert` row per applicable node --
holds and failures alike, so the record doubles as the audit trail. A reporting
tool filters to `not verdict.ok` for a failure-only view.

## 3. How the files are written

One YAML mapping per file, with `id`/`description`/`rule`/`severity`/`scope`
metadata, an optional `fix` hint, and two row lists: `when` rows gate whether
the check applies at a given node; `assert` rows are the actual rule conditions.
Rows are one of two shapes, distinguished by their keys (`measure` signals the
aggregate form; anything else is a comparison):

```yaml
id: offline_ordering
description: The dedicated Offline run meets Section 5.7.2's two clauses.
rule:
  "Endpoints Policy Section 5.7.2, Relationship to Maximum Supported
  Concurrency."
fix:
  Either raise the dedicated run's system_tps to 0.98x the C_max point's, or
  elect the C_max point as Offline.
severity: warn
scope: point
when:
  - left: point.config.offline
    op: eq
    right: dedicated
assert:
  - left: derived.offline_ordering
    op: eq
    right: true
```

An aggregate row reduces a collection first, then compares it to a value:

```yaml
assert:
  - over: point.results.accuracy
    measure: count
    op: ge
    value: 1
```

Operators (`op`) for comparison rows: `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `in`,
`matches`, `between`, `same_multiset` (each with its own operand arity --
`between` takes `lo` and `hi`; `exists`, `absent`, `not_null` take `left`
alone). Operators for aggregate rows (`op` after `measure`): `eq`, `ne`, `gt`,
`ge`, `lt`, `le`. Measures (`measure`): `count`, `sum`, `mean`, `min`, `max`,
`any`, `all`. Optional `abs_tol`/`rel_tol` (op-tolerant comparisons only --
`eq`/`ne`) compare within a tolerance.

A check may name upstream checks by `id` in `needs` (gating: the dependent fires
only when every named upstream chain holds) or `optional_needs` (recorded but
non-gating). Any dotted string whose first segment is a scope root resolves to a
fact at run time; everything else is a literal value (`5`, `true`,
`"dedicated"`).

A second example crossing `needs` and the aggregate form:

```yaml
id: accuracy_sane
description: At least one accuracy result exists at the mandatory points.
rule: "Endpoints Policy Section 4.3, Accuracy Metric."
severity: fail
scope: system
needs: [offline_ordering]
assert:
  - over: point.results.accuracy
    measure: count
    op: ge
    value: 1
```

## 4. Possible keys

Every dotted operand resolves through six scope roots. The keys below are the
full decoded vocabulary today.

| Key pattern                                                     | Represents                                                                                                                                                                                                                                                                                     | Source file / producer                                                                              |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `submission.config.<key>`                                       | Top-level submission-level settings                                                                                                                                                                                                                                                            | `config.yaml` at the submission root                                                                |
| `submission.client_config.<key>`                                | Client runtime settings                                                                                                                                                                                                                                                                        | `config.yaml` at the submission root                                                                |
| `submission.system_desc.<key>`                                  | System description fields (`max_supported_concurrency`, `model_name`, `division`, ...)                                                                                                                                                                                                         | `system_desc.json`                                                                                  |
| `submission.system_power.<key>`                                 | Power declaration (`provisioned_power_kw`, ...)                                                                                                                                                                                                                                                | `system_power.json`                                                                                 |
| `submission.results.<key>`                                      | Results summary rollup                                                                                                                                                                                                                                                                         | `results.json`                                                                                      |
| `system.<key>`, `curve.<key>`, `point.<key>`                    | The same three artifact families above at each node kind (`system_desc` at `system`, `point.yaml`'s fields at `point`, ...)                                                                                                                                                                    | `system_desc.json`, `point.yaml`, `result_summary.json`, `accuracy_results.json` per node directory |
| `derived.<name>`                                                | Named-derivation values computed from the submission (`derived.offline_ordering` = Section 5.7.2's pair-of-clauses verdict; `derived.point_cap` = Section 5.6's `<=32`; `derived.point_count_min` = Section 5.3's minimum count; `derived.accuracy_run_count_requirement` = Section 4.3's `N`) | Python-derived (see `DESIGN.md` for each derivation's source formula)                               |
| `ruleset.models.<name>.<key>` / `ruleset.datasets.<name>.<key>` | Model/dataset definitions the ruleset declares                                                                                                                                                                                                                                                 | `models.yaml` / `datasets.yaml` under the ruleset directory                                         |
| `regions.coverage`                                              | The interval of `concurrency` values a valid region covers (including the 10% margin)                                                                                                                                                                                                          | Computed from Section 5.5's reference algorithm (`compute_regions`)                                 |
| `regions.by_kind.<kind>.<lo\|hi>`                               | One of the five computed regions (`low_latency`, `low_concurrency`, `med_concurrency`, `high_concurrency`, `margin`) as an interval's bounds                                                                                                                                                   | Computed from Section 5.5's reference algorithm                                                     |

The per-node artifact families' field names follow each schema's own field table
verbatim: `point.yaml`'s `concurrency`, `offline`, `dataset_name`, `model_name`,
`steady_state.status`, `nodes_used`, `dp_shortfall`, etc. are addressed as
`point.config.concurrency`, `point.config.offline`, and so on (`point.yaml`'s
map is `config`), and `result_summary.json`'s raw metrics appear as
`point.result_summary.<metric>`. Everything named here is the full decoded
vocabulary; a `Ref` addressing any other dotted string is treated as a literal
value.

## Implementation details

The loader, tree walk, row operators, region derivation and the check runner are
documented in [`DESIGN.md`](DESIGN.md) - "what happens under the hood", kept
strictly separate from this how-to-use page.
