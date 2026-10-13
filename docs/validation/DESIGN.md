# Validation submodule - implementation notes

"What happens under the hood" for `inference_endpoint.validation`; the
user-facing how-to is [`README.md`](README.md). Content-here is the loader,
row/operator model, tree walk, derivation sources and the runner's mechanics --
for maintainers and reviewers, not check authors.

## Module map

| Module          | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `check.py`      | `Check` (the YAML block's typed shape: `id`/`description`/`rule`/`severity`/`scope`/`fix: str\|None`/`needs`/`optional_needs`/`when`/`assert`, `scope: RuleScopes`, `_decode_row` picks `Compare` vs `Aggregate` by `measure` presence), `Cohort.from_directory` (one `*.yaml` per `Check` sorted by filename; duplicate `id` -> `ValueError`), `Severity` wire-enum (Values `warn`/`fail`).                                                                                                                                                                           |
| `compare.py`    | `Ref` (dotted-fact resolution with `ROOTS` = `submission`/`system`/`curve`/`point`/`derived`/`ruleset`/`regions`/`cohort`; `found`/`MISSING` grid: absent-vs-stored-null distinguished), `Compare` (op row model, `Op`-driven arity), `Aggregate` (collection row with `__new__` measure-dispatch; `Count`/`Sum`/`Mean`/`Min`/`Max`/`AnyTrue`/`AllTrue` family), `Verdict.evaluate` (boundary: an escaping `TypeError`/`ValueError`/`OverflowError` becomes a failing `why`-naming-both-sides verdict).                                                                |
| `operations.py` | The `Op` family (13 ops) with `__new__` self-registration on `Op(name)` (`Eq`, `Ne`, `Gt`, `Ge`, `Lt`, `Le`, `In`, `Matches`, `SameMultiset`, `Between`, `Exists`, `Absent`, `NotNull`); `Predicate(subject, target)` (`target` can be an `Interval`/`Regions`); `Interval` (bounded/unbounded/half-open range with `__contains__`), `Op.tolerant` ClassVar (True on `Eq`/`Ne` only; `Compare._check_shape` rejects nonzero tolerances on non-tolerant ops).                                                                                                           |
| `ranges.py`     | `Interval(lo, hi=None, lo_inclusive, hi_inclusive)`; `Interval.for_accuracy_gate(reference_score, tolerance)` (Section 2.9.8's `score >= factor x reference`).                                                                                                                                                                                                                                                                                                                                                                                                         |
| `regions.py`    | `RegionKind` wire-enum (five `compute_regions` keys: `low_latency`/`low_concurrency`/`med_concurrency`/`high_concurrency`/`margin`); `Regions.for_bounds` (Section 5.5's reference algorithm verbatim); `Regions.for_submission` (Section-verified `max_supported_concurrency` + Low-Latency-point derivation); `Regions.covers` / `Regions.coverage`; Section 5.4 Example C's two hand-typo slips noted inline (port uses the formula both times).                                                                                                                    |
| `tree.py`       | `Node` (kind-registered subclasses via `__init_subclass__(kind=...)` + `__new__` dispatch: `SubmissionNode`/`SystemNode`/`CurveNode`/`PointNode`); `_StrictLoader` (duplicate-keys/merge-keys/aliases -> parse error); `_read_artifact` (single bounded symlink-safe byte channel: `os.open(…, O_RDONLY\|O_NOFOLLOW)`, byte-cap before decode, `UnicodeDecodeError` funneled to `(value, error)`); `Node.scopes()` (facts-projection map).                                                                                                                             |
| `sentinels.py`  | The shared `MISSING` sentinel (`value is MISSING` = the fact's absence).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `catalog.py`    | `Catalog.from_directory` (ruleset `models.yaml`/`datasets.yaml` -> plain dicts; `Catalog.scopes()` = the `ruleset` root's payload).                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `submission.py` | `Point` (frozen: `offline`/`concurrency`/`system_tps`/`node` + `point_power_kw` derivation = `provisioned_power_kw x Y/N`, Section 4.5.2/4.5.3, 2dp-rounded-once); `Submission` (frozen `system_desc` + ordered `points`; `offline_ordering` = Section 5.7.2's two clauses with `None`-when-`elected`/agentic; `point_cap` = Section 5.6's `<=32`; `point_count_min` = Section 5.3's `8`-or-`7`; `accuracy_run_count_requirement` = Section 4.3's `5`/`4`; `out_of_range_points` = Section 5.4's Concurrency-in-range row; `derived_fields()` = the facts-projection). |
| `runner.py`     | `Finding` (audit-trail record: `check_id`/`severity`/`scope`/`node_directory`/`verdict`/`rule`), `Finding.collect` (3 phases: `_collect_candidates` validates `needs` names; `_needs_gate` recursively memoized `holds_self` with `resolving`-set cycle-guard; `_evaluate_rows` emits one `Finding` per row where `when` holds at each applicable node -- holds and failures alike).                                                                                                                                                                                   |

## Row-operator mechanics

`Op` self-registers via
`__init_subclass__(name=..., slots=..., needs_found=..., tolerant=...)` and
dispatches through `Op(name)` (like `Node`'s kind dispatch). Arity per op lives
in one place (`Op.slots`); `Compare._check_shape` validates `Compare`'s
`right`/`lo`/`hi` field-presence against `Op.slots` (so `right: null` remains a
legal eq-against-null operand, per the W1 absent-vs-null grid).
`Predicate(subject, target)` is the resolved-operand carrier; `target` widens to
an `Interval` for `between` (whose `slots` are `("lo", "hi")` -- the only
interval-bound consumer) or a `Regions` (for `covered_by`-flavor semantics via
`Regions.covers`).

`_close_to` short-circuits exact-`int == int` at zero tolerance (math.isclose's
float coercion conflates distinct large integers) and lands tolerance math only
for `eq`/`ne` per `Op.tolerant`. Order ops (`gt`/`ge`/`lt`/`le`) have no
tolerance semantics (no standard meaning of an ordered comparison under
`abs_tol`).

## Runner's cascade and emission

`Finding.collect(cohort, submission, regions)`:

1. `_collect_candidates(cohort)` validates `needs`/`optional_needs` id
   references against the cohort (`ValueError` on an unknown id -- a load-level
   mistake, not a report item).
2. `_needs_gate` memoized-recursively-resolves `holds_self(check_id)`: the
   check's own `assert` rows holding at its `when`-applicable scope nodes, plus
   its `needs` chain passing first. A `needs` cycle trips a `resolving`-set
   guard into a `ValueError`. The gate-selects-checks-whose-NEEDS-pass
   (never-gating a check's own Findings on its own row-status -- a check whose
   rows fail still emits its Findings; only its dependents stay silent).
3. `_evaluate_rows` emits one `Finding` per row at each `when`-applicable node
   of the gate-approved checks (holds and failures alike -- the report side
   filters `not verdict.ok` for a failure summary).

## Derivation formulas (verbatim-anchored)

| Named derivation                                                               | Formula / rule source                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Interval.for_accuracy_gate(reference_score, tolerance)`                       | `[tolerance * reference_score, inf)`, Endpoints Policy Section 2.9.8                                                                                                                                                                      |
| `Regions.for_bounds(c_max, c_min)`                                             | Section 5.5's `compute_regions` reference algorithm (verbatim; Section 5.4 Example C's two hand-typo slips noted inline)                                                                                                                  |
| `Regions.for_submission(system_desc, concurrencies)`                           | `C_max = system_desc["max_supported_concurrency"]` (field name verbatim, `endpoints_rules.md`'s Max-concurrency-declared row); `C_min` = the lone `concurrency` entry in `[1, 32]` (the Low Latency point)                                |
| `Submission.offline_ordering()`                                                | `system_tps(Offline) >= 0.98 * system_tps(C_max)` AND `concurrency(Offline) >= C_max`, Section 5.7.2's two-clause wording; `None`-when-`elected`-or-agentic                                                                               |
| `Submission.point_cap()`                                                       | `len(points) <= 32` (Offline counted), Section 5.6                                                                                                                                                                                        |
| `Submission.point_count_min()`                                                 | `8` if a dedicated `offline` is declared (non-agentic), `7` otherwise (elected-or-agentic), Section 5.3 + `:1292`'s one-fewer-distinct-run derivation                                                                                     |
| `Submission.accuracy_run_count_requirement()`                                  | `5` if any `offline` is declared (non-agentic), `4` otherwise (agentic), Endpoints Policy Section 4.3's `N`                                                                                                                               |
| `Submission.out_of_range_points(regions)`                                      | `tuple(p for p in points if p.concurrency not in regions.coverage())`, Section 5.4's Concurrency-in-range row                                                                                                                             |
| `Point.point_power_kw(provisioned_power_kw, engaged_nodes, provisioned_nodes)` | `round(provisioned_power_kw * engaged_nodes / provisioned_nodes, 2)`, Section 4.5.2/4.5.3's `P_prov x Y/N`, rounded-2dp-once per `:897` (verified against Appendix C's `152.10 x 16/18 = 135.2` and `152.10 x 18/18 = 152.1` golden rows) |

## Notable engineering calls (for reviewers)

- The `strict loader` refuses duplicate keys, `<<` merge keys and all YAML
  aliases (alias fan-out can re-walk shared subtrees exponentially; silent
  last-wins keys hide data from the review).
- `_read_artifact`'s bounded byte-read runs its cap **before** decode and
  funnels `UnicodeDecodeError` (a `ValueError`) alongside `OSError` into
  `(value, error)` -- one channel where "nothing here aborts the run" holds for
  real.
- `Verdict.evaluate` is the single place incommensurable-pair handling is
  pinned: a `TypeError`/`ValueError`/`OverflowError` escaping any row's `holds`
  converts to a failing verdict naming both sides and the exception class
  (matching the README's garbage-grid "fails, `why` shows both" claim and
  `Interval.__contains__`'s return-fail-shape, per the review-council finding).
- One `MISSING` sentinel (`sentinels.py`) is shared by `Node.read`, `Predicate`
  and the presence ops so the absent-vs-stored-null grid is one rule, not a
  per-module pair of conventions.
