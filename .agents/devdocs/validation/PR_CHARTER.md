# Validation PR Charter — Python vs YAML, and the two-phase split

Working draft (`.agents/devdocs/`, first-revision notes per standing rule 1).
Committed together with PR 1 as `docs/validation/PR_CHARTER.md` if the reviewer
wants it shipped with the machinery.

## The boundary rule (when a check lives in Python vs in YAML)

1. **Python carries the arithmetic.** Anything computing a value: ratios,
   margins, tiles, ordering across points, sub-collection extraction, log-space
   boundary math, pass-boundary derivation. Named derivation on its subject
   (method or classmethod-producer), exposed as fields `Ref` reads.
   Examples already landed: `Regions.for_bounds` (Endpoints Policy Section
   5.5 reference algorithm), `Regions.coverage`, `Regions.covers`,
   `Interval.for_accuracy_gate` (Section 2.9.8's `score >= factor x reference`),
   `Submission.offline_ordering` (Section 5.7.2's pair of clauses),
   `Submission.point_cap` (Section 5.6's `<= 32` cap).

2. **YAML carries the policy.** Anything answering *which* fields, at *which*
   threshold, under *which* prose, on *which* node kind: `when`/`assert` rows
   over the existing `Compare`/`Aggregate`/`Op`/`Measure` vocabulary, plus the
   `id` / `description` / `rule` / `fix` / `severity` / `root` metadata. This
   is the maintainer-edit surface (the non-Python goal: change a threshold or
   add a check without touching the code).

3. **New condition semantics** live in Python as one `Op` subclass per new
   word (`operations.py`, `__new__` dispatch on the YAML `op` name). Anything
   expressible with the existing word set reuses it.

## Two-phase PR split (standing directive)

| Phase | Ships | Does not ship |
|---|---|---|
| **PR 1 — machinery + stubs** | the `validation/` module (rows, ops, ranges, regions, catalog, tree, submission, check), its unit grids, the W6 check-runner/`Verdict`-per-node core, and 1-2 representative stub check YAMLs under `tests/assets/cohorts/stub/` to pin the loading shape | the full compliance checklist encoded as YAML |
| **PR 2 — full cohort** | the assembled checklist as check-block YAMLs (one per compliance row below, prose `rule`/`fix` verbatim-cited from `endpoints_rules.md`'s own rows), plus named derivations added **only** where the boundary rule item 1 says a row cannot express the check on its own | further machinery refactors (those go with the row that needs them, keeping PR-1's surface held) |

## Tests policy

- Every Python layer carries its own behavioral grid (this branch's existing
  pattern: `Op`/`Aggregate`/`Regions`/`Interval`/`Submission` each hold their
  computed-value grid).
- YAML encoding gets 1-2 representative checks decoded end-to-end through
  `Cohort.from_directory` (the stubs ship exactly these). The full checklist in
  PR 2 is *not* test-matrixed row-by-row; it is machine-readable configuration
  reviewed once against its cited `endpoints_rules.md` rows.

## Compliance-row to check-block mapping (PR 2's working grid)

Source table: `endpoints_rules.md` §6-ish compliance-checklist
(`<https://github.com/mlcommons/endpoints_policies/blob/d062b470abd2/endpoints_rules.md>`),
section title "Compliance validation" (row-relevant anchors cited per row).

| Checklist row (severity from its rightmost column) | Check-block encoding needs | Python derivation crossing (per rule 1, when needed) |
|---|---|---|
| Point count ≥ 8 or ≥ 7 (Reject) | `when: agentic-or-not` contrast + `assert` on `len(points)` | its own named count shape (separate from Point cap: strictly `>`-vs-`<` bounds); cap stays `Submission.point_cap` |
| Point cap ≤ 32 (Reject) | one row asserting `point_cap` field-passes | landed (`Submission.point_cap`, Section 5.6) |
| Offline point present, exactly one declaration (Reject) | count-of-`offline`-declarations row + elected-on-$C_{max}$-point linkage | exercised explicitly in `offline_ordering`'s `ValueError` guard but its own row is separate |
| Ultra-Low/Low/Med/High concurrency coverage ≥ 1 pt (Reject each) | `Aggregate over: <points' concurrency values>` with `measure: count, op: ge, value: 1` per `regions.by_kind.<kind>` interval | `Regions.for_bounds` + `regions.by_kind` exposure (landed) |
| Accuracy run count = `N` (Reject) | `Aggregate count` over `accuracy_results` entries against rather-`N` (round's `N` = 5 non-agentic / 4 agentic) | none pending |
| Max concurrency declared `M > 32` in `system_desc.json` (Reject) | one `Compare` row on `system.max_supported_concurrency` | none (`Re gions.for_submission` shares the same field) |
| Point cap / concurrency-in-range rows | one arguable single row per point over `regions.coverage` | `Regions.coverage` (landed) |
| Offline ordering (`system_tps(Offline) ≥ 0.98 × system_tps(C_max)` + `concurrency(Offline) ≥ C_max`, dedicated only; Flag) | a two-clause `when: dedicated`-guard + `assert: submission.offline_ordering` | `Submission.offline_ordering` (landed, Section 5.7.2) |
| Per-point denominator (`system_tps_per_kw == system_tps / point_power_kw`, Flag) | arithmetic across power + tps fields (per-row-clause …) | named derivation needed (power normalization, Section 4.5.2/§4.5.3 — deferred) |
| Maximal engagement (`accelerators engaged == DP × A_replica`, Flag) | parallelism-configuration arithmetic | named derivation needed (Section 4.5.3-based; deferred) |
| Load pattern (Reject on non-conforming) | `runtime_settings.load_pattern`-contrast row (`offline` point vs rest) | none |
| Run duration (Flag) / Minimum query count (Flag) | rows over `steady_state` field + per-region minimums | named rewrite of §6.2/§6.4 bounds deferred |

## Known policy-text divergences to keep visible (each marked `SEEN-BY-PORT`, one TODO each filed upstream where applicable)

- Endpoints Policy Section 5.4's **Example C** carries two hand-typo slips
  against Section 5.5's own authoritative pseudocode: its Low Concurrency
  start reads `16` where `c_min + 1` says `17` (its own Med-start of `27`
  agrees), and its `round(16 + 2^6.652) = 116` reads `100.4` where
  `2*span_c = 6.65152` yields `100.53` (rounds to `117`). Verified with
  `python -c`; the port follows the formula in both cases. Example B matches
  the formula exactly; Example A matches after the formula's own rounding.
- The point-YAML `region` field's five declared values
  (`low_latency, low_concurrency, med_concurrency, high_concurrency,
  submitters_choice`) differ from the `compute_regions` key set
  (`low_latency, low_concurrency, med_concurrency, high_concurrency, margin`).
  Two valid vocabularies, declaration-vs-computed; `PointRegion` (a second
  enum mirroring the declaration set) is deliberately deferred until a check
  needs it (YAGNI).
- The `Point-cap` row (≤ 32) and the `Point count ≥ 7/8` minimum are separate
  rows with opposite bounds; the minimum's agentic-or-elected applicability
  contrast is its own named derivation, deliberately not folded into
  `Submission.point_cap`.