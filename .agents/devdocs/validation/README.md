# validation rewrite v2 — design notes (DRAFT, high churn)

> Scratch notes, per standing rules. Not documentation. Mirrors
> `docs/validation/` layout so it graduates in place later.

## Goal (user's words, 2026-10-10)

Implement the submission checks from `endpoints-submission-cli` (oracle
`32060e6`) in a new submodule `inference_endpoint.validation`. A cohort
maintainer must be able to (a) find and understand a check, (b) change its
threshold or condition, (c) add a check — without touching Python. Occam's
Razor: prefer built-ins and code reuse; do not over-engineer the way the Codex
port (6.3k lines) did. Rules pinned in hivemind `standing-implementation-rules`
(draft docs in `.agents/devdocs/`, base-first component waves, non-oneshot +
200-LOC cap + review gate, layer-correct tests, templated fixtures).

## Architecture decision (user, 2026-10-10)

NO expression language / ast sandbox. v1 (on the posterity branch) built
`expr.py` — a whitelisted-Python evaluator — and the reviews paid for it
(missing primitives betrayed by always-false idioms, `explain()`-derived
messages, sandbox type-by-convention). v2 instead:

- checks ARE YAML decoding into typed models (repo convention: pydantic for
  config; `BeforeValidator`s are the "custom decode hooks");
- each condition is one comparison ROW (`op` + operands), operands being either
  a literal or a dotted `Ref` into the scoped facts (`point.…`, `curve.…`,
  `ruleset.…`, `cohort.…`);
- genuinely algorithmic derivations (regions, accuracy slots, power math) are
  named Python functions the rows call BY NAME — rule authors tune numbers and
  comparisons, never algorithms;
- messages are owned by the comparison (each op returns `Verdict(ok, why)` with
  the actual values in `why`) — no AST re-derivation, no whispering a lie
  expression to force a failure.

## Component waves (base first, consumers after)

| W   | Component                                                                                                                                                                                                                              | Depends on | State                                                                                                                                                                                                       |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W1  | `compare.py`: `Ref` (incl. `from_string`/`resolve`), operand decode hooks, `Compare` row model, `Verdict.evaluate` = `Verdict(ok, why)`                                                                                                | —          | **reviewed; ownership refactor applied per `oop-codegen-method-ownership` (constructors own construction, heavy ops on their subject, producers on product; single-backtick docstrings). Re-gate pending.** |
| W2  | `tree.py`: Submission/System/Curve/Point nodes (kind-registry via `__new__`/`__init_subclass__`), hardened artifact load (unique-key YAML, size cap, no-follow containment), `MISSING` sentinel reads, `scopes()` for `Ref` resolution | stdlib     | accepted at gate 3                                                                                                                                                                                          |
| W3  | collection rows: `Aggregate` (count/sum/mean/min/max/any/all over a list or mapping values) + `same_multiset` compare op; `Op` family in `operations.py` (`__new__` dispatch, self-registering subclasses); blank-vs-null grid pinned  | W1         | **this wave: done pending review**                                                                                                                                                                          |
| W4  | check block model + cohort directory loader (metadata: id/description/rule/fix/severity, `when`/`assert` rows, per-node iteration)                                                                                                     | W1         |                                                                                                                                                                                                             |
| W5  | derivations: regions, accuracy slots/scores, shared paths (named functions) + the catalog view (models/datasets dict lookups — no pseudo-enums, sizes/values as plain data)                                                            | W2         |                                                                                                                                                                                                             |
| W7  | ported check families from the oracle (one family per change, e.g. metric-consistency, load-pattern, accuracy gates) + power module                                                                                                    | W3,W6      |                                                                                                                                                                                                             |
| W8  | CLI + report printing + golden/parity fixture harness (templated generation — standing rule 5)                                                                                                                                         | W7         |                                                                                                                                                                                                             |

Design notes for W6 (kept off the wave table's rows so the grid stays
parseable):

- **`Finding`** keeps no `fix` field: `Check.fix: str | None` is the optional
  YAML hint (the oracle's `Message.fix` where-declared semantics
  `endpoints-submission-cli@32060e6`'s `submission_checker/messages.py`), joined
  at render time where the catalog supplies one. `Finding` emits one record per
  _evaluated_ `assert` row (holds and failures alike) — the audit trail's
  document-format, matching `CheckResult`'s all-messages design and
  `messages.yaml`'s carrying both `pass:` and `fail:` texts. The report side
  filters `not verdict.ok` for failure-only summaries.
- **`region`-fold single source**: `Node.scopes` owns folding the `regions` root
  (fed by `SubmissionNode.from_directory(root, regions=...)`); the runner reads
  `node.scopes()` directly and does not re-fold.

Per-item FILTERS are deliberately NOT in the row language (kept out to avoid a
generator mini-language): filters over collections (e.g. "entries whose name is
declared") become derived facts in W5. Open question for the gate: names so far
match the v1 collected (problem) lists (accuracy/shared-path bands, errors) plus
two typed predicate facts (`drafter`, `line_factor`).

## W1 semantics grid (pins the sharp cells reviews kept tripping on)

Resolving `Ref` returns `(found: bool, value)`; `found` distinguishes ABSENT
from a present `None` (null) — the fact `default()` burned v1 on.

| operand state    | binary ops (eq/ge/…)                                   | `exists` | `absent` | `not_null` |
| ---------------- | ------------------------------------------------------ | -------- | -------- | ---------- |
| missing          | fail, `why` says `missing <ref>`                       | False    | **True** | False      |
| present, `null`  | compare `None` (eq true vs literal null)               | True     | False    | False      |
| present, garbage | compares naturally (str vs num: fail `why` shows both) | True     | False    | True       |

- `eq` over floats takes `abs_tol`/`rel_tol` (`math.isclose` — built-in, rule
  2).
- `in`: membership; mappings contribute their keys.
- `matches`: `re.fullmatch` on a repo-authored pattern.
- `between`: `lo <= x <= hi`.
- `why` text is ~80-char bounded `repr` of both sides + the op — a summary of
  the comparison as computed, not a re-parse of a written sentence.

## Open questions for the review gate

1. `for_each`-like iteration shape (needed by many checks: one finding per
   entry/statement) — planned as W3/W4 field, e.g. `each: <Ref to list>`.
2. `needs` semantics: v1 counted a need whose `when` was false as satisfied
   (hand-over accepted it). v2 proposal: need must have run and passed unless
   marked `optional: true`. Ask.
3. Parity harness: do we still need `expected_deltas.yaml`-style encoding of
   intended oracle differences, or is golden-output parity per case enough for
   v2? Occam leans golden-only; losing the named deltas is a real auditability
   cost though (`# DISCREPANCY Dn` trail maps to them).
4. Severity as the repo `Severity` enum (validation engines share it with
   nothing else) — smallest honest choice.

## History

- v1 lives on `feat/submission-checker-rewrite` (kept untouched, per user).
- This worktree: `feat/validation-v2` off `main` `d3c57f2e`.
- W1 (`compare.py`) REVIEWED GATE 1-5: ownership refactor (constructors own
  construction, heavy ops on their subject, producers on product), ASCII +
  example docstrings, `Ref.ROOTS` class constant, `row` -> `comparison`,
  `*operands` call, failure-branch-only message building, plain public verbs
  (`Compare.run`/`Compare.operands`) after a dunder flip. 39 tests green. W2
  (tree loader) in progress.
