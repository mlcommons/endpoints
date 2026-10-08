# Validation fixtures

The submission cases are synthetic test inputs, not historical submissions or
measured results. Their accuracy scores, performance summaries, steady-state
windows, system configurations, and power disclosures exercise validation behavior.
They must not be used to compare hardware or serving software.

`cases/` contains one readable YAML file per submission. Each file has:

- `defaults`: common artifact fields shared by its measurement points.
- `files`: submission-level artifacts and small source/documentation files.
- `points`: point paths and artifact fields that differ from the defaults.

The pytest fixtures in `tests/unit/validation/conftest.py` write complete JSON and
YAML artifacts into a temporary directory. Nested dictionaries merge recursively;
scalar values and lists replace their defaults. An empty override uses all defaults.
An omitted artifact is absent; `null` is a value, not a deletion instruction.
Missing fields must be omitted from defaults and supplied only by points that need
them. Tests can copy and mutate generated trees without changing the case files.

`valid_standardized` supplies a complete standardized layout. `invalid_submission`
exercises invalid declarations and missing requirements. `sub_a` through `sub_j`
exercise different curves, models, configurations, and power paths. Their names do
not guarantee that every case passes all rules in the current cohort.

Invented systems use neutral identifiers (`system_a` through `system_l`). Real
model, dataset, accelerator, and processor names exercise published policy keys
and hardware power lookups; they do not establish that those configurations ran.

`policy-fingerprints.json` pins the catalog and rule definitions independently of
these cases, so policy edits require an explicit expected-value update.
