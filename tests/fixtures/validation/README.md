# Validation fixtures

The submission trees are synthetic test cases, not historical submissions or
measured results. Their accuracy scores, performance summaries, steady-state
windows, system configurations, and power disclosures are constructed to exercise
validation behavior. They must not be used to compare hardware or serving software.

Each invented system has a neutral identifier (`system_a` through `system_l`).
Real model, dataset, accelerator, and processor names remain where they exercise
published policy keys and hardware power lookups; these names do not establish
that the configurations were run on those products.

`valid_standardized` supplies a complete standardized submission layout.
`invalid_submission` exercises invalid declarations and missing requirements.
`sub_a` through `sub_j` exercise different curve, model, configuration, and power
paths. These are checker inputs: the directory names do not guarantee that every
case passes all rules in the current cohort.

The corpus was adapted from the submission checker test corpus. Its fixture
regeneration tool explicitly describes the data as synthetic and constructs
accuracy values, steady-state windows, and power disclosures. The copies here
are maintained as native validation fixtures and require no external checker.

`policy-fingerprints.json` pins the catalog and rule definitions independently of
these submission trees, so policy edits require an explicit expected-value update.
