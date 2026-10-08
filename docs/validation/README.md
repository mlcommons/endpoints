# Cohort policy coverage

The 2026-10-C1 revision 1 bundle contains 89 definitions across five files in
`src/inference_endpoint/validation/policies/2026-10-C1/`.
Catalogs, enrollment, exceptions and catalog diagnostics live in catalog.yaml.
All files declare the same cohort and revision; parser intervals are inclusive.

## Current interfaces

The artifact validation API loads the five-file cohort policy, classifies the
submission, plans applicable checks, and executes registered evaluators. Its
regression suite covers artifact parsing, disclosures, seeds, accuracy, metric
consistency, coverage, and provisioned/per-point power.

Check selection uses typed classifications and prerequisites. Ready means selected
with available prerequisites; evaluators determine compliance. Requirement field
names and references are validated by the versioned policy parser.

## Conditions and evidence

- Each check declares when it applies, when it is excluded, and which evidence it
  requires.
- All condition fields must match. Within an enum list, any listed value can match.
  A matching exclusion prevents the check from running.
- Missing classification information blocks checks that need it. An empty set
  means the information is known and none of the listed facts are present.
- Exceptions for points within a Pareto curve create separate plans for the
  affected points. Overlapping exceptions for the same point cause an error.
- Each point retains its own classification information, including unknown values.
- An invalid artifact blocks checks that depend on it; independent checks can
  still run.
- Dataset entries default to `is_legacy: false` and `sample_unit: sample` when those
  fields are omitted.
- Seeds come from the bundled published cohort catalog.

## Unresolved policy inputs

The approved_client_revisions list is empty. Published full client SHAs must be
supplied before client revision approval can pass.
Speculative-head approval publication cohorts are unspecified.
The policy catalog is proposed, not an official MLCommons publication.

## Evaluator kinds

- A rule's YAML `kind` selects its callable evaluator class.
- The rule supplies the evidence addresses, catalogs, thresholds, and other
  requirements used by that evaluator. Some kinds also use an `operation` to
  select a specific behavior.
- The planner selects applicable checks and verifies prerequisites before
  evaluators run. Evaluators return findings for the validation report.

| Evaluator kind      | How it works                                                                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `accuracy_coverage` | Checks that accuracy results cover the required concurrency bands and offline points.                                                                               |
| `accuracy_gate`     | Compares model accuracy scores, issued accuracy sample counts, or output-length statistics with policy thresholds; supports per-point and aggregated results.       |
| `accuracy_presence` | Checks that enough model curves contain standalone or embedded accuracy results.                                                                                    |
| `artifact_binding`  | Matches a submitted client SHA or checkpoint repository and revision against approved catalog entries.                                                              |
| `artifact_schema`   | Checks that an artifact parsed successfully into its typed model and, when required, is nonempty.                                                                   |
| `catalog_integrity` | Checks that a required seed or speculative-head catalog loaded successfully and is available.                                                                       |
| `cohort_identifier` | Checks the target cohort string has the expected year, month, and C0/C1 format.                                                                                     |
| `collection_size`   | Checks that the number of points or other collection members falls within the configured limits.                                                                    |
| `comparison`        | Compares two artifact values or constants using equality, greater-than, or greater-than-or-equal.                                                                   |
| `consistency`       | Checks equality between two values or across declarations in a collection, optionally excluding specified fields.                                                   |
| `coverage`          | Counts performance points in a concurrency band and checks the required minimum.                                                                                    |
| `dataset_count`     | Checks that a declared count equals the dataset sample count or is a positive multiple of it.                                                                       |
| `dataset_minimum`   | Checks that completed samples meet the threshold for the point's dataset.                                                                                           |
| `derived_metric`    | Recalculates throughput, interactivity, utilization, or throughput per kW and compares it with the reported value using a tolerance.                                |
| `disclosure`        | Checks that required configuration disclosure fields are present and nonempty.                                                                                      |
| `drafter_binding`   | Checks speculative decoding head approval by model, repository, and revision, or checks approval lead time.                                                         |
| `duration`          | Checks runtime against the threshold for the concurrency band, using the configured precedence for window and whole-run durations.                                  |
| `field_constraints` | Checks that configured fields are absent or exactly equal to prescribed values, including model-specific constraints.                                               |
| `issuance`          | Checks that the load pattern is allowed and, when required, concurrency is positive.                                                                                |
| `membership`        | Checks that a value belongs to a configured catalog of allowed values.                                                                                              |
| `numeric_validity`  | Checks numeric type and configured requirements for presence, finiteness, and positivity.                                                                           |
| `offline`           | Checks required offline points, elected-offline concurrency, or throughput ordering relative to other points.                                                       |
| `path_resolution`   | Resolves shared source and documentation paths and checks existence, directory type, and allowed traversal or symlink behavior.                                     |
| `power`             | Checks system power descriptors and sources, reports estimated power, calculates per-point power from engaged nodes, or checks accelerator capacity and engagement. |
| `presence`          | Checks that required files, directories, declarations, or collection members exist.                                                                                 |
| `region_basis`      | Reports the minimum concurrency used to derive regions and checks that readable concurrency values are available.                                                   |
| `region_boundaries` | Checks the minimum and maximum concurrency requirements and that region boundaries were computed.                                                                   |
| `region_placement`  | Checks that a point lies in an allowed computed region or that its declared region matches.                                                                         |
| `report`            | Includes a resolved artifact value in the report without applying a pass/fail constraint.                                                                           |
| `seed_binding`      | Checks published seed-set membership, runtime seed values, adoption windows, or legacy seed field usage.                                                            |
| `steady_state`      | Checks status and verdict vocabulary, window consistency, or reporting basis; reports drift and unavailable official windows.                                       |
| `warmup`            | Checks that warmup salt is disabled or log retention is declared, with configured exemptions for disabled warmup.                                                   |

## Definitions

Rules are grouped by scope; each group lists its policy file once.

### Submission

Policy file: `submission_checks.yaml`.

| Rule ID                   | Evaluator kind      |
| ------------------------- | ------------------- |
| `accuracy-present`        | `accuracy_presence` |
| `path-exists`             | `presence`          |
| `required-dir`            | `presence`          |
| `src-dir`                 | `presence`          |
| `system-name-consistency` | `consistency`       |
| `system-results-dir`      | `presence`          |

### Implementation

Policy file: `submission_checks.yaml`.

| Rule ID      | Evaluator kind |
| ------------ | -------------- |
| `src-readme` | `presence`     |

### System

Policy file: `system_checks.yaml`.

| Rule ID                            | Evaluator kind |
| ---------------------------------- | -------------- |
| `benchmark-model-dir`              | `presence`     |
| `power-descriptor`                 | `power`        |
| `power-estimated`                  | `power`        |
| `system-description-across-curves` | `consistency`  |

### Pareto curve

Policy file: `curve_checks.yaml`.

| Rule ID                          | Evaluator kind      |
| -------------------------------- | ------------------- |
| `accuracy-coverage`              | `accuracy_coverage` |
| `accuracy-gate`                  | `accuracy_gate`     |
| `accuracy-sample-count`          | `accuracy_gate`     |
| `agentic-accuracy`               | `accuracy_gate`     |
| `agentic-accuracy-inline`        | `accuracy_gate`     |
| `agentic-accuracy-swebench`      | `accuracy_gate`     |
| `agentic-osl-range`              | `accuracy_gate`     |
| `benchmark-type-consistency`     | `consistency`       |
| `config-consistency-dataset`     | `consistency`       |
| `config-consistency-model`       | `consistency`       |
| `high-concurrency-coverage`      | `coverage`          |
| `low-concurrency-coverage`       | `coverage`          |
| `max-concurrency-declared`       | `report`            |
| `measurement-points-present`     | `presence`          |
| `med-concurrency-coverage`       | `coverage`          |
| `model-name-consistency`         | `consistency`       |
| `model-name-valid`               | `membership`        |
| `offline-ordering`               | `offline`           |
| `offline-point-present`          | `offline`           |
| `point-cap`                      | `collection_size`   |
| `point-count`                    | `collection_size`   |
| `point-dirs`                     | `presence`          |
| `region-basis`                   | `region_basis`      |
| `region-computation`             | `region_boundaries` |
| `seed-set-consistency`           | `consistency`       |
| `system-description-consistency` | `consistency`       |
| `tps-utilization`                | `derived_metric`    |
| `ultra-low-concurrency-coverage` | `coverage`          |

### Point

Policy file: `point_checks.yaml`.

| Rule ID                            | Evaluator kind      |
| ---------------------------------- | ------------------- |
| `accuracy-valid`                   | `artifact_schema`   |
| `agentic-metric-consistency`       | `derived_metric`    |
| `agentic-trajectory-count`         | `dataset_count`     |
| `approved-checkpoint`              | `artifact_binding`  |
| `approved-drafter`                 | `drafter_binding`   |
| `concurrency-in-range`             | `region_placement`  |
| `drafter-approval-lead-time`       | `drafter_binding`   |
| `endpoints-client-sha`             | `artifact_binding`  |
| `load-pattern`                     | `issuance`          |
| `maximal-engagement`               | `power`             |
| `metric-consistency-accounting`    | `comparison`        |
| `metric-consistency-duration`      | `comparison`        |
| `metric-consistency-output-tokens` | `comparison`        |
| `metric-consistency-system-tps`    | `derived_metric`    |
| `metric-consistency-tpot-p90`      | `numeric_validity`  |
| `metric-consistency-tps-per-kw`    | `derived_metric`    |
| `metric-consistency-tps-per-user`  | `derived_metric`    |
| `min-query-count`                  | `dataset_minimum`   |
| `nodes-used`                       | `power`             |
| `offline-declared`                 | `membership`        |
| `point-config-valid`               | `artifact_schema`   |
| `point-dirname-concurrency`        | `consistency`       |
| `point-disclosure-complete`        | `disclosure`        |
| `point-duration`                   | `duration`          |
| `point-power`                      | `power`             |
| `point-rules-skipped`              | `report`            |
| `region-declared`                  | `membership`        |
| `region-placement`                 | `region_placement`  |
| `result-file-valid`                | `artifact_schema`   |
| `result-summary-present`           | `presence`          |
| `seed-config-legacy`               | `seed_binding`      |
| `seed-runtime-match`               | `seed_binding`      |
| `seed-set-adoption`                | `seed_binding`      |
| `seed-set-membership`              | `seed_binding`      |
| `shared-path-resolution`           | `path_resolution`   |
| `steady-state-basis`               | `steady_state`      |
| `steady-state-consistency`         | `steady_state`      |
| `steady-state-valid`               | `steady_state`      |
| `streaming-config`                 | `report`            |
| `submission-flags`                 | `field_constraints` |
| `swebench-instance-count`          | `dataset_count`     |
| `swebench-template`                | `field_constraints` |
| `system-description-present`       | `presence`          |
| `system-description-valid`         | `artifact_schema`   |
| `target-cohort`                    | `cohort_identifier` |
| `warmup-logs-retained`             | `warmup`            |
| `warmup-present`                   | `presence`          |
| `warmup-salt`                      | `warmup`            |

### Validator catalogs

Policy file: `catalog.yaml`.

| Rule ID                 | Evaluator kind      |
| ----------------------- | ------------------- |
| `drafter-list-registry` | `catalog_integrity` |
| `seed-set-registry`     | `catalog_integrity` |
