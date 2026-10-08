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

- A rule's YAML `kind` selects a callable evaluator class; its other fields supply
  evidence addresses, catalogs, thresholds, and requirements.
- Each kind below has its own registry entry. The groups organize related checks;
  they do not represent class inheritance.
- Nested operations under a kind share its evaluator class. For example, client
  SHA and checkpoint approval both use `artifact_binding`.
- The planner selects applicable checks and verifies prerequisites before
  evaluators return findings for the report.

- **Artifact structure and required evidence**
  - `presence`: requires files, directories, declarations, or collection members.
  - `artifact_schema`: requires successful typed parsing and, when configured,
    nonempty contents.
  - `disclosure`: requires nonempty configuration fields.
  - `path_resolution`: resolves shared source and documentation paths and checks
    existence, directory type, and allowed traversal or symlink behavior.
  - `catalog_integrity`: requires a successfully loaded seed or speculative-head
    catalog.
- **Values and collections**
  - `membership`: checks a value against an allowed catalog.
  - `field_constraints`: requires fields to be absent or exactly equal to
    prescribed values, including model-specific constraints.
  - `numeric_validity`: checks numeric type, presence, finiteness, and positivity.
  - `comparison`: compares artifact values or constants using equality,
    greater-than, or greater-than-or-equal.
  - `consistency`: checks equality between two values or across a collection of
    declarations, optionally excluding specified fields.
  - `collection_size`: checks the number of members against configured limits.
  - `report`: includes a resolved value in the report without a pass/fail
    constraint.
- **Run classification and Pareto curves**
  - `cohort_identifier`: checks the cohort string's year, month, and C0/C1 format.
  - `issuance`: checks allowed load patterns and, when required, positive
    concurrency.
  - `region_basis`: reports the minimum concurrency used to derive regions and
    requires readable concurrency values.
  - `region_boundaries`: checks minimum and maximum concurrency requirements and
    that region boundaries were computed.
  - `region_placement`:
    - Checks that a point lies within an allowed computed region.
    - Checks that the declared region matches the computed region.
  - `coverage`: counts performance points in a concurrency band and checks the
    required minimum.
  - `offline`:
    - Checks required offline points and elected-offline concurrency.
    - Checks offline throughput ordering relative to other points.
- **Datasets and accuracy**
  - `dataset_minimum`: compares completed samples with the dataset's threshold.
  - `dataset_count`: requires the dataset's exact sample count or a positive
    multiple of it.
  - `accuracy_presence`: requires accuracy results in enough model curves.
  - `accuracy_coverage`: requires accuracy results across concurrency bands and
    offline points.
  - `accuracy_gate`:
    - Checks that an agentic accuracy profile is available.
    - Compares single-turn scores and issued accuracy sample counts with model
      requirements.
    - Compares agentic scores per point or as a mean of concurrency-band means.
    - Checks output-length statistics against a model-specific range.
- **Approved artifacts and seeds**
  - `artifact_binding`:
    - Matches the submitted client SHA against the approved list.
    - Matches checkpoint repository and revision against the model's approvals.
  - `drafter_binding`:
    - Matches a speculative decoding head by model, repository, and revision.
    - Checks its approval lead time in cohorts.
  - `seed_binding`:
    - Checks published seed-set membership.
    - Compares runtime seed values with the published set.
    - Checks the seed set's adoption window.
    - Reports legacy seed field usage.
- **Measurements and runtime**
  - `duration`: compares runtime with the concurrency band's threshold, using the
    configured precedence for window and whole-run durations.
  - `derived_metric`: recalculates a metric and compares it with the reported
    value using a tolerance. Operations cover throughput, interactivity,
    utilization, and throughput per kW.
  - `steady_state`:
    - Checks status and verdict vocabulary.
    - Checks window and reported super-pass consistency.
    - Checks reporting basis and reports drift or unavailable official windows.
  - `warmup`:
    - Requires disabled warmup salt.
    - Checks declared log retention, with configured exemptions for disabled
      warmup.
- **Power**
  - `power`:
    - Checks system power descriptors and sources.
    - Reports estimated power that uses component defaults.
    - Calculates per-point power from engaged nodes.
    - Checks declared accelerator capacity and maximal engagement.

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
