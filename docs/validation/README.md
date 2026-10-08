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

## Definitions

| Rule ID                            | Scope          | Evaluator kind      | Policy file              |
| ---------------------------------- | -------------- | ------------------- | ------------------------ |
| `accuracy-coverage`                | curve          | `accuracy_coverage` | `curve_checks.yaml`      |
| `accuracy-gate`                    | curve          | `accuracy_gate`     | `curve_checks.yaml`      |
| `accuracy-present`                 | submission     | `accuracy_presence` | `submission_checks.yaml` |
| `accuracy-sample-count`            | curve          | `accuracy_gate`     | `curve_checks.yaml`      |
| `accuracy-valid`                   | point          | `artifact_schema`   | `point_checks.yaml`      |
| `agentic-accuracy`                 | curve          | `accuracy_gate`     | `curve_checks.yaml`      |
| `agentic-accuracy-inline`          | curve          | `accuracy_gate`     | `curve_checks.yaml`      |
| `agentic-accuracy-swebench`        | curve          | `accuracy_gate`     | `curve_checks.yaml`      |
| `agentic-metric-consistency`       | point          | `derived_metric`    | `point_checks.yaml`      |
| `agentic-osl-range`                | curve          | `accuracy_gate`     | `curve_checks.yaml`      |
| `agentic-trajectory-count`         | point          | `dataset_count`     | `point_checks.yaml`      |
| `approved-checkpoint`              | point          | `artifact_binding`  | `point_checks.yaml`      |
| `approved-drafter`                 | point          | `drafter_binding`   | `point_checks.yaml`      |
| `benchmark-model-dir`              | system         | `presence`          | `system_checks.yaml`     |
| `benchmark-type-consistency`       | curve          | `consistency`       | `curve_checks.yaml`      |
| `concurrency-in-range`             | point          | `region_placement`  | `point_checks.yaml`      |
| `config-consistency-dataset`       | curve          | `consistency`       | `curve_checks.yaml`      |
| `config-consistency-model`         | curve          | `consistency`       | `curve_checks.yaml`      |
| `drafter-approval-lead-time`       | point          | `drafter_binding`   | `point_checks.yaml`      |
| `drafter-list-registry`            | validator      | `catalog_integrity` | `catalog.yaml`           |
| `endpoints-client-sha`             | point          | `artifact_binding`  | `point_checks.yaml`      |
| `high-concurrency-coverage`        | curve          | `coverage`          | `curve_checks.yaml`      |
| `load-pattern`                     | point          | `issuance`          | `point_checks.yaml`      |
| `low-concurrency-coverage`         | curve          | `coverage`          | `curve_checks.yaml`      |
| `max-concurrency-declared`         | curve          | `report`            | `curve_checks.yaml`      |
| `maximal-engagement`               | point          | `power`             | `point_checks.yaml`      |
| `measurement-points-present`       | curve          | `presence`          | `curve_checks.yaml`      |
| `med-concurrency-coverage`         | curve          | `coverage`          | `curve_checks.yaml`      |
| `metric-consistency-accounting`    | point          | `comparison`        | `point_checks.yaml`      |
| `metric-consistency-duration`      | point          | `comparison`        | `point_checks.yaml`      |
| `metric-consistency-output-tokens` | point          | `comparison`        | `point_checks.yaml`      |
| `metric-consistency-system-tps`    | point          | `derived_metric`    | `point_checks.yaml`      |
| `metric-consistency-tpot-p90`      | point          | `numeric_validity`  | `point_checks.yaml`      |
| `metric-consistency-tps-per-kw`    | point          | `derived_metric`    | `point_checks.yaml`      |
| `metric-consistency-tps-per-user`  | point          | `derived_metric`    | `point_checks.yaml`      |
| `min-query-count`                  | point          | `dataset_minimum`   | `point_checks.yaml`      |
| `model-name-consistency`           | curve          | `consistency`       | `curve_checks.yaml`      |
| `model-name-valid`                 | curve          | `membership`        | `curve_checks.yaml`      |
| `nodes-used`                       | point          | `power`             | `point_checks.yaml`      |
| `offline-declared`                 | point          | `membership`        | `point_checks.yaml`      |
| `offline-ordering`                 | curve          | `offline`           | `curve_checks.yaml`      |
| `offline-point-present`            | curve          | `offline`           | `curve_checks.yaml`      |
| `path-exists`                      | submission     | `presence`          | `submission_checks.yaml` |
| `point-cap`                        | curve          | `collection_size`   | `curve_checks.yaml`      |
| `point-config-valid`               | point          | `artifact_schema`   | `point_checks.yaml`      |
| `point-count`                      | curve          | `collection_size`   | `curve_checks.yaml`      |
| `point-dirname-concurrency`        | point          | `consistency`       | `point_checks.yaml`      |
| `point-dirs`                       | curve          | `presence`          | `curve_checks.yaml`      |
| `point-disclosure-complete`        | point          | `disclosure`        | `point_checks.yaml`      |
| `point-duration`                   | point          | `duration`          | `point_checks.yaml`      |
| `point-power`                      | point          | `power`             | `point_checks.yaml`      |
| `point-rules-skipped`              | point          | `report`            | `point_checks.yaml`      |
| `power-descriptor`                 | system         | `power`             | `system_checks.yaml`     |
| `power-estimated`                  | system         | `power`             | `system_checks.yaml`     |
| `region-basis`                     | curve          | `region_basis`      | `curve_checks.yaml`      |
| `region-computation`               | curve          | `region_boundaries` | `curve_checks.yaml`      |
| `region-declared`                  | point          | `membership`        | `point_checks.yaml`      |
| `region-placement`                 | point          | `region_placement`  | `point_checks.yaml`      |
| `required-dir`                     | submission     | `presence`          | `submission_checks.yaml` |
| `result-file-valid`                | point          | `artifact_schema`   | `point_checks.yaml`      |
| `result-summary-present`           | point          | `presence`          | `point_checks.yaml`      |
| `seed-config-legacy`               | point          | `seed_binding`      | `point_checks.yaml`      |
| `seed-runtime-match`               | point          | `seed_binding`      | `point_checks.yaml`      |
| `seed-set-adoption`                | point          | `seed_binding`      | `point_checks.yaml`      |
| `seed-set-consistency`             | curve          | `consistency`       | `curve_checks.yaml`      |
| `seed-set-membership`              | point          | `seed_binding`      | `point_checks.yaml`      |
| `seed-set-registry`                | validator      | `catalog_integrity` | `catalog.yaml`           |
| `shared-path-resolution`           | point          | `path_resolution`   | `point_checks.yaml`      |
| `src-dir`                          | submission     | `presence`          | `submission_checks.yaml` |
| `src-readme`                       | implementation | `presence`          | `submission_checks.yaml` |
| `steady-state-basis`               | point          | `steady_state`      | `point_checks.yaml`      |
| `steady-state-consistency`         | point          | `steady_state`      | `point_checks.yaml`      |
| `steady-state-valid`               | point          | `steady_state`      | `point_checks.yaml`      |
| `streaming-config`                 | point          | `report`            | `point_checks.yaml`      |
| `submission-flags`                 | point          | `field_constraints` | `point_checks.yaml`      |
| `swebench-instance-count`          | point          | `dataset_count`     | `point_checks.yaml`      |
| `swebench-template`                | point          | `field_constraints` | `point_checks.yaml`      |
| `system-description-across-curves` | system         | `consistency`       | `system_checks.yaml`     |
| `system-description-consistency`   | curve          | `consistency`       | `curve_checks.yaml`      |
| `system-description-present`       | point          | `presence`          | `point_checks.yaml`      |
| `system-description-valid`         | point          | `artifact_schema`   | `point_checks.yaml`      |
| `system-name-consistency`          | submission     | `consistency`       | `submission_checks.yaml` |
| `system-results-dir`               | submission     | `presence`          | `submission_checks.yaml` |
| `target-cohort`                    | point          | `cohort_identifier` | `point_checks.yaml`      |
| `tps-utilization`                  | curve          | `derived_metric`    | `curve_checks.yaml`      |
| `ultra-low-concurrency-coverage`   | curve          | `coverage`          | `curve_checks.yaml`      |
| `warmup-logs-retained`             | point          | `warmup`            | `point_checks.yaml`      |
| `warmup-present`                   | point          | `presence`          | `point_checks.yaml`      |
| `warmup-salt`                      | point          | `warmup`            | `point_checks.yaml`      |
