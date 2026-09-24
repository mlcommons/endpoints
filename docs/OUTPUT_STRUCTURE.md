# Run Output Structure

What a benchmark run writes to its report directory. The focus is the **combined run**
(`--mode both`) — one process that measures performance _and_ scores accuracy into a single
directory.

For the shape of the perf report itself see [metrics/report_design.md](metrics/report_design.md);
for the compliance-audit subtree see [compliance_audit_plan.md](compliance_audit_plan.md).

## Where the output goes

Point a run at a directory of your choosing with the top-level `report_dir` field in the
benchmark config file:

```yaml
report_dir: results/gptoss_120b_offline
```

Every benchmark subcommand also takes `--report-dir` on the command line, including
`benchmark from-config`, where it overrides the config file's value. If neither is set, the run
writes to a timestamped directory under the system temp dir: `$TMPDIR/reports_<YYYYMMDD_HHMMSS>`.

| Set via                            | Result                       |
| ---------------------------------- | ---------------------------- |
| `report_dir:` in the config file   | as given                     |
| `--report-dir` on the command line | as given; wins over the file |
| neither                            | timestamped temp directory   |

The directory is resolved once, before any work starts, created during setup, and `config.yaml` is
written into it immediately — so it exists even if the run fails early. Artifacts accumulate there
as the run progresses and are **salvaged on interrupt**: a Ctrl-C'd or timed-out run still leaves
`events.jsonl`, `report.txt`, and a `result_summary.json` marked `state: "interrupted"`,
`complete: false`.

## Layout of a combined run

```
<report_dir>/
├── config.yaml                     # resolved config actually used, secrets redacted
├── report.txt                      # human-readable report (perf + accuracy headline)
├── events.jsonl                    # per-sample event log, all non-warmup phases
├── sample_idx_map.json             # {phase/dataset name: {sample_uuid: sample_index}}
├── performance/
│   └── result_summary.json         # machine-readable perf report (performance phase only)
├── accuracy/
│   └── accuracy_results.json       # per-dataset scores + weighted average
├── metrics/
│   ├── .ready                      # aggregator startup marker (touched once, at startup)
│   └── final_snapshot.json         # terminal metrics snapshot; the source of the perf report
└── profiling.json                  # only when settings.profiling.engine is set
```

Scorer-specific and audit artifacts add to this tree; see [Scorer-specific
artifacts](#scorer-specific-artifacts) and [Compliance audit subtree](#compliance-audit-subtree).

## Scorer-specific artifacts

Some scorers write their own files into `report_dir` in addition to the standard set:

| Path                                                         | Produced by                                          |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| `scores.json`                                                | agentic inline scorer (per-turn / per-domain detail) |
| `per_entry_scores.json`                                      | BFCL v4 multi-turn CLI                               |
| `swe_bench_results.json`, `swe_bench_runs/<run_id>/`         | SWE-bench scorer                                     |
| `vbench_videos/`, `vbench_results/`, `vbench_subprocess.log` | VBench (video-gen) scorer                            |
| `deepseek_eval/`, `deepseek_eval_subprocess.log`             | legacy MLPerf DeepSeek-R1 scorer                     |

## Compliance audit subtree

When the config carries an `audit:` block, the audit runs after the main benchmark and shares the
same `report_dir`, nesting each of its phases as a complete run directory of its own:

```
<report_dir>/
├── …                                        # the main run's artifacts, as above
└── audit/
    ├── audit_<test_id>.json                 # e.g. audit_output_caching_test.json
    ├── verify_<TEST_ID>.txt                 # "Performance check pass: true/false"
    ├── reference/                           # full run dir for the reference phase
    └── output_caching/                      # full run dir for the audit phase
```

## Optional plots

Plots are not produced by a run. Generate them afterwards from a finished report directory:

```bash
python scripts/plot_results.py <report_dir>        # writes <report_dir>/plots/*.png
```

It reads `accuracy/accuracy_results.json`, `scores.json`, and
`performance/result_summary.json`, and skips any distribution the run did not record.

## Downstream consumers

| Tool                                             | Reads                                                                              |
| ------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `scripts/check_compliance.py`                    | `config.yaml`, `accuracy/accuracy_results.json` / `scores.json`                    |
| `scripts/publish_submission.py`                  | `performance/result_summary.json`, `accuracy/accuracy_results.json`, `config.yaml` |
| `scripts/plot_results.py`                        | as above                                                                           |
| `scripts/early_stopping_estimate_from_events.py` | `events.jsonl`                                                                     |
