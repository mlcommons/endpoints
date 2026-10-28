# Benchmark Suite (status:proposed)

The Benchmark Suite is a Git-versioned record of reusable models, datasets, and the complete contract for each benchmark. A Ruleset is the approved grouping published for one cohort: benchmark IDs, a benchmark suite version, and seed sets. A submission names a cohort, seed set and benchmark, and the checker verifies the run against that benchmark contract.

## Definitions

- Benchmark suite - the git-versioned collection of models, datasets, and benchmarks available in the repository. The suite provides specifications for entire benchmarks (models, datasets, requirements etc), but can also be used for non-submission benchmarking. It is meant as a repository for supported items - everything in the suite at any given time should be supported by the client to conduct benchmarking runs.
- Cohort - a logical collection of submissions made during a certain duration. A cohort is a submission round defined by an identifier such as `2026-09-C0` . The current proposal is to have two cohorts per month, so `C0` and `C1` are the two cohorts for each month.
- Ruleset - defines the valid submissions for a cohort. Each ruleset is tied to a cohort and defines the benchmarks that are valid for that cohort. While datasets, models, benchmark updates can happen at any time, the inclusion in a ruleset for a cohort signals that the update is now a valid submission via rolling submission. A ruleset also includes other cohort specific items such as the seed sets that are valid for submissions in that cohort.
- Phase - each benchmark execution is defined as a series of phases. Currently they can be divided into four groups:
  - Pre-processing - executed to generate the complete configuration files, as well as perform any sanity checks (for instance current repo matches the SHA for the cohort tag ).
  - Performance - measured execution of the performance dataset with the user specified concurrency.
  - Accuracy - correctness checks performed over the accuracy dataset and scored by the designated scorer and aggregated to component scores. Each accuracy dataset can be run multiple times (repeats) and there can be different ways of aggregating the scores from different runs (best of K or average over K). As there can be multiple accuracy datasets, we can also specify aggregation across accuracy datasets (unweighted average), or alternatively leave the scores per dataset separate.
  - Post-processing - performs integrity checks on the datasets (SHA verification), number of samples issued, model checkpoints. It will also ensure that the accuracy scores satisfies the gates specified in the benchmark specification. There can also be requirements on the performance phase such as minimum duration, minimum number of samples issued, time-to-first-token constraints etc.

## Submitter file

The submitter will use a minimalist file to specify the submission:

```yaml
config.yaml   # submission configuration; replaces today's submission_ref{model, ruleset}
  type: submission
  submission_ref:
    cohort:  2026-09-C0  # frozen selection (YYYY-MM-C0/C1, submission rules §4.2)
    seed_set_id: B
    benchmark:     gpt-oss-120b
    # TODO - add path overrides.
  endpoint_config:
    endpoints: [ https://my-host/v1 ]
    api_type:  openai           # must equal the benchmark's locked api_type
    # response_mode comes from benchmark.execution
  settings:
    load_pattern: # submission-owned; concurrency is the only supported online type
      type: concurrency
      target_concurrency: 64 # one Pareto point; re-run across values to trace the curve
```

### Folder structure:

The `type: submission` entry in the yaml file above directs the front-end to load the submission details from the benchmark suite which is stored in the following folder structure:

```
benchmark_suite/
├── suite.yaml             version; the loader globs the three directories below
├── models/<id>.yaml       id, reference_id, tokenizer?, processor?, chat_template?,
│                          metadata{}
├── datasets/<id>.yaml     id, description, samples, source{url, sha256},
│                          prompts{}, ground_truth_or_tests{}, evaluation_recipes{}?,
│                          metadata{}
└── benchmarks/<id>.yaml   id, family, model, api_type,
                           execution{}       # response_mode - streaming vs non-streaming
                           preprocessing{}   # input transform + optional validation hooks
                           performance{}     # datasets, generation, runtime/sample/latency
                                             # constraints
                           accuracy{}        # datasets, recipes, generation,
                                             # repeat_aggregation
                                             # dataset_aggregation, thresholds
                           postprocessing{}  # report, integrity, and compliance gates
rulesets/<cohort>.yaml
├── cohort, benchmark_suite_version
├── seed_sets[]    # id, scheduler_rng_seed, sample_index_rng_seed, model_seed
└── benchmarks[]   # complete benchmark IDs; no accuracy overrides


```

## Sample entries:

### Cohort specification(rulesets/2026-09-C0.yaml):

```yaml
rulesets/2026-09-C0.yaml
  cohort: 2026-09-C0                      # submission round; YYYY-MM-C0 (1st Wed) / -C1 (3rd Wed)
  benchmark_suite_version: "1.0"

  seed_sets:                              # a submission selects exactly one, by id
    - id: A
      scheduler_rng_seed:    16159082839903944936
      sample_index_rng_seed: 2747215439041700203
      model_seed:            42
    - id: B
      scheduler_rng_seed:    <generated at ruleset publication>
      sample_index_rng_seed: <generated at ruleset publication>
      model_seed:            <generated at ruleset publication>

  benchmarks:
    - deepseek-r1
    - gpt-oss-120b

```

## DeepSeek-R1

### Dataset:

```yaml
datasets:
  - id: mlperf_deepseek_r1
    description: >-
      Custom dataset curated by MLCommons for DeepSeek R1, specifically for the
      MLPerf Inference benchmark
    samples: 4388
    source:
      # provenance; not fetch-enforced
      url: https://<public-host>/mlperf_deepseek_r1_dataset_4388_fp8_eval.jsonl
      # verified against bytes loaded
      sha256: <64 hex>
    prompts: { column: question }
    ground_truth_or_tests: { column: ground_truth }
    evaluation_recipes:
      mlperf_exact_match:
        scorer: legacy_mlperf_deepseek_r1
        ground_truth: ground_truth
```

### Model

```yaml
models:
  - id: deepseek-r1
    reference_id: deepseek-ai/DeepSeek-R1-0528
```

### Benchmark:

```yaml
benchmarks:
  - id: deepseek-r1
    family: text
    model: deepseek-r1
    api_type: openai
    execution: { response_mode: streaming } # benchmark-wide; applies to every phase

    preprocessing:
      input_transform: chat_messages

    performance:
      datasets: [{ dataset: mlperf_deepseek_r1 }]
      generation:
        temperature: 0.0
        top_p: 1.0
        top_k: 1
        max_new_tokens: 20000
      requirements:
        min_duration_ms: 600000
        min_sample_count:
          <TBD> # authoritative online value;
          # freeze before the Ruleset is published

    accuracy:
      datasets:
        - dataset: mlperf_deepseek_r1
          evaluation_recipe: mlperf_exact_match
          num_repeats: 1
      generation:
        temperature: 0.0
        top_p: 1.0
        top_k: 1
        max_new_tokens: 20000
      dataset_aggregation: { method: single_dataset }
      thresholds:
        exact_match: { reference: 81.3582, lower_factor: 0.99 }
        TOKENS_PER_SAMPLE:
          { reference: 3886.2274, lower_factor: 0.9, upper_factor: 1.1 }

    postprocessing:
      gates:
        [
          dataset_integrity,
          workload_integrity,
          accuracy_gate,
          requirements_lock,
        ]
```

## GPT-OSS-120b full specification

```yaml
# MODEL
models:
  - id: gpt-oss-120b
    reference_id: openai/gpt-oss-120b

# DATASETS
datasets:
  - id: aime25
    description: AIME 2025 (AIME2025-I + AIME2025-II, test split)
    samples: 30
    source: { url: <TBD>, sha256: <TBD> }
    prompts: { column: question }
    ground_truth_or_tests: { column: answer }
    evaluation_recipes:
      mlperf_exact_match:
        {
          scorer: pass_at_1,
          extractor: boxed_math_extractor,
          ground_truth: answer,
        }

  - id: gpqa
    description: GPQA Diamond
    samples: 198
    source: { url: <TBD>, sha256: <TBD> }
    prompts: { column: question }
    ground_truth_or_tests: { column: ground_truth }
    evaluation_recipes:
      mlperf_exact_match:
        {
          scorer: pass_at_1,
          extractor: abcd_extractor,
          ground_truth: ground_truth,
        }

  - id: livecodebench
    description: LiveCodeBench release_v6
    samples: 1055
    source: { url: <TBD>, sha256: <TBD> }
    prompts: { column: question }
    ground_truth_or_tests: { test_cases: test }
    evaluation_recipes:
      mlperf_code:
        { scorer: code_bench_scorer, extractor: python_code_extractor }

  - id: gpt_oss_perf
    description: <TBD>
    samples: 6396
    source: { url: <TBD>, sha256: <TBD> }
    prompts: { column: input }

# BENCHMARK
benchmarks:
  - id: gpt-oss-120b
    family: text
    model: gpt-oss-120b
    api_type: openai
    execution: { response_mode: streaming } # benchmark-wide; applies to every phase

    preprocessing:
      input_transform: chat_messages

    performance:
      datasets: [{ dataset: gpt_oss_perf }]
      generation:
        temperature: 1.0
        top_p: 1.0
        top_k: 0
        reasoning_effort: low
        max_new_tokens: 10240
      requirements:
        min_duration_ms: 600000
        min_sample_count: 6396
        min_context_length: 65536
        max_ttft_latency_ms: 3000
        max_tpot_latency_ms: 80

    accuracy:
      datasets:
        - dataset: aime25
          evaluation_recipe: mlperf_exact_match
          generation: # sparse merge over the phase generation block
            reasoning_effort: high
            max_new_tokens: 32768
          num_repeats: 8
          repeat_aggregation: { method: mean_correctness } # illustrative; WG freezes
        - dataset: gpqa
          evaluation_recipe: mlperf_exact_match
          generation:
            reasoning_effort: high
            max_new_tokens: 32768
          num_repeats: 5
          repeat_aggregation: { method: mean_correctness }
        - dataset: livecodebench
          evaluation_recipe: mlperf_code
          generation:
            reasoning_effort: high
            max_new_tokens: 32768
          num_repeats: 3
          repeat_aggregation: { method: mean_correctness }
      dataset_aggregation:
        method: micro_average
        weighting: base_questions
        components: [aime25, gpqa, livecodebench]
      thresholds:
        exact_match:
          scale: percentage
          reference: 83.13
          lower_factor: 0.99 # resolved threshold: 82.2987 (display 82.30)
      requirements:
        min_accuracy_sample_count: 4395

    postprocessing:
      gates:
        [
          dataset_integrity,
          workload_integrity,
          accuracy_gate,
          requirements_lock,
        ]
```

## Submission flow:

1. Submitter writes a config yaml file that identifies the cohort, benchmark, seed and optional overrides (model paths, dataset paths etc) as well as the client specific settings (endpoint urls, worker configs, network configs, api keys etc).
2. The front end will perform initial validation on the submission metadata to check valid cohort, benchmark membership, and verify repository versions match the remote tags for that cohort.
3. Once validated, the specification is inlined into the yaml file to reveal an expanded configuration file which specifies exactly what needs to be executed. This ensures that the benchmark execution is correct by construction. The generated yaml file will be saved as part of the output artifacts.
4. The generated yaml is passed on to the execution phases which will run the phases in order and produce associated artifacts. For instance, the dataloader will produce an implementation specific SHA for the data loaded which will be logged.
5. Towards the end of the execution of the benchmark phases, post-processing step will apply the checks needed to verify submission integrity:
   a) Dataset integrity - verify that the dataset loaded include a SHA from the dataloader and that matches the reference SHA. This requires the dataloader to output a SHA for each dataset loaded that would be format dependent. 
   b) Model integrity - verify that the model id corresponds to the same model as specified in the benchmark suite. This should not be a hard error as we might have quantized models here. If this pass fails, we can flag it as a warning and advise the submitter to submit a document explaining the difference. (TODO - need a reliable way of determining the version of the model, computing the SHA is too expensive for large models.) For huggingface models, we can use the git SHA for the model root as an identifier.
   c) Generation integrity - ensure that for each dataset, the generation config used in the config file is compliant with the benchmark specification. This includes api-type, generation parameters (top k, top p, max output sequence length, reasoning effort etc).
   d) Accuracy gating - verify that the accuracy setup conforms to the benchmark specification including the aggregation methods. Compare the reported scores(s) to the benchmark specification.
   e) Requirement gating - verify that the run satisfies the run requirements for the benchmark. This includes the duration requirements, sample counts, metrics (TPOT/TTFT) availability.

## Next steps

1. Address review comments.
2. Implementation plan (schema design + POC).
3. Reference implementations (DeepSeek-R1, GPT-OSS-120b, Llama3.1-8b).
4. Submission checker testing.
5. Adoption of submission process.
