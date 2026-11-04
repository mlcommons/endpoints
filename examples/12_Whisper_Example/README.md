# MLPerf speech2text — whisper-large-v3 on LibriSpeech

Benchmarks an OpenAI-compatible transcription endpoint
(`POST /v1/audio/transcriptions`) against the MLPerf
[speech2text](https://github.com/mlcommons/inference/tree/master/speech2text)
workload and scores Word Error Rate.

|                  |                                                                             |
| ---------------- | --------------------------------------------------------------------------- |
| Model            | `openai/whisper-large-v3`                                                   |
| Dataset          | LibriSpeech dev-clean + dev-other, repackaged to ~30 s clips (1633 samples) |
| Metric           | WER (reported as `100 - WER%`)                                              |
| Reference        | WER 2.0671% → accuracy **97.9329%**                                         |
| Submission floor | 99% of reference → **96.9536**                                              |
| Scenario         | Offline (`max_throughput`)                                                  |

## 1. Serve the model

```bash
vllm serve openai/whisper-large-v3 --port 8000
```

Whisper is an encoder-decoder ASR model, so it is served only on
`/v1/audio/transcriptions` — not `/v1/chat/completions`.

## 2. The dataset

Nothing to do: the first run downloads the MLCommons-prepared corpus into
`dataset_cache/whisper_librispeech/source/` (several GB of audio) and caches the
parsed rows as a parquet beside it. The ~30 s repackaging is part of the
benchmark definition, so the prepared artifact is fetched rather than rebuilt
here: different clip boundaries would produce a WER that is not comparable to
the reference.

```
<source>/dev-all-repack.json      # manifest: files[0].fname, original_duration, transcript
<source>/dev-all-repack/*.wav     # 16 kHz mono clips
```

The manifest's `fname` entries keep the upstream pipeline's `./data/` prefix,
which the hosted artifact does not reproduce. The loader probes the first clip
and drops any leading components that aren't present under the root, so both
layouts resolve with no configuration.

To use a copy you already have, set `datasets[*].generate_params.manifest` or
`$WHISPER_LIBRISPEECH_MANIFEST`; either skips the download.
`generate_params.data_dir` overrides the root that `fname` resolves against; it
defaults to the manifest's own directory.

The parsed-row cache is keyed on the dataset ID alone, so if you change
`manifest` or `data_dir` after a first run, also pass
`generate_params.force: true` once — otherwise the rows built from the previous
manifest are reused.

Keep the WAVs on **fast local storage**. The worker reads each file inside the
measured request, and that read is synchronous on its event loop: a slow read
stalls every other in-flight request in the same worker process, so a network
mount or cold cache inflates latency beyond the clip that caused it.

## 3. Run

Edit `endpoint_config.endpoints` first. The config issues the full 1633-sample
corpus in both the performance and accuracy phases, which is what a submission
needs.

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/12_Whisper_Example/offline_whisper.yaml
```

To check the wiring first, set `runtime.n_samples_to_issue` and both `samples`
values to 8. The dataset truncates in manifest order, so the smoke subset is a
prefix of the full corpus — but a partial perf phase is not a reportable result.

## 4. Read the result

`<report_dir>/accuracy/accuracy_results.json`:

```json
{
  "score": 97.93,
  "complete": true,
  "breakdown": {
    "wer_percent": 2.07,
    "edit_distance": 1241,
    "reference_words": 60000,
    "empty_responses": 0,
    "missing_samples": 0
  }
}
```

A submission needs `complete: true` and `missing_samples: 0`. An issued sample
that never produced a response is scored as an empty transcript — its reference
words count in the denominator and all of them count as errors — so a partial
run shows up as a depressed score rather than as a flattering one, and
`complete` goes `false`. `empty_responses` is the separate case of a sample that
did answer, with nothing in it.

A WER far above 2.07% usually points at the dataset rather than the endpoint —
most often a manifest that is not the repackaged `dev-all-repack.json`.

## How it maps onto the framework

| Piece                    | Where                                                                  |
| ------------------------ | ---------------------------------------------------------------------- |
| Route + multipart upload | `openai/transcriptions_adapter.py` (`api_type: openai_transcriptions`) |
| Dataset                  | `dataset_manager/predefined/whisper_librispeech/`                      |
| Metric                   | `evaluation/wer.py` + `WERScorer` (`eval_method: wer`)                 |
| Reference numbers        | `config/rulesets/mlcommons/models.py` (`Whisper_large_v3`)             |

Notes on the run config. All three constraints below are enforced by config
validation keyed on `api_type: openai_transcriptions`, so a hand-written config
gets an error rather than a silently odd run:

- **Non-streaming.** OpenAI does not support `stream` for Whisper-class models.
  `streaming: on` is rejected and `auto` resolves to `off` even in online mode.
  There is no TTFT/TPOT; the metrics are QPS and per-request latency.
- **`metrics_isl` is forced off.** Audio rows carry no text prompt, so there is
  no input sequence length. One "ISL is unavailable" warning per run is
  expected.
- **Warmup salt is rejected.** Salting prepends a random hex prefix to a text
  prompt to defeat prefix caching; an audio upload has no prompt, and its cache
  key is the audio.
- **`language: en`** is sent on every request (the adapter's `DEFAULT_LANGUAGE`;
  a dataset `language` column overrides it). It makes the server build Whisper's
  `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>` prompt instead of
  auto-detecting the language per clip, which is what the MLPerf reference does.

## Reference numbers

The reference accuracy, the 0.99 factor and the 1633-sample count are registered
as reference data in `config/rulesets/mlcommons/models.py` (`Whisper_large_v3`)
and `datasets.py` (`LibriSpeechDevAll`), so a `submission_ref` run resolves them
from the ruleset instead of the numbers living only in this file. They
transcribe the MLPerf Inference submission checker's own constants, where the
gate is `"whisper": ("ACCURACY", (100.0 - 2.0671) * 0.99)` — the same
`100 - WER` scale `WERScorer` reports on.

The in-repo `compliance/checker.py` does **not** cover this workload: its
accuracy gate resolves BFCL metric keys only, and its config-lock expects a
single-stream run (`num_workers: 1`), which this example is not. Checking a
whisper submission against the gate is the external MLPerf Inference submission
checker's job.

### Expect a score a little above the reference

A healthy run scores slightly better than 97.9329%, not equal to it. The scoring
is bit-exact with upstream — `evaluation/wer.py` reproduces
`compute_wer_with_concatenation` from the MLCommons reference
(https://github.com/mlcommons/inference/tree/master/speech2text) with zero
per-sample disagreement over the full 1633-clip corpus, and the two candidate
`EnglishTextNormalizer` implementations (`transformers` and `openai-whisper`)
give the same corpus WER. The difference is the serving path, and it is inherent
to benchmarking an endpoint:

|                | MLPerf reference SUT                                                         | This example                                           |
| -------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------ |
| Engine         | vLLM V0, offline `LLM.generate()`                                            | any OpenAI-compatible server, over HTTP                |
| Decoder prompt | explicit `<\|startoftranscript\|><\|en\|><\|transcribe\|><\|notimestamps\|>` | built by the server's `/v1/audio/transcriptions` layer |
| Audio decode   | `librosa.load(path, sr=16000)` in-process                                    | server-side, from the multipart upload                 |
| Sampling       | `temperature=0, top_p=1.0, max_tokens=200`                                   | `temperature=0`, length left to the server             |

That reference's own SUT pins `dtype = "bfloat16"`, which is why
`Whisper_large_v3.golden_accuracy` is registered as `bfloat16` rather than fp32.
Serving fp32 instead moves the score by well under 0.1 points, so a gap of that
size is engine variance rather than a scoring fault. `max_tokens=200` never
binds in practice — observed output lengths top out near 150 tokens.
