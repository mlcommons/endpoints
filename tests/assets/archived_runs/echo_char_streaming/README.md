# Archived run: echo server, `char` tokenizer, streaming

Frozen report directory from one real run of commit `93ec321e`, before the
benchmark read server-reported token counts. Every token count in it comes from
the client tokenizer, and no record carries `usage`.
`tests/unit/metrics/test_archived_run_compat.py` checks that current code still
decodes these files and recomputes the numbers this run reported. Do not
regenerate or edit them: the test pins values from these exact files.

Kept from the report directory: `events.jsonl`, `sample_idx_map.json`,
`metrics/final_snapshot.json` and `performance/result_summary.json`.
`dataset.jsonl` is the input (five prompts, the last one empty, so three samples
have empty output).

Produced from the repository root, with the echo server on a free local port:

```bash
python -m inference_endpoint.testing.echo_server --port 18765 &
inference-endpoint benchmark online \
  --endpoints http://127.0.0.1:18765 \
  --model tests/assets/tokenizers/char \
  --dataset tests/assets/archived_runs/echo_char_streaming/dataset.jsonl \
  --load-pattern concurrency --concurrency 2 --num-samples 15 \
  --streaming on --steady-state \
  --workers 1 --no-cpu-affinity --max-connections 32 \
  --report-dir "$OUT"
kill %1
```
