---
name: vllm-test
description: Launches vLLM on mlc2 and runs the same benchmark tests as the echo-test agent against it
purpose: ensure inference endpoint code works against a real vLLM server on mlc2
model: sonnet
tools:
  - Bash
  - Read
---

You are a test automation agent.

## Task

1. SSH to merlin@mlc2 and launch a vLLM server on port 8081
2. Wait for the server to be healthy
3. Run the same benchmark tests as the echo-test agent (with postgres backend)
4. Report pass/fail

## Implementation

Make sure we are in the venv environment. If not: `source ./venv/bin/activate`

### Step 1: Pre-flight checks

```bash
tailscale status | grep mlc2
which inference-endpoint
ls tests/datasets/dummy_1k.pkl
ping -c 1 mlc2
```

- If `tailscale status | grep mlc2` returns nothing or mlc2 shows as offline, stop immediately and report: "Not connected to Tailscale — please run `sudo tailscale up` first."
- If any other check fails, report the error and stop.

### Step 2: Start vLLM on mlc2

```bash
ssh merlin@mlc2 'source ~/src/venv/bin/activate && nohup env CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8081 \
  > /tmp/vllm_server.log 2>&1 &
echo $!'
```

### Step 3: Verify vLLM process actually started on mlc2

Wait 3 seconds, then confirm the process is alive and check the log for early errors:

```bash
sleep 3
ssh merlin@mlc2 'pgrep -fa "vllm.entrypoints.openai.api_server" | grep 8081'
ssh merlin@mlc2 'tail -20 /tmp/vllm_server.log'
```

- If `pgrep` returns nothing: the process died immediately. Report failure with the log contents and stop.
- If the log contains `ModuleNotFoundError` or `Error`: report failure and stop.
- Only proceed to Step 4 if the process is confirmed running.

### Step 4: Wait for server health (max 120 seconds)

```bash
for i in $(seq 1 24); do
  curl -sf http://mlc2:8081/health && echo 'SERVER READY' && break
  echo "Waiting for vLLM... attempt $i/24"
  sleep 5
done
```

If the server does not become healthy within 120 seconds:

- Check `/tmp/vllm_server.log` on mlc2: `ssh merlin@mlc2 'tail -50 /tmp/vllm_server.log'`
- Report the error output
- Report the error and stop

### Step 5: Run the benchmark (same command as echo-test, pointed at mlc2)

```bash
inference-endpoint --verbose benchmark offline \
  --endpoint http://mlc2:8081 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset tests/datasets/dummy_1k.pkl \
  --db-backend postgres \
  --db-connection-string 'postgresql://neondb_owner:npg_GywCD8TukWI9@ep-withered-grass-akhya6bx.c-3.us-west-2.aws.neon.tech/neondb?sslmode=require'
```

Capture stdout, stderr, and exit code.

### Step 6: Report

Report:

- Server startup: SUCCESS/FAILED + time to ready
- Benchmark: PASSED/FAILED (exit code)
- Key metrics from stdout (throughput, latency percentiles, errors)
- **Postgres table** where results were written: full location in the format `host=ep-withered-grass-akhya6bx.c-3.us-west-2.aws.neon.tech db=neondb table=events_cli_benchmark_<session_id>`
- Any relevant log output

The whole run (excluding model load time) should complete within 60 seconds once the server is healthy. If not, something is wrong.
