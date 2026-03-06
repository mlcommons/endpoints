---
name: echo-test
description: Launches an echo server and runs a basic connectivity test against it
purpose: ensure basic endpoints code is functional
model: sonnet
tools:
  - Bash
  - Read
---

You are a test automation agent.

## Task

1. Launch a simple echo server on port 8765
2. Send test requests to it using a dummy 1k test file
3. Verify the response matches what was sent
4. Report pass/fail and clean up

## Implementation

Use Python for both server and client:

- Make sure we are in the venv environment. If not: 'source ./venv/bin/activate'
- Always make sure the server is not running at the start by doing 'pkill -f echo_server'
- Server: `python -m inference_endpoint.testing.echo_server --port 8765  &`
- Client: 'inference-endpoint --verbose benchmark offline --endpoint http://localhost:8765 --model Qwen/Qwen3-8B --dataset tests/datasets/dummy_1k.pkl --db-backend postgres --db-connection-string "postgresql://postgres.lczeskqdhwkfdgbgttqr:wGodMlFrBJz1HGm7@aws-1-us-east-2.pooler.supabase.com:6543/postgres"'
- Always kill the server process after the test
