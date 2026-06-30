#!/bin/bash
# Complete benchmark workflow for Qwen2.5-0.5B-Instruct
# Supports both vLLM and SGLang inference servers

set -eo pipefail  # Exit on error, including failures in piped benchmark commands

echo "========================================"
echo "Qwen2.5-0.5B Benchmark Runner"
echo "========================================"
echo ""

# Parse arguments
SERVER_TYPE="${1:-vllm}"     # vllm or sglang
BENCHMARK_TYPE="${2:-offline}"  # offline or online

# Validate server type
if [[ "$SERVER_TYPE" != "vllm" && "$SERVER_TYPE" != "sglang" ]]; then
    echo "ERROR: Invalid server type: $SERVER_TYPE"
    echo "Usage: bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh [vllm|sglang] [offline|online]"
    exit 1
fi

# Validate benchmark type
if [[ "$BENCHMARK_TYPE" != "offline" && "$BENCHMARK_TYPE" != "online" ]]; then
    echo "ERROR: Invalid benchmark type: $BENCHMARK_TYPE"
    echo "Usage: bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh [vllm|sglang] [offline|online]"
    exit 1
fi

# Configuration
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Set server-specific configuration
if [[ "$SERVER_TYPE" == "vllm" ]]; then
    CONTAINER_NAME="vllm-qwen"
    SERVER_PORT=8000
    CONFIG_PREFIX=""
    DOCKER_IMAGE="vllm/vllm-openai:latest"
else
    CONTAINER_NAME="sglang-qwen"
    SERVER_PORT=30000
    CONFIG_PREFIX="sglang_"
    DOCKER_IMAGE="lmsysorg/sglang:latest"
fi

echo "Configuration:"
echo "  Server: $SERVER_TYPE"
echo "  Benchmark: $BENCHMARK_TYPE"
echo "  Container: $CONTAINER_NAME"
echo "  Port: $SERVER_PORT"
echo ""

# Check if running from repo root
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Please run this script from the repository root"
    echo "Usage: bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh [vllm|sglang] [offline|online]"
    exit 1
fi

# Step 1: Prepare dataset
echo "Step 1: Preparing dataset..."
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at .venv/"
    echo "Please create it first: python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

source .venv/bin/activate
python examples/08_Qwen2.5-0.5B_Example/prepare_dataset.py
echo "✅ Dataset prepared"
echo ""

# Step 2: Check if container is already running
echo "Step 2: Checking for existing $SERVER_TYPE container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Found existing container: ${CONTAINER_NAME}"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is running. Skipping server launch."
    else
        echo "Container exists but not running. Starting..."
        docker start ${CONTAINER_NAME}
        sleep 15
    fi
else
    echo "No existing container found. Launching $SERVER_TYPE server..."

    # Step 3: Launch server (vLLM or SGLang)
    echo "Step 3: Launching $SERVER_TYPE server..."

    if [[ "$SERVER_TYPE" == "vllm" ]]; then
        # Launch vLLM
        docker run --runtime nvidia --gpus all \
          -v ${HF_HOME}:/root/.cache/huggingface \
          -e PYTORCH_ALLOC_CONF=expandable_segments:True \
          -p ${SERVER_PORT}:8000 \
          --ipc=host \
          --name ${CONTAINER_NAME} \
          -d \
          ${DOCKER_IMAGE} \
          --model ${MODEL_NAME} \
          --gpu-memory-utilization 0.85
    else
        # Launch SGLang
        docker run --runtime nvidia --gpus all \
          --net host \
          -v ${HF_HOME}:/root/.cache/huggingface \
          --ipc=host \
          --name ${CONTAINER_NAME} \
          -d \
          ${DOCKER_IMAGE} \
          python3 -m sglang.launch_server \
          --model-path ${MODEL_NAME} \
          --host 0.0.0.0 \
          --port ${SERVER_PORT} \
          --mem-fraction-static 0.9 \
          --attention-backend flashinfer
    fi

    echo "Waiting for server to start..."
    sleep 20
fi
echo ""

# Step 4: Wait for server to be ready
echo "Step 4: Waiting for server to be ready..."
MAX_RETRIES=40
RETRY_COUNT=0

# Different ready indicators for vLLM vs SGLang
if [[ "$SERVER_TYPE" == "vllm" ]]; then
    READY_PATTERN="Uvicorn running|Application startup complete"
else
    READY_PATTERN="Uvicorn running|Server is ready"
fi

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker logs ${CONTAINER_NAME} 2>&1 | grep -qE "$READY_PATTERN"; then
        echo "✅ Server is ready!"
        break
    fi
    if docker logs ${CONTAINER_NAME} 2>&1 | grep -qE "ERROR.*failed|CUDA out of memory|RuntimeError"; then
        echo "❌ Server failed to start. Check logs:"
        docker logs ${CONTAINER_NAME} 2>&1 | tail -20
        exit 1
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 5
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Server did not start within expected time"
    docker logs ${CONTAINER_NAME} 2>&1 | tail -30
    exit 1
fi
echo ""

# Step 5: Verify server
echo "Step 5: Verifying server..."
sleep 5  # Give it a moment to fully initialize

if curl -s http://localhost:${SERVER_PORT}/v1/models 2>/dev/null | grep -q "${MODEL_NAME}"; then
    echo "✅ Server is responding correctly"
elif curl -s http://localhost:${SERVER_PORT}/health 2>/dev/null | grep -q "ok\|healthy"; then
    echo "✅ Server health check passed"
else
    echo "⚠️  Warning: Server may not be fully ready, but proceeding..."
fi
echo ""

# Step 6: Run benchmark
echo "Step 6: Running ${SERVER_TYPE} ${BENCHMARK_TYPE} benchmark..."
CONFIG_FILE="examples/08_Qwen2.5-0.5B_Example/${CONFIG_PREFIX}${BENCHMARK_TYPE}_qwen_benchmark.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls examples/08_Qwen2.5-0.5B_Example/*.yaml
    exit 1
fi

source .venv/bin/activate
if [[ "$BENCHMARK_TYPE" == "online" ]]; then
    python scripts/concurrency_sweep/run.py \
      --config "$CONFIG_FILE" 2>&1 | tee benchmark_${SERVER_TYPE}_${BENCHMARK_TYPE}.log
else
    inference-endpoint benchmark from-config -c "$CONFIG_FILE" 2>&1 | tee benchmark_${SERVER_TYPE}_${BENCHMARK_TYPE}.log
fi

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
echo "Server: $SERVER_TYPE"
echo "Benchmark Type: $BENCHMARK_TYPE"
echo ""
echo "Results saved to:"
if [[ "$SERVER_TYPE" == "vllm" ]]; then
    if [ "$BENCHMARK_TYPE" = "offline" ]; then
        RESULT_DIR="results/qwen_offline_benchmark/"
        SWEEP_DIR=""
    else
        RESULT_DIR="results/qwen_online_benchmark/"
        SWEEP_DIR="${RESULT_DIR}concurrency_sweep/"
    fi
else
    if [ "$BENCHMARK_TYPE" = "offline" ]; then
        RESULT_DIR="results/qwen_sglang_offline_benchmark/"
        SWEEP_DIR=""
    else
        RESULT_DIR="results/qwen_sglang_online_benchmark/"
        SWEEP_DIR="${RESULT_DIR}concurrency_sweep/"
    fi
fi

echo "  ${RESULT_DIR}"
echo ""
if [[ "$BENCHMARK_TYPE" == "online" ]]; then
    echo "Summarize sweep results (tables + CSV + Markdown + plot):"
    echo "  python scripts/concurrency_sweep/summarize.py ${SWEEP_DIR}"
else
    echo "View summary:"
    echo "  cat ${RESULT_DIR}report.txt"
fi
echo ""
echo "Benchmark log:"
echo "  cat benchmark_${SERVER_TYPE}_${BENCHMARK_TYPE}.log"
echo ""
echo "To stop the server:"
echo "  docker stop ${CONTAINER_NAME}"
echo ""
