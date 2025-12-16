#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Setup script to create a vLLM virtualenv for benchmark comparison
# This isolates vLLM dependencies from the main inference-endpoint environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-${SCRIPT_DIR}/vllm_venv}"
VLLM_VERSION="0.11.2"

echo "Creating vLLM virtualenv at: ${VENV_DIR}"
echo "vLLM version: ${VLLM_VERSION}"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for fast installation..."
    uv venv "${VENV_DIR}"
    echo "Installing vLLM[bench]==${VLLM_VERSION}..."
    uv pip install --python "${VENV_DIR}/bin/python" "vllm[bench]==${VLLM_VERSION}"
else
    echo "uv not found, using pip..."
    # Create virtualenv (without pip to avoid ensurepip issues)
    python3 -m venv --without-pip "${VENV_DIR}"

    # Install pip using get-pip.py
    echo "Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV_DIR}/bin/python3"

    # Activate venv
    source "${VENV_DIR}/bin/activate"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing vLLM[bench]==${VLLM_VERSION}..."
    pip install "vllm[bench]==${VLLM_VERSION}"
fi

echo ""
echo "vLLM virtualenv setup complete!"
echo "Location: ${VENV_DIR}"
echo ""
echo "To use with compare_with_vllm.py:"
echo "  python compare_with_vllm.py --model <model> --vllm-venv-dir ${VENV_DIR}"
echo ""
echo "Or if using the default location (vllm_venv in this directory):"
echo "  python compare_with_vllm.py --model <model>"
