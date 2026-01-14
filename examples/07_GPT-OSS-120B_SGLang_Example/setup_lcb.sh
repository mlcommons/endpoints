# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required under the License agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Setup script to clone the LiveCodeBench repository and set up the environment, since LCB
# has different dependencies than the rest of Inference Endpoint. In the future, this will
# be migrated to a containerized environment.

set -euo pipefail

# Default installation directory
DEFAULT_LCB_ROOT="/opt/LiveCodeBench"

# Parse command line arguments
LCB_ROOT="${1:-$DEFAULT_LCB_ROOT}"

# Display installation path
echo "LiveCodeBench will be installed to: ${LCB_ROOT}"

# Check if directory already exists
if [ -d "${LCB_ROOT}" ]; then
    echo "Warning: Directory ${LCB_ROOT} already exists."
    read -p "Do you want to remove it and reinstall? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing directory..."
        rm -rf "${LCB_ROOT}"
    else
        echo "Installation cancelled."
        exit 1
    fi
fi

echo "Cloning LiveCodeBench repository..."
git clone https://github.com/LiveCodeBench/LiveCodeBench.git "${LCB_ROOT}"

echo "Installing dependencies..."
cd "${LCB_ROOT}"
pip install datasets==3.6.0  # LCB requires HF datasets < 4.0.0
pip install -e .
