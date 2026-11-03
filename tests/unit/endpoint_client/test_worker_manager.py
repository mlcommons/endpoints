# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import signal
from unittest.mock import MagicMock

import pytest
from inference_endpoint.endpoint_client import worker_manager as worker_manager_module
from inference_endpoint.endpoint_client.worker_manager import WorkerManager


@pytest.mark.unit
def test_spawned_worker_inherits_sigint_blocked(monkeypatch):
    blocked = []

    class FakeProcess:
        pid = 1

        def __init__(self, **kwargs):
            pass

        def start(self):
            current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
            blocked.append(signal.SIGINT in current_mask)

    monkeypatch.setattr(worker_manager_module, "Process", FakeProcess)
    manager = object.__new__(WorkerManager)
    manager.http_config = MagicMock()
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    manager._spawn_worker(0, MagicMock())

    assert blocked == [True]
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == previous_mask
