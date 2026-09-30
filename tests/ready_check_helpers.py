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

"""Ready-check test helpers.

Kept separate from tests/test_helpers.py so that modules imported by spawned
child processes do not pull in its heavy dependencies.
"""

from inference_endpoint.async_utils.transport.zmq.ready_check import (
    ReadyCheckReceiver,
)

_DELIVERY_TIMEOUT_MS = 5000


async def wait_until_ready_signal_queued(receiver: ReadyCheckReceiver) -> None:
    """Wait until a sent ready signal is receivable, so timed waits don't race IPC."""
    queued = await receiver._sock.poll(timeout=_DELIVERY_TIMEOUT_MS)
    assert (
        queued
    ), f"ready signal was not delivered within {_DELIVERY_TIMEOUT_MS / 1000}s"
