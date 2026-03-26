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

"""Transport abstraction for worker IPC."""

from typing import Annotated, TypeAlias

from pydantic import Field

from .protocol import (
    ReceiverTransport,
    SenderTransport,
    TransportConfig,
    WorkerConnector,
    WorkerPoolTransport,
)

# ZMQ implementation
from .zmq import ZMQTransportConfig, ZmqWorkerPoolTransport

# Discriminated union of transport backends, dispatched on the ``type`` field.
# To add a new transport: define a TransportConfig subclass with a Literal type
# field and add it to this union.
AnyTransportConfig: TypeAlias = Annotated[  # noqa: UP040
    ZMQTransportConfig,  # extend with: ZMQTransportConfig | ShmTransportConfig
    Field(discriminator="type"),
]

__all__ = [
    # Config
    "AnyTransportConfig",
    "TransportConfig",
    "ZMQTransportConfig",
    # Protocols
    "ReceiverTransport",
    "SenderTransport",
    "WorkerConnector",
    "WorkerPoolTransport",
    # Default implementation
    "ZmqWorkerPoolTransport",
]
