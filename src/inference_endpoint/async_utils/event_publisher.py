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
# See the specific language governing permissions and
# limitations under the License.


import uuid

from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqEventRecordPublisher


class EventPublisherService(ZmqEventRecordPublisher):
    """Publisher for publishing event records over ZMQ PUB socket.

    Wraps ZmqEventRecordPublisher with LoopManager integration and
    auto-generated socket names.
    """

    def __init__(
        self,
        managed_zmq_context: ManagedZMQContext,
        extra_eager: bool = False,
        isolated_event_loop: bool = False,
        send_threshold: int = 1000,
    ):
        """Creates a new EventPublisherService.

        Args:
            managed_zmq_context: The managed ZMQ context to use.
            extra_eager: If True, publish() blocks until the message is sent.
                Useful for testing or when EventRecords are used as a
                synchronization mechanism (e.g., ENDED as a stop signal).
            isolated_event_loop: If True, runs on a separate event loop thread.
            send_threshold: Minimum number of buffered records before an
                automatic flush is triggered. See ZmqEventRecordPublisher.
        """
        if extra_eager:
            loop = None
        elif isolated_event_loop:
            loop = LoopManager().create_loop("ev_pub")
        else:
            loop = LoopManager().default_loop
        self.socket_name = f"ev_pub_{uuid.uuid4().hex[:8]}"
        super().__init__(
            self.socket_name,
            managed_zmq_context,
            loop=loop,
            send_threshold=send_threshold,
        )
