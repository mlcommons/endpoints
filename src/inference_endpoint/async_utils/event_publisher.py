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
from inference_endpoint.utils import SingletonMixin


class EventPublisherService(SingletonMixin, ZmqEventRecordPublisher):
    """Singleton publisher for publishing event records."""

    def __init__(
        self,
        managed_zmq_context: ManagedZMQContext,
        extra_eager: bool = False,
        isolated_event_loop: bool = False,
    ):
        """Creates a new EventPublisherService.

        By default, the publisher will be run on the main thread's event loop (i.e. the default loop).

        Args:
            managed_zmq_context (ManagedZMQContext): The managed ZMQ context to use for the publisher.
            extra_eager (bool): If True, the publisher will be a blocking call and calls to .publish()
                will block until the message has been successfully sent. In most cases, this should not
                be turned on, but it is useful for testing, or specifically in the use case where
                EventRecords are being used as a synchronization mechanism (i.e. sending a specific
                EventRecord as a STOP signal to subscribers to ensure the ordering of cleanup.)
            isolated_event_loop (bool): If True, the publisher will be run in a separate event loop.
        """
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # Set up event loop settings
        if extra_eager:
            loop = None
        elif isolated_event_loop:
            loop = LoopManager().create_loop("ev_pub")
        else:
            loop = LoopManager().default_loop
        super().__init__(
            f"ev_pub_{uuid.uuid4().hex[:8]}", managed_zmq_context, loop=loop
        )
