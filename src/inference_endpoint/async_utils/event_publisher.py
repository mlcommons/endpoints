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


import os
import uuid

from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqEventRecordPublisher
from inference_endpoint.async_utils.transport.zmq.transport import ZMQ_SOCKET_DIR


class EventPublisherService(ZmqEventRecordPublisher):
    """Singleton publisher for publishing event records.

    By default, the publisher will be run on the main thread's event loop. Since the
    publisher is created at startup, several environment variables are used to configure
    the publisher service:
     - EV_PUB_EXTRA_EAGER=1 will make it so that calls to .publish() will not be buffered and non-async.
       This means that the publisher will ignore any event loop and .publish() will block until the
       message has been successfully sent. In most cases, this should not be turned on.
     - EV_PUB_SEP_THREAD=1 will create a separate event loop specific to the publisher on its own
       thread.
     - EV_PUB_SOCK_DIR: If set, is used as the base directory for pub-sub IPC sockets. Otherwise,
       an auto-generated temp directory is used.
    """

    _instance: "EventPublisherService | None" = None

    def __new__(cls) -> "EventPublisherService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # Set up sockets
        sock_dir = os.environ.get("EV_PUB_SOCK_DIR")
        if not sock_dir:
            sock_dir = ZMQ_SOCKET_DIR
        bind_addr = f"ipc://{sock_dir}/ev_pub_{uuid.uuid4().hex[:8]}"

        # Set up event loop settings
        if os.environ.get("EV_PUB_EXTRA_EAGER", "0") == "1":
            loop = None
        elif os.environ.get("EV_PUB_SEP_THREAD", "0") == "1":
            loop = LoopManager().create_loop("ev_pub")
        else:
            loop = LoopManager().default_loop
        super().__init__(bind_addr, loop=loop)
