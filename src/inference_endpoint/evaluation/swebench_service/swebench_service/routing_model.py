# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-swe-agent model wrapper for trajectory-stable endpoint routing."""

from __future__ import annotations

import uuid
from typing import Any

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig


class SessionRoutingModelConfig(LitellmModelConfig):
    """Add endpoint routing header names to mini-swe-agent's model config."""

    routing_headers: tuple[str, ...] = ()


class SessionRoutingLitellmModel(LitellmModel):
    """Attach one opaque routing ID to every request in this model trajectory."""

    def __init__(self, **kwargs: Any):
        super().__init__(config_class=SessionRoutingModelConfig, **kwargs)
        self.routing_session_id = uuid.uuid4().hex

        if not self.config.routing_headers:
            return
        model_kwargs = dict(self.config.model_kwargs)
        extra_headers = dict(model_kwargs.get("extra_headers") or {})
        extra_headers.update(
            dict.fromkeys(self.config.routing_headers, self.routing_session_id)
        )
        model_kwargs["extra_headers"] = extra_headers
        self.config.model_kwargs = model_kwargs
