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

"""Vendor-agnostic power/energy monitoring.

A benchmark run can spawn a *sidecar* that streams power telemetry to a trace
file while the performance phase runs; at finalization the trace is sliced to
the measurement window, integrated into energy, and written to a sibling
``power.json`` (the ``Report`` is never mutated, mirroring the profiling
precedent).

The agnosticism boundary is the *process boundary*: any source that can print
one sample per line (``nvidia-smi``, a Prometheus/DCGM exporter, an
IPMI/redfish/PDU scraper, a cloud metrics CLI, or a user script) plugs in via
the ``command`` source with a field mapping — no Python, no core edits.
"""

from inference_endpoint.power.collector import PowerCollector
from inference_endpoint.power.render import write_power_section
from inference_endpoint.power.sources import ResolvedSource, power_source, resolve
from inference_endpoint.power.window import build_power_report

__all__ = [
    "PowerCollector",
    "ResolvedSource",
    "build_power_report",
    "power_source",  # decorator: register a custom source
    "resolve",
    "write_power_section",
]
