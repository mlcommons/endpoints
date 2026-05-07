# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HTTPClientConfig construction on non-Linux platforms.

NUMA probing is Linux-only; auto-detecting num_workers must fall back
gracefully so HTTPClientConfig() can be constructed anywhere.
"""

from unittest.mock import patch

from inference_endpoint.endpoint_client import config as cfg
from inference_endpoint.endpoint_client.cpu_affinity import UnsupportedPlatformError


class TestAutoNumWorkersNonLinux:
    def _clear_cache(self):
        cfg._get_auto_num_workers.cache_clear()

    def test_get_current_numa_node_unsupported_falls_back_to_min(self):
        self._clear_cache()
        with patch.object(
            cfg, "get_current_numa_node", side_effect=UnsupportedPlatformError("darwin")
        ):
            assert cfg._get_auto_num_workers() == 10

    def test_get_cpus_in_numa_node_unsupported_falls_back_to_min(self):
        self._clear_cache()
        with (
            patch.object(cfg, "get_current_numa_node", return_value=0),
            patch.object(
                cfg,
                "get_cpus_in_numa_node",
                side_effect=UnsupportedPlatformError("darwin"),
            ),
        ):
            assert cfg._get_auto_num_workers() == 10

    def test_http_client_config_constructs_when_numa_unsupported(self):
        self._clear_cache()
        with patch.object(
            cfg, "get_current_numa_node", side_effect=UnsupportedPlatformError("darwin")
        ):
            c = cfg.HTTPClientConfig()
        assert c.num_workers == 10
