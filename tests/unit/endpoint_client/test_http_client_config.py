# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HTTPClientConfig construction on non-Linux platforms.

NUMA probing is Linux-only; auto-detecting num_workers must fall back
gracefully so HTTPClientConfig() can be constructed anywhere.
"""

from unittest.mock import patch

import pytest
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


class TestSourceIpsBudgetScaling:
    """source_ips multiplies the ephemeral-port budget by the IP count."""

    # num_workers is pinned (>=1) so config resolution skips NUMA auto-probe.

    def test_blank_source_ips_are_dropped(self):
        c = cfg.HTTPClientConfig(source_ips=["127.0.0.1", "  ", "", "127.0.0.2"])
        assert c.source_ips == ["127.0.0.1", "127.0.0.2"]

    def test_auto_max_connections_scales_by_source_ip_count(self):
        with (
            patch.object(cfg, "get_ephemeral_port_range", return_value=(32768, 60999)),
            patch.object(cfg, "get_ephemeral_port_limit", return_value=10000),
        ):
            c = cfg.HTTPClientConfig(
                endpoint_urls=["http://localhost:8000"],
                num_workers=10,
                source_ips=["1.1.1.1", "2.2.2.2", "3.3.3.3"],
            )
        assert c.max_connections == 30000  # 10000 available x 3 source IPs

    def test_auto_max_connections_unchanged_without_source_ips(self):
        with (
            patch.object(cfg, "get_ephemeral_port_range", return_value=(32768, 60999)),
            patch.object(cfg, "get_ephemeral_port_limit", return_value=10000),
        ):
            c = cfg.HTTPClientConfig(
                endpoint_urls=["http://localhost:8000"], num_workers=10
            )
        assert c.max_connections == 10000  # single-IP budget, unchanged

    def test_explicit_max_connections_within_scaled_budget_ok(self):
        # 25000 exceeds the single-IP budget (10000) but fits 3 IPs (30000).
        with (
            patch.object(cfg, "get_ephemeral_port_range", return_value=(32768, 60999)),
            patch.object(cfg, "get_ephemeral_port_limit", return_value=10000),
        ):
            c = cfg.HTTPClientConfig(
                endpoint_urls=["http://localhost:8000"],
                num_workers=10,
                max_connections=25000,
                source_ips=["1.1.1.1", "2.2.2.2", "3.3.3.3"],
            )
        assert c.max_connections == 25000

    def test_explicit_max_connections_exceeding_scaled_budget_raises(self):
        with (
            patch.object(cfg, "get_ephemeral_port_range", return_value=(32768, 60999)),
            patch.object(cfg, "get_ephemeral_port_limit", return_value=10000),
        ):
            with pytest.raises(RuntimeError, match="exceeds ephemeral port"):
                cfg.HTTPClientConfig(
                    endpoint_urls=["http://localhost:8000"],
                    num_workers=10,
                    max_connections=40000,
                    source_ips=["1.1.1.1", "2.2.2.2", "3.3.3.3"],
                )
