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


class TestEndpointBudgetScaling:
    """max_connections budget scales with the number of distinct endpoints.

    Ephemeral-port capacity is per (source IP, destination) pair, so each
    distinct endpoint contributes its configured port-range budget. num_workers
    is pinned (>=1) so config resolution skips the NUMA auto-probe.
    """

    def test_auto_budget_scales_with_distinct_endpoints(self):
        with patch.object(cfg, "get_ephemeral_port_range", return_value=(1, 10000)):
            c = cfg.HTTPClientConfig(
                endpoint_urls=[
                    "http://10.0.0.1:8000",
                    "http://10.0.0.2:8000",
                    "http://10.0.0.3:8000",
                ],
                num_workers=10,
            )
        assert c.max_connections == 30000  # 10000 ports x 3 distinct endpoints

    def test_single_endpoint_budget_unchanged(self):
        with patch.object(cfg, "get_ephemeral_port_range", return_value=(1, 10000)):
            c = cfg.HTTPClientConfig(
                endpoint_urls=["http://10.0.0.1:8000"], num_workers=10
            )
        assert c.max_connections == 10000  # single endpoint -> unchanged

    def test_live_socket_usage_does_not_reject_overall_connection_limit(self):
        with (
            patch.object(cfg, "get_ephemeral_port_range", return_value=(1, 10000)),
            patch.object(
                cfg, "get_ephemeral_port_limit", return_value=0, create=True
            ) as live_port_limit,
        ):
            c = cfg.HTTPClientConfig(
                endpoint_urls=["http://10.0.0.1:8000"],
                num_workers=10,
                max_connections=10,
            )
        assert c.max_connections == 10
        live_port_limit.assert_not_called()

    def test_duplicate_endpoints_do_not_inflate_budget(self):
        # Same (host, port) repeated (even with different paths) is one
        # destination -> one budget, since the 4-tuple ignores path.
        with patch.object(cfg, "get_ephemeral_port_range", return_value=(1, 10000)):
            c = cfg.HTTPClientConfig(
                endpoint_urls=[
                    "http://10.0.0.1:8000/v1/a",
                    "http://10.0.0.1:8000/v1/b",
                    "http://10.0.0.1:8000",
                ],
                num_workers=10,
            )
        assert c.max_connections == 10000  # 1 distinct (host, port)

    def test_explicit_max_connections_within_scaled_budget_ok(self):
        # 25000 exceeds one endpoint's budget (10000) but fits 3 (30000).
        with patch.object(cfg, "get_ephemeral_port_range", return_value=(1, 10000)):
            c = cfg.HTTPClientConfig(
                endpoint_urls=[
                    "http://10.0.0.1:8000",
                    "http://10.0.0.2:8000",
                    "http://10.0.0.3:8000",
                ],
                num_workers=10,
                max_connections=25000,
            )
        assert c.max_connections == 25000

    def test_explicit_max_connections_exceeding_scaled_budget_raises(self):
        with patch.object(cfg, "get_ephemeral_port_range", return_value=(1, 10000)):
            with pytest.raises(RuntimeError, match="exceeds the ephemeral"):
                cfg.HTTPClientConfig(
                    endpoint_urls=["http://10.0.0.1:8000", "http://10.0.0.2:8000"],
                    num_workers=10,
                    max_connections=40000,  # > 2 x 10000
                )


@pytest.mark.unit
class TestEndpointDestination:
    """Distinct-destination identity used for the ephemeral-port budget."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("http://10.0.0.1:8000", ("10.0.0.1", 8000)),
            ("https://host:9000", ("host", 9000)),
            ("http://host", ("host", 80)),
            ("https://host", ("host", 443)),
            ("10.0.0.1:8000", ("10.0.0.1", 8000)),  # schemeless host:port
            ("host:9000", ("host", 9000)),
            ("http://[::1]:8000", ("::1", 8000)),  # IPv6
        ],
    )
    def test_resolves_host_and_port(self, url, expected):
        assert cfg._endpoint_destination(url) == expected

    def test_schemeless_urls_count_as_distinct(self):
        # Bare host:port must not collapse to (None, None) and inflate to 1.
        keys = {cfg._endpoint_destination(u) for u in ("a:8000", "b:8000")}
        assert len(keys) == 2

    def test_http_and_https_same_host_are_distinct(self):
        # Default ports differ (80 vs 443) -> two destinations, not one.
        keys = {cfg._endpoint_destination(u) for u in ("http://h", "https://h")}
        assert len(keys) == 2

    def test_schemeless_budget_scales_with_distinct_hosts(self):
        with patch.object(cfg, "get_ephemeral_port_range", return_value=(1, 10000)):
            c = cfg.HTTPClientConfig(
                endpoint_urls=["10.0.0.1:8000", "10.0.0.2:8000"],
                num_workers=10,
            )
        assert c.max_connections == 20000  # 10000 ports x 2 distinct hosts
