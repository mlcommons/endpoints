# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from inference_endpoint.endpoint_client.cpu_affinity import (
    AffinityPlan,
    compute_affinity_plan,
    pin_loadgen,
    set_cpu_affinity,
)


class TestAffinityPlan:
    def test_get_worker_cpus_round_robin(self):
        plan = AffinityPlan(
            loadgen_cpus=[0, 1],
            worker_cpu_sets=[[2, 3], [4, 5]],
        )
        assert plan.get_worker_cpus(0) == [2, 3]
        assert plan.get_worker_cpus(1) == [4, 5]
        assert plan.get_worker_cpus(2) == [2, 3]  # round-robin
        assert plan.get_worker_cpus(3) == [4, 5]

    def test_get_worker_cpus_empty(self):
        plan = AffinityPlan(loadgen_cpus=[0], worker_cpu_sets=[])
        assert plan.get_worker_cpus(0) == []


class TestComputeAffinityPlan:
    @patch("inference_endpoint.endpoint_client.cpu_affinity.get_all_online_cpus")
    @patch("inference_endpoint.endpoint_client.cpu_affinity.get_physical_core_id")
    @patch("inference_endpoint.endpoint_client.cpu_affinity.get_numa_node")
    @patch(
        "inference_endpoint.endpoint_client.cpu_affinity.get_cpus_ranked_by_performance"
    )
    def test_compute_affinity_plan(self, mock_rank, mock_numa, mock_phys, mock_online):
        # 8 CPUs, 4 physical cores (hyperthreading)
        mock_online.return_value = {0, 1, 2, 3, 4, 5, 6, 7}
        mock_phys.side_effect = lambda cpu: cpu % 4
        mock_numa.return_value = 0
        mock_rank.return_value = [0, 1, 2, 3, 4, 5, 6, 7]

        plan = compute_affinity_plan(num_workers=2, loadgen_cores=2)

        assert len(plan.loadgen_cpus) == 4
        assert len(plan.worker_cpu_sets) == 2


class TestPinLoadgen:
    @patch("inference_endpoint.endpoint_client.cpu_affinity.compute_affinity_plan")
    @patch("os.sched_setaffinity")
    @patch("os.getpid", return_value=12345)
    def test_pin_loadgen(self, mock_getpid, mock_setaffinity, mock_plan):
        mock_plan.return_value = AffinityPlan(
            loadgen_cpus=[0, 1],
            worker_cpu_sets=[[2, 3]],
            _loadgen_physical_cores=1,
        )

        result = pin_loadgen(num_workers=1)

        assert result is not None
        mock_setaffinity.assert_called_once_with(12345, {0, 1})


class TestSetCpuAffinity:
    @patch("os.sched_setaffinity")
    def test_set_cpu_affinity(self, mock_setaffinity):
        success = set_cpu_affinity(12345, {0, 1})
        assert success
        mock_setaffinity.assert_called_with(12345, {0, 1})

    @patch("os.sched_setaffinity")
    def test_set_cpu_affinity_empty(self, mock_setaffinity):
        success = set_cpu_affinity(12345, set())
        assert not success
        mock_setaffinity.assert_not_called()
