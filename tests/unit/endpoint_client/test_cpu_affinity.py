# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from inference_endpoint.endpoint_client.cpu_affinity import (
    AffinityPlan,
    compute_affinity_plan,
    get_all_online_cpus,
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


class TestGetAllOnlineCpus:
    """Test get_all_online_cpus with cgroup intersection logic."""

    @patch("os.sched_getaffinity")
    @patch("pathlib.Path.read_text")
    def test_sysfs_intersected_with_cgroup(self, mock_read, mock_getaffinity):
        """Test that sysfs CPUs are intersected with cgroup-allowed CPUs."""
        # Sysfs reports 0-7, but cgroup restricts to 0-3
        mock_read.return_value = "0-7\n"
        mock_getaffinity.return_value = {0, 1, 2, 3}

        cpus = get_all_online_cpus()

        # Should return intersection: only CPUs allowed by both sysfs AND cgroup
        assert cpus == {0, 1, 2, 3}

    @patch("os.sched_getaffinity", side_effect=OSError("Permission denied"))
    @patch("pathlib.Path.read_text")
    def test_sysfs_without_cgroup_check(self, mock_read, mock_getaffinity):
        """Test sysfs alone when sched_getaffinity fails."""
        mock_read.return_value = "0-3,8-11\n"
        cpus = get_all_online_cpus()
        assert cpus == {0, 1, 2, 3, 8, 9, 10, 11}

    @patch("os.sched_getaffinity")
    @patch("pathlib.Path.read_text", side_effect=OSError("No such file"))
    def test_fallback_cpu_directories_intersected(self, mock_read, mock_getaffinity):
        """Test cpu directory fallback intersected with cgroup."""
        mock_getaffinity.return_value = {0, 2}  # Cgroup restricts to 0, 2

        # Create mock entries with proper name attribute
        cpu0 = MagicMock()
        cpu0.name = "cpu0"
        cpu1 = MagicMock()
        cpu1.name = "cpu1"
        cpu2 = MagicMock()
        cpu2.name = "cpu2"
        cpufreq = MagicMock()
        cpufreq.name = "cpufreq"

        with patch("pathlib.Path.iterdir", return_value=[cpu0, cpu1, cpu2, cpufreq]):
            cpus = get_all_online_cpus()
            # Should return intersection of {0,1,2} (dirs) and {0,2} (cgroup)
            assert cpus == {0, 2}

    @patch("os.sched_getaffinity")
    @patch("pathlib.Path.iterdir", side_effect=OSError("Permission denied"))
    @patch("pathlib.Path.read_text", side_effect=OSError("No such file"))
    def test_fallback_sched_getaffinity_only(
        self, mock_read, mock_iterdir, mock_getaffinity
    ):
        """Test final fallback: uses sched_getaffinity directly."""
        mock_getaffinity.return_value = {0, 2, 4, 6}

        cpus = get_all_online_cpus()

        assert cpus == {0, 2, 4, 6}

    @patch("os.sched_getaffinity", side_effect=OSError("Permission denied"))
    @patch("pathlib.Path.iterdir", side_effect=OSError("Permission denied"))
    @patch("pathlib.Path.read_text", side_effect=OSError("No such file"))
    def test_all_methods_fail_returns_empty(
        self, mock_read, mock_iterdir, mock_getaffinity
    ):
        """Test that empty set is returned when all methods fail."""
        cpus = get_all_online_cpus()
        assert cpus == set()
