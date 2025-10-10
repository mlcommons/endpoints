# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for profiler_utils.py shared utilities."""

import os


class TestOutputPathGeneration:
    """Test get_output_path functionality."""

    def test_generates_unique_paths(self, temp_profile_dir):
        """Test get_output_path generates unique file paths."""
        from inference_endpoint.profiling.profiler_utils import get_output_path

        path1 = get_output_path(temp_profile_dir, "test_profiler", "txt")
        path2 = get_output_path(temp_profile_dir, "test_profiler", "txt")

        # Same directory
        assert path1.parent == path2.parent == temp_profile_dir
        # Correct suffix
        assert path1.suffix == ".txt"
        assert path2.suffix == ".txt"
        # Contains PID
        assert f"pid{os.getpid()}" in path1.name

    def test_accepts_string_and_path(self, temp_profile_dir):
        """Test get_output_path works with both string and Path."""
        from inference_endpoint.profiling.profiler_utils import get_output_path

        path1 = get_output_path(str(temp_profile_dir), "profiler", "txt")
        path2 = get_output_path(temp_profile_dir, "profiler", "txt")

        assert path1.parent == temp_profile_dir
        assert path2.parent == temp_profile_dir

    def test_custom_suffix(self, temp_profile_dir):
        """Test get_output_path with custom suffix."""
        from inference_endpoint.profiling.profiler_utils import get_output_path

        path = get_output_path(temp_profile_dir, "profiler", "callgrind")
        assert path.suffix == ".callgrind"


class TestProfileCollection:
    """Test collect_all_profiles functionality."""

    def test_separates_main_and_worker_profiles(self, temp_profile_dir):
        """Test collect_all_profiles separates main and worker files."""
        from inference_endpoint.profiling.profiler_utils import collect_all_profiles

        main_pid = os.getpid()
        main_file = temp_profile_dir / f"profile_pid{main_pid}_12345.txt"
        worker_file = temp_profile_dir / "profile_pid99999_12346.txt"

        main_file.write_text("main")
        worker_file.write_text("worker")

        main_files, worker_files = collect_all_profiles("test", temp_profile_dir)

        assert len(main_files) == 1
        assert len(worker_files) == 1
        assert main_file in main_files
        assert worker_file in worker_files

    def test_handles_empty_directory(self, temp_profile_dir):
        """Test collect_all_profiles with empty directory."""
        from inference_endpoint.profiling.profiler_utils import collect_all_profiles

        main_files, worker_files = collect_all_profiles("test", temp_profile_dir)
        assert main_files == []
        assert worker_files == []

    def test_handles_nonexistent_directory(self):
        """Test collect_all_profiles with nonexistent directory."""
        from inference_endpoint.profiling.profiler_utils import collect_all_profiles

        main_files, worker_files = collect_all_profiles("test", "/nonexistent/path")
        assert main_files == []
        assert worker_files == []

    def test_sorts_files(self, temp_profile_dir):
        """Test collect_all_profiles returns sorted lists."""
        from inference_endpoint.profiling.profiler_utils import collect_all_profiles

        # Create multiple files
        for i in range(3):
            f = temp_profile_dir / f"profile_pid99999_{i}.txt"
            f.write_text(f"profile {i}")

        main_files, worker_files = collect_all_profiles("test", temp_profile_dir)

        # Worker files should be sorted
        assert len(worker_files) == 3
        assert worker_files == sorted(worker_files)


class TestProfilePrinting:
    """Test print_all_profiles functionality."""

    def test_prints_profile_content(self, temp_profile_dir, capsys):
        """Test print_all_profiles outputs content."""
        from inference_endpoint.profiling.profiler_utils import print_all_profiles

        main_pid = os.getpid()
        profile_file = temp_profile_dir / f"profile_pid{main_pid}_12345.txt"
        profile_file.write_text("Test profile content")

        print_all_profiles("test_profiler", temp_profile_dir)

        captured = capsys.readouterr()
        assert "TEST_PROFILER" in captured.err
        assert "Test profile content" in captured.err

    def test_noop_when_empty(self, temp_profile_dir, capsys):
        """Test print_all_profiles does nothing when no profiles."""
        from inference_endpoint.profiling.profiler_utils import print_all_profiles

        print_all_profiles("test_profiler", temp_profile_dir)

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_handles_read_errors(self, temp_profile_dir, capsys):
        """Test print_all_profiles handles file read errors."""
        from inference_endpoint.profiling.profiler_utils import print_all_profiles

        # Create directory instead of file
        main_pid = os.getpid()
        bad_file = temp_profile_dir / f"profile_pid{main_pid}_12345.txt"
        bad_file.mkdir()

        print_all_profiles("test_profiler", temp_profile_dir)

        captured = capsys.readouterr()
        assert "Error reading" in captured.err or "TEST_PROFILER" in captured.err


class TestProcessDetection:
    """Test process type detection utilities."""

    def test_is_pytest_mode(self):
        """Test is_pytest_mode detects pytest."""
        from inference_endpoint.profiling.profiler_utils import is_pytest_mode

        assert is_pytest_mode() is True

    def test_is_worker_process(self):
        """Test is_worker_process identifies process type."""
        from inference_endpoint.profiling.profiler_utils import is_worker_process

        # Should return bool indicating if we're in a worker
        result = is_worker_process()
        assert isinstance(result, bool)
        # In test process (MainProcess), should be False
        assert result is False


class TestLogProfilerStart:
    """Test log_profiler_start utility."""

    def test_logs_profiler_start(self, capsys):
        """Test log_profiler_start outputs to stderr."""
        from inference_endpoint.profiling.profiler_utils import log_profiler_start

        log_profiler_start("test_profiler")

        captured = capsys.readouterr()
        # Should log profiler name and PID
        assert "Test_profiler" in captured.err
        assert str(os.getpid()) in captured.err


class TestRunOutputDir:
    """Test get_run_output_dir functionality."""

    def test_creates_and_returns_directory(self, temp_profile_dir):
        """Test get_run_output_dir creates directory."""
        from inference_endpoint.profiling import profiler_utils

        profiler_utils._run_output_dir = None
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        output_dir = profiler_utils.get_run_output_dir()

        assert output_dir == temp_profile_dir
        assert output_dir.exists()

        # Clean up
        del os.environ["PROFILER_OUT_DIR"]
        profiler_utils._run_output_dir = None

    def test_caches_directory(self, temp_profile_dir):
        """Test get_run_output_dir caches result."""
        from inference_endpoint.profiling import profiler_utils

        profiler_utils._run_output_dir = None
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        dir1 = profiler_utils.get_run_output_dir()
        dir2 = profiler_utils.get_run_output_dir()

        assert dir1 == dir2

        # Clean up
        del os.environ["PROFILER_OUT_DIR"]
        profiler_utils._run_output_dir = None
