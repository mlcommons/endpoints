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

"""Unit tests for LoopManager."""

import time

import pytest
from inference_endpoint.async_utils.loop_manager import LoopManager, ManagedLoop


@pytest.fixture(autouse=True)
def reset_loop_manager():
    instance = LoopManager._instance
    yield
    LoopManager._instance = instance
    if instance is not None:
        instance._initialized = True


@pytest.fixture
def fresh_loop_manager(reset_loop_manager):
    LoopManager._instance = None
    manager = LoopManager()
    yield manager
    for name in list(manager.loops.keys()):
        if name != "default":
            try:
                manager.stop_loop(name, immediate=True)
            except Exception:
                pass


class TestLoopManagerSingleton:
    def test_singleton_same_instance(self, fresh_loop_manager):
        m1 = LoopManager()
        m2 = LoopManager()
        assert m1 is m2
        assert m1 is fresh_loop_manager

    def test_initialized_has_default_loop_and_loop_runnable(self, fresh_loop_manager):
        assert fresh_loop_manager.default_loop is not None
        assert fresh_loop_manager.loops.get("default") is not None
        result = None

        async def set_result():
            nonlocal result
            result = "ok"

        fresh_loop_manager.default_loop.run_until_complete(set_result())
        assert result == "ok"


class TestLoopManagerDefaultLoop:
    def test_default_loop_none_when_no_default_entry(self, reset_loop_manager):
        LoopManager._instance = None
        manager = LoopManager.__new__(LoopManager)
        manager.loops = {}
        manager._initialized = True
        with pytest.raises(RuntimeError, match="Default loop not found"):
            _ = manager.default_loop


class TestLoopManagerCreateLoop:
    def test_create_loop_default_idempotent_returns_same_loop(self, fresh_loop_manager):
        loop_before = fresh_loop_manager.get_loop("default")
        created = fresh_loop_manager.create_loop("default")
        assert created is loop_before
        assert fresh_loop_manager.get_loop("default") is loop_before

    def test_create_loop_new_name_returns_loop_and_managed_has_thread(
        self, fresh_loop_manager
    ):
        loop = fresh_loop_manager.create_loop("worker-1")
        assert loop is not None
        assert fresh_loop_manager.get_loop("worker-1") is loop
        managed = fresh_loop_manager.loops["worker-1"]
        assert isinstance(managed, ManagedLoop)
        assert managed.loop is loop
        assert managed.thread is not None
        assert managed.thread.is_alive()

    def test_create_loop_asyncio_backend_returns_loop_and_get_loop_matches(
        self, fresh_loop_manager
    ):
        loop = fresh_loop_manager.create_loop("asyncio-loop", backend="asyncio")
        assert loop is not None
        assert fresh_loop_manager.get_loop("asyncio-loop") is loop

    def test_create_loop_invalid_backend_raises(self, fresh_loop_manager):
        with pytest.raises(ValueError, match="Invalid backend"):
            fresh_loop_manager.create_loop("bad", backend="invalid")


class TestLoopManagerGetLoop:
    def test_get_loop_default_equals_default_loop_property(self, fresh_loop_manager):
        assert fresh_loop_manager.get_loop("default") is fresh_loop_manager.default_loop

    def test_get_loop_by_name_returns_created_loop_distinct_from_default(
        self, fresh_loop_manager
    ):
        fresh_loop_manager.create_loop("aux")
        loop = fresh_loop_manager.get_loop("aux")
        assert loop is not None
        assert loop is not fresh_loop_manager.default_loop


class TestLoopManagerStopLoop:
    def test_stop_loop_default_raises(self, fresh_loop_manager):
        with pytest.raises(ValueError, match="cannot be stopped"):
            fresh_loop_manager.stop_loop("default")

    def test_stop_loop_immediate_removes_loop_from_manager(self, fresh_loop_manager):
        fresh_loop_manager.create_loop("to-stop")
        assert "to-stop" in fresh_loop_manager.loops
        fresh_loop_manager.stop_loop("to-stop", immediate=True)
        assert "to-stop" not in fresh_loop_manager.loops

    def test_stop_loop_non_immediate_schedules_stop_no_raise(self, fresh_loop_manager):
        fresh_loop_manager.create_loop("to-stop-deferred")
        fresh_loop_manager.stop_loop("to-stop-deferred", immediate=False)
        time.sleep(0.05)
        if "to-stop-deferred" in fresh_loop_manager.loops:
            fresh_loop_manager.stop_loop("to-stop-deferred", immediate=True)
