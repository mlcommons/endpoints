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

"""Abstract interface for all storage backends."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any


class StorageBackend(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None:
        """Open the underlying connection or session."""

    @abc.abstractmethod
    def write(self, key: str, data: Any, **kwargs) -> None:
        """Write data identified by key."""

    @abc.abstractmethod
    def read(self, key: str, **kwargs) -> Any:
        """Read data identified by key."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Delete the item identified by key."""

    @abc.abstractmethod
    def exists(self, key: str, **kwargs) -> bool:
        """Return True if the item identified by key exists."""

    @abc.abstractmethod
    def list(self, prefix: str = "", **kwargs) -> Iterator[str]:
        """Iterate over available keys, optionally filtered by prefix."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the connection or session and release resources."""

    def __enter__(self) -> StorageBackend:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
