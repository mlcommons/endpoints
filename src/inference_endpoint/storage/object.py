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

"""Object storage backends.

Currently implemented: S3 (requires boto3).
GCS and local filesystem backends can be added here following the same pattern.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

from .base import StorageBackend


class S3ObjectBackend(StorageBackend):
    """Object storage backed by AWS S3 (requires boto3)."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._client = None

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def connect(self) -> None:
        import boto3

        self._client = boto3.client("s3")

    def write(self, key: str, data: Any, **kwargs) -> None:
        self._client.upload_fileobj(io.BytesIO(data), self.bucket, self._key(key))

    def read(self, key: str, **kwargs) -> bytes:
        buf = io.BytesIO()
        self._client.download_fileobj(self.bucket, self._key(key), buf)
        return buf.getvalue()

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def exists(self, key: str) -> bool:
        import botocore.exceptions

        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except botocore.exceptions.ClientError:
            return False

    def list(self, prefix: str = "") -> Iterator[str]:
        full_prefix = self._key(prefix) if prefix else (self.prefix or "")
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if self.prefix:
                    k = k[len(self.prefix) + 1 :]
                yield k

    def close(self) -> None:
        self._client = None
