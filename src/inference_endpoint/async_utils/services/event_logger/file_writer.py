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

"""JSONL file writer for event records.

If additional file-based writers are needed in the future, the shared file I/O
logic (open, flush, close, flush_interval) should be refactored out of
JSONLWriter into a ``StreamedFileWriter`` base class sitting between
``RecordWriter`` and the concrete writers.
"""

from pathlib import Path

import msgspec
from inference_endpoint.core.record import EventRecord, EventType

from .writer import RecordWriter


class JSONLWriter(RecordWriter):
    """Writes event records to a JSONL file."""

    extension = ".jsonl"

    def __init__(
        self,
        file_path: Path,
        mode: str = "w",
        flush_interval: int | None = None,
        **kwargs: object,
    ):
        super().__init__(flush_interval=flush_interval)
        self.file_path = Path(file_path).with_suffix(self.extension)
        self.file_obj = self.file_path.open(mode=mode)  # type: ignore[assignment]
        self.encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)

    def _write_record(self, record: EventRecord) -> None:
        if self.file_obj is not None:
            self.file_obj.write(self.encoder.encode(record).decode("utf-8") + "\n")

    def flush(self) -> None:
        if self.file_obj is not None:
            self.file_obj.flush()
        super().flush()

    def close(self) -> None:
        if self.file_obj is not None:
            try:
                self.flush()
                self.file_obj.close()
            except (OSError, FileNotFoundError):
                pass
            finally:
                self.file_obj = None  # type: ignore[assignment]
