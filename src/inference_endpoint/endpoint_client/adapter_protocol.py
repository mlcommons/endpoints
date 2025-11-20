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

"""Base class for HTTP request adapters."""

import re
from abc import ABC, abstractmethod

from inference_endpoint.core.types import Query, QueryResult


class HttpRequestAdapter(ABC):
    """
    Abstract base class for HTTP request adapters.

    Adapters convert between internal Query/QueryResult types and
    endpoint-specific formats (e.g., OpenAI, custom formats).
    """

    # SSE (Server-Sent Events) is an HTTP standard
    # Pre-compiled regex for extracting SSE data fields with JSON content
    # Matches "data: {json content}" and captures the JSON part
    SSE_DATA_PATTERN: re.Pattern[bytes] = re.compile(rb"data:\s*(\{[^\n]+\})")

    @staticmethod
    @abstractmethod
    def encode_query(query: Query) -> bytes:
        """
        Encode a Query to bytes for HTTP transmission.

        Args:
            query: Input query with prompt and parameters

        Returns:
            Encoded request bytes ready for HTTP POST
        """
        ...

    @staticmethod
    @abstractmethod
    def decode_response(response_bytes: bytes, query_id: str) -> QueryResult:
        """
        Decode HTTP response bytes to QueryResult.

        Args:
            response_bytes: Raw bytes from HTTP response
            query_id: ID for the query (to associate with result)

        Returns:
            QueryResult with extracted content
        """
        ...

    @staticmethod
    @abstractmethod
    def decode_sse_message(json_bytes: bytes) -> str:
        """
        Decode SSE message and extract content string.

        Args:
            json_bytes: Raw JSON bytes from SSE stream

        Returns:
            Content string from the SSE message
        """
        ...
