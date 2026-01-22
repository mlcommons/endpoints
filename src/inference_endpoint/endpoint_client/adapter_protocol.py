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

"""Base class for HTTP request adapters."""

import re
from abc import ABC, abstractmethod

from inference_endpoint.config.schema import ModelParams
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.dataset_manager.transforms import Transform


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

    @classmethod
    @abstractmethod
    def dataset_transforms(
        cls,
        model_params: ModelParams,
    ) -> list[Transform]:
        """Returns a list of transforms to apply to the dataset such that each row,
        when converted to a dictionary, can be used as the `.data` field of a Query.

        It is expected that these transforms will be applied after other transforms,
        such that the input dataframe will contain a column `prompt` (and optionally
        `system`). There can be any arbitrary number of extraneous columns in the
        dataframe, which must be filtered out. As such, all adapter dataset transforms
        should include a `ColumnFilter` transform to ensure that when a row is converted
        to a dictionary, only the necessary keys are present.

        Args:
            model_params: The model parameters for the endpoint to use
        """
        raise NotImplementedError("dataset_transforms not implemented")

    @classmethod
    @abstractmethod
    def encode_query(cls, query: Query) -> bytes:
        """
        Encode a Query to bytes for HTTP transmission.

        Args:
            query: Input query with prompt and parameters

        Returns:
            Encoded request bytes ready for HTTP POST
        """
        raise NotImplementedError("encode_query not implemented")

    @classmethod
    @abstractmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        """
        Decode HTTP response bytes to QueryResult.

        Args:
            response_bytes: Raw bytes from HTTP response
            query_id: ID for the query (to associate with result)

        Returns:
            QueryResult with extracted content
        """
        raise NotImplementedError("decode_response not implemented")

    @classmethod
    @abstractmethod
    def decode_sse_message(cls, json_bytes: bytes) -> str:
        """
        Decode SSE message and extract content string.

        Args:
            json_bytes: Raw JSON bytes from SSE stream

        Returns:
            Content string from the SSE message
        """
        raise NotImplementedError("decode_sse_message not implemented")

    @classmethod
    def parse_sse_chunk(cls, buffer: bytes, end_pos: int) -> list[str]:
        """
        Parse SSE chunk and extract all content strings.

        Extracts JSON documents from SSE stream and decodes them to content strings.
        Silently ignores non-content SSE messages (role, finish_reason, etc).

        Args:
            buffer: Byte buffer containing SSE data
            end_pos: End position in buffer to parse up to

        Returns:
            List of content strings extracted from the SSE chunk
        """
        json_docs = cls.SSE_DATA_PATTERN.findall(buffer[:end_pos])
        parsed_contents = []

        try:
            for json_doc in json_docs:
                content = cls.decode_sse_message(json_doc)
                parsed_contents.append(content)
        except Exception:
            # Normal for non-content SSE messages (role, finish_reason, etc)
            pass

        return parsed_contents
