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


import datetime

import openai_harmony as harmony
from transformers import AutoTokenizer, PreTrainedTokenizer


class Harmonizer:
    """Utility class for using the OpenAI Harmony library to pre-process user prompts.

    See https://cookbook.openai.com/articles/openai-harmony for more details.
    """

    _tokenizers: dict[str, PreTrainedTokenizer] = {}
    _encodings: dict[str, "harmony.HarmonyEncoding"] = {}

    @classmethod
    def get_tokenizer(cls, tokenizer_name: str) -> PreTrainedTokenizer:
        """Get a tokenizer from the cache, or load from HuggingFace.

        Args:
            tokenizer_name: The name of the tokenizer to use.

        Returns:
            The tokenizer.
        """
        if tokenizer_name not in cls._tokenizers:
            cls._tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(
                tokenizer_name
            )
        return cls._tokenizers[tokenizer_name]

    @classmethod
    def get_encoding(cls, encoding_name: str) -> "harmony.HarmonyEncoding":
        """Get an encoding from the cache, or load from OpenAI Harmony.

        Args:
            encoding_name: The name of the encoding to use.

        Returns:
            The encoding.
        """
        if encoding_name not in cls._encodings:
            _enc_name = getattr(harmony.HarmonyEncodingName, encoding_name)
            cls._encodings[encoding_name] = harmony.load_harmony_encoding(_enc_name)
        return cls._encodings[encoding_name]

    def __init__(
        self,
        tokenizer_name: str = "openai/gpt-oss-120b",
        encoding_name: str = "HARMONY_GPT_OSS",
        reasoning_effort: str = "high",
        conversation_start_date: str | None = None,
    ):
        """
        Args:
            tokenizer_name: The name of the tokenizer to use for the dataset.
                (Default: openai/gpt-oss-120b)
            encoding_name: The name of the HarmonyEncoding enum member to use. If not a valid
                enum member name, the string value will be used as-is. (Default: HARMONY_GPT_OSS)
            reasoning_effort: Low, Medium, or High. Case-insensitive. (Default: High)
            conversation_start_date: An ISO format date string for the start of the conversation.
                If None, the current date will be used. (Default: None)
        """
        self.tokenizer_name = tokenizer_name
        self._encoding_name = encoding_name
        self._reasoning_effort = reasoning_effort
        self._conversation_start_date = (
            conversation_start_date or datetime.date.today().isoformat()
        )
        self.tokenizer: PreTrainedTokenizer | None = None
        self.encoding: "harmony.HarmonyEncoding | None" = None
        self.system_message: "harmony.SystemContent | None" = None

    def _ensure_loaded(
        self,
    ) -> "tuple[PreTrainedTokenizer, harmony.HarmonyEncoding, harmony.SystemContent]":
        if self.tokenizer is None:
            tokenizer = self.__class__.get_tokenizer(self.tokenizer_name)
            encoding = self.__class__.get_encoding(self._encoding_name)

            _effort = getattr(harmony.ReasoningEffort, self._reasoning_effort.upper())

            system_message = (
                harmony.SystemContent.new()
                .with_reasoning_effort(_effort)
                .with_conversation_start_date(self._conversation_start_date)
            )

            # Assign to instance variables only after all steps succeed to ensure atomicity
            self.tokenizer = tokenizer
            self.encoding = encoding
            self.system_message = system_message

        assert self.tokenizer is not None
        assert self.encoding is not None
        assert self.system_message is not None
        return self.tokenizer, self.encoding, self.system_message

    def __call__(self, user_prompt: str, tokenize: bool = True) -> str | list[int]:
        """Convert a user prompt to a Harmony-compatible format.

        Args:
            user_prompt: The user prompt to convert.
            tokenize: Whether to tokenize the user prompt. If True, the user prompt will be returned
                as a tokenized list of integers. Otherwise, the user prompt will be returned as a string.
                (Default: True)

        Returns:
            The Harmony-compatible format of the user prompt.
        """
        tokenizer, encoding, system_message = self._ensure_loaded()
        conv = harmony.Conversation.from_messages(
            [
                harmony.Message.from_role_and_content(
                    harmony.Role.SYSTEM, system_message
                ),
                harmony.Message.from_role_and_content(harmony.Role.USER, user_prompt),
            ]
        )
        toks = encoding.render_conversation_for_completion(conv, harmony.Role.ASSISTANT)

        if tokenize:
            return toks
        else:
            return tokenizer.decode(toks, skip_special_tokens=False)

    def to_text(self, toks: list[int]) -> str:
        """Convert a tokenized sequence to a string.

        Args:
            toks: The tokenized sequence to convert.

        Returns:
            The string representation of the sequence.
        """
        tokenizer, _, _ = self._ensure_loaded()
        return tokenizer.decode(toks, skip_special_tokens=False)

    def to_tokens(self, text: str) -> list[int]:
        """Convert a string to a tokenized sequence.

        Args:
            text: The string to convert.

        Returns:
            The tokenized sequence.
        """
        tokenizer, _, _ = self._ensure_loaded()
        return tokenizer.encode(text, add_special_tokens=True)
