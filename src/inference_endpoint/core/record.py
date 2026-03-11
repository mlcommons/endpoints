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

import enum
import time
from typing import Any, ClassVar, Final

import msgspec

from .types import OUTPUT_TYPE, ErrorData

TOPIC_FRAME_SIZE: Final[int] = 40
"""int: Fixed bytesize for the encoded topic string. PUB messages will be prefixed by a
topic string corresponding to the EventType. This topic will be null-padded to this fixed
size to allow the publisher to use single-frame sends for performance optimization.
Using a fixed size also allows subscribers to strip a fixed prefix size without needing
to inspect the message itself.

The fixed sized is chosen to be the smallest multiple of 8 bytes that is greater than or
equal to the length of the longest topic string.
"""


class EventTypeMeta(enum.EnumMeta):
    """Metaclass for event kind enums classes.

    This must be done with a metaclass rather than mixin-classes, decorators, or __init_subclass__
    mechanisms, since in those cases, the defined child classes will be valid subclasses of enum.Enum
    or enum.StrEnum.
    If that is the case, then custom enc_hooks cannot be used since msgspec will always see that the
    object is a built-in type (enum.Enum) and return before it reaches custom user-defined hooks.

    Using a custom metaclass won't match any built-in encoders since msgspec's built-in type support
    explicitly checks for Enum type-coercion based on if the Enum's metaclass is explicitly enum.<metaclass>:
    https://github.com/jcrist/msgspec/blob/a9ed8f12f11269704aa25680a2273287485c9f5a/src/msgspec/_core.c#L13448-L13450
    """

    _REGISTRY: ClassVar[dict[str, "EventType"]] = {}

    def __new__(mcls, cls_name, bases, classdict, **kwargs):
        enum_cls = super().__new__(mcls, cls_name, bases, classdict, **kwargs)
        if cls_name == "EventType":
            return enum_cls

        category = classdict.get("__category__")
        if category is None:
            raise ValueError(
                "EventTypeEnum should define class-level '__category__' attribute."
            )

        for member in enum_cls:
            member.topic = f"{category}.{member._value_}"
            member.topic_bytes = member.topic.encode("utf-8")
            if len(member.topic_bytes) > TOPIC_FRAME_SIZE:
                raise TypeError(
                    f"Topic '{member.topic}' is too long to fit in the topic frame size of {TOPIC_FRAME_SIZE} bytes."
                )
            member.topic_bytes_padded = member.topic_bytes.ljust(
                TOPIC_FRAME_SIZE, b"\0"
            )

            # Update _value2member_map_ so that ChildEventType("category.value") will
            # return the correct member.
            enum_cls._value2member_map_[member.topic] = member

            # If in the future, msgspec no longer filters Enums based on metaclass, then in here,
            # we can do: `member._value_ = member.topic` to force the value of the enum to be the
            # message routing topic so that `Enum.MEMBER.value` returns "category.value"

        # Save a reference by category to the enum class.
        EventTypeMeta._REGISTRY[category] = enum_cls

        # Add a classmethod to get category
        enum_cls.category = classmethod(lambda cls: category)
        return enum_cls

    @classmethod
    def from_topic(cls, topic: str) -> "EventType":
        parts = topic.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid topic '{topic}'")
        category, value = parts
        if category not in cls._REGISTRY:
            raise ValueError(f"Unknown category '{category}'")
        enum_cls = cls._REGISTRY[category]
        # MyPy doesn't recognize that enum_cls is an enum.Enum
        return enum_cls(value)  # type: ignore[operator]


class EventType(enum.Enum, metaclass=EventTypeMeta):
    @classmethod
    def encode_hook(cls, obj: Any) -> Any:
        if isinstance(obj, cls):
            # MyPy doesn't recognize that the .topic attribute is defined by __new__ in the metaclass.
            return obj.topic  # type: ignore[attr-defined]
        raise NotImplementedError(f"Encoding {obj} to type {cls} is not supported")

    @classmethod
    def decode_hook(cls, type: type, obj: Any) -> Any:
        if type is cls and isinstance(obj, str):
            return EventTypeMeta.from_topic(obj)
        raise NotImplementedError(
            f"Decoding topic '{obj}' to type {type} is not supported"
        )


class SessionEventType(EventType):
    __category__ = "session"

    STARTED = "started"
    ENDED = "ended"
    STOP_LOADGEN = "stop_loadgen"
    STOP_PERFORMANCE_TRACKING = "stop_performance_tracking"


class ErrorEventType(EventType):
    __category__ = "error"

    GENERIC = "generic"
    LOADGEN = "loadgen"
    SESSION = "session"
    CLIENT = "client"


class SampleEventType(EventType):
    __category__ = "sample"

    ISSUED = "issued"
    COMPLETE = "complete"
    RECV_FIRST = "recv_first"
    RECV_NON_FIRST = "recv_non_first"
    CLIENT_SEND = "client_send"
    CLIENT_RESP_DONE = "client_resp_done"
    TRANSPORT_SENT = "transport_sent"
    TRANSPORT_RECV = "transport_recv"


class EventRecord(msgspec.Struct, kw_only=True, frozen=True, gc=False):  # type: ignore[call-arg]
    """A record of an event that occurs throughout the inference process."""

    event_type: EventType
    timestamp_ns: int = msgspec.field(default_factory=time.monotonic_ns)
    sample_uuid: str = ""
    data: OUTPUT_TYPE | ErrorData | None = None


_ENCODER = msgspec.msgpack.Encoder(enc_hook=EventType.encode_hook)
_DECODER = msgspec.msgpack.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)


def encode_event_record(event_record: EventRecord) -> tuple[bytes, bytes]:
    """Encodes an EventRecord into a tuple of (topic_bytes_padded, payload_bytes).

    Args:
        event_record: The EventRecord to encode.

    Returns:
        A tuple of (topic_bytes_padded, payload_bytes).
    """
    # MyPy doesn't recognize custom attributes defined by __new__ in the metaclass.
    return event_record.event_type.topic_bytes_padded, _ENCODER.encode(event_record)  # type: ignore[attr-defined]


def decode_event_record(payload: bytes) -> EventRecord:
    return _DECODER.decode(payload)
