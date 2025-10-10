import re
import time

import msgspec
from inference_endpoint.core.types import Query, QueryResult

from .openai_types_gen import (
    ChatCompletionResponseMessage,
    Choice,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    FinishReason,
    Logprobs,
    ModelIdsShared,
    Object7,
    ReasoningEffort,
    Role5,
    Role6,
    ServiceTier,
)


# msgspec structs for typed SSE message parsing (OpenAI streaming format)
class SSEDelta(msgspec.Struct):
    """SSE delta object containing content."""

    content: str = ""


class SSEChoice(msgspec.Struct):
    """SSE choice object containing delta."""

    delta: SSEDelta = msgspec.field(default_factory=SSEDelta)
    finish_reason: str | None = None


class SSEMessage(msgspec.Struct):
    """SSE message structure for OpenAI streaming responses."""

    choices: list[SSEChoice] = msgspec.field(default_factory=list)


class OpenAIAdapter:
    """Adapter for OpenAI API."""

    # Pre-compiled regex for extracting SSE data fields with JSON content
    # Matches "data: {json content}" and captures the JSON part
    SSE_DATA_PATTERN = re.compile(rb"data:\s*(\{[^\n]+\})", re.MULTILINE)

    @staticmethod
    def to_openai_request(query: Query) -> CreateChatCompletionRequest:
        """Convert a Query to an OpenAI request."""
        if "prompt" not in query.data:
            raise ValueError("prompt not found in json_value")

        return CreateChatCompletionRequest(
            model=ModelIdsShared(query.data.get("model", "no-model-name")),
            service_tier=ServiceTier.auto,
            reasoning_effort=ReasoningEffort.medium,
            messages=[{"role": Role5.user, "content": query.data["prompt"]}],
            stream=query.data.get("stream", False),
            max_completion_tokens=query.data.get("max_completion_tokens", 100),
            temperature=query.data.get("temperature", 0.7),
        )

    @staticmethod
    def from_openai_request(request: CreateChatCompletionRequest) -> Query:
        """Convert an OpenAI request to a Query."""
        if not request.messages or len(request.messages) == 0:
            raise ValueError("Request must contain at least one message")
        return Query(
            data={
                "prompt": request.messages[0].root.content,
                "model": request.model,
                "stream": request.stream,
            },
        )

    @staticmethod
    def from_openai_response(response: CreateChatCompletionResponse) -> QueryResult:
        """Convert an OpenAI response to a QueryResult."""
        if not response.choices:
            raise ValueError("Response must contain at least one choice")
        return QueryResult(
            id=response.id,
            response_output=response.choices[0].message.content,
        )

    @staticmethod
    def to_openai_response(result: QueryResult) -> CreateChatCompletionResponse:
        """Convert a QueryResult to an OpenAI response."""
        return CreateChatCompletionResponse(
            id=result.id,
            choices=[
                Choice(
                    finish_reason=FinishReason.stop,
                    index=0,
                    message=ChatCompletionResponseMessage(
                        content=result.response_output, role=Role6.assistant, refusal=""
                    ),
                    logprobs=Logprobs(content=[], refusal=[]),
                )
            ],
            created=int(time.time()),
            model="model",
            object=Object7.chat_completion,
            service_tier=ServiceTier.auto,
        )
