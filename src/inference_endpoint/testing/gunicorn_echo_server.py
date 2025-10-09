import asyncio
import json
import logging
import multiprocessing
import os
import signal
import time
import uuid

import gunicorn.app.base
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from inference_endpoint.core.types import Query
from inference_endpoint.testing.echo_server import OpenAIAdapter, QueryResult


class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(
        self,
        options=None,
        num_workers: int | None = None,
        max_osl: int | None = None,
        start_event=None,
        shutdown_event=None,
    ):
        print(
            f"Initializing StandaloneApplication [{num_workers} workers] options: {options} max_osl: {max_osl}"
        )
        self.max_osl = max_osl
        self.options = options or {}
        self._server_thread = None
        self.application = FastAPI()
        self.shutdown_event = shutdown_event
        self.start_event = start_event
        self.application.post("/v1/chat/completions")(self.chat_completions)
        self.num_workers = (
            num_workers
            if num_workers is not None
            else multiprocessing.cpu_count() * 2 + 1
        )
        super().__init__()
        print(
            f"Initialized StandaloneApplication [{num_workers} workers] options: {options} max_osl: {max_osl}"
        )

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            print(f"Setting {key} to {value}")
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

    def start(self):
        try:
            if self.shutdown_event:

                def shutdown_checker():
                    while not self.shutdown_event.is_set():
                        time.sleep(1)
                    self.stop()

                self.shutdown_task = asyncio.create_task(shutdown_checker())
            if self.start_event:
                self.start_event.set()
            print(
                f"Starting StandaloneApplication [{self.num_workers} workers] options: {self.options} max_osl: {self.max_osl} at {self.options.get('bind')}"
            )
            self.run()
        except Exception as e:
            print(f"FAILED :: Error starting StandaloneApplication: {e}")
            raise e
        finally:
            print("FINISHED:: Starting StandaloneApplication finished")

    def stop(self):
        os.kill(os.getpid(), signal.SIGKILL)

    def url(self):
        return self.options.get("bind")

    async def chat_completions(self, request: Request):
        max_osl = self.max_osl
        json_payload = await request.json()
        completion_request = OpenAIAdapter.from_json(json_payload)
        if (
            completion_request.data.get("prompt")
            and len(completion_request.data.get("prompt")) > 0
        ):
            raw_request = completion_request.data.get("prompt")
        else:
            raise ValueError("Request must contain at least one message")
        id = json_payload.get("id", str(uuid.uuid4()))
        raw_response = raw_request
        logging.debug(f"Content of request: {raw_request} - response : {raw_response}")

        if max_osl is not None and len(raw_response) > 0:
            # if max_osl is specified, it can be either larger or smaller than the length of the prompt
            # if max_osl is larger, we can repeate the prompt until we reach the max_osl
            if len(raw_response) > max_osl:
                raw_response = raw_response[:max_osl]
            # if max_osl is smaller, we can truncate the prompt
            else:
                raw_response = raw_response * (max_osl // len(raw_response) + 1)
                raw_response = raw_response[:max_osl]

        # Check if this is a streaming request
        logging.debug(f"Streaming response: {completion_request.data.get('stream')}\n")
        if completion_request.data.get("stream") is True:
            # Return SSE (Server-Sent Events) format for streaming
            return StreamingResponse(
                self._handle_streaming_response(id, completion_request, raw_response),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming: return QueryResult as before
            response = QueryResult(
                id=id,
                response_output=raw_response,
            )
            echo_response = OpenAIAdapter.to_openai_response(response).model_dump(
                mode="json"
            )
            echo_response["id"] = id
            logging.debug(f"Echo response (non-streaming): {echo_response}")
            return echo_response

    async def _handle_streaming_response(
        self,
        id: str,
        completion_request: Query,
        content: str,
    ):
        """
        Generate OpenAI-compatible streaming response in SSE format.

        Yields strings in the format: "data: {json}\n\n"
        """
        try:
            raw_response = content
            model_name = completion_request.data.get("model", "unspecified-model")

            # Send content in chunks (word by word for echo server)
            words = raw_response.split() if raw_response else []

            # Send chunks
            for i, word in enumerate(words):
                # Add space before word (except first)
                chunk_content = f" {word}" if i > 0 else word

                chunk_data = {
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_content},
                            "finish_reason": None,
                        }
                    ],
                }

                # Yield as SSE format string
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Send final chunk with finish_reason
            final_chunk = {
                "id": id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            # Yield final chunk and done marker
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logging.error(f"Error handling streaming response: {e}")
            raise e


if __name__ == "__main__":
    num_workers = 8
    options = {
        "bind": "0.0.0.0:12345",
        "workers": num_workers,
        "worker_class": "uvicorn.workers.UvicornWorker",
    }
    app = StandaloneApplication(options, num_workers=num_workers)
    print(f"Starting app :: {app} url: {app.url()}")
    app.start()
