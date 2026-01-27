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

"""WebSocket server for LiveCodeBench evaluation service.

This server provides a WebSocket endpoint for running LiveCodeBench code evaluations
with real-time progress updates via callbacks.
"""

import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from lib.lcb_serve import LCBServe
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class EvaluationRequest(BaseModel):
    """Schema for evaluation request."""

    codes_dict: dict[str, list[str]] = Field(
        description="Dictionary mapping question IDs to lists of code samples"
    )
    timeout_sec: int = Field(
        60,
        description="Timeout in seconds for each test case execution",
        ge=1,
        le=3600,
    )

    @field_validator("codes_dict")
    @classmethod
    def validate_codes_dict(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        """Validate that codes_dict has correct structure."""
        if not v:
            raise ValueError("codes_dict cannot be empty")

        code_list_length = None
        for qid, code_list in v.items():
            if not isinstance(qid, str):
                raise ValueError(f"Question ID must be str, got {type(qid)}")
            if not isinstance(code_list, list):
                raise ValueError(
                    f"Code samples for question {qid} must be a list, got {type(code_list)}"
                )
            if not code_list:
                raise ValueError(f"Code list for question {qid} cannot be empty")
            if not all(isinstance(code, str) for code in code_list):
                raise ValueError(f"All code samples for question {qid} must be strings")
            if code_list_length is None:
                code_list_length = len(code_list)
            else:
                if len(code_list) != code_list_length:
                    raise ValueError("All code lists must have the same length.")

        return v


class ProgressMessage(BaseModel):
    """Message format for progress updates."""

    status: Literal["started", "progress", "completed", "error"]
    total_samples: int | None = None
    completed_samples: int | None = None
    error: str | None = None
    result: dict | None = None


# Create FastAPI app with docs disabled
app = FastAPI(
    title="LiveCodeBench Evaluation Service",
    description="WebSocket service for running LiveCodeBench code evaluations",
    version="1.0.0",
    docs_url=None,  # Disable /docs
    redoc_url=None,  # Disable /redoc
    openapi_url=None,  # Disable /openapi.json
)


# Global LCBServe instance (initialized on startup)
lcb_serve: LCBServe | None = None


class EvaluationSession:
    """Manages a single evaluation session with progress tracking and WebSocket communication."""

    def __init__(
        self,
        websocket: WebSocket,
        request: EvaluationRequest,
        lcb_serve_instance: LCBServe,
        event_loop: asyncio.AbstractEventLoop,
    ):
        self.websocket = websocket
        self.request = request
        self.lcb_serve = lcb_serve_instance
        self.event_loop = event_loop

        # Calculate total samples upfront.
        # Pydantic validation ensures all code lists have the same length and are not empty.
        # Note: next(iter(...)) is a common pattern to quickly get the first value of an iterator.
        code_samples_per_problem = len(next(iter(request.codes_dict.values())))
        self.total_samples = len(request.codes_dict) * code_samples_per_problem
        logger.info(f"- Evaluating {self.total_samples} code samples")
        logger.info(f"    # problem IDs: {len(self.request.codes_dict)}")
        logger.info(f"    code samples per problem: {code_samples_per_problem}")
        logger.info(f"    timeout: {request.timeout_sec}")

        self.completed_samples = 0

        # Queue for progress updates from callback (thread-safe)
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    def on_problem_complete(self, question_ids: list[str]) -> None:
        """Callback invoked when problems complete evaluation.

        This runs in the executor thread and safely enqueues progress updates.
        """
        self.completed_samples += len(question_ids)

        # Put progress update in queue (thread-safe)
        # Use the stored event loop since this runs in a thread pool executor
        asyncio.run_coroutine_threadsafe(
            self.progress_queue.put(
                {
                    "status": "progress",
                    "total_samples": self.total_samples,
                    "completed_samples": self.completed_samples,
                }
            ),
            self.event_loop,
        )

    async def send_message(self, message: ProgressMessage) -> None:
        """Send a progress message to the client."""
        await self.websocket.send_json(message.model_dump())

    async def run_evaluation(self) -> dict:
        """Run evaluation in executor and return result.

        Returns:
            dict with keys: "success" (bool), "result" (dict) or "error" (str)
        """
        try:
            result = await self.event_loop.run_in_executor(
                None,
                self.lcb_serve.evaluate,
                self.request.codes_dict,
                self.request.timeout_sec,
                self.on_problem_complete,
            )
            return {"success": True, "result": result}
        except Exception as e:
            tb_string = traceback.format_exc()
            logger.error(f"Evaluation failed with exception: {e}")
            logger.error(tb_string)
            return {
                "success": False,
                "error": str(e),
                "traceback": tb_string,
            }

    async def stream_progress_updates(self, eval_task: asyncio.Task) -> None:
        """Stream progress updates to client while evaluation is running."""
        while not eval_task.done():
            try:
                # Wait for progress update with timeout
                progress_update = await asyncio.wait_for(
                    self.progress_queue.get(), timeout=0.1
                )
                await self.websocket.send_json(progress_update)
            except TimeoutError:
                # No progress update yet, continue waiting
                continue

        # Drain any remaining progress updates
        while not self.progress_queue.empty():
            progress_update = await self.progress_queue.get()
            await self.websocket.send_json(progress_update)

    async def send_final_result(self, eval_result: dict) -> None:
        """Process and send final evaluation result to client."""
        if not eval_result["success"]:
            # Evaluation failed
            await self.send_message(
                ProgressMessage(
                    status="error",
                    error=f"Evaluation failed: {eval_result['error']}",
                )
            )
        else:
            # Evaluation succeeded
            result_dict = eval_result["result"]

            # Calculate total passed samples
            total_passed = sum(
                sum(1 for passed in results if passed)
                for results in result_dict.values()
            )

            logger.info(f"Completed {self.total_samples} ({total_passed} passed)")
            logger.info(f"lcb_serve cache stats: {self.lcb_serve.cache_info()!r}")

            # Send final result
            await self.send_message(
                ProgressMessage(
                    status="completed",
                    total_samples=self.total_samples,
                    completed_samples=self.total_samples,
                    result={
                        "total_samples": self.total_samples,
                        "total_passed": total_passed,
                        "results": result_dict,  # qid -> list[bool]
                    },
                )
            )

    async def execute(self) -> None:
        """Execute the evaluation session with progress streaming."""
        # Send started message
        await self.send_message(
            ProgressMessage(
                status="started",
                total_samples=self.total_samples,
                completed_samples=0,
            )
        )

        # Start evaluation task
        eval_task = asyncio.create_task(self.run_evaluation())

        # Stream progress updates while evaluation runs
        await self.stream_progress_updates(eval_task)

        # Get and send final result
        eval_result = await eval_task
        await self.send_final_result(eval_result)


@app.on_event("startup")
async def startup_event():
    """Initialize LCBServe on server startup.

    Configuration is read from environment variables with defaults:
    - LCB_VERSION_TAG (default: "release_v6")
    - LCB_USE_LITE (default: "true")
    - LCB_N_WORKERS (default: None, auto-detect)
    - LCB_DATASETS_DIR (default: "/opt/LiveCodeBench_Datasets")
    - LCB_AUTO_GENERATE_DATASET (default: "true")
    - LCB_SERVER_DEBUG (default: not set, enables DEBUG logging if set)
    - LCB_TEST_CACHE_SIZE (default: None, no limit)
    - LCB_PRELOAD_TESTS (default: "false")
    """
    global lcb_serve

    # Configure logging level
    debug_mode = os.getenv("LCB_SERVER_DEBUG")
    if debug_mode:
        logging.basicConfig(level=logging.DEBUG, force=True)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled via LCB_SERVER_DEBUG")
    else:
        logging.basicConfig(level=logging.INFO, force=True)
        logger.setLevel(logging.INFO)

    # Read configuration from environment variables with defaults
    version_tag = os.getenv("LCB_VERSION_TAG", "release_v6")

    # Parse n_workers (None or int)
    n_workers_str = os.getenv("LCB_N_WORKERS")
    n_workers = int(n_workers_str) if n_workers_str else None

    # Parse paths
    datasets_dir = Path(os.getenv("LCB_DATASETS_DIR", "/opt/LiveCodeBench_Datasets"))

    # Parse boolean for auto_generate_dataset
    auto_generate_str = os.getenv("LCB_AUTO_GENERATE_DATASET", "false").lower()
    auto_generate_dataset = auto_generate_str in ("true", "1", "yes", "on")

    # Parse test_case_cache_limit (None or int)
    test_cache_size_str = os.getenv("LCB_TEST_CACHE_SIZE")
    if not test_cache_size_str or test_cache_size_str.lower() in (
        "none",
        "inf",
        "infinity",
        "unlimited",
    ):
        test_suite_cache_limit = None
    else:
        test_suite_cache_limit = int(test_cache_size_str)
        if test_suite_cache_limit <= 0:
            raise ValueError("LCB_TEST_CACHE_SIZE must be a positive integer")

    # Parse boolean for preload_test_cases
    preload_tests_str = os.getenv("LCB_PRELOAD_TESTS", "true").lower()
    preload_test_cases = preload_tests_str in ("true", "1", "yes", "on")

    logger.info("Initializing LCBServe with configuration:")
    logger.info(f"  version_tag: {version_tag}")
    logger.info(f"  n_workers: {n_workers or 'auto-detect'}")
    logger.info(f"  datasets_dir: {datasets_dir}")
    logger.info(f"  auto_generate_dataset: {auto_generate_dataset}")
    logger.info(f"  test_suite_cache_limit: {test_suite_cache_limit or 'unlimited'}")
    logger.info(f"  preload_test_cases: {preload_test_cases}")

    lcb_serve = LCBServe(
        version_tag=version_tag,
        n_workers=n_workers,
        datasets_dir=datasets_dir,
        auto_generate_dataset=auto_generate_dataset,
        test_suite_cache_limit=test_suite_cache_limit,
        preload_test_cases=preload_test_cases,
    )
    logger.info("LCBServe initialized successfully")


@app.get("/info")
async def get_info():
    """Get information about the LCBServe instance.

    Returns:
        JSON response with version tag, number of workers, and dataset size.
    """
    if lcb_serve is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "LCBServe not initialized",
            },
        )

    try:
        return JSONResponse(
            status_code=200,
            content={
                "version_tag": lcb_serve.version_tag,
                "n_workers": lcb_serve.n_workers,
                "dataset_size": len(lcb_serve.df),
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Error retrieving info: {e!s}",
                "traceback": traceback.format_exc(),
            },
        )


@app.websocket("/evaluate")
async def websocket_evaluate(websocket: WebSocket):
    """WebSocket endpoint for LiveCodeBench evaluation.

    Accepts JSON payload with codes_dict and timeout_sec.
    Streams progress updates in real-time and returns final results.
    The actual code is run in subprocesses to avoid corrupting the main process.
    """
    await websocket.accept()

    try:
        # Receive and validate request
        raw_data = await websocket.receive_text()
        try:
            data_dict = json.loads(raw_data)
            request = EvaluationRequest(**data_dict)
        except (json.JSONDecodeError, ValueError) as e:
            await websocket.send_json(
                ProgressMessage(
                    status="error",
                    error=f"Invalid request format: {e!s}",
                ).model_dump()
            )
            await websocket.close()
            return

        # Create and execute evaluation session
        loop = asyncio.get_event_loop()
        session = EvaluationSession(websocket, request, lcb_serve, loop)
        await session.execute()
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_evaluate: {e}", exc_info=True)
        try:
            await websocket.send_json(
                ProgressMessage(
                    status="error",
                    error=f"Unexpected error: {e!s}",
                ).model_dump()
            )
        except Exception as send_error:
            logger.warning(f"Failed to send error message to client: {send_error}")
    finally:
        try:
            await websocket.close()
        except Exception as close_error:
            logger.warning(f"Failed to close websocket: {close_error}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=13835,
        log_level="info",
        timeout_keep_alive=7200,  # 2 hours - allow long-running evaluations
        ws_ping_interval=30,  # Send WebSocket ping every 30 seconds
        ws_ping_timeout=10,  # Wait 10 seconds for pong response
    )
