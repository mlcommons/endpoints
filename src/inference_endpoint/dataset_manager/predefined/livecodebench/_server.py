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
import os
import shutil
import traceback
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from lib.lcb_serve import LCBServe


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

        # Calculate total samples upfront
        self.total_samples = sum(len(codes) for codes in request.codes_dict.values())
        samples_per_problem = len(list(request.codes_dict.values())[0])
        print(f"- Evaluating {self.total_samples} code samples")
        print(f"    # problem IDs: {len(self.request.codes_dict)}")
        print(f"    code samples per problem: {samples_per_problem}")
        print(f"    timeout: {request.timeout_sec}")

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
            print(tb_string)
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
    """
    global lcb_serve

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

    print("Initializing LCBServe with configuration:")
    print(f"  version_tag: {version_tag}")
    print(f"  n_workers: {n_workers or 'auto-detect'}")
    print(f"  datasets_dir: {datasets_dir}")
    print(f"  auto_generate_dataset: {auto_generate_dataset}")

    lcb_serve = LCBServe(
        version_tag=version_tag,
        use_lite=True,  # Non-lite version of LCB is not supported
        n_workers=n_workers,
        datasets_dir=datasets_dir,
        auto_generate_dataset=auto_generate_dataset,
    )
    print("LCBServe initialized successfully")


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


@app.get("/copy_dataset")
async def copy_dataset():
    """Copy dataset file from LCB_DATASETS_DIR to /mnt/datasets.

    Returns:
        JSON response with status and details about the copy operation.
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
        # Get the source dataset directory and version tag
        datasets_dir = lcb_serve.datasets_dir
        version_tag = lcb_serve.version_tag

        # Construct the source file path
        source_file = datasets_dir / f"livecodebench_{version_tag}.parquet"

        # Check if source file exists
        if not source_file.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"Dataset file not found: {source_file}",
                },
            )

        # Check if destination directory exists
        dest_dir = Path("/mnt/datasets")
        if not dest_dir.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"Destination directory does not exist: {dest_dir}",
                },
            )

        # Construct destination file path
        dest_file = dest_dir / source_file.name

        # Copy the file
        shutil.copy2(source_file, dest_file)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Dataset copied successfully",
            },
        )

    except PermissionError as e:
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": f"Permission denied: {e!s}",
            },
        )
    except OSError as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"OS error during copy: {e!s}",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Unexpected error: {e!s}",
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
        # Client disconnected, nothing to do
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                ProgressMessage(
                    status="error",
                    error=f"Unexpected error: {e!s}",
                ).model_dump()
            )
        except Exception:
            # If we can't send the error, just close
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


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
