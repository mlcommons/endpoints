"""LoadGenerator integration for HTTPEndpointClient."""

import asyncio
import logging
import threading
from typing import Any

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.load_generator import SampleIssuer
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.load_generator.sample import Sample
from inference_endpoint.profiling import profile

logger = logging.getLogger(__name__)


class HttpClientSampleIssuer(SampleIssuer):
    """SampleIssuer using HTTPEndpointClient with async response handling.

    Responsibilities:
    - Send queries via HTTP client
    - Route responses to appropriate sample callbacks
    """

    def __init__(
        self,
        http_client: HTTPEndpointClient,
    ):
        super().__init__()
        self.http_client = http_client

        # Task for handling responses
        self.response_task: asyncio.Task | None = None

        # Map query ID -> Sample for routing
        # Only accessed from event loop thread (lock-free)
        self.query_id_to_sample: dict[str, Sample] = {}

        # Signals when all pending queries complete
        self._all_complete_event = threading.Event()

        # Shutdown flag for response handler
        self._shutdown = False

    def start(self):
        """Start response handler on the HTTP client's event loop."""
        self.response_task = asyncio.run_coroutine_threadsafe(
            self.handle_responses(), self.http_client.loop
        )

    @profile
    async def handle_responses(self):
        """Handle all responses in async loop. Routes responses to sample callbacks."""
        while not self._shutdown:
            try:
                response = await self.http_client.get_ready_responses_async()
                if response is None:  # timed out without a response
                    continue

                # Safe: single-threaded access (event loop thread)
                sample = self.query_id_to_sample.get(response.id)

                assert (
                    sample is not None
                ), f"Sample not found for response: {response.id}"

                # Route to appropriate callback based on response type
                match response:
                    case StreamChunk(is_complete=False):
                        if response.metadata.get("first_chunk", False):
                            sample.on_first_chunk(response)
                        else:
                            sample.on_non_first_chunk(response)

                    case StreamChunk(is_complete=True):
                        raise NotImplementedError(
                            "StreamChunk(is_complete=True) should not be received, QueryResult is expected instead"
                        )

                    case QueryResult(error=err) if err:
                        logger.error(f"Error in request {response.id}: {err}")
                        self.query_id_to_sample.pop(response.id, None)

                    case QueryResult():
                        # Final response for both streaming and non-streaming
                        sample.on_complete(response)

                        # Remove from map and check if all complete
                        self.query_id_to_sample.pop(response.id, None)
                        if len(self.query_id_to_sample) == 0:
                            self._all_complete_event.set()

                    case _:
                        raise ValueError(f"Unexpected response type: {type(response)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in response handler: {e}", exc_info=True)
                continue

    @profile
    def issue(self, sample: Sample):
        """Issue sample to HTTP endpoint.

        Flow:
        1. Create Query from sample data, and to track response
        2. Issue via HTTP client
        4. Call REQUEST_SENT callback on Query
        """
        # Convert int uuid to string for consistency with Query.id type
        query_id: str = str(sample.uuid)
        query = Query(id=query_id, data=sample.data)

        # Schedule state mutation on event loop thread
        # This ensures all dict access happens from single thread
        def _add_to_map(query_id: str, sample: Sample):
            self.query_id_to_sample[query_id] = sample

        self.http_client.loop.call_soon_threadsafe(_add_to_map, query_id, sample)

        # Issue via HTTP client (thread-safe, non-blocking)
        self.http_client.issue_query(query)

        # Notify that request was issued
        sample.callbacks[SampleEvent.REQUEST_SENT](query)

    def wait_for_all_complete(self, timeout: float | None = None):
        """Wait (blocking) for all pending queries to complete.

        Args:
            timeout: Maximum time to wait in seconds. None = wait forever.

        Returns:
            True if all complete, False if timeout
        """
        result = self._all_complete_event.wait(timeout=timeout)
        return result

    def shutdown(self):
        """Shutdown issuer response handler."""
        self._shutdown = True

        if self.response_task:
            self.response_task.cancel()

    def process_sample_data(self, s_uuid: int, sample_data: Any):
        raise NotImplementedError(
            "HttpClientSampleIssuer does not implement process_sample_data"
        )
