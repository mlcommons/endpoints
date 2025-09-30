import logging

import aiohttp
import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.dataset_manager.dataloader import (
    DeepSeekR1ChatCompletionDataLoader,
)
from inference_endpoint.openai.openai_adapter import OpenAIAdapter
from inference_endpoint.openai.openai_types_gen import CreateChatCompletionResponse
from inference_endpoint.testing.echo_server import EchoServer


class OracleServer(EchoServer):
    def __init__(self, file_path):
        """
        Initialize the Oracle server with a dataset and load predefined prompt-response mappings.

        The server loads chat completion samples from the specified file path using a custom parser.
        Each sample is mapped from its input prompt to its reference output, allowing subsequent
        retrieval of responses based on exact prompt matching.

        Args:
            file_path (str): Path to the dataset file containing chat completion samples
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.file_path = file_path

        def parser(x):
            """
            Extract the prompt and reference output from a dataset sample object.

            Converts a dataset sample into a dictionary with 'prompt' and 'output' keys,
            using the sample's text input as the prompt and reference output as the response.

            Returns:
                dict: A dictionary with 'prompt' and 'output' keys derived from the input sample.
            """
            return {"prompt": x.text_input, "output": x.ref_output}

        self.parser = parser
        data_loader = DeepSeekR1ChatCompletionDataLoader(
            self.file_path, parser=self.parser
        )
        data_loader.load()
        self.data = {}
        for i in range(data_loader.num_samples()):
            sample = data_loader.load_sample(i)
            self.data[sample["prompt"]] = sample["output"]

    def get_response(self, request: str) -> str:
        """
        Retrieve a predefined response for a given request from the loaded dataset.

        Returns the stored output corresponding to the input request. If no matching
        response is found, returns a default "No response found" message.

        Args:
            request (str): The input prompt to look up in the dataset.

        Returns:
            str: The matching output for the request, or a default message if not found.
        """
        return self.data.get(request, "No response found")


@pytest.fixture
def mock_http_oracle_server(ds_pickle_dataset_path):
    """
    Pytest fixture that creates and manages a mock HTTP oracle server for dataset-driven testing.

    Creates an OracleServer instance from a specified dataset pickle file, starts the server
    on a dynamically allocated port, and manages its lifecycle during testing.

    Args:
        ds_pickle_dataset_path (str): Path to the dataset pickle file containing chat completion samples

    Yields:
        OracleServer: A running mock HTTP server serving predefined responses from the dataset

    Raises:
        RuntimeError: If any errors occur during server setup or execution
    """
    # Create and start the server with dynamic port allocation (port=0)
    server = OracleServer(ds_pickle_dataset_path)
    server.start()

    try:
        yield server
    except Exception as e:
        raise RuntimeError(f"Mock Oracle Server error: {e}") from e
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_ds_chat_completion_data_loader_with_oracle_server(
    ds_pickle_dataset_path, mock_http_oracle_server
):
    """
    Test the DeepSeekR1ChatCompletionDataLoader by performing a roundtrip request through a mock HTTP Oracle server.

    Validates the end-to-end flow of loading dataset samples, transforming requests to OpenAI format,
    sending requests to a mock server, and verifying the server's responses match expected outputs.

    The test iterates through each sample in the dataset, sends an HTTP POST request to the mock server,
    and checks that the server returns a response matching the sample's reference output.
    """

    def parser(x):
        return {"prompt": x.text_input, "output": x.ref_output}

    ds_chat_completion_data_loader = DeepSeekR1ChatCompletionDataLoader(
        ds_pickle_dataset_path, parser=parser
    )
    ds_chat_completion_data_loader.load()
    assert ds_chat_completion_data_loader.num_samples() == 5
    for i in range(ds_chat_completion_data_loader.num_samples()):
        sample = ds_chat_completion_data_loader.load_sample(i)
        async with aiohttp.ClientSession() as session:
            payload = OpenAIAdapter.to_openai_request(
                Query(
                    id="test-chat-completions",
                    data={"prompt": str(sample["prompt"]), "model": "test-model"},
                )
            ).model_dump(mode="json")

            async with session.post(
                f"{mock_http_oracle_server.url}/v1/chat/completions", json=payload
            ) as response:
                assert response.status == 200

                response_data = await response.json()
                assert (
                    OpenAIAdapter.from_openai_response(
                        CreateChatCompletionResponse(**response_data)
                    ).response_output
                    == sample["output"]
                )
                logging.debug(
                    f"Sample {i} passed : in:\n {sample['prompt'][0:30]} out:\n {sample['output'][0:30]}"
                )
