import base64
import random
import time
import logging
from typing import List, Optional

import requests
from openai import OpenAI
from openai import APIConnectionError, APIError, InternalServerError
from openai.pagination import SyncPage
from openai.types.model import Model

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 15

def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""
    try:
        with requests.get(content_url) as response:
            response.raise_for_status()
            result = base64.b64encode(response.content).decode("utf-8")
        return result
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch content from URL {content_url}: {str(e)}") from e

def get_first_model(client: OpenAI) -> str:
    """
    Get the first model from the vLLM server.
    """
    for attempt in range(MAX_RETRIES):
        try:
            models: SyncPage[Model] = client.models.list()
            if len(models.data) == 0:
                raise RuntimeError(f"No models found on the vLLM server at {client.base_url}")
            return models.data[0].id
        except APIConnectionError as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    "Failed to get the list of models from the vLLM server at "
                    f"{client.base_url} with API key {client.api_key}. Check\n"
                    "1. the server is running\n"
                    "2. the server URL is correct\n"
                    "3. the API key is correct"
                ) from e
            delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            logging.warning(f"Failed to get models, retrying in {delay} seconds...")
            time.sleep(delay)

        except Exception as e:
            raise RuntimeError(f"Unexpected error while getting models: {str(e)}") from e

    raise RuntimeError("Failed to get models after all retry attempts") 

def run_single_image(image_url: str, model: str, prompt: str, num_completions: int = 1) -> List[str]:
    """Run inference on a single image with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                model=model,
                max_completion_tokens=4096,
                temperature=0.0,
                n=num_completions,
            )
            results = []
            for choice in chat_completion.choices:
                if choice.message.content is not None:
                    results.append(choice.message.content)
                else:
                    results.append("")
            return results
        except (APIError, InternalServerError) as e:
            if attempt == MAX_RETRIES - 1:
                logging.error(f"Failed to run inference after {MAX_RETRIES} attempts: {str(e)}")
                raise RuntimeError(f"Failed to run inference after {MAX_RETRIES} attempts: {str(e)}") from e
            delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            logging.warning(f"API error occurred, retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Unexpected error during inference: {str(e)}")
            raise RuntimeError(f"Unexpected error during inference: {str(e)}") from e

    raise RuntimeError("Failed to run inference after all retry attempts")

def generate(image_url: str, prompt: str, num_completions: int = 1) -> Optional[List[str]]:
    """Generate completions with error handling."""
    try:
        model = get_first_model(client)
        return run_single_image(image_url, model, prompt, num_completions)
    except Exception as e:
        logging.error(f"Failed to generate completions: {str(e)}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = generate("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", "What's in this image?")
    if result:
        for i in result:
            print(i)
