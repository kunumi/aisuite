import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse


class VllmProvider(Provider):
    """
    vLLM Provider that makes HTTP calls instead of using SDK.
    It uses the /v1/chat/completions endpoint.
    Read more here - https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server
    If VLLM_API_URL is not set and not passed in config, then it will default to "http://localhost:8000"
    """

    _CHAT_COMPLETION_ENDPOINT = "/v1/chat/completions"
    _CONNECT_ERROR_MESSAGE = "vLLM is likely not running. Start vLLM by running `vllm serve` on your host."

    def __init__(self, **config):
        """
        Initialize the vLLM provider with the given configuration.
        """
        self.url = config.get("api_url") or os.getenv(
            "VLLM_API_URL", "http://localhost:8000"
        )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the chat completions endpoint using httpx.
        """
        kwargs["stream"] = False
        data = {
            "model": model,
            "messages": messages,
            **kwargs,  # Pass any additional arguments to the API
        }

        try:
            response = httpx.post(
                self.url.rstrip("/") + self._CHAT_COMPLETION_ENDPOINT,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError:  # Handle connection errors
            raise LLMError(f"Connection failed: {self._CONNECT_ERROR_MESSAGE}")
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"vLLM request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        # Return the normalized response
        return self._normalize_response(response.json())

    def _normalize_response(self, response_data):
        """
        Normalize the API response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["choices"][0]["message"]["content"]
        return normalized_response
