import pytest
from unittest.mock import patch, MagicMock
from aisuite.providers.google_provider import GoogleProvider
import json


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "path-to-service-account-json")
    monkeypatch.setenv("GOOGLE_PROJECT_ID", "vertex-project-id")
    monkeypatch.setenv("GOOGLE_REGION", "us-central1")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-api-key")

def test_missing_env_vars():
    """Test that an error is raised if required environment variables are missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            GoogleProvider(vertexai=True)
        assert "Missing one or more required Google environment variables" in str(
            exc_info.value
        )

def test_missing_env_vars_gemini():
    """Test that an error is raised if required environment variables are missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            GoogleProvider()
        assert "Gemini API key is missing. Please provide it in the config or set the GEMINI_API_KEY environment variable." in str(
            exc_info.value
        )


def test_google_genai_interface():
    """High-level test that the interface is initialized and chat completions are requested successfully."""

    # Test case 1: Regular text response
    def test_text_response():
        user_greeting = "Hello!"
        message_history = [{"role": "user", "content": user_greeting}]
        selected_model = "our-favorite-model"
        response_text_content = "mocked-text-response-from-model"

        interface = GoogleProvider()
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = response_text_content
        # Ensure function_call attribute doesn't exist
        del mock_response.candidates[0].content.parts[0].function_call

        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_client.models = mock_models
        mock_models.generate_content.return_value = mock_response
        interface.client = mock_client

        response = interface.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=0.7,
        )

        # Assert the response is in the correct format
        assert response.choices[0].message.content == response_text_content

    # Test case 2: Function call response
    def test_function_call():
        user_greeting = "What's the weather?"
        message_history = [{"role": "user", "content": user_greeting}]
        selected_model = "our-favorite-model"

        interface = GoogleProvider(vertexai=True)
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]

        # Mock the function call response
        function_call_mock = MagicMock()
        function_call_mock.name = "get_weather"
        function_call_mock.args = {"location": "San Francisco"}
        mock_response.candidates[0].content.parts[0].function_call = function_call_mock
        mock_response.candidates[0].content.parts[0].text = None

        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_client.models = mock_models
        mock_models.generate_content.return_value = mock_response
        interface.client = mock_client

        response = interface.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=0.7,
        )

        # Assert the response contains the function call
        assert response.choices[0].message.content is None
        assert response.choices[0].message.tool_calls[0].type == "function"
        assert (
            response.choices[0].message.tool_calls[0].function.name == "get_weather"
        )
        assert json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        ) == {"location": "San Francisco"}
        assert response.choices[0].finish_reason == "tool_calls"

#     # Run both test cases
    test_text_response()
    test_function_call()
