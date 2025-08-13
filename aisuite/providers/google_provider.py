import os
from google.genai import Client as GenAIClient
from google.genai.types import *
from typing import List, Dict, Any, Tuple
import json
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse, Message

class GoogleMessageConverter:
    @staticmethod
    def convert_user_role_message(message: Dict[str, Any]) -> Content:
        """Convert user or system messages to Google Vertex AI format."""
        parts = [Part.from_text(text=message["content"])]
        return Content(role="user", parts=parts)

    @staticmethod
    def convert_assistant_role_message(message: Dict[str, Any]) -> Content:
        """Convert assistant messages to Google Vertex AI format."""
        if "tool_calls" in message and message["tool_calls"]:
            # Handle function calls
            tool_call = message["tool_calls"][
                0
            ]  # Assuming single function call for now
            function_call = tool_call["function"]

            # Create a Part from the function call
            parts = [
                Part.from_function_call(
                    name=function_call["name"],
                    args=json.loads(function_call["arguments"]),
                )
            ]
        else:
            # Handle regular text messages
            parts = [Part.from_text(text=message["content"])]

        return Content(role="model", parts=parts)

    @staticmethod
    def convert_tool_role_message(message: Dict[str, Any]) -> Content:
        """Convert tool messages to Google Vertex AI format."""
        if "content" not in message:
            raise ValueError("Tool result message must have a content field")

        try:
            content_json = json.loads(message["content"])
            part = Part.from_function_response(
                name=message["name"], response=content_json
            )
            return Content(role="model", parts=[part])
        except json.JSONDecodeError:
            raise ValueError("Tool result message must be valid JSON")

    @staticmethod
    def convert_request(messages: List[Dict[str, Any]]) -> Tuple[str, List[Content]]:
        """Convert messages to Google Vertex AI format."""
        # Convert all messages to dicts if they're Message objects

        # TODO: This is a temporary solution to extract the system message.
        # User can pass multiple system messages, which can mingled with other messages.
        # This needs to be fixed to handle this case.
        system_message = None
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages.pop(0)

        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message
            for message in messages
        ]

        formatted_messages = []
        for message in messages:
            if message["role"] == "tool":
                vertex_message = GoogleMessageConverter.convert_tool_role_message(
                    message
                )
                if vertex_message:
                    formatted_messages.append(vertex_message)
            elif message["role"] == "assistant":
                formatted_messages.append(
                    GoogleMessageConverter.convert_assistant_role_message(message)
                )
            else:  # user or system role
                formatted_messages.append(
                    GoogleMessageConverter.convert_user_role_message(message)
                )

        return system_message, formatted_messages
    
    @staticmethod
    def convert_response(response) -> ChatCompletionResponse:
        """Normalize the response from Vertex AI to match OpenAI's response format."""
        openai_response = ChatCompletionResponse()

        # TODO: We need to go through each part, because function call may not be the first part.
        #       Currently, we are only handling the first part, but this is not enough.
        #
        # This is a valid response:
        # candidates {
        #   content {
        #     role: "model"
        #     parts {
        #       text: "The current temperature in San Francisco is 72 degrees Celsius. \n\n"
        #     }
        #     parts {
        #       function_call {
        #         name: "is_it_raining"
        #         args {
        #           fields {
        #             key: "location"
        #             value {
        #               string_value: "San Francisco"
        #             }
        #           }
        #         }
        #       }
        #     }
        #   }
        #   finish_reason: STOP

        # Check if the response contains function calls
        # Note: Just checking if the function_call attribute exists is not enough,
        #       it is important to check if the function_call is not None.
        if (
            hasattr(response.candidates[0].content.parts[0], "function_call")
            and response.candidates[0].content.parts[0].function_call
        ):
            function_call = response.candidates[0].content.parts[0].function_call

            # args is a MapComposite.
            # Convert the MapComposite to a dictionary
            args_dict = {}
            # Another way to try is: args_dict = dict(function_call.args)
            for key, value in function_call.args.items():
                args_dict[key] = value

            openai_response.choices[0].message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"call_{hash(function_call.name)}",  # Generate a unique ID
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(args_dict),
                        },
                    }
                ],
                "refusal": None,
            }
            openai_response.choices[0].message = Message(
                **openai_response.choices[0].message
            )
            openai_response.choices[0].finish_reason = "tool_calls"
        else:
            # Handle regular text response
            openai_response.choices[0].message.content = (
                response.candidates[0].content.parts[0].text
            )
            openai_response.choices[0].finish_reason = "stop"

        return openai_response


class GoogleProvider(Provider):
    def __init__(self, vertexai = False, **config):

        if vertexai:
            self.initilize_vertex_ai(config)
        else:
            self.initialize_gemini(config)
    
    def initilize_vertex_ai(self, config: Dict[str, Any]):

        self.project_id = config.get("project") or os.getenv("GOOGLE_PROJECT_ID")
        self.location = config.get("location") or os.getenv("GOOGLE_REGION", "us-central1")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

        if not self.project_id or not self.location:
            raise EnvironmentError(
                "Missing one or more required Google environment variables: "
                "GOOGLE_PROJECT_ID, GOOGLE_REGION. "
                "Please refer to the setup guide: /guides/google.md."
            )

        self.client = GenAIClient(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )

    def initialize_gemini(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Gemini API key is missing. Please provide it in the config or set the GEMINI_API_KEY environment variable."
            )
        self.client = GenAIClient(api_key=self.api_key)

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            system_message, converted_messages = GoogleMessageConverter.convert_request(messages)
            response = self.client.models.generate_content(
                model=model,
                contents=converted_messages,
                config=GenerateContentConfig(
                    system_instruction=system_message if system_message else None,
                    **kwargs
                )
            )
            return GoogleMessageConverter.convert_response(response)
        except Exception as e:
            raise LLMError(f"Error in chat_completions_create: {str(e)}")

    def list_models(self):
        try:
            response = self.client.models.list()
            return [model.name for model in response]
        except Exception as e:
            raise LLMError(f"Error in list_models: {str(e)}")