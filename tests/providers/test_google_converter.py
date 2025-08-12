import unittest
from unittest.mock import MagicMock
from aisuite.providers.google_provider import GoogleMessageConverter
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function
from aisuite.framework import ChatCompletionResponse


class TestGoogleMessageConverter(unittest.TestCase):

    def setUp(self):
        self.converter = GoogleMessageConverter()

    def test_convert_request_with_system_instruction(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather today?"}
        ]
        system_instruction, converted_messages = self.converter.convert_request(messages)

        self.assertEqual(len(converted_messages), 1)
        self.assertEqual(converted_messages[0].role, "user")
        self.assertEqual(system_instruction, "You are a helpful assistant.")
        self.assertEqual(
            converted_messages[0].parts[0].text, "What is the weather today?"
        )

    def test_convert_request_user_message(self):
        messages = [{"role": "user", "content": "What is the weather today?"}]
        _, converted_messages = self.converter.convert_request(messages)

        self.assertEqual(len(converted_messages), 1)
        self.assertEqual(converted_messages[0].role, "user")
        self.assertEqual(
            converted_messages[0].parts[0].text, "What is the weather today?"
        )

    def test_convert_request_tool_result_message(self):
        messages = [
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"temperature": "15", "unit": "Celsius"}',
            }
        ]
        _, converted_messages = self.converter.convert_request(messages)

        self.assertEqual(len(converted_messages), 1)
        self.assertEqual(converted_messages[0].parts[0].function_response.name, "get_weather")
        self.assertEqual(
            converted_messages[0].parts[0].function_response.response,
            {"temperature": "15", "unit": "Celsius"},
        )

    def test_convert_request_assistant_message(self):
        messages = [
            {
                "role": "assistant",
                "content": "The weather is sunny with a temperature of 25 degrees Celsius.",
            }
        ]
        _, converted_messages = self.converter.convert_request(messages)

        self.assertEqual(len(converted_messages), 1)
        self.assertEqual(converted_messages[0].role, "model")
        self.assertEqual(
            converted_messages[0].parts[0].text,
            "The weather is sunny with a temperature of 25 degrees Celsius.",
        )

if __name__ == "__main__":
    unittest.main()
