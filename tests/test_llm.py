import json
from io import BytesIO
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from scheduler.llm import ChatMessage, ModelClientError, OpenAIModelClient


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class OpenAIModelClientTests(unittest.TestCase):
    def test_omits_empty_tool_calls_from_messages_sent_to_provider(self) -> None:
        response = {"choices": [{"message": {"content": "How can I help?"}}]}
        client = OpenAIModelClient(api_key="test-key")
        messages = [
            ChatMessage(role="system", content="Scheduling assistant"),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ]

        with patch("scheduler.llm.urlopen", return_value=FakeHttpResponse(response)) as mock_urlopen:
            client.complete(messages=messages, tools=[], model="test-model")

        request_payload = json.loads(mock_urlopen.call_args.args[0].data)
        for message in request_payload["messages"]:
            self.assertNotIn("tool_calls", message)

    def test_http_error_includes_provider_diagnostic_without_request_headers(self) -> None:
        response = {
            "error": {
                "message": "The selected model does not support this parameter.",
                "type": "invalid_request_error",
                "param": "tools",
                "code": "unsupported_parameter",
            }
        }
        error = HTTPError(
            "https://example.test/v1/chat/completions",
            400,
            "Bad Request",
            hdrs=None,
            fp=BytesIO(json.dumps(response).encode("utf-8")),
        )
        client = OpenAIModelClient(api_key="secret-test-key")

        with patch("scheduler.llm.urlopen", side_effect=error):
            with self.assertRaises(ModelClientError) as raised:
                client.complete(messages=[], tools=[], model="test-model")

        self.assertIn("HTTP 400", str(raised.exception))
        self.assertIn("unsupported_parameter", str(raised.exception))
        self.assertIn("tools", str(raised.exception))
        self.assertNotIn("secret-test-key", str(raised.exception))

    def test_parses_a_provider_tool_call_into_normalized_model_turn(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_available_slots",
                                    "arguments": '{"specialty":"cardiology"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client = OpenAIModelClient(api_key="test-key", base_url="https://example.test/v1")

        with patch("scheduler.llm.urlopen", return_value=FakeHttpResponse(payload)) as urlopen_mock:
            turn = client.complete(
                messages=[ChatMessage(role="user", content="Find cardiology slots")],
                tools=[],
                model="test-model",
            )

        self.assertEqual(turn.tool_calls[0].name, "search_available_slots")
        self.assertEqual(turn.tool_calls[0].arguments, {"specialty": "cardiology"})
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")

    def test_rejects_invalid_tool_arguments_from_provider(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "book_appointment", "arguments": "[]"},
                            }
                        ]
                    }
                }
            ]
        }
        client = OpenAIModelClient(api_key="test-key")

        with patch("scheduler.llm.urlopen", return_value=FakeHttpResponse(payload)):
            with self.assertRaises(ModelClientError):
                client.complete(messages=[], tools=[], model="test-model")


if __name__ == "__main__":
    unittest.main()
