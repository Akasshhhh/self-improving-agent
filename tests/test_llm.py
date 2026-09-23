import json
import unittest
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
