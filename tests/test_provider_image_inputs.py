from __future__ import annotations

from pathlib import Path

from titan.provider import OpenAICompatProvider
from titan.types import Message, Role


def test_chat_payload_converts_quoted_local_image_path_to_image_part(tmp_path: Path):
    image = tmp_path / "frontcover.PNG"
    image.write_bytes(b"fake-png")
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    payload = provider._chat_messages_payload([
        Message(role=Role.USER, content=f"'{image}' What's in this image?"),
    ])

    content = payload[0]["content"]
    assert content[0] == {"type": "text", "text": f"'{image}' What's in this image?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_responses_payload_converts_local_image_path_to_input_image(tmp_path: Path):
    image = tmp_path / "cover.jpg"
    image.write_bytes(b"fake-jpg")
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    content = provider._responses_message_content(
        Message(role=Role.USER, content=f"{image} describe this"),
    )

    assert content[0] == {"type": "input_text", "text": f"{image} describe this"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_responses_payload_converts_unquoted_screenshot_path_with_spaces(tmp_path: Path):
    image = tmp_path / "Screenshot 2026-05-12 at 5.47.12 AM.png"
    image.write_bytes(b"fake-png")
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    content = provider._responses_message_content(
        Message(role=Role.USER, content=f"examine the prompt in {image} and execute"),
    )

    assert isinstance(content, list)
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_chat_payload_converts_backticked_screenshot_path_with_spaces(tmp_path: Path):
    image = tmp_path / "Screenshot 2026-05-12 at 5.47.12 AM.png"
    image.write_bytes(b"fake-png")
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    payload = provider._chat_messages_payload([
        Message(role=Role.USER, content=f"You shared `{image}`. Describe it."),
    ])

    content = payload[0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_non_image_text_stays_plain_for_chat_payload():
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    payload = provider._chat_messages_payload([Message(role=Role.USER, content="hello")])

    assert payload == [{"role": "user", "content": "hello"}]
