from __future__ import annotations

from pathlib import Path
import base64

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


def test_responses_payload_converts_wrapped_local_image_path(tmp_path: Path):
    image = tmp_path / "Screenshot 2026-05-12 at 6.29.31\u202fAM.png"
    image.write_bytes(b"fake-png")
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    wrapped = str(image).replace("/", "/\n", 1)
    content = provider._responses_message_content(
        Message(role=Role.USER, content=f"run prompt in {wrapped}"),
    )

    assert isinstance(content, list)
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_responses_payload_resolves_macos_ampm_spacing_variant(tmp_path: Path):
    image = tmp_path / "Screenshot 2026-05-12 at 5.47.12\u202fAM.png"
    image.write_bytes(b"fake-png")
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    pasted_variant = str(image).replace("\u202fAM", "AM")
    content = provider._responses_message_content(
        Message(role=Role.USER, content=f"execute from image {pasted_variant}"),
    )

    assert isinstance(content, list)
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_provider_uses_preprocess_image_for_data_url(monkeypatch, tmp_path: Path):
    image = tmp_path / "cover.png"
    image.write_bytes(b"ignored")
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    monkeypatch.setattr(
        "titan.provider.preprocess_image_for_attachment",
        lambda _p: ("image/jpeg", b"abc"),
    )

    payload = provider._chat_messages_payload([
        Message(role=Role.USER, content=f"'{image}' analyze"),
    ])
    content = payload[0]["content"]
    assert isinstance(content, list)
    url = content[1]["image_url"]["url"]
    assert url == f"data:image/jpeg;base64,{base64.b64encode(b'abc').decode('ascii')}"


def test_non_image_text_stays_plain_for_chat_payload():
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    payload = provider._chat_messages_payload([Message(role=Role.USER, content="hello")])

    assert payload == [{"role": "user", "content": "hello"}]


def test_chat_payload_bridges_read_file_image_descriptor_into_input_image(tmp_path: Path):
    image = tmp_path / "bridge.png"
    image.write_bytes(b"fake-png")
    provider = OpenAICompatProvider(api_base="https://example.test/v1", api_key="token")

    tool_msg = Message(
        role=Role.TOOL,
        tool_name="read_file",
        tool_call_id="call_1",
        content=(
            "{"
            '"type":"image_file",'
            f'"path":"{image.resolve()}",'
            '"mime":"image/png",'
            '"size_bytes":8'
            "}"
        ),
    )

    payload = provider._chat_messages_payload([
        Message(role=Role.USER, content="analyze the screenshot"),
        tool_msg,
    ])

    assert len(payload) == 3
    assert payload[2]["role"] == "user"
    bridged = payload[2]["content"]
    assert isinstance(bridged, list)
    assert bridged[1]["type"] == "image_url"
    assert bridged[1]["image_url"]["url"].startswith("data:image/png;base64,")
