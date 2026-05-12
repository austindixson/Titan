from urllib import error

from titan.provider import OpenAICompatProvider, ProviderError
from titan.types import Message, Role


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        import json

        return json.dumps(self.payload).encode()


class _StreamingResponse:
    def __init__(self, lines):
        self.lines = [line.encode() for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_chat_completion_timeout_is_retryable(monkeypatch):
    provider = OpenAICompatProvider(api_base="https://api.openai.com/v1", api_key="token")

    def _boom(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("titan.provider.request.urlopen", _boom)

    try:
        provider.generate("gpt", [Message(role=Role.USER, content="hi")], [])
        assert False, "expected ProviderError"
    except ProviderError as e:
        assert e.retryable is True
        assert "timed out" in str(e)


def test_chat_completion_http_400_is_non_retryable(monkeypatch):
    provider = OpenAICompatProvider(api_base="https://api.openai.com/v1", api_key="token")

    class _Err:
        code = 400

        def read(self):
            return b"bad request"

    def _boom(*args, **kwargs):
        raise error.HTTPError("http://x", 400, "bad", hdrs=None, fp=None)

    monkeypatch.setattr("titan.provider.request.urlopen", _boom)

    try:
        provider.generate("gpt", [Message(role=Role.USER, content="hi")], [])
        assert False, "expected ProviderError"
    except ProviderError as e:
        assert e.retryable is False
        assert "http 400" in str(e)


def test_chat_completion_missing_choices_returns_empty_final(monkeypatch):
    provider = OpenAICompatProvider(api_base="https://api.openai.com/v1", api_key="token")
    payload = {"choices": []}

    monkeypatch.setattr("titan.provider.request.urlopen", lambda *args, **kwargs: _FakeResponse(payload))

    out = provider.generate("gpt", [Message(role=Role.USER, content="hi")], [])
    assert out.text == ""
    assert out.tool_calls == []


def test_chat_completion_streaming_emits_deltas_and_final_text(monkeypatch):
    provider = OpenAICompatProvider(api_base="https://api.openai.com/v1", api_key="token")
    events = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
        'data: [DONE]\n',
    ]
    seen = []

    monkeypatch.setattr("titan.provider.request.urlopen", lambda *args, **kwargs: _StreamingResponse(events))
    out = provider.generate_with_callback(
        "gpt",
        [Message(role=Role.USER, content="hi")],
        [],
        on_event=lambda event_type, **payload: seen.append((event_type, payload)),
    )

    assert out.text == "Hello"
    assert [(t, p["text"]) for t, p in seen] == [("stream_delta", "Hel"), ("stream_delta", "lo")]


def test_codex_completed_event_text_not_duplicated(monkeypatch):
    provider = OpenAICompatProvider(api_base="https://chatgpt.com/backend-api/codex", api_key="token")
    events = [
        'data: {"type":"response.output_text.delta","delta":"Hello"}\n',
        'data: {"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"Hello"}]}]}}\n',
        'data: [DONE]\n',
    ]

    monkeypatch.setattr("titan.provider.request.urlopen", lambda *args, **kwargs: _StreamingResponse(events))
    out = provider.generate("gpt", [Message(role=Role.USER, content="hi")], [])
    assert out.text == "Hello"


def test_codex_error_event_is_non_retryable(monkeypatch):
    provider = OpenAICompatProvider(api_base="https://chatgpt.com/backend-api/codex", api_key="token")
    events = [
        'data: {"type":"error","error":{"message":"bad tool schema"}}\n',
        'data: [DONE]\n',
    ]

    monkeypatch.setattr("titan.provider.request.urlopen", lambda *args, **kwargs: _StreamingResponse(events))

    try:
        provider.generate("gpt", [Message(role=Role.USER, content="hi")], [])
        assert False, "expected ProviderError"
    except ProviderError as e:
        assert e.retryable is False
        assert "codex_error" in str(e)


def test_codex_function_arguments_delta_accumulates(monkeypatch):
    provider = OpenAICompatProvider(api_base="https://chatgpt.com/backend-api/codex", api_key="token")
    events = [
        'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"item1","call_id":"call1","name":"read_file"}}\n',
        'data: {"type":"response.function_call_arguments.delta","item_id":"item1","delta":"{\"path\""}\n',
        'data: {"type":"response.function_call_arguments.delta","item_id":"item1","delta":":\"a.txt\"}"}\n',
        'data: {"type":"response.completed","response":{"output":[{"type":"function_call","call_id":"call1","name":"read_file"}]}}\n',
        'data: [DONE]\n',
    ]

    monkeypatch.setattr("titan.provider.request.urlopen", lambda *args, **kwargs: _StreamingResponse(events))
    out = provider.generate("gpt", [Message(role=Role.USER, content="hi")], [])
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "read_file"
    assert out.tool_calls[0].arguments == {"path": "a.txt"}
