from __future__ import annotations
import base64
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request, error

from .types import AssistantResponse, Message, Role, ToolCall
from typing import Any
from .auth import resolve_provider_credentials, provider_default_base_url
from .config import RetryConfig, HarnessConfig
from .image_paths import candidate_image_paths_from_text
from .image_preprocess import preprocess_image_for_attachment


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class Provider:
    def generate(self, model: str, messages: list[Message], tools: list[dict]) -> AssistantResponse:
        raise NotImplementedError

    def generate_with_callback(self, model: str, messages: list[Message], tools: list[dict], on_event=None) -> AssistantResponse:
        return self.generate(model, messages, tools)


def build_provider_from_config(cfg: HarnessConfig) -> Provider:
    provider_name = (cfg.provider or "openai-codex").strip().lower()

    if provider_name == "mock":
        from .mock_provider import make_tool_then_final_script, MockProvider

        return MockProvider(script=make_tool_then_final_script())

    config_key = (cfg.api_keys.get(provider_name, "") or "").strip()
    if config_key:
        creds = None
    elif provider_name == "openai-codex":
        creds = resolve_provider_credentials(
            provider_name,
            api_key_env=cfg.oauth_token_env,
            base_url=cfg.api_base or None,
        )
    elif provider_name == "openai":
        creds = resolve_provider_credentials(
            provider_name,
            api_key_env=cfg.api_key_env,
            base_url=cfg.api_base or None,
        )
    else:
        creds = resolve_provider_credentials(provider_name, base_url=cfg.api_base or None)

    resolved_base = (
        (creds.base_url if creds and creds.base_url else "")
        or (cfg.api_base or "")
        or provider_default_base_url(provider_name)
    )

    return OpenAICompatProvider(
        api_base=resolved_base,
        api_key=(config_key or (creds.token if creds else cfg.api_key())),
    )


def retry_call(fn, retry: RetryConfig):
    attempts = 0
    while True:
        attempts += 1
        try:
            return fn()
        except ProviderError as e:
            if not e.retryable or attempts > retry.max_retries + 1:
                raise
            delay = min(retry.max_delay_ms, retry.base_delay_ms * (2 ** (attempts - 1)))
            time.sleep((delay + random.randint(0, delay)) / 1000)


@dataclass
class OpenAICompatProvider(Provider):
    api_base: str
    api_key: str = ""

    def generate(self, model: str, messages: list[Message], tools: list[dict]) -> AssistantResponse:
        return self.generate_with_callback(model, messages, tools, on_event=None)

    def generate_with_callback(self, model: str, messages: list[Message], tools: list[dict], on_event=None) -> AssistantResponse:
        token = (self.api_key or "").strip()
        base = (self.api_base or "").strip() or "https://api.openai.com/v1"
        if not token:
            raise ProviderError("missing provider credentials", retryable=False)

        # Codex OAuth backend requires /responses with stream=true and store=false.
        if "chatgpt.com/backend-api/codex" in base:
            return self._generate_codex_responses(base, token, model, messages, tools, on_event=on_event)

        return self._generate_chat_completions(base, token, model, messages, tools, on_event=on_event)

    def _read_http_error_body(self, e: error.HTTPError) -> str:
        try:
            return e.read().decode(errors="ignore")
        except Exception:
            return getattr(e, "reason", "") or getattr(e, "msg", "") or ""

    def _parse_sse_data(self, data_str: str) -> dict[str, Any] | None:
        try:
            evt = json.loads(data_str)
            return evt if isinstance(evt, dict) else None
        except Exception:
            # Some OpenAI-compatible streams emit argument delta fragments with
            # unescaped JSON quotes inside the SSE JSON envelope. Preserve the
            # envelope fields we need instead of dropping the tool-call delta.
            if '"type":"response.function_call_arguments.delta"' not in data_str:
                return None
            item_marker = '"item_id":"'
            delta_marker = '"delta":"'
            item_start = data_str.find(item_marker)
            delta_start = data_str.find(delta_marker)
            if item_start < 0 or delta_start < 0:
                return None
            item_start += len(item_marker)
            item_end = data_str.find('"', item_start)
            if item_end < 0:
                return None
            delta_start += len(delta_marker)
            delta_end = data_str.rfind('"')
            if delta_end < delta_start:
                return None
            return {
                "type": "response.function_call_arguments.delta",
                "item_id": data_str[item_start:item_end],
                "delta": data_str[delta_start:delta_end].replace('\\"', '"'),
            }

    def _chat_message_content(self, message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"text", "output_text"} and item.get("text"):
                    text_parts.append(str(item.get("text")))
            return "".join(text_parts)
        return ""

    def _candidate_paths_from_text(self, text: str) -> list[Path]:
        return candidate_image_paths_from_text(text)

    def _image_data_url(self, path: Path) -> str:
        mime, image_bytes = preprocess_image_for_attachment(path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _chat_content_for_message(self, message: Message) -> str | list[dict[str, Any]]:
        if message.role != Role.USER:
            return message.content
        images = self._candidate_paths_from_text(message.content)
        if not images:
            return message.content
        content: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": self._image_data_url(image)}})
        return content

    def _assistant_tool_calls_payload(self, message: Message) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for tc in message.tool_calls:
            rows.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments or {}),
                    },
                }
            )
        return rows

    def _chat_message_payload(self, message: Message) -> dict[str, Any]:
        row: dict[str, Any] = {"role": message.role.value, "content": self._chat_content_for_message(message)}
        if message.role == Role.ASSISTANT and message.tool_calls:
            row["tool_calls"] = self._assistant_tool_calls_payload(message)
            if not str(message.content or "").strip():
                row["content"] = None
        if message.role == Role.TOOL:
            if message.tool_call_id:
                row["tool_call_id"] = message.tool_call_id
            if message.tool_name:
                row["name"] = message.tool_name
        return row

    def _chat_messages_payload(self, messages: list[Message]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for m in messages:
            payload.append(self._chat_message_payload(m))
            tool_image = self._tool_image_path_from_descriptor(m)
            if tool_image is not None:
                bridge = self._tool_image_bridge_user_message(tool_image)
                payload.append(self._chat_message_payload(bridge))
        return payload

    def _responses_message_content(self, message: Message) -> str | list[dict[str, Any]]:
        if message.role != Role.USER:
            return message.content
        images = self._candidate_paths_from_text(message.content)
        if not images:
            return message.content
        content: list[dict[str, Any]] = [{"type": "input_text", "text": message.content}]
        for image in images:
            content.append({"type": "input_image", "image_url": self._image_data_url(image)})
        return content

    def _tool_image_path_from_descriptor(self, message: Message) -> Path | None:
        if message.role != Role.TOOL:
            return None
        if (message.tool_name or "") != "read_file":
            return None
        raw = (message.content or "").strip()
        if not raw.startswith("{"):
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if not isinstance(payload, dict) or payload.get("type") != "image_file":
            return None
        raw_path = str(payload.get("path", "")).strip()
        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser()
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate.resolve()

    def _tool_image_bridge_user_message(self, image_path: Path) -> Message:
        return Message(
            role=Role.USER,
            content=(
                f"[tool-image] Local image read via read_file: {image_path}\n"
                "Analyze this attached image for the current task."
            ),
        )

    def _codex_function_call_item(self, call_id: str, name: str, arguments: str) -> dict[str, Any]:
        return {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }

    def _codex_append_assistant(self, items: list[dict[str, Any]], message: Message, seen: set[str]) -> None:
        if str(message.content or "").strip() or not message.tool_calls:
            items.append({"role": message.role.value, "content": self._responses_message_content(message)})
        for tc in message.tool_calls:
            items.append(self._codex_function_call_item(tc.id, tc.name, json.dumps(tc.arguments or {})))
            seen.add(tc.id)

    def _codex_append_tool(self, items: list[dict[str, Any]], message: Message, seen: set[str]) -> None:
        call_id = message.tool_call_id or ""
        if call_id not in seen:
            items.append(self._codex_function_call_item(call_id, message.tool_name or "tool", "{}"))
            seen.add(call_id)
        items.append({"type": "function_call_output", "call_id": call_id, "output": message.content})
        tool_image = self._tool_image_path_from_descriptor(message)
        if tool_image is None:
            return
        bridge = self._tool_image_bridge_user_message(tool_image)
        items.append({"role": "user", "content": self._responses_message_content(bridge)})

    def _codex_input_items(self, messages: list[Message]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for message in messages:
            if message.role == Role.ASSISTANT:
                self._codex_append_assistant(items, message, seen)
            elif message.role == Role.USER:
                items.append({"role": message.role.value, "content": self._responses_message_content(message)})
            elif message.role == Role.TOOL:
                self._codex_append_tool(items, message, seen)
        return items

    def _generate_chat_completions(self, base: str, token: str, model: str, messages: list[Message], tools: list[dict], on_event=None) -> AssistantResponse:
        url = f"{base.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": self._chat_messages_payload(messages),
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }
        if on_event:
            payload["stream"] = True
        req = request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        if on_event:
            req.add_header("Accept", "text/event-stream")
        req.add_header("Authorization", f"Bearer {token}")
        body = json.dumps(payload).encode()

        try:
            with request.urlopen(req, data=body, timeout=90) as resp:
                if on_event:
                    return self._parse_chat_completions_stream(resp, on_event)
                data = json.loads(resp.read().decode())
        except error.HTTPError as e:
            txt = self._read_http_error_body(e)
            retryable = e.code in (408, 409, 429, 500, 502, 503, 504)
            raise ProviderError(f"http {e.code}: {txt}", retryable=retryable)
        except Exception as e:
            raise ProviderError(str(e), retryable=True)

        choice = (data.get("choices") or [{}])[0].get("message", {})
        text = self._chat_message_content(choice)
        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except Exception:
                parsed = {}
            tool_calls.append(ToolCall(id=tc.get("id", "call_unknown"), name=fn.get("name", ""), arguments=parsed if isinstance(parsed, dict) else {}))
        usage = data.get("usage") or {}
        return AssistantResponse(text=text, tool_calls=tool_calls, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0))

    def _parse_chat_completions_stream(self, resp, on_event) -> AssistantResponse:
        text_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        for raw in resp:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                evt = json.loads(data_str)
            except Exception:
                continue
            choice = (evt.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            if content:
                text_parts.append(content)
                on_event("stream_delta", text=content, kind="text")
            for tc in delta.get("tool_calls") or []:
                index = int(tc.get("index", 0))
                slot = tool_calls_by_index.setdefault(index, {"id": tc.get("id") or f"call_{index}", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc.get("id")
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn.get("name")
                    on_event("stream_tool_call", id=slot["id"], name=slot["name"], kind="tool_call")
                if fn.get("arguments"):
                    slot["arguments"] = (slot.get("arguments") or "") + fn.get("arguments")
        parsed_tool_calls: list[ToolCall] = []
        for _index, tc in sorted(tool_calls_by_index.items()):
            args_raw = tc.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            parsed_tool_calls.append(ToolCall(id=tc.get("id", "call_unknown"), name=tc.get("name", ""), arguments=args if isinstance(args, dict) else {}))
        return AssistantResponse(text="".join(text_parts), tool_calls=parsed_tool_calls, input_tokens=0, output_tokens=0)

    def _extract_text_from_completed_event(self, evt: dict[str, Any]) -> str:
        resp = evt.get("response") or {}
        out = resp.get("output") or []
        text_parts: list[str] = []
        for item in out:
            if item.get("type") != "message":
                continue
            for c in item.get("content") or []:
                if c.get("type") == "output_text" and c.get("text"):
                    text_parts.append(c["text"])
        return "".join(text_parts)

    def _merge_tool_call(self, by_call_id: dict[str, dict[str, Any]], item: dict[str, Any]) -> None:
        if (item.get("type") or "") != "function_call":
            return
        call_id = item.get("call_id") or item.get("id")
        if not call_id:
            return
        slot = by_call_id.setdefault(call_id, {"id": item.get("id") or call_id, "name": item.get("name", ""), "arguments": ""})
        if item.get("name"):
            slot["name"] = item.get("name")
        if item.get("arguments"):
            slot["arguments"] = item.get("arguments")

    def _codex_system_text(self, messages: list[Message]) -> str:
        system_text = "You are a helpful coding assistant."
        for message in messages:
            if message.role.value == "system" and message.content.strip():
                return message.content.strip()
        return system_text

    def _codex_tools_payload(self, tools: list[dict] | None) -> list[dict[str, Any]]:
        codex_tools: list[dict[str, Any]] = []
        for tool in tools or []:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function") or {}
            codex_tools.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return codex_tools

    def _codex_on_text_delta(self, evt: dict[str, Any], text_parts: list[str], on_event) -> None:
        delta = evt.get("delta") or ""
        if not delta:
            return
        text_parts.append(delta)
        if on_event:
            on_event("stream_delta", text=delta, kind="text")

    def _codex_on_output_item(
        self,
        evt: dict[str, Any],
        tool_calls_by_id: dict[str, dict[str, Any]],
        item_to_call_id: dict[str, str],
        on_event,
    ) -> None:
        item = evt.get("item") or {}
        self._merge_tool_call(tool_calls_by_id, item)
        if item.get("type") != "function_call":
            return
        item_id = item.get("id")
        call_id = item.get("call_id") or item_id
        if item_id and call_id:
            item_to_call_id[item_id] = call_id
        if on_event:
            on_event("stream_tool_call", id=call_id or "", name=item.get("name", ""), kind="tool_call")

    def _codex_slot(self, tool_calls_by_id: dict[str, dict[str, Any]], call_id: str) -> dict[str, Any]:
        return tool_calls_by_id.setdefault(call_id, {"id": call_id, "name": "", "arguments": ""})

    def _codex_on_args_delta(
        self,
        evt: dict[str, Any],
        tool_calls_by_id: dict[str, dict[str, Any]],
        item_to_call_id: dict[str, str],
    ) -> None:
        item_id = evt.get("item_id")
        delta = evt.get("delta") or ""
        if not (item_id and delta):
            return
        call_id = item_to_call_id.get(item_id)
        if not call_id:
            return
        slot = self._codex_slot(tool_calls_by_id, call_id)
        slot["arguments"] = (slot.get("arguments") or "") + delta

    def _codex_on_args_done(
        self,
        evt: dict[str, Any],
        tool_calls_by_id: dict[str, dict[str, Any]],
        item_to_call_id: dict[str, str],
    ) -> None:
        item_id = evt.get("item_id")
        args = evt.get("arguments")
        if not item_id or args is None:
            return
        call_id = item_to_call_id.get(item_id)
        if not call_id:
            return
        self._codex_slot(tool_calls_by_id, call_id)["arguments"] = args

    def _codex_on_completed(
        self,
        evt: dict[str, Any],
        text_parts: list[str],
        tool_calls_by_id: dict[str, dict[str, Any]],
    ) -> None:
        if not text_parts:
            completed_text = self._extract_text_from_completed_event(evt)
            if completed_text:
                text_parts.append(completed_text)
        for item in ((evt.get("response") or {}).get("output") or []):
            self._merge_tool_call(tool_calls_by_id, item)

    def _dispatch_codex_event(
        self,
        evt: dict[str, Any],
        text_parts: list[str],
        tool_calls_by_id: dict[str, dict[str, Any]],
        item_to_call_id: dict[str, str],
        on_event,
    ) -> None:
        event_type = evt.get("type")
        if event_type == "response.output_text.delta":
            self._codex_on_text_delta(evt, text_parts, on_event)
            return
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            self._codex_on_output_item(evt, tool_calls_by_id, item_to_call_id, on_event)
            return
        if event_type == "response.function_call_arguments.delta":
            self._codex_on_args_delta(evt, tool_calls_by_id, item_to_call_id)
            return
        if event_type == "response.function_call_arguments.done":
            self._codex_on_args_done(evt, tool_calls_by_id, item_to_call_id)
            return
        if event_type == "response.completed":
            self._codex_on_completed(evt, text_parts, tool_calls_by_id)
            return
        if event_type == "error":
            raise ProviderError(f"codex_error: {json.dumps(evt.get('error') or evt)}", retryable=False)

    def _iter_codex_events(self, resp):
        for raw in resp:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                evt = self._parse_sse_data(data_str)
            except Exception:
                evt = None
            if evt:
                yield evt

    def _consume_codex_stream(self, resp, on_event) -> tuple[list[str], dict[str, dict[str, Any]]]:
        text_parts: list[str] = []
        tool_calls_by_id: dict[str, dict[str, Any]] = {}
        item_to_call_id: dict[str, str] = {}
        for evt in self._iter_codex_events(resp):
            self._dispatch_codex_event(evt, text_parts, tool_calls_by_id, item_to_call_id, on_event)
        return text_parts, tool_calls_by_id

    def _codex_parsed_tool_calls(self, tool_calls_by_id: dict[str, dict[str, Any]]) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for call_id, tc in tool_calls_by_id.items():
            args_raw = tc.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            parsed.append(ToolCall(id=call_id, name=tc.get("name", ""), arguments=args if isinstance(args, dict) else {}))
        return parsed

    def _generate_codex_responses(self, base: str, token: str, model: str, messages: list[Message], tools: list[dict], on_event=None) -> AssistantResponse:
        url = f"{base.rstrip('/')}/responses"
        payload = {
            "model": model,
            "instructions": self._codex_system_text(messages),
            "input": self._codex_input_items(messages),
            "stream": True,
            "store": False,
            "tools": self._codex_tools_payload(tools),
        }
        req = request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream")
        req.add_header("Authorization", f"Bearer {token}")
        try:
            with request.urlopen(req, data=json.dumps(payload).encode(), timeout=120) as resp:
                text_parts, tool_calls_by_id = self._consume_codex_stream(resp, on_event)
        except ProviderError:
            raise
        except error.HTTPError as e:
            txt = self._read_http_error_body(e)
            retryable = e.code in (408, 409, 429, 500, 502, 503, 504)
            raise ProviderError(f"http {e.code}: {txt}", retryable=retryable)
        except Exception as e:
            raise ProviderError(str(e), retryable=True)
        return AssistantResponse(
            text="".join(text_parts).strip(),
            tool_calls=self._codex_parsed_tool_calls(tool_calls_by_id),
            input_tokens=0,
            output_tokens=0,
        )
