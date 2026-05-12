from __future__ import annotations
from dataclasses import dataclass
from .types import Message, Role, AssistantResponse, ToolCall
from .provider import Provider


@dataclass
class MockProvider(Provider):
    script: list[AssistantResponse]
    idx: int = 0

    def generate(self, model: str, messages: list[Message], tools: list[dict]) -> AssistantResponse:
        if self.idx >= len(self.script):
            return AssistantResponse(text="")
        r = self.script[self.idx]
        self.idx += 1
        return r


def make_tool_then_final_script() -> list[AssistantResponse]:
    return [
        AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "echo hello"})]),
        AssistantResponse(text="Done. Tool executed."),
    ]
