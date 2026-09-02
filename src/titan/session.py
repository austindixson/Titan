from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from .types import Message


def _tool_calls_row(msg: Message) -> list[dict] | None:
    if not msg.tool_calls:
        return None
    return [
        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
        for tc in msg.tool_calls
    ]


class SessionStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_id = uuid.uuid4().hex[:12]
        self.checkpoints_path = self.path.with_name("checkpoints.jsonl")

    def append(self, msg: Message) -> None:
        row = {
            "ts": int(time.time() * 1000),
            "trace_id": self.trace_id,
            "role": msg.role.value,
            "content": msg.content,
            "tool_call_id": msg.tool_call_id,
            "tool_name": msg.tool_name,
            "is_error": msg.is_error,
        }
        tool_calls = _tool_calls_row(msg)
        if tool_calls is not None:
            row["tool_calls"] = tool_calls
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def checkpoint(self, state: str, turn: int, note: str = "") -> None:
        row = {
            "ts": int(time.time() * 1000),
            "trace_id": self.trace_id,
            "state": state,
            "turn": turn,
            "note": note,
        }
        with self.checkpoints_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
