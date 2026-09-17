from __future__ import annotations
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any
from .types import Message, Role, ToolCall


def new_entry_id() -> str:
    return uuid.uuid4().hex[:12]


def synthesize_legacy_id(row: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(row.get("trace_id") or ""),
            str(row.get("ts") or ""),
            str(row.get("role") or ""),
            str(row.get("tool_call_id") or ""),
            hashlib.sha1(str(row.get("content") or "").encode("utf-8", errors="ignore")).hexdigest()[:8],
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _tool_calls_from_row(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    out: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        out.append(ToolCall(id=str(item.get("id") or "call_unknown"), name=str(item.get("name") or ""), arguments=args))
    return out


def message_from_entry(entry: dict[str, Any]) -> Message | None:
    if entry.get("type", "message") != "message":
        return None
    role_raw = str(entry.get("role") or "user")
    try:
        role = Role(role_raw)
    except ValueError:
        return None
    return Message(
        role=role,
        content=str(entry.get("content") or ""),
        id=str(entry.get("id") or ""),
        tool_call_id=entry.get("tool_call_id"),
        tool_name=entry.get("tool_name"),
        is_error=bool(entry.get("is_error", False)),
        tool_calls=_tool_calls_from_row(entry.get("tool_calls")),
        input_tokens=int(entry.get("input_tokens") or 0),
        output_tokens=int(entry.get("output_tokens") or 0),
    )


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

    def load_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if not row.get("type"):
                row["type"] = "message"
            if not row.get("id"):
                row["id"] = synthesize_legacy_id(row)
            rows.append(row)
        return rows

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "type": "compaction",
            "id": entry_id or new_entry_id(),
            "ts": int(time.time() * 1000),
            "trace_id": self.trace_id,
            "summary": summary,
            "first_kept_entry_id": first_kept_entry_id,
            "tokens_before": tokens_before,
            "details": details or {},
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row
