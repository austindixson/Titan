from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .config import CompactionConfig
from .image_paths import candidate_image_paths_from_text
from .session import message_from_entry
from .types import Message, Role, ToolCall


COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between a user "
    "and an AI coding assistant, then produce a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """Use this EXACT format:

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [Requirements mentioned by user]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues, if any]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [What should happen next]

## Critical Context
- [Data needed to continue]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""

CONTEXT_WINDOWS = {
    ("grok", "grok-4.6"): 500_000,
    ("openai-codex", "gpt-6-astra"): 1_050_000,
    ("astra", "gpt-6-astra"): 1_050_000,
    ("litert", "gemma4-12b"): 128_000,
}


def context_window_for(provider: str, model: str) -> int:
    return CONTEXT_WINDOWS.get((provider.strip().lower(), model.strip()), 128_000)


def should_compact(context_tokens: int, context_window: int, settings: CompactionConfig) -> bool:
    return bool(settings.enabled) and context_tokens > context_window - settings.reserve_tokens


def _has_image_payload(message: Message) -> bool:
    if candidate_image_paths_from_text(message.content or ""):
        return True
    if "[tool-image]" in (message.content or ""):
        return True
    raw = (message.content or "").strip()
    if raw.startswith("{") and '"image_file"' in raw:
        return True
    return False


def estimate_tokens(message: Message) -> int:
    chars = len(message.content or "")
    for tc in message.tool_calls:
        chars += len(tc.name) + len(json.dumps(tc.arguments))
    tokens = math.ceil(chars / 4) if chars else 0
    if _has_image_payload(message):
        tokens += 1200
    return tokens


def estimate_context_tokens(messages: list[Message]) -> int:
    last_usage_index = -1
    usage_tokens = 0
    for i, msg in enumerate(messages):
        if msg.role == Role.ASSISTANT and msg.input_tokens > 0:
            last_usage_index = i
            usage_tokens = msg.input_tokens
    if last_usage_index < 0:
        return sum(estimate_tokens(m) for m in messages)
    trailing = sum(estimate_tokens(m) for m in messages[last_usage_index + 1 :])
    return usage_tokens + trailing


def _entry_role(entry: dict[str, Any]) -> str:
    if entry.get("type") == "compaction":
        return "compaction"
    return str(entry.get("role") or "")


def _is_valid_cut(entry: dict[str, Any]) -> bool:
    kind = entry.get("type", "message")
    if kind == "compaction":
        return False
    role = _entry_role(entry)
    return role in {Role.USER.value, Role.ASSISTANT.value}


def find_cut_point(entries: list[dict[str, Any]], start_index: int, end_index: int, keep_recent_tokens: int) -> dict[str, int | bool]:
    accumulated = 0
    cut_index = start_index
    for i in range(end_index - 1, start_index - 1, -1):
        msg = message_from_entry(entries[i])
        if msg is None and entries[i].get("type") == "compaction":
            accumulated += math.ceil(len(str(entries[i].get("summary") or "")) / 4)
        elif msg is not None:
            accumulated += estimate_tokens(msg)
        if accumulated >= keep_recent_tokens:
            cut_index = i
            break
    else:
        cut_index = start_index

    first_kept = cut_index
    for i in range(cut_index, end_index):
        if _is_valid_cut(entries[i]):
            first_kept = i
            break
    turn_start = find_turn_start_index(entries, first_kept, start_index)
    is_split = _entry_role(entries[first_kept]) != Role.USER.value if entries else False
    return {
        "first_kept_entry_index": first_kept,
        "turn_start_index": turn_start,
        "is_split_turn": bool(is_split and turn_start >= 0),
    }


def find_turn_start_index(entries: list[dict[str, Any]], entry_index: int, start_index: int) -> int:
    for i in range(entry_index, start_index - 1, -1):
        if _entry_role(entries[i]) == Role.USER.value:
            return i
    return -1


def serialize_conversation(messages: list[Message], tool_result_limit: int = 2000) -> str:
    lines: list[str] = []
    for m in messages:
        if m.role == Role.USER:
            lines.append(f"[User]: {m.content}")
        elif m.role == Role.ASSISTANT:
            if m.content:
                lines.append(f"[Assistant]: {m.content}")
            if m.tool_calls:
                calls = "; ".join(
                    f"{tc.name}({json.dumps(tc.arguments, separators=(',', ':'))})" for tc in m.tool_calls
                )
                lines.append(f"[Assistant tool calls]: {calls}")
        elif m.role == Role.TOOL:
            body = m.content or ""
            if len(body) > tool_result_limit:
                omitted = len(body) - tool_result_limit
                body = body[:tool_result_limit] + f"\n[truncated {omitted} characters]"
            lines.append(f"[Tool result]: {body}")
        elif m.role == Role.SYSTEM:
            lines.append(f"[System]: {m.content}")
    return "\n".join(lines)


def extract_file_ops_from_message(message: Message, read_files: set[str], modified_files: set[str]) -> None:
    if message.role != Role.ASSISTANT:
        return
    for tc in message.tool_calls:
        args = tc.arguments or {}
        path = str(args.get("path") or "").strip()
        if tc.name in {"read_file"} and path:
            read_files.add(path)
        if tc.name in {"write_file", "edit_file"} and path:
            modified_files.add(path)
        if tc.name == "shell":
            cmd = str(args.get("command") or "")
            if any(tok in cmd for tok in ("rm ", "mv ", "sed ", "tee ")):
                modified_files.add(cmd[:80])


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    parts: list[str] = []
    if read_files:
        parts.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        parts.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    return "\n".join(parts)


def wrap_summary_user_message(summary: str) -> Message:
    return Message(
        role=Role.USER,
        content=COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX,
    )


def rebuild_llm_context(system: Message, entries: list[dict[str, Any]]) -> list[Message]:
    last_comp_index = -1
    for i, entry in enumerate(entries):
        if entry.get("type") == "compaction":
            last_comp_index = i
    out: list[Message] = [system]
    if last_comp_index >= 0:
        summary = str(entries[last_comp_index].get("summary") or "")
        out.append(wrap_summary_user_message(summary))
        first_kept = str(entries[last_comp_index].get("first_kept_entry_id") or "")
        start = last_comp_index + 1
        if first_kept:
            for i, entry in enumerate(entries):
                if str(entry.get("id") or "") == first_kept:
                    start = i
                    break
        for entry in entries[start:]:
            if entry.get("type") == "compaction":
                continue
            msg = message_from_entry(entry)
            if msg is not None:
                out.append(msg)
        return out
    for entry in entries:
        msg = message_from_entry(entry)
        if msg is not None:
            if msg.role == Role.SYSTEM:
                continue
            out.append(msg)
    return out


@dataclass
class CompactionPreparation:
    first_kept_entry_id: str
    messages_to_summarize: list[Message]
    turn_prefix_messages: list[Message]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None
    read_files: list[str]
    modified_files: list[str]
    settings: CompactionConfig


def prepare_compaction(entries: list[dict[str, Any]], settings: CompactionConfig, system: Message) -> CompactionPreparation | None:
    if not entries:
        return None
    if entries[-1].get("type") == "compaction":
        return None
    prev_index = -1
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("type") == "compaction":
            prev_index = i
            break
    previous_summary = None
    boundary_start = 0
    if prev_index >= 0:
        previous_summary = str(entries[prev_index].get("summary") or "")
        first_kept = str(entries[prev_index].get("first_kept_entry_id") or "")
        found = -1
        if first_kept:
            for i, entry in enumerate(entries):
                if str(entry.get("id") or "") == first_kept:
                    found = i
                    break
        boundary_start = found if found >= 0 else prev_index + 1
    rebuilt = rebuild_llm_context(system, entries)
    tokens_before = estimate_context_tokens(rebuilt)
    cut = find_cut_point(entries, boundary_start, len(entries), settings.keep_recent_tokens)
    first_kept_index = int(cut["first_kept_entry_index"])
    first_kept_entry = entries[first_kept_index]
    first_kept_id = str(first_kept_entry.get("id") or "")
    if not first_kept_id:
        return None
    history_end = int(cut["turn_start_index"]) if cut["is_split_turn"] else first_kept_index
    messages_to_summarize: list[Message] = []
    for i in range(boundary_start, max(history_end, boundary_start)):
        msg = message_from_entry(entries[i])
        if msg is not None:
            messages_to_summarize.append(msg)
    turn_prefix: list[Message] = []
    if cut["is_split_turn"]:
        for i in range(int(cut["turn_start_index"]), first_kept_index):
            msg = message_from_entry(entries[i])
            if msg is not None:
                turn_prefix.append(msg)
    read_files: set[str] = set()
    modified_files: set[str] = set()
    if prev_index >= 0:
        details = entries[prev_index].get("details") or {}
        if isinstance(details, dict):
            for p in details.get("read_files") or details.get("readFiles") or []:
                read_files.add(str(p))
            for p in details.get("modified_files") or details.get("modifiedFiles") or []:
                modified_files.add(str(p))
    for msg in messages_to_summarize + turn_prefix:
        extract_file_ops_from_message(msg, read_files, modified_files)
    return CompactionPreparation(
        first_kept_entry_id=first_kept_id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix,
        is_split_turn=bool(cut["is_split_turn"]),
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        read_files=sorted(read_files),
        modified_files=sorted(modified_files),
        settings=settings,
    )


def _complete_user_turns(entries: list[dict[str, Any]]) -> list[int]:
    starts: list[int] = []
    for i, entry in enumerate(entries):
        if _entry_role(entry) == Role.USER.value:
            starts.append(i)
    return starts


def emergency_drop_compaction(
    entries: list[dict[str, Any]],
    system: Message,
    window: int,
    settings: CompactionConfig,
) -> dict[str, Any] | None:
    if not entries or entries[-1].get("type") != "compaction":
        return None
    prev = entries[-1]
    tokens_before = estimate_context_tokens(rebuild_llm_context(system, entries))
    first_kept = str(prev.get("first_kept_entry_id") or "")
    start = 0
    if first_kept:
        for i, entry in enumerate(entries):
            if str(entry.get("id") or "") == first_kept:
                start = i
                break
        else:
            start = len(entries) - 1
    kept = entries[start:]
    turn_starts = _complete_user_turns(kept)
    if not turn_starts:
        return None
    target = window - settings.reserve_tokens
    chosen = turn_starts[0]
    for ts in turn_starts:
        trial_first = str(kept[ts].get("id") or "")
        trial_entries = list(entries)
        trial_entries.append(
            {
                "type": "compaction",
                "id": "trial",
                "summary": prev.get("summary") or "",
                "first_kept_entry_id": trial_first,
            }
        )
        est = estimate_context_tokens(rebuild_llm_context(system, trial_entries))
        if est < target:
            chosen = ts
            break
        chosen = ts
    first_id = str(kept[chosen].get("id") or "")
    if not first_id:
        return None
    trial = list(entries) + [
        {
            "type": "compaction",
            "id": "trial",
            "summary": prev.get("summary") or "",
            "first_kept_entry_id": first_id,
        }
    ]
    if estimate_context_tokens(rebuild_llm_context(system, trial)) >= target and chosen == turn_starts[-1]:
        # last remaining turn still over
        if len(turn_starts) == 1:
            return None
    details = dict(prev.get("details") or {})
    details["emergency_drop"] = True
    return {
        "summary": str(prev.get("summary") or ""),
        "first_kept_entry_id": first_id,
        "tokens_before": tokens_before,
        "details": details,
    }


class CompactionController:
    def __init__(self, provider, session, settings: CompactionConfig, model: str, window: int):
        self.provider = provider
        self.session = session
        self.settings = settings
        self.model = model
        self.window = window

    def _system_from_history(self, history: list[Message]) -> Message:
        for m in history:
            if m.role == Role.SYSTEM:
                return m
        return Message(role=Role.SYSTEM, content="You are Titan.")

    def _summarize(self, messages: list[Message], prompt: str, previous_summary: str | None, max_tokens: int, custom: str = "") -> str:
        conversation = serialize_conversation(messages)
        body = f"<conversation>\n{conversation}\n</conversation>\n\n"
        if previous_summary:
            body += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        body += prompt
        if custom:
            body += f"\n\nAdditional focus: {custom}"
        history = [
            Message(role=Role.SYSTEM, content=SUMMARIZATION_SYSTEM_PROMPT),
            Message(role=Role.USER, content=body),
        ]
        resp = self.provider.generate(self.model, history, [])
        return (resp.text or "").strip()

    def maybe_compact(
        self,
        history: list[Message],
        *,
        reason: str = "auto",
        custom_instructions: str = "",
        force: bool = False,
    ) -> dict[str, Any] | None:
        if self.session is None:
            return None
        entries = self.session.load_entries()
        system = self._system_from_history(history)
        rebuilt = rebuild_llm_context(system, entries) if entries else list(history)
        tokens = estimate_context_tokens(rebuilt)
        if not force and not should_compact(tokens, self.window, self.settings):
            return None
        prep = prepare_compaction(entries, self.settings, system)
        if prep is None:
            return None
        try:
            history_text = "No prior history."
            if prep.messages_to_summarize:
                history_text = self._summarize(
                    prep.messages_to_summarize,
                    UPDATE_SUMMARIZATION_PROMPT if prep.previous_summary else SUMMARIZATION_PROMPT,
                    prep.previous_summary,
                    math.floor(0.8 * self.settings.reserve_tokens),
                    custom_instructions,
                )
            if prep.is_split_turn and prep.turn_prefix_messages:
                prefix_text = self._summarize(
                    prep.turn_prefix_messages,
                    TURN_PREFIX_SUMMARIZATION_PROMPT,
                    None,
                    math.floor(0.5 * self.settings.reserve_tokens),
                    custom_instructions,
                )
                history_text = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}"
            files = format_file_operations(prep.read_files, prep.modified_files)
            if files:
                history_text = f"{history_text}\n\n{files}"
        except Exception:
            return None
        details = {"read_files": prep.read_files, "modified_files": prep.modified_files}
        self.session.append_compaction(history_text, prep.first_kept_entry_id, prep.tokens_before, details)
        after_entries = self.session.load_entries()
        rebuilt_after = rebuild_llm_context(system, after_entries)
        return {
            "reason": reason,
            "tokens_before": prep.tokens_before,
            "tokens_after": estimate_context_tokens(rebuilt_after),
            "first_kept_entry_id": prep.first_kept_entry_id,
            "split_turn": prep.is_split_turn,
            "model": self.model,
            "window": self.window,
            "context_tokens": tokens,
        }

    def emergency_drop(self, history: list[Message]) -> dict[str, Any] | None:
        if self.session is None:
            return None
        system = self._system_from_history(history)
        entries = self.session.load_entries()
        drop = emergency_drop_compaction(entries, system, self.window, self.settings)
        if drop is None:
            return None
        self.session.append_compaction(
            drop["summary"],
            drop["first_kept_entry_id"],
            drop["tokens_before"],
            drop["details"],
        )
        after = rebuild_llm_context(system, self.session.load_entries())
        return {
            "reason": "emergency_drop",
            "tokens_before": drop["tokens_before"],
            "tokens_after": estimate_context_tokens(after),
            "first_kept_entry_id": drop["first_kept_entry_id"],
            "split_turn": False,
            "model": self.model,
            "window": self.window,
            "context_tokens": drop["tokens_before"],
        }
