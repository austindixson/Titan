from __future__ import annotations

import time
import json
import shlex
import subprocess
import tempfile
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from urllib import request, error
from urllib.parse import unquote, urlparse

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual import events
from textual.css.query import NoMatches
from textual.message import Message as TextualMessage
from textual.widgets import Button, Input, Markdown, RichLog, Static, TextArea

from .auth import (
    canonical_provider,
    ordered_provider_options,
    provider_default_base_url,
    provider_default_model,
    provider_display_name,
    provider_family,
    provider_model_options,
    resolve_provider_credentials,
    supported_openai_compat_providers,
)
from .compaction import CompactionController, context_window_for, estimate_context_tokens, rebuild_llm_context
from .config import load_harness_config, resolve_config_path, update_config_key
from .prompt import TITAN_SYSTEM_PROMPT
from .titan import TitanHarness
from .loop import AgentEvent
from .provider import build_provider_from_config
from .session import SessionStore
from .tools import default_registry
from .types import Message, Role, RunOutcome, RunStopReason
from .slash_commands import execute_slash_command


@dataclass
class UiState:
    pending: bool = False
    started_at: float | None = None
    state: str = "PLAN"
    turn: int = 0
    tool_calls: int = 0
    turn_tool_calls: int = 0
    thinking_dots: int = 0
    pending_tool_names: list[str] = field(default_factory=list)
    pending_tool_count: int = 0


class LoopEventMsg(TextualMessage):
    def __init__(self, event: AgentEvent) -> None:
        self.event = event
        super().__init__()


class LoopDoneMsg(TextualMessage):
    def __init__(self, outcome: RunOutcome) -> None:
        self.outcome = outcome
        super().__init__()


class LoopFailedMsg(TextualMessage):
    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__()


def _plain_renderable(renderable) -> str:
    plain = getattr(renderable, "plain", None)
    if isinstance(plain, str):
        return plain
    return str(renderable)


class SelectableRichLog(TextArea):
    """Read-only text pane with reliable mouse drag-selection in terminal UIs."""

    def __init__(self, *args, **kwargs) -> None:
        wrap = kwargs.pop("wrap", None)
        kwargs.setdefault("read_only", True)
        kwargs.setdefault("soft_wrap", True if wrap is None else bool(wrap))
        kwargs.setdefault("show_cursor", False)
        kwargs.setdefault("show_line_numbers", False)
        super().__init__("", *args, **kwargs)
        self.selection_lines: list[str] = []

    def write_selectable(self, renderable, plain: str | None = None) -> None:
        display_line = _plain_renderable(renderable)
        selection_line = str(plain) if plain is not None else display_line
        self.selection_lines.append(selection_line)
        chunk = f"\n{display_line}" if self.text else display_line
        self.insert(chunk, location=self.document.end, maintain_selection_offset=False)

    def set_lines(self, lines: list[str]) -> None:
        self.selection_lines = [str(line) for line in lines]
        self.load_text("\n".join(self.selection_lines))

    def clear(self) -> None:
        self.selection_lines.clear()
        self.load_text("")


class ComposerTextArea(TextArea):
    PASTE_FILE_CHARS = 2000
    PASTE_FILE_LINES = 10
    class Submitted(TextualMessage):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.paste_payloads: dict[str, str] = {}
        self.message_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""

    def _path_from_token(self, token: str) -> str | None:
        raw = token.strip().strip('"').strip("'")
        if not raw:
            return None
        is_file_uri = raw.startswith("file://")
        if is_file_uri:
            parsed = urlparse(raw)
            if parsed.netloc not in {"", "localhost"}:
                return None
            raw = unquote(parsed.path)
        raw = raw.replace("\\ ", " ")
        looks_pathlike = is_file_uri or raw.startswith(("/", "~/", "./", "../")) or "/" in raw
        if not looks_pathlike:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        try:
            if path.exists():
                return str(path)
        except (OSError, ValueError):
            return None
        return None

    def _normalize_file_drop(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        tokens: list[str]
        try:
            tokens = shlex.split(stripped)
        except ValueError:
            tokens = stripped.splitlines()
        single = self._path_from_token(stripped)
        if single:
            return single
        paths = [self._path_from_token(token) for token in tokens]
        if paths and all(paths):
            return "\n".join(paths)
        paths = [self._path_from_token(line) for line in stripped.splitlines() if line.strip()]
        return "\n".join(paths) if paths and all(paths) else None

    def normalize_paste_for_display(self, text: str) -> str:
        path_text = self._normalize_file_drop(text)
        if path_text:
            return path_text
        lines = text.splitlines()
        if len(text) > self.PASTE_FILE_CHARS or len(lines) > self.PASTE_FILE_LINES:
            directory = Path.cwd() / ".titan" / "pastes"
            directory.mkdir(parents=True, exist_ok=True)
            # Keep files for prompt history/session replay; never expand them back
            # into the submitted prompt. NamedTemporaryFile gives unique private files.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".txt", prefix="paste-", dir=directory, delete=False) as pasted:
                pasted.write(text)
                return str(Path(pasted.name).resolve())
        return text

    def expand_paste_tokens(self, text: str) -> str:
        expanded = text
        for token, payload in self.paste_payloads.items():
            expanded = expanded.replace(token, payload)
        return expanded

    def record_history(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        if not self.message_history or self.message_history[-1] != stripped:
            self.message_history.append(stripped)
        self._history_index = None

    def recall_previous_history(self) -> bool:
        if not self.message_history:
            return False
        if self._history_index is None:
            self._history_draft = self.text
            self._history_index = len(self.message_history) - 1
        else:
            self._history_index = max(0, self._history_index - 1)
        self.load_text(self.message_history[self._history_index])
        return True

    def recall_next_history(self) -> bool:
        if not self.message_history or self._history_index is None:
            return False
        self._history_index += 1
        if self._history_index == len(self.message_history):
            self._history_index = None
            self.load_text(self._history_draft)
        else:
            self.load_text(self.message_history[self._history_index])
        return True

    async def _on_paste(self, event: events.Paste) -> None:
        event.stop()
        event.prevent_default()
        try:
            normalized = self.normalize_paste_for_display(event.text)
        except OSError as exc:
            self.app.notify(f"Could not save pasted text: {exc}. Keeping it inline.", severity="error")
            normalized = event.text
        self.insert(normalized)

    async def _on_key(self, event: events.Key) -> None:
        if event.key in {"shift+enter", "alt+enter"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.expand_paste_tokens(self.text)))
            return
        if event.key == "up" and self.cursor_location[0] == 0:
            if self.recall_previous_history():
                event.stop()
                event.prevent_default()
                return
        if event.key == "down" and self.cursor_location[0] == self.document.line_count - 1:
            if self.recall_next_history():
                event.stop()
                event.prevent_default()
                return
        await super()._on_key(event)


class ConversationLog(VerticalScroll):
    """A rendered conversation; semantic copy text is independent of decoration."""

    can_focus = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.selection_lines: list[str] = []

    def on_mount(self) -> None:
        # Layout may change repeatedly after Markdown.update() returns. Anchor
        # across those layout passes rather than sampling a transient scroll end.
        # Textual releases this on user scroll and resumes on reaching bottom.
        self.anchor()

    @property
    def text(self) -> str:
        return "\n\n".join(self.selection_lines)

    def write_selectable(self, renderable, plain: str | None = None) -> None:
        text = plain if plain is not None else _plain_renderable(renderable)
        self.write_message("", text)

    def write_message(self, speaker: str, body: str, plain: str | None = None) -> None:
        self.selection_lines.append(plain if plain is not None else body)
        content = Static(body, markup=False) if speaker in {"You", ""} else Markdown(body)
        message = Container(
            Static(speaker or "Activity", classes="message-speaker", markup=False),
            content,
            classes="message user-message" if speaker == "You" else "message assistant-message" if speaker else "message activity-message",
        )
        self.mount(message)


class TitanTui(App[None]):
    TITLE = "Titan"
    CSS_PATH = "titan_tui.tcss"
    BINDINGS = [
        ("ctrl+t", "copy_trace", "Copy trace"),
        ("ctrl+d", "toggle_top_tab", "Trace/Diff"),
        ("ctrl+y", "copy_chat", "Copy chat"),
        ("ctrl+f", "operator_input", "Focus input"),
        ("ctrl+r", "toggle_trace_verbosity", "Trace mode"),
        ("ctrl+p", "cycle_provider", "Provider"),
        ("ctrl+l", "cycle_model", "Model"),
        ("escape", "dismiss", "Back / Stop"),
        ("ctrl+n", "cycle_theme", "Theme"),
        ("ctrl+g", "stop", "Stop"),
        ("ctrl+c", "handle_ctrl_c", "Cancel/Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+end", "follow_latest", "Latest"),
    ]
    BINDINGS = [Binding(*binding, priority=True, show=False) for binding in BINDINGS]

    def __init__(self) -> None:
        super().__init__()
        self.theme = "textual-dark"
        self.cfg = load_harness_config()
        self.provider_options = ordered_provider_options(["grok", "openai-codex", "homebase", "litert"])
        current = canonical_provider(str(self.cfg.provider).strip().lower())
        family = provider_family(current)
        if family == "grok":
            current = "grok"
        elif family == "codex":
            current = "openai-codex"
        if current and current not in self.provider_options:
            self.provider_options.insert(0, current)
        self.model_options = self._models_for_provider(self.cfg.provider, self.cfg.model)
        self._model_button_models: dict[str, str] = {}
        provider = build_provider_from_config(self.cfg)
        tools = default_registry()
        session_path = str(tools.cwd / ".titan" / "session.jsonl")
        self.harness = TitanHarness(provider=provider, tools=tools, config=self.cfg, session_store=SessionStore(session_path))
        system = Message(role=Role.SYSTEM, content=TITAN_SYSTEM_PROMPT)
        entries = self.harness.session_store.load_entries()
        self.history = rebuild_llm_context(system, entries) if entries else [system]
        self.ui = UiState()
        self.compact_ui = False
        self.trace_verbosity_levels = ["compact", "normal", "full"]
        self.trace_verbosity_index = self.trace_verbosity_levels.index("normal")
        self.plan_shown_this_run = False
        self.trace_lines: list[str] = []
        self.diff_lines: list[str] = []
        self.chat_lines: list[str] = []
        self.active_top_tab = "trace"
        self.top_tab_expanded = False
        self.progress_update_interval_seconds = 15.0
        self.progress_events: list[str] = []
        self.last_progress_update_at = 0.0
        self.last_progress_signature = ""
        self.pending_api_key_provider: str | None = None
        self.ctrl_c_quit_armed = False
        self.themes = [
            {"name": "ocean", "plan": "cyan", "titan": "green"},
            {"name": "sunset", "plan": "yellow", "titan": "magenta"},
            {"name": "ember", "plan": "red", "titan": "yellow"},
            {"name": "violet", "plan": "blue", "titan": "cyan"},
        ]
        self.theme_index = 0
        self.activity = "Ready"
        self.stream_text = ""
        self.stream_dirty = False
        self.stream_widget: Markdown | None = None
        self.tool_targets: dict[str, str] = {}
        self._diff_refresh_requested = False
        self._diff_worker_running = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="masthead"):
            yield Static("TITAN", id="brand")
            yield Static(self.harness.workspace.label, id="workspace", markup=False)
            with Horizontal(id="top_tabs"):
                yield Button("Trace", id="tab-trace")
                yield Button("Diff", id="tab-diff")
        with Container(id="stage"):
            with Container(id="welcome"):
                yield Static("✳", id="welcome-mark")
                yield Static("What are we building?", id="welcome-title")
                # Quick-start instructions: show provider setup if not configured.
                provider_label = self.cfg.provider or "(not set)"
                model_label = self.cfg.model or "(not set)"
                has_key = self._provider_has_key(self.cfg.provider)
                if not has_key:
                    subtitle = (
                        f"Set up a provider first:\n"
                        f"  1) /config set provider {provider_label}\n"
                        f"  2) /config set model {model_label}\n"
                        f"  3) Click 'New key' or paste an API key, then press Enter"
                    )
                else:
                    subtitle = (
                        f"Configured: {provider_label} / {model_label}. "
                        f"Type a task and press Enter."
                    )
                yield Static(subtitle, id="welcome-subtitle")
                yield Static("/help  commands     ↑  prompt history", id="welcome-hint")
            yield ConversationLog(id="output")
            with Container(id="top"):
                yield SelectableRichLog(id="trace", soft_wrap=True)
                yield SelectableRichLog(id="diff", soft_wrap=True)
        with Container(id="provider_menu"):
            for p in self.provider_options:
                yield Button(p, id=f"provider-opt-{p}")
        with Container(id="model_menu"):
            for index in range(8):
                yield Button("", id=f"model-opt-{index}")
        yield Static("", id="assistant_line", markup=False)
        yield ComposerTextArea(
            "",
            id="input",
            soft_wrap=True,
            show_line_numbers=False,
            compact=True,
            placeholder="Ask Titan to build something…",
        )
        with Horizontal(id="controls"):
            yield Button(provider_display_name(self.cfg.provider), id="btn-provider")
            yield Button(self.cfg.model, id="btn-model")
            yield Button("New key", id="btn-new-key")
            yield Button("Stop", id="btn-stop")
            yield Static("enter send  ·  alt+enter newline", id="input-hint")
        yield Static("", id="api_key_prompt")
        yield Input("", id="api_key_input", password=True, placeholder="Paste API key and press Enter")
        yield Static("", id="status_line", markup=False)

    def on_mount(self) -> None:
        self._set_top_tab("trace")
        self.query_one("#api_key_prompt", Static).display = False
        self.query_one("#provider_menu", Container).styles.display = "none"
        self.query_one("#model_menu", Container).styles.display = "none"
        self._rebuild_model_menu()
        self._refresh_workspace_label()
        self.query_one("#api_key_input", Input).display = False
        self.query_one("#btn-new-key", Button).styles.display = "none"
        self.query_one("#input", ComposerTextArea).focus()
        self._apply_responsive_layout(self.size.width)
        self._refresh_status()
        for message in self.history:
            if message.role in {Role.USER, Role.ASSISTANT} and message.content:
                self._write_chat_box("You" if message.role == Role.USER else "Titan", message.content, "")
        self.set_interval(0.2, self._tick)

    def on_resize(self) -> None:
        self._apply_responsive_layout(self.size.width)

    def _refresh_workspace_label(self) -> None:
        try:
            widget = self.query_one("#workspace", Static)
        except NoMatches:
            return
        ws = self.harness.workspace
        widget.update(ws.label)
        widget.tooltip = str(ws.cwd)

    def _apply_responsive_layout(self, width: int) -> None:
        compact = width < 80
        if compact:
            self.add_class("compact")
        else:
            self.remove_class("compact")
        self.compact_ui = compact

        self.query_one("#btn-stop", Button).label = "Stop"

        provider = self.harness.config.provider
        self.query_one("#btn-provider", Button).label = f"{provider_display_name(provider)} ▾"
        self.query_one("#btn-model", Button).label = f"{self.harness.config.model} ▾"
        self.query_one("#btn-new-key", Button).label = "New key"
        self._refresh_top_tab_labels()

    def _refresh_top_tab_labels(self) -> None:
        try:
            trace_tab = self.query_one("#tab-trace", Button)
            diff_tab = self.query_one("#tab-diff", Button)
        except NoMatches:
            return
        trace_tab.label = "Activity ×" if self.top_tab_expanded and self.active_top_tab == "trace" else "Activity"
        diff_tab.label = "Diff ×" if self.top_tab_expanded and self.active_top_tab == "diff" else "Diff"
        trace_tab.set_class(self.top_tab_expanded and self.active_top_tab == "trace", "active-tab")
        diff_tab.set_class(self.top_tab_expanded and self.active_top_tab == "diff", "active-tab")

    def _set_top_tab_expanded(self, expanded: bool) -> None:
        self.top_tab_expanded = expanded
        try:
            top = self.query_one("#top", Container)
            output = self.query_one("#output", ConversationLog)
        except NoMatches:
            return
        top.set_class(self.top_tab_expanded, "expanded")
        output.set_class(self.top_tab_expanded, "trace-hidden")
        self.query_one("#welcome").display = not self.chat_lines and not expanded
        if expanded and self.active_top_tab == "diff":
            self._refresh_diff_tab()
        self._refresh_top_tab_labels()

    def _set_top_tab(self, tab: str) -> None:
        if tab not in {"trace", "diff"}:
            return
        self.active_top_tab = tab
        try:
            trace = self.query_one("#trace", SelectableRichLog)
            diff = self.query_one("#diff", SelectableRichLog)
        except NoMatches:
            return
        trace.display = tab == "trace"
        diff.display = tab == "diff"
        if tab == "diff":
            self._refresh_diff_tab()
        self._refresh_top_tab_labels()

    def _toggle_active_top_tab_expansion(self) -> None:
        self._set_top_tab_expanded(not self.top_tab_expanded)

    def _style_diff_line(self, line: str) -> Text:
        if line.startswith("+") and not line.startswith("+++"):
            return Text(line, style="green")
        if line.startswith("-") and not line.startswith("---"):
            return Text(line, style="red")
        if line.startswith("@@"):
            return Text(line, style="bold cyan")
        if line.startswith("diff --git"):
            return Text(line, style="bold magenta")
        if line.startswith("+++") or line.startswith("---"):
            return Text(line, style="yellow")
        return Text(line, style="dim") if line.startswith(" ") else Text(line)

    def _collect_git_diff(self) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", "--no-ext-diff", "--"],
                cwd=str(self.harness.tools.cwd),
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception as e:
            return f"diff unavailable: {e}"
        if result.returncode != 0:
            return (result.stderr or result.stdout or "diff unavailable").strip()
        return result.stdout.strip() or "No working-tree diff."

    def _refresh_diff_tab(self) -> None:
        self._diff_refresh_requested = True
        if self._diff_worker_running:
            return
        self._diff_worker_running = True

        async def refresh() -> None:
            try:
                while self._diff_refresh_requested:
                    self._diff_refresh_requested = False
                    text = await asyncio.to_thread(self._collect_git_diff)
                    self._display_diff(text)
            except Exception as exc:
                self._display_diff(f"Diff unavailable: {exc}")
            finally:
                self._diff_worker_running = False

        self.run_worker(refresh(), group="diff")

    def _display_diff(self, text: str) -> None:
        try:
            diff = self.query_one("#diff", SelectableRichLog)
        except NoMatches:
            return
        lines = text.splitlines() or ["No working-tree diff."]
        max_lines = 400
        if len(lines) > max_lines:
            omitted = len(lines) - max_lines
            lines = [f"... {omitted} earlier diff lines omitted ..."] + lines[-max_lines:]
        self.diff_lines = lines
        diff.set_lines(lines)

    def _record_progress_event(self, note: str) -> None:
        note = self._compact(note, 120)
        if not note:
            return
        self.progress_events.append(note)
        self.progress_events = self.progress_events[-8:]

    def _progress_summary(self) -> str:
        recent = "; ".join(self.progress_events[-2:]) if self.progress_events else "working through the current step"
        return (
            f"{recent}. State {self.ui.state}, turn {self.ui.turn}, "
            f"tools this turn {self.ui.turn_tool_calls}, total tools {self.ui.tool_calls}."
        )

    def _write_chat_progress(self, summary: str) -> None:
        plain = f"progress> {summary}"
        self.chat_lines.append(plain)
        self.query_one("#output", ConversationLog).write_selectable(
            Text(plain, style="dim #9aa0a6"),
            plain,
        )

    def _maybe_emit_progress_update(self, *, force: bool = False) -> None:
        if not self.harness.config.chat_recaps_enabled:
            return
        if not self.ui.pending:
            return
        now = time.time()
        if not force and now - self.last_progress_update_at < self.progress_update_interval_seconds:
            return
        summary = self._progress_summary()
        if summary == self.last_progress_signature:
            return
        self._write_chat_progress(summary)
        self.last_progress_update_at = now
        self.last_progress_signature = summary

    def _tick(self) -> None:
        if self.stream_dirty:
            self.stream_dirty = False
            output = self.query_one("#output", ConversationLog)
            if self.stream_widget is None:
                self.stream_widget = Markdown(self.stream_text, classes="streaming-message")
                output.mount(self.stream_widget)
            else:
                self.stream_widget.update(self.stream_text)
        if self.ui.pending:
            self.ui.thinking_dots = (self.ui.thinking_dots + 1) % 4
            dots = ("◐", "◓", "◑", "◒")[self.ui.thinking_dots]
            try:
                elapsed = time.time() - self.ui.started_at if self.ui.started_at else 0
                self.query_one("#assistant_line", Static).update(f"{dots} {self.activity} · {elapsed:.0f}s")
            except NoMatches:
                return
            self._maybe_emit_progress_update()
        self._refresh_status()

    def _clear_stream(self) -> None:
        if self.stream_widget is not None:
            self.stream_widget.remove()
            self.stream_widget = None
        self.stream_text = ""
        self.stream_dirty = False

    def _trace_emit(self, trace: RichLog, line: str, payload: dict | None = None) -> None:
        mode = self.trace_verbosity_levels[self.trace_verbosity_index]
        raw_line = line if payload is None or mode != "full" else f"{line} | {json.dumps(payload, ensure_ascii=False)}"
        self.trace_lines.append(raw_line)
        if mode == "compact":
            self._write_trace(line)
            return
        if mode == "normal":
            self._write_trace(line)
            return
        if payload is None:
            self._write_trace(line)
            return
        self._write_trace(raw_line)

    def _write_trace(self, text: str) -> None:
        self.query_one("#trace", SelectableRichLog).write_selectable(text)

    def _compact(self, text: str, limit: int = 140) -> str:
        compact = " ".join(text.split())
        return compact if len(compact) <= limit else compact[: limit - 1] + "…"

    def _tool_trace_label(self, name: str, arguments: object) -> tuple[str, str, str]:
        args = arguments if isinstance(arguments, dict) else {}
        if name == "read_file":
            return ("📖", "read", str(args.get("path", "")))
        if name == "search_files":
            return ("🔎", "grep", str(args.get("pattern", "")))
        if name == "patch":
            return ("🔧", "patch", str(args.get("path", "")) or str(args.get("mode", "")))
        if name == "write_file":
            return ("📝", "write", str(args.get("path", "")))
        if name == "edit_file":
            return ("🔧", "edit", str(args.get("path", "")))
        if name in {"shell", "terminal"}:
            return ("💻", "bash", self._compact(str(args.get("command", "")), 100))
        return ("🔧", name or "tool", self._compact(str(arguments), 100))

    def _brief_chat_text(self, text: str, max_chars: int = 1400, max_lines: int = 12, grace_chars: int = 320) -> str:
        """Briefly trim long responses for the chat box, with a grace window."""
        stripped = text.strip()
        if not stripped:
            return stripped
        # If already under max_chars, return as-is.
        if len(stripped) <= max_chars:
            return stripped
        # Count lines; if under max_lines, return as-is (the grace window
        # exists so the last complete sentence is preserved).
        lines = stripped.splitlines()
        if len(lines) <= max_lines:
            return stripped
        # Truncate to max_chars worth of lines, then extend by grace_chars
        # to avoid cutting mid-sentence.
        truncated = "\n".join(lines[:max_lines])
        if len(truncated) <= max_chars - grace_chars:
            # Add a few more lines within the grace window.
            extra = max_lines
            while extra <= len(lines) and len(truncated) < max_chars - grace_chars:
                truncated += "\n" + lines[extra]
                extra += 1
        return truncated

    def _theme_color_for_speaker(self, speaker: str, requested: str) -> str:
        theme = self.themes[self.theme_index]
        if speaker.startswith("Titan plan"):
            return str(theme["plan"])
        if speaker.startswith("Titan"):
            return str(theme["titan"])
        return requested

    def _chat_renderable(self, speaker: str, body: str, border_style: str):
        if speaker == "You":
            return Text(f"• {body}", style="bold")
        return body

    def _chat_plain_text(self, speaker: str, body: str) -> str:
        return f"• {body}" if speaker == "You" else f"{speaker}: {body}"

    def _write_chat_box(self, speaker: str, text: str, border_style: str) -> None:
        body = self._brief_chat_text(text)
        plain = self._chat_plain_text(speaker, body)
        self.chat_lines.append(plain)
        self.query_one("#welcome").display = False
        output = self.query_one("#output", ConversationLog)
        output.add_class("has-messages")
        output.write_message(speaker, body, plain)

    def _write_chat_plain(self, text: str) -> None:
        self.chat_lines.append(text)
        self.query_one("#output", ConversationLog).write_selectable(text)

    def _copy_to_clipboard(self, text: str, label: str) -> None:
        if not text.strip():
            self._write_trace(f"copy {label}: nothing to copy")
            return
        try:
            subprocess.run(["pbcopy"], input=text, text=True, check=True, timeout=5)
            self._write_trace(f"copied {label} to clipboard")
            self.query_one("#status_line", Static).update(f"copied {label} to clipboard")
        except Exception as e:
            self._write_trace(f"copy {label} failed: {e}")

    def _emit_chat_trace(self, note: str) -> None:
        if not self.harness.config.chat_recaps_enabled:
            return
        self._write_chat_plain(f"trace> {note}")

    def _chat_trace_mode(self) -> str:
        return self.trace_verbosity_levels[self.trace_verbosity_index]

    def _flush_tool_summary_to_chat(self) -> None:
        if self.ui.pending_tool_count <= 0:
            return
        names: list[str] = []
        for name in self.ui.pending_tool_names:
            if name and name not in names:
                names.append(name)
        shown = ", ".join(names[:3]) if names else "unknown"
        suffix = "" if len(names) <= 3 else f", +{len(names) - 3}"
        self._emit_chat_trace(f"tools {self.ui.pending_tool_count} call(s): {shown}{suffix}")
        self.ui.pending_tool_names.clear()
        self.ui.pending_tool_count = 0

    def _refresh_status(self) -> None:
        window = context_window_for(self.harness.config.provider, self.harness.config.model)
        ctx_tokens = estimate_context_tokens(self.history)
        status_parts = [f"{ctx_tokens / window:.0%} context · {self.ui.tool_calls} tools"]

        if self.ui.pending:
            status_parts.append(f"turn {self.ui.turn}")
            # Show remaining wall-clock time.
            if self.ui.started_at is not None:
                elapsed = time.time() - self.ui.started_at
                total_ms = self.harness.config.max_wall_clock_ms
                remaining_ms = max(0, total_ms - elapsed * 1000)
                remaining_s = remaining_ms / 1000
                if remaining_s > 0:
                    mins, secs = divmod(int(remaining_s), 60)
                    status_parts.append(f"{mins}:{secs:02d}s remaining")
                else:
                    status_parts.append("no time left")
            status_parts.append(f"state {self.ui.state}")
        else:
            # Show provider status when idle.
            provider_name = self.harness.config.provider
            has_key = self._provider_has_key(provider_name)
            if has_key:
                status_parts.append("✓ connected")
            else:
                status_parts.append("⚠ no key for " + provider_name)
            status_parts.append("/help commands")

        if self.size.width >= 110:
            status_parts.append("ctrl+d details · ctrl+end latest · ctrl+q quit")
        elif self.size.width >= 80:
            status_parts.append("ctrl+end latest")
        try:
            self.query_one("#status_line", Static).update("   ".join(status_parts))
            self.query_one("#btn-stop").display = self.ui.pending
            self.query_one("#btn-provider", Button).disabled = self.ui.pending
            self.query_one("#btn-model", Button).disabled = self.ui.pending
        except NoMatches:
            return

    async def on_composer_text_area_submitted(self, event: ComposerTextArea.Submitted) -> None:
        composer = self.query_one("#input", ComposerTextArea)
        task = event.value.strip()
        if self.ui.pending:
            self.notify("Titan is working. Your draft is saved; Esc stops the run.", timeout=3)
            return
        if task and not self.ui.pending:
            composer.record_history(task)
        composer.load_text("")
        composer.paste_payloads.clear()
        try:
            await self._submit_task(task)
        except Exception as exc:
            composer.load_text(task)
            self.on_loop_failed_msg(LoopFailedMsg(exc))

    def _run_local_operation(self, label, work, finished) -> None:
        self.ui.pending = True
        self.ui.started_at = time.time()
        self.ctrl_c_quit_armed = False
        self.activity = label
        self._refresh_status()

        async def perform() -> None:
            try:
                result = await asyncio.to_thread(work)
                finished(result)
                self.query_one("#assistant_line", Static).update(f"✓ {label} finished")
            except Exception as exc:
                self.on_loop_failed_msg(LoopFailedMsg(exc))
            finally:
                self.ui.pending = False
                self.ui.started_at = None
                self.activity = "Ready"
                self._refresh_status()

        self.run_worker(perform(), group="workspace-operation")

    async def _submit_task(self, task: str) -> None:
        if not task or self.ui.pending:
            return
        self.action_follow_latest()

        if task.startswith("/"):
            res = execute_slash_command(task, run_pending=self.ui.pending, registry=self.harness.tools, harness=self.harness)
            if res.handled:
                if res.message == "trace-toggle":
                    self.action_toggle_trace_verbosity()
                    self._write_chat_box("You", task, "cyan")
                    self._write_chat_box("Titan", "toggled trace verbosity", "green")
                elif res.message.startswith("compact:"):
                    if self.ui.pending:
                        self._write_chat_box("Titan", "busy; /compact refused while a run is active", "yellow")
                        return
                    instructions = res.message.split(":", 1)[1]
                    cfg = self.harness.config
                    cc = CompactionController(
                        self.harness.provider,
                        self.harness.session_store,
                        cfg.compaction,
                        cfg.model,
                        context_window_for(cfg.provider, cfg.model),
                    )
                    self._write_chat_box("You", task, "cyan")

                    def compacted(result) -> None:
                        if result:
                            self.history[:] = rebuild_llm_context(
                                Message(role=Role.SYSTEM, content=TITAN_SYSTEM_PROMPT),
                                self.harness.session_store.load_entries(),
                            )
                        self._write_chat_box("Titan", "already compacted; nothing to compact" if not result else f"compacted tokens_before={result['tokens_before']}", "green")

                    self._run_local_operation("Compacting context", lambda: cc.maybe_compact(self.history, reason="manual", custom_instructions=instructions, force=True), compacted)
                    return
                elif res.message.startswith("undo:"):
                    if self.ui.pending:
                        self._write_chat_box("Titan", "busy; /undo refused while a run is active", "yellow")
                        return
                    checkpoint_id = res.message.split(":", 1)[1] or None
                    self._write_chat_box("You", task, "cyan")

                    def _undo():
                        from .git_checkpoint import GitCheckpointError, restore_checkpoint

                        try:
                            result = restore_checkpoint(checkpoint_id=checkpoint_id or None)
                            return {"ok": result.ok, "error": None if result.ok else result.message}
                        except GitCheckpointError as exc:
                            return {"ok": False, "error": str(exc)}

                    self._run_local_operation(
                        "Restoring checkpoint",
                        _undo,
                        lambda undone: self._write_chat_box("Titan", "undo ok" if undone.get("ok") else f"undo failed: {undone.get('error')}", "green" if undone.get("ok") else "red"),
                    )
                    return
                else:
                    if task.startswith("/config"):
                        self.cfg = load_harness_config()
                        self.harness.config = self.cfg
                        self.harness.provider = build_provider_from_config(self.cfg)
                        self._hide_model_menu()
                        self._rebuild_model_menu()
                        self._apply_responsive_layout(self.size.width)
                    self._write_chat_box("You", task, "cyan")
                    prefix = "Titan error" if res.is_error else "Titan"
                    self._write_chat_box(prefix, res.message, "red" if res.is_error else "green")
                    if task.startswith("/cd") or task.startswith("/pwd"):
                        self._refresh_workspace_label()
                return

        self._write_chat_box("You", task, "cyan")
        self.query_one("#output", ConversationLog).refresh()
        self.activity = "Thinking"
        self.query_one("#assistant_line", Static).update("◐ Thinking")

        self.ui.pending = True
        self.ctrl_c_quit_armed = False
        self.ui.started_at = time.time()
        self.ui.tool_calls = 0
        self.ui.turn_tool_calls = 0
        self.ui.turn = 0
        self.ui.state = "PLAN"
        self.ui.thinking_dots = 0
        self.ui.pending_tool_names.clear()
        self.ui.pending_tool_count = 0
        self.progress_events.clear()
        self.last_progress_update_at = time.time()
        self.last_progress_signature = ""
        self.plan_shown_this_run = False
        self.tool_targets.clear()
        self._refresh_status()

        def _run_blocking() -> None:
            def cb(ev: AgentEvent) -> None:
                self.post_message(LoopEventMsg(ev))

            try:
                out = self.harness.run_with_callback(task, self.history, on_event=cb)
            except Exception as exc:
                self.post_message(LoopFailedMsg(exc))
            else:
                self.post_message(LoopDoneMsg(out))

        self.run_worker(_run_blocking, thread=True)

    def on_loop_event_msg(self, msg: LoopEventMsg) -> None:
        ev = msg.event
        trace = self.query_one("#trace", SelectableRichLog)

        if ev.type == "run_started":
            pass
        elif ev.type == "route_decision":
            state = str(ev.payload.get("state", self.ui.state))
            reason = str(ev.payload.get("reason", ""))
            self.ui.state = state
            self._record_progress_event(f"routed to {state}: {reason}")
            pass
        elif ev.type == "plan_budget":
            phases = ev.payload.get("phases", [])
            phase_summary = ", ".join(
                f"{p.get('name')}:{p.get('iterations')}" for p in phases if isinstance(p, dict)
            )
            max_iterations = ev.payload.get("max_iterations")
            self._record_progress_event(f"planned iteration budget {phase_summary} within {max_iterations} turns")
            pass
        elif ev.type == "iteration_started":
            self.ui.turn = int(ev.payload.get("iteration", self.ui.turn))
            self.ui.turn_tool_calls = 0
            pass
        elif ev.type == "provider_request":
            self._clear_stream()
            self.activity = "Thinking"
            self._record_progress_event(
                f"asking {ev.payload.get('provider', self.harness.config.provider)} {ev.payload.get('model', self.harness.config.model)} for next step"
            )
            pass
        elif ev.type == "provider_stream_delta":
            if ev.payload.get("kind", "text") == "text":
                self.stream_text += str(ev.payload.get("text", ""))
                self.stream_dirty = True
        elif ev.type == "provider_stream_tool_call":
            pass
        elif ev.type == "budget_finalization_requested":
            remaining = ev.payload.get("remaining_iterations")
            self.ui.state = "FINALIZE"
            self._record_progress_event(f"using reserved finalization turn ({remaining} remaining) instead of taking more tool actions")
            pass
            self._maybe_emit_progress_update(force=True)
        elif ev.type == "empty_turn_recovery":
            self._record_progress_event("recovering from an empty assistant turn after tool use")
            pass
            if self._chat_trace_mode() in ("normal", "full"):
                self._emit_chat_trace("recovering from empty post-tool turn")
        elif ev.type == "on_state_enter":
            prev_state = self.ui.state
            prev_turn = self.ui.turn
            self.ui.state = str(ev.payload.get("state", self.ui.state))
            self.ui.turn = int(ev.payload.get("turn", self.ui.turn))
            if self.ui.turn != prev_turn:
                self.ui.turn_tool_calls = 0
            self._record_progress_event(f"entered {self.ui.state} phase on turn {self.ui.turn}")
            pass
            if self.ui.state != prev_state and self._chat_trace_mode() in ("normal", "full"):
                self._emit_chat_trace(f"state {prev_state} -> {self.ui.state} (turn {self.ui.turn})")
        elif ev.type == "on_transition":
            from_state = ev.payload.get("from_state")
            to_state = ev.payload.get("to_state")
            self._record_progress_event(f"finished {from_state} and moved to {to_state}")
            pass
            self._maybe_emit_progress_update(force=True)
        elif ev.type == "assistant_message":
            text = str(ev.payload.get("text", "")).strip()
            if text:
                compact = self._compact(text, 180)
                self._trace_emit(trace, f"reasoning: {compact}", ev.payload)
                self._flush_tool_summary_to_chat()
                state = str(ev.payload.get("state", self.ui.state))
                has_tool_calls = bool(ev.payload.get("has_tool_calls"))
                if has_tool_calls:
                    self._clear_stream()
                    self._write_chat_box("Titan", text, "yellow")
                    self.plan_shown_this_run = True
                    self._record_progress_event("planned the approach and started executing tools")
                self._emit_chat_trace(f"reasoning {compact}")
        elif ev.type == "tool_batch_started":
            self._record_progress_event(f"started {ev.payload.get('count')} tool call(s)")
            pass
        elif ev.type == "tool_batch_rejected":
            self._record_progress_event("stopped an over-budget tool batch before execution")
            pass
        elif ev.type == "tool_call":
            name = str(ev.payload.get("name", ""))
            args_obj = ev.payload.get("arguments", {})
            self.ui.tool_calls = int(ev.payload.get("tool_calls_total", ev.payload.get("count", self.ui.tool_calls + 1)))
            self.ui.turn_tool_calls = int(ev.payload.get("tool_calls_this_turn", self.ui.turn_tool_calls + 1))
            self.ui.pending_tool_count += 1
            if name:
                self.ui.pending_tool_names.append(name)
            self._record_progress_event(f"running tool {name or 'unknown'}")
            icon, verb, target = self._tool_trace_label(name, args_obj)
            self.activity = self._compact(f"{verb.capitalize()} {target}", max(20, self.size.width - 18))
            self.tool_targets[str(ev.payload.get("id", ""))] = f"{verb} · {target}" if target else verb
            self._trace_emit(
                trace,
                f"┊ {icon} {verb:<9} {target}  [{self.ui.turn_tool_calls}/{self.ui.tool_calls}]",
                ev.payload,
            )
        elif ev.type == "tool_call_rejected":
            name = str(ev.payload.get("name", ""))
            args = str(ev.payload.get("arguments", ""))
            compact_args = self._compact(args, 120)
            self._record_progress_event(f"rejected tool {name or 'unknown'} due to policy or budget")
            self._trace_emit(
                trace,
                (
                    f"tool-call rejected {ev.payload.get('index')}/{ev.payload.get('count')} "
                    f"{name} args={compact_args}"
                ),
                ev.payload,
            )
        elif ev.type == "tool_result":
            name = str(ev.payload.get("name", ""))
            is_error = bool(ev.payload.get("is_error"))
            content = str(ev.payload.get("content", "")).strip()
            compact_content = self._compact(content, 140)
            self._record_progress_event(f"finished tool {name or 'unknown'} {'with an error' if is_error else 'successfully'}")
            status_icon = "❌" if is_error else "✅"
            label = self.tool_targets.pop(str(ev.payload.get("id", "")), name or "tool")
            self._write_chat_plain(f"{'×' if is_error else '✓'} {label}" + (f" · {self._compact(content, 100)}" if is_error else ""))
            self._trace_emit(trace, f"┊ {status_icon} {name or 'tool'} {('ERR' if is_error else 'OK')} {compact_content}", ev.payload)
            if name == "cd" and not is_error:
                from .workspace import inspect_workspace

                self.harness.workspace = inspect_workspace(self.harness.tools.cwd)
                self._refresh_workspace_label()
            if self._chat_trace_mode() == "full":
                self._emit_chat_trace(
                    f"tool-result {name or 'unknown'} {'ERR' if is_error else 'OK'} {compact_content}"
                )
            if self.active_top_tab == "diff" and self.top_tab_expanded:
                self._refresh_diff_tab()
            self._maybe_emit_progress_update()
        elif ev.type == "on_skill_created":
            self._trace_emit(trace, f"skill-created {ev.payload.get('path')}", ev.payload)
        elif ev.type == "git_checkpoint":
            self._trace_emit(trace, f"checkpoint {ev.payload.get('id', '')}", ev.payload)
        elif ev.type == "compaction":
            self._trace_emit(
                trace,
                f"compacted {ev.payload.get('tokens_before')}->{ev.payload.get('tokens_after')} tokens",
                ev.payload,
            )
        elif ev.type == "compaction_failed":
            self._trace_emit(trace, f"compaction failed: {ev.payload.get('error')}", ev.payload)
        elif ev.type == "verify":
            skipped = ev.payload.get("skipped_reason") or ""
            cmd = ev.payload.get("command") or skipped or "verify"
            self._trace_emit(trace, f"verify {cmd} exit={ev.payload.get('exit_code')}", ev.payload)

        self._refresh_status()

    def _next_step_for_final(self, final_text: str) -> str:
        lowered = final_text.lower()
        if "fail" in lowered or "gap" in lowered:
            return "Fix the first failing/gap item above, then rerun the targeted test."
        if "pass" in lowered:
            return "Run one more focused dogfood task against the next highest-risk behavior."
        return "Continue with the smallest concrete follow-up that verifies or improves the result."

    def _should_append_final_summary(self, out: RunOutcome) -> bool:
        if not self.harness.config.chat_recaps_enabled:
            return False
        return not (
            out.stop.reason == RunStopReason.AssistantFinal
            and out.stop.iterations <= 1
            and out.stop.tool_calls_total == 0
        )

    def _with_final_summary(self, final_text: str, out: RunOutcome) -> str:
        if not self._should_append_final_summary(out):
            return final_text
        if "Next best step:" in final_text and "Summary:" in final_text:
            return final_text
        summary = f"Finished with {out.stop.reason.value}; turns={out.stop.iterations}, tools={out.stop.tool_calls_total}."
        next_step = self._next_step_for_final(final_text)
        return (
            f"{final_text.rstrip()}\n\n"
            "Summary:\n"
            f"- {summary}\n\n"
            "Next best step:\n"
            f"- {next_step}\n"
            "- I can do that next if you say: do it."
        )

    def _budget_iteration_fallback_text(self, out: RunOutcome) -> str:
        return (
            "Summary:\n"
            "- Paused cleanly at the configured iteration ceiling before Titan produced a final answer.\n"
            f"- Progress used {out.stop.iterations} turns and {out.stop.tool_calls_total} tool calls.\n\n"
            "Next best step:\n"
            "- Continue the same task with the current context, or raise max_iterations for larger work."
        )

    def on_loop_done_msg(self, msg: LoopDoneMsg) -> None:
        partial = self.stream_text
        self._clear_stream()
        out = msg.outcome

        if partial.strip() and out.stop.reason != RunStopReason.AssistantFinal:
            self._write_chat_box("Titan · partial response", partial, "")

        if out.text.strip():
            final_text = out.text.strip()
        elif out.stop.reason == RunStopReason.BudgetIterations:
            final_text = self._budget_iteration_fallback_text(out)
        else:
            final_text = f"Stopped: {out.stop.reason.value} ({out.stop.notes or 'no details'})"
        final_text = self._with_final_summary(final_text, out)
        self._record_progress_event(
            f"run stopped at {out.stop.reason.value} after {out.stop.iterations} turns and {out.stop.tool_calls_total} tools"
        )
        self._maybe_emit_progress_update(force=True)
        self._flush_tool_summary_to_chat()
        self._write_chat_box("Titan", final_text, "green")
        duration = max(out.stop.elapsed_ms / 1000, time.time() - self.ui.started_at if self.ui.started_at is not None else 0)
        label = (
            "Completed" if out.stop.reason == RunStopReason.AssistantFinal
            else "Stopped" if out.stop.reason == RunStopReason.Interrupted
            else "Limit reached" if out.stop.reason.value.startswith("Budget")
            else "Run failed"
        )
        self.query_one("#assistant_line", Static).update(f"{'✓' if out.stop.reason == RunStopReason.AssistantFinal else '·'} {label} · {duration:.1f}s")

        # Keep stop/trace details minimal in trace view; tool/path/reasoning lines are primary.
        if self.active_top_tab == "diff" and self.top_tab_expanded:
            self._refresh_diff_tab()

        self.ui.pending = False
        self.ui.started_at = None
        self.ui.turn = out.stop.iterations
        self.ui.tool_calls = out.stop.tool_calls_total
        self.activity = "Ready"
        self._refresh_status()

    def on_loop_failed_msg(self, msg: LoopFailedMsg) -> None:
        partial = self.stream_text
        self._clear_stream()
        if partial.strip():
            self._write_chat_box("Titan · partial response", partial, "")
        detail = f"{type(msg.error).__name__}: {msg.error}"
        self._write_chat_box("Titan error", detail, "red")
        self._write_trace(detail)
        self.ui.pending = False
        self.ui.started_at = None
        self.ui.pending_tool_names.clear()
        self.ui.pending_tool_count = 0
        self.activity = "Ready"
        self.query_one("#api_key_input", Input).disabled = False
        self.query_one("#assistant_line", Static).update("× Run failed · you can retry")
        self._refresh_status()

    def action_stop(self) -> None:
        self.harness.interrupt()
        self.activity = "Stopping"
        self.query_one("#assistant_line", Static).update("· Stopping — waiting for the current operation")
        self._write_trace("stop requested")

    def action_handle_ctrl_c(self) -> None:
        if self.ctrl_c_quit_armed:
            self.exit()
            return
        if self.ui.pending:
            self.action_stop()
            self.ctrl_c_quit_armed = True
            self._write_trace("Ctrl+C: cancelled current task; press Ctrl+C again to quit Titan")
            return
        self.ctrl_c_quit_armed = True
        self._write_trace("Ctrl+C: no active task; press Ctrl+C again to quit Titan")

    def action_close_titan_instance(self) -> None:
        self.action_stop()

    def action_operator_input(self) -> None:
        self._write_trace("operator input: type in bottom box and press Enter")
        self.query_one("#input", ComposerTextArea).focus()

    def action_follow_latest(self) -> None:
        self._set_top_tab_expanded(False)
        self.query_one("#output", ConversationLog).anchor()

    def action_clear_input(self) -> None:
        composer = self.query_one("#input", ComposerTextArea)
        composer.load_text("")
        composer.paste_payloads.clear()
        composer.focus()
        self._write_trace("input cleared")

    def action_toggle_trace_verbosity(self) -> None:
        self.trace_verbosity_index = (self.trace_verbosity_index + 1) % len(self.trace_verbosity_levels)
        mode = self.trace_verbosity_levels[self.trace_verbosity_index]
        self._apply_responsive_layout(self.size.width)
        self._write_trace(f"trace verbosity -> {mode}")
        self.notify(f"Activity detail: {mode}", timeout=2)
        self._refresh_status()

    def action_toggle_top_tab(self) -> None:
        if not self.top_tab_expanded:
            self._set_top_tab_expanded(True)
        elif self.active_top_tab == "trace":
            self._set_top_tab("diff")
        else:
            self._set_top_tab_expanded(False)
            self._set_top_tab("trace")
        self._refresh_status()

    def action_dismiss(self) -> None:
        if self.pending_api_key_provider:
            self._hide_provider_key_prompt()
        elif any(self.query_one(f"#{name}").display for name in ("provider_menu", "model_menu")):
            self._hide_provider_menu()
            self._hide_model_menu()
        elif self.top_tab_expanded:
            self._set_top_tab_expanded(False)
        elif self.ui.pending:
            self.action_stop()
        self.query_one("#input", ComposerTextArea).focus()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "input":
            event.text_area.styles.height = min(8, max(3, event.text_area.document.line_count + 2))

    def action_copy_trace(self) -> None:
        self._copy_to_clipboard("\n".join(self.trace_lines), "trace")

    def action_copy_chat(self) -> None:
        self._copy_to_clipboard("\n\n".join(self.chat_lines), "chat")

    def action_cycle_theme(self) -> None:
        self.theme_index = (self.theme_index + 1) % len(self.themes)
        colors = ["#b9a178", "#d69aab", "#d39970", "#b1a0d0"]
        self.query_one("#brand").styles.color = colors[self.theme_index]
        self.query_one("#input").styles.border = ("round", colors[self.theme_index])
        self._write_trace(f"theme -> {self.themes[self.theme_index]['name']}")

    def _provider_has_key(self, provider: str) -> bool:
        if provider == "mock":
            return True
        # Local no-auth OpenAI-compat endpoints (Home Base brain, LiteRT).
        if canonical_provider(provider) in {"homebase", "litert"}:
            return True
        if self.harness.config.api_keys.get(provider):
            return True
        try:
            return resolve_provider_credentials(provider, base_url=self.harness.config.api_base or None) is not None
        except Exception:
            return False

    def _ensure_local_provider_key(self, provider: str) -> None:
        """Seed a placeholder key so Authorization headers exist for no-auth local servers."""
        key = canonical_provider(provider)
        if key not in {"homebase", "litert"}:
            return
        if (self.harness.config.api_keys.get(key) or "").strip():
            return
        self._save_provider_key(key, "local")

    def _prompt_for_provider_key(self, provider: str) -> None:
        self.pending_api_key_provider = provider
        prompt = self.query_one("#api_key_prompt", Static)
        key_input = self.query_one("#api_key_input", Input)
        prompt.update(f"API key required for {provider}. Paste key and press Enter; input is hidden.")
        prompt.display = True
        key_input.value = ""
        key_input.display = True
        key_input.focus()
        self._write_trace(f"provider {provider} needs a saved API key")

    def _show_new_key_button_forever(self) -> None:
        """Keep the 'New key' button visible until a key is configured."""
        try:
            self.query_one("#btn-new-key", Button).styles.display = "block"
        except NoMatches:
            pass

    def _validate_provider_key(self, provider: str, key: str) -> tuple[bool, str]:
        try:
            base = provider_default_base_url(provider).rstrip("/")
        except Exception as exc:
            return False, f"unknown provider: {exc}"

        req = request.Request(
            f"{base}/models",
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=8) as resp:
                if getattr(resp, "status", 200) >= 400:
                    return False, f"http {getattr(resp, 'status', 'error')}"
                return True, "ok"
        except error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode(errors="ignore")
            except Exception:
                pass
            return False, f"http {e.code}: {body or e.reason}"
        except Exception as e:
            return False, str(e)

    def _hide_provider_key_prompt(self) -> None:
        self.pending_api_key_provider = None
        self.query_one("#api_key_prompt", Static).display = False
        key_input = self.query_one("#api_key_input", Input)
        key_input.value = ""
        key_input.display = False
        self.query_one("#btn-new-key", Button).styles.display = "none"

    def _save_provider_key(self, provider: str, key: str) -> None:
        self.harness.config.api_keys[provider] = key
        self.cfg.api_keys[provider] = key
        update_config_key(resolve_config_path(), f"api_keys.{provider}", key)
        self.harness.provider = build_provider_from_config(self.harness.config)

    def _provider_default_model(self, provider: str) -> str:
        return provider_default_model(provider)

    def _models_for_provider(self, provider: str, current_model: str | None = None) -> list[str]:
        models = provider_model_options(provider)
        current = str(current_model or "").strip()
        if current and current not in models:
            models = [current, *models]
        return models

    def _rebuild_model_menu(self) -> None:
        self.model_options = self._models_for_provider(self.harness.config.provider, self.harness.config.model)
        self._model_button_models = {}
        for index in range(8):
            try:
                button = self.query_one(f"#model-opt-{index}", Button)
            except NoMatches:
                continue
            if index < len(self.model_options):
                model = self.model_options[index]
                button_id = f"model-opt-{index}"
                self._model_button_models[button_id] = model
                button.label = model
                button.display = True
            else:
                button.display = False

    def _apply_model_selection(self, next_model: str) -> None:
        if self.ui.pending:
            return
        next_model = str(next_model).strip()
        if not next_model:
            return
        if next_model not in self._models_for_provider(self.harness.config.provider, self.harness.config.model):
            self.notify("That model belongs to a different provider. Choose the provider first.", timeout=4)
            self._rebuild_model_menu()
            return
        self.harness.config.model = next_model
        self.cfg.model = next_model
        update_config_key(resolve_config_path(), "model", next_model)
        self._hide_model_menu()
        self._apply_responsive_layout(self.size.width)
        self._write_trace(f"model -> {next_model}")
        self._refresh_status()

    def _apply_provider_selection(self, next_provider: str) -> None:
        if self.ui.pending:
            return
        next_provider = canonical_provider(str(next_provider).strip().lower())
        if not next_provider:
            return
        self._hide_model_menu()
        if provider_family(next_provider) != provider_family(self.harness.config.provider):
            self.harness.config.api_base = ""
            self.cfg.api_base = ""
            update_config_key(resolve_config_path(), "api_base", "")
        self.harness.config.provider = next_provider
        self.cfg.provider = next_provider
        models = self._models_for_provider(next_provider, None)
        if self.harness.config.model not in models:
            default_model = self._provider_default_model(next_provider)
            self.harness.config.model = default_model
            self.cfg.model = default_model
            update_config_key(resolve_config_path(), "model", default_model)
        update_config_key(resolve_config_path(), "provider", next_provider)
        self._ensure_local_provider_key(next_provider)
        self._show_new_key_button_forever()
        self.harness.provider = build_provider_from_config(self.harness.config)
        if not self._provider_has_key(next_provider):
            self._prompt_for_provider_key(next_provider)
        self._hide_provider_menu()
        self._rebuild_model_menu()
        self._apply_responsive_layout(self.size.width)
        self._write_trace(f"provider -> {next_provider} model={self.harness.config.model}")
        self._refresh_status()

    def _show_provider_menu(self) -> None:
        self._hide_model_menu()
        menu = self.query_one("#provider_menu", Container)
        menu.styles.display = "block"
        self.query_one("#api_key_prompt", Static).update("Pick provider")
        self.query_one("#api_key_prompt", Static).display = True

    def _hide_provider_menu(self) -> None:
        self.query_one("#provider_menu", Container).styles.display = "none"
        if self.pending_api_key_provider is None:
            try:
                if str(self.query_one("#model_menu", Container).styles.display) == "none":
                    self.query_one("#api_key_prompt", Static).display = False
            except NoMatches:
                self.query_one("#api_key_prompt", Static).display = False

    def _show_model_menu(self) -> None:
        self._hide_provider_menu()
        self._rebuild_model_menu()
        menu = self.query_one("#model_menu", Container)
        menu.styles.display = "block"
        self.query_one("#api_key_prompt", Static).update("Pick model")
        self.query_one("#api_key_prompt", Static).display = True

    def _hide_model_menu(self) -> None:
        try:
            self.query_one("#model_menu", Container).styles.display = "none"
        except NoMatches:
            return
        if self.pending_api_key_provider is None:
            try:
                if str(self.query_one("#provider_menu", Container).styles.display) == "none":
                    self.query_one("#api_key_prompt", Static).display = False
            except NoMatches:
                self.query_one("#api_key_prompt", Static).display = False

    def action_cycle_provider(self) -> None:
        if self.ui.pending:
            self._write_trace("provider switch blocked while run is active")
            return
        idx = self.provider_options.index(self.harness.config.provider) if self.harness.config.provider in self.provider_options else 0
        next_provider = self.provider_options[(idx + 1) % len(self.provider_options)]
        self._apply_provider_selection(next_provider)

    def action_cycle_model(self) -> None:
        if self.ui.pending:
            self._write_trace("model switch blocked while run is active")
            return
        models = self._models_for_provider(self.harness.config.provider, self.harness.config.model)
        current = self.harness.config.model
        idx = models.index(current) if current in models else 0
        self._apply_model_selection(models[(idx + 1) % len(models)])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "api_key_input" or not self.pending_api_key_provider or self.ui.pending:
            return
        key = event.value.strip()
        provider = self.pending_api_key_provider
        if not key:
            self._write_trace(f"provider {provider} key not saved: empty input")
            return
        event.input.disabled = True

        def validated(result) -> None:
            self.query_one("#api_key_input", Input).disabled = False
            if self.pending_api_key_provider != provider:
                return  # The user dismissed the credential prompt while waiting.
            valid, detail = result
            if not valid:
                self._prompt_for_provider_key(provider)
                self.query_one("#api_key_prompt", Static).update(
                    f"Invalid key for {provider}; not saved. ({self._compact(detail, 120)})"
                )
                self._write_trace(f"provider {provider} key invalid; existing credentials preserved")
                return
            self._save_provider_key(provider, key)
            self._hide_provider_key_prompt()
            self._write_trace(f"saved valid API key for {provider}")
            self.query_one("#input", ComposerTextArea).focus()

        self._run_local_operation("Validating credentials", lambda: self._validate_provider_key(provider, key), validated)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if not bid:
            return
        if bid.startswith("provider-opt-"):
            self._apply_provider_selection(bid.removeprefix("provider-opt-"))
            return
        if bid.startswith("model-opt-"):
            model = self._model_button_models.get(bid, event.button.label)
            self._apply_model_selection(str(model))
            return
        if bid == "btn-stop":
            self.action_stop()
        elif bid == "tab-trace":
            if self.active_top_tab != "trace":
                self._set_top_tab("trace")
                self._set_top_tab_expanded(True)
            else:
                self._toggle_active_top_tab_expansion()
        elif bid == "tab-diff":
            if self.active_top_tab != "diff":
                self._set_top_tab("diff")
                self._set_top_tab_expanded(True)
            else:
                self._toggle_active_top_tab_expansion()
        elif bid == "btn-clear":
            self.action_clear_input()
        elif bid == "btn-trace":
            self.action_toggle_trace_verbosity()
        elif bid == "btn-provider":
            self._show_provider_menu()
        elif bid == "btn-model":
            self._show_model_menu()
        elif bid == "btn-new-key":
            self._prompt_for_provider_key(self.harness.config.provider)
        elif bid == "btn-theme":
            self.action_cycle_theme()
        elif bid == "btn-quit":
            self.exit()


def run() -> None:
    # Keep Textual mouse support enabled so buttons and Textual's internal text selection work.
    # Conversation copy uses semantic text; activity panes support text selection.
    TitanTui().run(mouse=True)


if __name__ == "__main__":
    run()
