from __future__ import annotations

import time
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from rich import box
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual import events
from textual.css.query import NoMatches
from textual.message import Message as TextualMessage
from textual.widgets import Button, Footer, Header, Input, RichLog, Static, TextArea

from .auth import resolve_provider_credentials, supported_openai_compat_providers
from .config import load_harness_config, resolve_config_path, update_config_key
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


class SelectableRichLog(RichLog):
    ALLOW_SELECT = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.selection_lines: list[str] = []

    def write_selectable(self, renderable, plain: str | None = None) -> None:
        self.selection_lines.append(str(renderable if plain is None else plain))
        self.write(renderable)

    def get_selection(self, selection):
        return selection.extract("\n".join(self.selection_lines)), "\n"

    def selection_updated(self, selection) -> None:
        self.refresh()


class ComposerTextArea(TextArea):
    class Submitted(TextualMessage):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.paste_payloads: dict[str, str] = {}
        self._paste_index = 0
        self.message_history: list[str] = []
        self._history_index: int | None = None

    def _path_from_token(self, token: str) -> str | None:
        raw = token.strip().strip('"').strip("'")
        if not raw:
            return None
        is_file_uri = raw.startswith("file://")
        if is_file_uri:
            parsed = urlparse(raw)
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
        except OSError:
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
        paths = [self._path_from_token(token) for token in tokens]
        if paths and all(paths):
            return "\n".join(paths)
        single = self._path_from_token(stripped)
        return single

    def normalize_paste_for_display(self, text: str) -> str:
        path_text = self._normalize_file_drop(text)
        if path_text:
            return path_text
        lines = text.splitlines()
        if len(lines) > 1:
            self._paste_index += 1
            token = f"[pasted {len(lines)} lines #{self._paste_index}]"
            self.paste_payloads[token] = text
            return token
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
            self._history_index = len(self.message_history) - 1
        else:
            self._history_index = (self._history_index - 1) % len(self.message_history)
        self.load_text(self.message_history[self._history_index])
        return True

    def recall_next_history(self) -> bool:
        if not self.message_history or self._history_index is None:
            return False
        self._history_index = (self._history_index + 1) % len(self.message_history)
        self.load_text(self.message_history[self._history_index])
        return True

    async def _on_paste(self, event: events.Paste) -> None:
        event.stop()
        event.prevent_default()
        self.insert(self.normalize_paste_for_display(event.text))

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.expand_paste_tokens(self.text)))
            return
        if event.key == "up":
            if self.recall_previous_history():
                event.stop()
                event.prevent_default()
                return
        if event.key == "down":
            if self.recall_next_history():
                event.stop()
                event.prevent_default()
                return
        await super()._on_key(event)


class TitanTui(App[None]):
    BINDINGS = [
        ("ctrl+t", "copy_trace", "Copy trace"),
        ("ctrl+d", "toggle_top_tab", "Trace/Diff"),
        ("ctrl+y", "copy_chat", "Copy chat"),
        ("ctrl+f", "operator_input", "Focus input"),
        ("ctrl+r", "toggle_trace_verbosity", "Trace mode"),
        ("ctrl+p", "cycle_provider", "Provider"),
        ("ctrl+g", "stop", "Stop"),
        ("ctrl+c", "handle_ctrl_c", "Cancel/Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    CSS = """
    Screen { layout: vertical; }
    #top {
        height: 6;
        min-height: 6;
    }
    #top.expanded {
        height: 1fr;
        min-height: 10;
    }
    #top_tabs {
        height: 1;
        padding: 0 1;
    }
    #top_tabs Button {
        width: auto;
        height: 1;
        min-width: 7;
        margin-right: 1;
        padding: 0 1;
        border: none;
        background: #202124;
        color: #9aa0a6;
        text-style: bold;
    }
    #top_tabs Button.active-tab {
        background: #263238;
        color: #ffffff;
    }
    #trace { height: 1fr; border: solid #3a3a3a; }
    #diff { height: 1fr; border: solid #3a3a3a; }
    #output {
        height: 1fr;
        min-height: 10;
        border: solid #3a3a3a;
    }
    #output.trace-hidden {
        height: 0;
        min-height: 0;
        border: none;
    }
    #controls {
        height: 3;
        padding: 0 1;
    }
    #controls Button {
        width: auto;
        height: 1;
        min-width: 8;
        margin-right: 1;
        padding: 0 1;
        border: none;
        color: #ffffff;
        background: #2b2b2b;
        text-style: bold;
        content-align: center middle;
    }
    #controls Button.-error {
        background: #b00020;
        color: #ffffff;
    }
    #assistant_line {
        height: 1;
        padding: 0 1;
    }
    #input {
        height: 4;
        min-height: 3;
        border: solid #3a3a3a;
    }
    #api_key_prompt { height: 1; padding: 0 1; color: #fbbc04; }
    #api_key_input { height: 3; border: solid #5f6368; }
    #status_line {
        height: 1;
        padding: 0 1;
        color: #9aa0a6;
    }

    Screen.compact #controls Button {
        min-width: 6;
        margin-right: 0;
        padding: 0 0;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.cfg = load_harness_config()
        self.provider_options = supported_openai_compat_providers()
        if self.cfg.provider not in self.provider_options:
            self.provider_options.insert(0, self.cfg.provider)
        provider = build_provider_from_config(self.cfg)
        self.harness = TitanHarness(provider=provider, tools=default_registry(), config=self.cfg, session_store=SessionStore(".titan/session.jsonl"))
        self.history = [Message(role=Role.SYSTEM, content="You are Titan.")]
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="top"):
            with Horizontal(id="top_tabs"):
                yield Button("Trace", id="tab-trace")
                yield Button("Diff", id="tab-diff")
            yield SelectableRichLog(id="trace", wrap=True)
            yield SelectableRichLog(id="diff", wrap=True)
        yield SelectableRichLog(id="output", wrap=True)
        with Horizontal(id="controls"):
            yield Button("Stop", id="btn-stop")
            yield Button(f"Provider: {self.cfg.provider}", id="btn-provider")
            yield Button("Clear", id="btn-clear")
            yield Button(f"Trace: {self._chat_trace_mode()}", id="btn-trace")
            yield Button("Quit", id="btn-quit", variant="error")
        yield Static("Titan: ready", id="assistant_line")
        yield ComposerTextArea(
            "",
            id="input",
            soft_wrap=True,
            show_line_numbers=False,
            compact=True,
            placeholder="Type task and press Enter",
        )
        yield Static("", id="api_key_prompt")
        yield Input("", id="api_key_input", password=True, placeholder="Paste API key and press Enter")
        yield Static("", id="status_line")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#trace", SelectableRichLog).write_selectable("trace ready")
        self._refresh_diff_tab()
        self._set_top_tab("trace")
        self.query_one("#api_key_prompt", Static).display = False
        self.query_one("#api_key_input", Input).display = False
        self.query_one("#input", ComposerTextArea).focus()
        self._apply_responsive_layout(self.size.width)
        self._refresh_status()
        self.set_interval(0.2, self._tick)

    def on_resize(self) -> None:
        self._apply_responsive_layout(self.size.width)

    def _apply_responsive_layout(self, width: int) -> None:
        compact = width < 120
        if compact:
            self.add_class("compact")
        else:
            self.remove_class("compact")
        self.compact_ui = compact

        self.query_one("#btn-stop", Button).label = "Stop"
        self.query_one("#btn-clear", Button).label = "Clear"
        self.query_one("#btn-trace", Button).label = f"Trace: {self._chat_trace_mode()}"
        self.query_one("#btn-quit", Button).label = "Quit"

        provider = self.harness.config.provider
        self.query_one("#btn-provider", Button).label = f"Provider: {provider}"
        self._refresh_top_tab_labels()

    def _refresh_top_tab_labels(self) -> None:
        try:
            trace_tab = self.query_one("#tab-trace", Button)
            diff_tab = self.query_one("#tab-diff", Button)
        except NoMatches:
            return
        if self.active_top_tab == "trace":
            trace_tab.label = "Trace ▾" if self.top_tab_expanded else "Trace ●"
            diff_tab.label = "Diff"
        else:
            trace_tab.label = "Trace"
            diff_tab.label = "Diff ▾" if self.top_tab_expanded else "Diff ●"
        trace_tab.set_class(self.active_top_tab == "trace", "active-tab")
        diff_tab.set_class(self.active_top_tab == "diff", "active-tab")

    def _set_top_tab_expanded(self, expanded: bool) -> None:
        self.top_tab_expanded = expanded
        try:
            top = self.query_one("#top", Container)
            output = self.query_one("#output", SelectableRichLog)
        except NoMatches:
            return
        top.set_class(self.top_tab_expanded, "expanded")
        output.set_class(self.top_tab_expanded, "trace-hidden")
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
                cwd=".",
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
        try:
            diff = self.query_one("#diff", SelectableRichLog)
        except NoMatches:
            return
        text = self._collect_git_diff()
        self.diff_lines = text.splitlines() or ["No working-tree diff."]
        diff.clear()
        diff.selection_lines.clear()
        for line in self.diff_lines:
            diff.write_selectable(self._style_diff_line(line), line)

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
        self.query_one("#output", SelectableRichLog).write_selectable(
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
        if self.ui.pending:
            self.ui.thinking_dots = (self.ui.thinking_dots + 1) % 4
            dots = "." * (self.ui.thinking_dots + 1)
            try:
                self.query_one("#assistant_line", Static).update(f"Titan: thinking{dots}")
            except NoMatches:
                return
            self._maybe_emit_progress_update()
        self._refresh_status()

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
        if name in {"shell", "terminal"}:
            return ("💻", "bash", self._compact(str(args.get("command", "")), 100))
        return ("🔧", name or "tool", self._compact(str(arguments), 100))

    def _brief_chat_text(self, text: str, max_chars: int = 1400, max_lines: int = 12, grace_chars: int = 320) -> str:
        return text.strip()

    def _chat_renderable(self, speaker: str, body: str, border_style: str):
        if speaker == "You":
            return Text(f"• {body}", style="bold")
        return Panel(
            Text(body),
            title=speaker,
            title_align="left",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _chat_plain_text(self, speaker: str, body: str) -> str:
        return f"• {body}" if speaker == "You" else f"{speaker}: {body}"

    def _write_chat_box(self, speaker: str, text: str, border_style: str) -> None:
        body = self._brief_chat_text(text)
        plain = self._chat_plain_text(speaker, body)
        self.chat_lines.append(plain)
        self.query_one("#output", SelectableRichLog).write_selectable(
            self._chat_renderable(speaker, body, border_style),
            plain,
        )

    def _write_chat_plain(self, text: str) -> None:
        self.chat_lines.append(text)
        self.query_one("#output", SelectableRichLog).write_selectable(text)

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
        elapsed_ms = int((time.time() - self.ui.started_at) * 1000) if self.ui.pending and self.ui.started_at else 0
        timer = f"{elapsed_ms/1000:.1f}s" if self.ui.pending else "0.0s"
        status = (
            f"state={self.ui.state} "
            f"turn={self.ui.turn}/{self.harness.config.max_iterations} "
            f"tools_used_this_turn={self.ui.turn_tool_calls} "
            f"provider={self.harness.config.provider} "
            f"thinking={'yes' if self.ui.pending else 'no'}({timer}) "
            f"trace={self.trace_verbosity_levels[self.trace_verbosity_index]}"
        )
        try:
            self.query_one("#status_line", Static).update(status)
        except NoMatches:
            return

    async def on_composer_text_area_submitted(self, event: ComposerTextArea.Submitted) -> None:
        composer = self.query_one("#input", ComposerTextArea)
        task = event.value.strip()
        if task and not self.ui.pending:
            composer.record_history(task)
        composer.load_text("")
        composer.paste_payloads.clear()
        await self._submit_task(task)

    async def _submit_task(self, task: str) -> None:
        if not task or self.ui.pending:
            return

        if task.startswith("/"):
            res = execute_slash_command(task)
            if res.handled:
                if res.message == "trace-toggle":
                    self.action_toggle_trace_verbosity()
                    self._write_chat_box("You", task, "cyan")
                    self._write_chat_box("Titan", "toggled trace verbosity", "green")
                else:
                    if task.startswith("/config"):
                        previous_provider = self.harness.config.provider
                        self.cfg = load_harness_config()
                        self.harness.config = self.cfg
                        if self.cfg.provider != previous_provider:
                            self.harness.provider = build_provider_from_config(self.cfg)
                        self._apply_responsive_layout(self.size.width)
                    self._write_chat_box("You", task, "cyan")
                    prefix = "Titan error" if res.is_error else "Titan"
                    self._write_chat_box(prefix, res.message, "red" if res.is_error else "green")
                return

        self._write_chat_box("You", task, "cyan")
        self.query_one("#assistant_line", Static).update("Titan: thinking.")

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
        self._refresh_status()

        def _run_blocking() -> None:
            def cb(ev: AgentEvent) -> None:
                self.post_message(LoopEventMsg(ev))

            out = self.harness.run_with_callback(task, self.history, on_event=cb)
            self.post_message(LoopDoneMsg(out))

        self.run_worker(_run_blocking, thread=True)

    def on_loop_event_msg(self, msg: LoopEventMsg) -> None:
        ev = msg.event
        trace = self.query_one("#trace", RichLog)

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
            self._record_progress_event(
                f"asking {ev.payload.get('provider', self.harness.config.provider)} {ev.payload.get('model', self.harness.config.model)} for next step"
            )
            pass
        elif ev.type == "provider_stream_delta":
            pass
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
                if state == "PLAN" and has_tool_calls and not self.plan_shown_this_run:
                    self._write_chat_box("Titan plan", text, "yellow")
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
            self._trace_emit(trace, f"┊ {status_icon} {name or 'tool'} {('ERR' if is_error else 'OK')} {compact_content}", ev.payload)
            if self._chat_trace_mode() == "full":
                self._emit_chat_trace(
                    f"tool-result {name or 'unknown'} {'ERR' if is_error else 'OK'} {compact_content}"
                )
            if self.active_top_tab == "diff":
                self._refresh_diff_tab()
            self._maybe_emit_progress_update()
        elif ev.type == "on_skill_created":
            self._trace_emit(trace, f"skill-created {ev.payload.get('path')}", ev.payload)

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
        outlog = self.query_one("#output", RichLog)
        out = msg.outcome

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
        self.query_one("#assistant_line", Static).update(f"Titan: {self._compact(final_text, 180)}")

        # Keep stop/trace details minimal in trace view; tool/path/reasoning lines are primary.
        if self.active_top_tab == "diff":
            self._refresh_diff_tab()

        self.ui.pending = False
        self.ui.started_at = None
        self._refresh_status()

    def action_stop(self) -> None:
        self._write_trace("stop requested (not yet wired to engine)")

    def action_handle_ctrl_c(self) -> None:
        if self.ui.pending:
            self.action_close_titan_instance()
            self.ctrl_c_quit_armed = True
            self._write_trace("Ctrl+C: cancelled current task; press Ctrl+C again to quit Titan")
            return
        if self.ctrl_c_quit_armed:
            self.exit()
            return
        self.ctrl_c_quit_armed = True
        self._write_trace("Ctrl+C: no active task; press Ctrl+C again to quit Titan")

    def action_close_titan_instance(self) -> None:
        self.ui.pending = False
        self.ui.started_at = None
        self.ui.pending_tool_names.clear()
        self.ui.pending_tool_count = 0
        self.ui.turn_tool_calls = 0
        self.query_one("#assistant_line", Static).update("Titan: instance closed")
        self._write_trace("current Titan instance closed; app remains open")
        self._refresh_status()

    def action_operator_input(self) -> None:
        self._write_trace("operator input: type in bottom box and press Enter")
        self.query_one("#input", ComposerTextArea).focus()

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
        self.query_one("#btn-trace", Button).label = f"Trace: {mode}"
        self._refresh_status()

    def action_toggle_top_tab(self) -> None:
        self._set_top_tab("diff" if self.active_top_tab == "trace" else "trace")
        self._refresh_status()

    def action_copy_trace(self) -> None:
        self._copy_to_clipboard("\n".join(self.trace_lines), "trace")

    def action_copy_chat(self) -> None:
        self._copy_to_clipboard("\n\n".join(self.chat_lines), "chat")

    def _provider_has_key(self, provider: str) -> bool:
        if provider == "mock":
            return True
        if self.harness.config.api_keys.get(provider):
            return True
        try:
            return resolve_provider_credentials(provider, base_url=self.harness.config.api_base or None) is not None
        except Exception:
            return False

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

    def _hide_provider_key_prompt(self) -> None:
        self.pending_api_key_provider = None
        self.query_one("#api_key_prompt", Static).display = False
        key_input = self.query_one("#api_key_input", Input)
        key_input.value = ""
        key_input.display = False

    def _save_provider_key(self, provider: str, key: str) -> None:
        self.harness.config.api_keys[provider] = key
        self.cfg.api_keys[provider] = key
        update_config_key(resolve_config_path(), f"api_keys.{provider}", key)
        self.harness.provider = build_provider_from_config(self.harness.config)

    def action_cycle_provider(self) -> None:
        if self.ui.pending:
            self._write_trace("provider switch blocked while run is active")
            return
        idx = self.provider_options.index(self.harness.config.provider) if self.harness.config.provider in self.provider_options else 0
        next_provider = self.provider_options[(idx + 1) % len(self.provider_options)]
        self.harness.config.provider = next_provider
        self.cfg.provider = next_provider
        if self._provider_has_key(next_provider):
            self._hide_provider_key_prompt()
            self.harness.provider = build_provider_from_config(self.harness.config)
        else:
            self._prompt_for_provider_key(next_provider)
        self._apply_responsive_layout(self.size.width)
        self._write_trace(f"provider -> {next_provider}")
        self._refresh_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "api_key_input" or not self.pending_api_key_provider:
            return
        key = event.value.strip()
        provider = self.pending_api_key_provider
        if not key:
            self._write_trace(f"provider {provider} key not saved: empty input")
            return
        self._save_provider_key(provider, key)
        self._hide_provider_key_prompt()
        self._write_trace(f"saved API key for {provider}")
        self.query_one("#input", ComposerTextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-stop":
            self.action_stop()
        elif bid == "tab-trace":
            if self.active_top_tab != "trace":
                self._set_top_tab("trace")
            else:
                self._toggle_active_top_tab_expansion()
        elif bid == "tab-diff":
            if self.active_top_tab != "diff":
                self._set_top_tab("diff")
            else:
                self._toggle_active_top_tab_expansion()
        elif bid == "btn-clear":
            self.action_clear_input()
        elif bid == "btn-trace":
            self.action_toggle_trace_verbosity()
        elif bid == "btn-provider":
            self.action_cycle_provider()
        elif bid == "btn-quit":
            self.exit()


def run() -> None:
    # Keep Textual mouse support enabled so buttons and Textual's internal text selection work.
    # Chat/trace use SelectableRichLog so selected copies contain semantic text, not padded cells.
    TitanTui().run(mouse=True)


if __name__ == "__main__":
    run()
