from __future__ import annotations

import time
import json
import subprocess
from dataclasses import dataclass, field

from rich import box
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual import events
from textual.css.query import NoMatches
from textual.message import Message as TextualMessage
from textual.widgets import Button, Footer, Header, RichLog, Static, TextArea

from .auth import supported_openai_compat_providers
from .config import load_harness_config
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

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        await super()._on_key(event)


class TitanTui(App[None]):
    BINDINGS = [
        ("ctrl+t", "copy_trace", "Copy trace"),
        ("ctrl+y", "copy_chat", "Copy chat"),
        ("ctrl+o", "operator_input", "Focus input"),
        ("ctrl+r", "toggle_trace_verbosity", "Trace mode"),
        ("ctrl+p", "cycle_provider", "Provider"),
        ("ctrl+g", "stop", "Stop"),
        ("ctrl+q", "quit", "Quit"),
    ]

    CSS = """
    Screen { layout: vertical; }
    #top {
        height: 6;
        min-height: 6;
    }
    #trace { height: 1fr; border: solid #3a3a3a; }
    #output {
        height: 1fr;
        min-height: 10;
        border: solid #3a3a3a;
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
        self.trace_verbosity_index = 0
        self.plan_shown_this_run = False
        self.trace_lines: list[str] = []
        self.chat_lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="top"):
            yield SelectableRichLog(id="trace", wrap=True)
        yield SelectableRichLog(id="output", wrap=True)
        with Horizontal(id="controls"):
            yield Button("Stop", id="btn-stop")
            yield Button(f"Provider: {self.cfg.provider}", id="btn-provider")
            yield Button("Operator", id="btn-operator")
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
        yield Static("", id="status_line")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#trace", SelectableRichLog).write_selectable("trace ready")
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
        self.query_one("#btn-operator", Button).label = "Operator"
        self.query_one("#btn-trace", Button).label = f"Trace: {self._chat_trace_mode()}"
        self.query_one("#btn-quit", Button).label = "Quit"

        provider = self.harness.config.provider
        self.query_one("#btn-provider", Button).label = f"Provider: {provider}"

    def _tick(self) -> None:
        if self.ui.pending:
            self.ui.thinking_dots = (self.ui.thinking_dots + 1) % 4
            dots = "." * (self.ui.thinking_dots + 1)
            try:
                self.query_one("#assistant_line", Static).update(f"Titan: thinking{dots}")
            except NoMatches:
                return
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

    def _brief_chat_text(self, text: str, max_chars: int = 1400, max_lines: int = 12) -> str:
        stripped = text.strip()
        lines = stripped.splitlines()
        clipped = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            clipped = True
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[: max_chars - 1].rstrip() + "…"
            clipped = True
        if clipped:
            out += "\n[truncated in chat; full details remain in trace/session logs]"
        return out

    def _write_chat_box(self, speaker: str, text: str, border_style: str) -> None:
        body = self._brief_chat_text(text)
        self.chat_lines.append(f"{speaker}: {body}")
        self.query_one("#output", SelectableRichLog).write_selectable(
            Panel(
                Text(body),
                title=speaker,
                title_align="left",
                border_style=border_style,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            f"{speaker}: {body}",
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
            f"tools={self.ui.tool_calls} "
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
        composer.load_text("")
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
        self.ui.started_at = time.time()
        self.ui.tool_calls = 0
        self.ui.turn = 0
        self.ui.state = "PLAN"
        self.ui.thinking_dots = 0
        self.ui.pending_tool_names.clear()
        self.ui.pending_tool_count = 0
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
            self._trace_emit(trace, "run started", ev.payload)
        elif ev.type == "route_decision":
            state = str(ev.payload.get("state", self.ui.state))
            reason = str(ev.payload.get("reason", ""))
            self.ui.state = state
            self._trace_emit(trace, f"route {state}: {reason}", ev.payload)
        elif ev.type == "iteration_started":
            self.ui.turn = int(ev.payload.get("iteration", self.ui.turn))
            self._trace_emit(trace, f"iteration {self.ui.turn}", ev.payload)
        elif ev.type == "provider_request":
            self._trace_emit(
                trace,
                (
                    f"provider request iteration={ev.payload.get('iteration')} "
                    f"state={ev.payload.get('state')} tools_available={ev.payload.get('tool_count')}"
                ),
                ev.payload,
            )
        elif ev.type == "provider_stream_delta":
            text = self._compact(str(ev.payload.get("text", "")), 120)
            if text:
                self._trace_emit(trace, f"stream {text}", ev.payload)
        elif ev.type == "provider_stream_tool_call":
            name = str(ev.payload.get("name", "")) or "tool"
            self._trace_emit(trace, f"stream tool-call {name}", ev.payload)
        elif ev.type == "empty_turn_recovery":
            self._trace_emit(
                trace,
                (
                    "empty-turn recovery "
                    f"attempt={ev.payload.get('consecutive_empty_turns')} "
                    f"tools={ev.payload.get('tool_calls_total')}"
                ),
                ev.payload,
            )
            if self._chat_trace_mode() in ("normal", "full"):
                self._emit_chat_trace("recovering from empty post-tool turn")
        elif ev.type == "on_state_enter":
            prev_state = self.ui.state
            self.ui.state = str(ev.payload.get("state", self.ui.state))
            self.ui.turn = int(ev.payload.get("turn", self.ui.turn))
            self._trace_emit(trace, f"enter {self.ui.state} turn={self.ui.turn}", ev.payload)
            if self.ui.state != prev_state and self._chat_trace_mode() in ("normal", "full"):
                self._emit_chat_trace(f"state {prev_state} -> {self.ui.state} (turn {self.ui.turn})")
        elif ev.type == "on_transition":
            self._trace_emit(trace, f"transition {ev.payload.get('from_state')} -> {ev.payload.get('to_state')}", ev.payload)
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
                self._emit_chat_trace(f"reasoning {compact}")
        elif ev.type == "tool_batch_started":
            self._trace_emit(trace, f"tool-batch count={ev.payload.get('count')}", ev.payload)
        elif ev.type == "tool_batch_rejected":
            self._trace_emit(
                trace,
                (
                    f"tool-batch rejected count={ev.payload.get('count')} "
                    f"max={ev.payload.get('max_tool_calls_per_iteration')}"
                ),
                ev.payload,
            )
        elif ev.type == "tool_call":
            name = str(ev.payload.get("name", ""))
            args = str(ev.payload.get("arguments", ""))
            compact_args = self._compact(args, 120)
            self.ui.pending_tool_count += 1
            if name:
                self.ui.pending_tool_names.append(name)
            self._trace_emit(trace, f"tool-call {name} args={compact_args}", ev.payload)
        elif ev.type == "tool_call_rejected":
            name = str(ev.payload.get("name", ""))
            args = str(ev.payload.get("arguments", ""))
            compact_args = self._compact(args, 120)
            self._trace_emit(
                trace,
                (
                    f"tool-call rejected {ev.payload.get('index')}/{ev.payload.get('count')} "
                    f"{name} args={compact_args}"
                ),
                ev.payload,
            )
        elif ev.type == "tool_result":
            self.ui.tool_calls += 1
            name = str(ev.payload.get("name", ""))
            is_error = bool(ev.payload.get("is_error"))
            content = str(ev.payload.get("content", "")).strip()
            compact_content = self._compact(content, 140)
            self._trace_emit(
                trace,
                f"tool-result {name} err={is_error} output={compact_content}",
                ev.payload,
            )
            if self._chat_trace_mode() == "full":
                self._emit_chat_trace(
                    f"tool-result {name or 'unknown'} {'ERR' if is_error else 'OK'} {compact_content}"
                )
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

    def on_loop_done_msg(self, msg: LoopDoneMsg) -> None:
        outlog = self.query_one("#output", RichLog)
        out = msg.outcome

        final_text = out.text.strip() if out.text.strip() else f"Stopped: {out.stop.reason.value} ({out.stop.notes or 'no details'})"
        final_text = self._with_final_summary(final_text, out)
        self._flush_tool_summary_to_chat()
        self._write_chat_box("Titan", final_text, "green")
        self.query_one("#assistant_line", Static).update(f"Titan: {self._compact(final_text, 180)}")

        # Keep stop/trace details out of chat panel; trace panel already has execution details.
        completion_line = f"completed stop={out.stop.reason.value} iter={out.stop.iterations} tools={out.stop.tool_calls_total}"
        self.trace_lines.append(completion_line)
        self._write_trace(completion_line)

        self.ui.pending = False
        self.ui.started_at = None
        self._refresh_status()

    def action_stop(self) -> None:
        self._write_trace("stop requested (not yet wired to engine)")

    def action_operator_input(self) -> None:
        self._write_trace("operator input: type in bottom box and press Enter")
        self.query_one("#input", ComposerTextArea).focus()

    def action_toggle_trace_verbosity(self) -> None:
        self.trace_verbosity_index = (self.trace_verbosity_index + 1) % len(self.trace_verbosity_levels)
        mode = self.trace_verbosity_levels[self.trace_verbosity_index]
        self._apply_responsive_layout(self.size.width)
        self._write_trace(f"trace verbosity -> {mode}")
        self.query_one("#btn-trace", Button).label = f"Trace: {mode}"
        self._refresh_status()

    def action_copy_trace(self) -> None:
        self._copy_to_clipboard("\n".join(self.trace_lines), "trace")

    def action_copy_chat(self) -> None:
        self._copy_to_clipboard("\n\n".join(self.chat_lines), "chat")

    def action_cycle_provider(self) -> None:
        if self.ui.pending:
            self._write_trace("provider switch blocked while run is active")
            return
        idx = self.provider_options.index(self.harness.config.provider) if self.harness.config.provider in self.provider_options else 0
        next_provider = self.provider_options[(idx + 1) % len(self.provider_options)]
        self.harness.config.provider = next_provider
        self.cfg.provider = next_provider
        self.harness.provider = build_provider_from_config(self.harness.config)
        self._apply_responsive_layout(self.size.width)
        self._write_trace(f"provider -> {next_provider}")
        self._refresh_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-stop":
            self.action_stop()
        elif bid == "btn-operator":
            self.action_operator_input()
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
