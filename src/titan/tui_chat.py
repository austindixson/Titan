from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message as TextualMessage
from textual.widgets import Footer, Header, Input, RichLog, Static

from .auth import resolve_openai_credentials
from .config import load_harness_config
from .loop import AgentEvent, AgentLoop
from .provider import OpenAICompatProvider
from .session import SessionStore
from .tools import default_registry
from .types import Message, Role, RunOutcome


@dataclass
class UiState:
    pending: bool = False
    run_started_at: Optional[float] = None
    iteration: int = 0
    tool_calls: int = 0
    last_stop: str = "ready"
    last_assistant_text: str = ""


class LoopEventMsg(TextualMessage):
    def __init__(self, event: AgentEvent) -> None:
        self.event = event
        super().__init__()


class LoopDoneMsg(TextualMessage):
    def __init__(self, outcome: RunOutcome) -> None:
        self.outcome = outcome
        super().__init__()


class FerroclawChatApp(App[None]):
    CSS = """
    Screen { layout: vertical; background: #0b0f14; color: #d7e3f4; }
    #root { layout: vertical; height: 1fr; }
    #status { height: 1; padding: 0 1; background: #101827; color: #97a9bf; }
    #main { height: 1fr; }
    #chat { width: 2fr; border: solid #1f2937; background: #0b0f14; padding: 0 1; }
    #right { width: 1fr; layout: vertical; }
    #trace { height: 1fr; border: solid #374151; background: #0b0f14; padding: 0 1; }
    #tools { height: 1fr; border: solid #374151; background: #0b0f14; padding: 0 1; }
    #composer { height: 3; border: solid #374151; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+end", "scroll_bottom", "Bottom"),
        ("ctrl+home", "scroll_top", "Top"),
        ("pageup", "scroll_page_up", "PgUp"),
        ("pagedown", "scroll_page_down", "PgDn"),
        ("shift+up", "scroll_up", "Up"),
        ("shift+down", "scroll_down", "Down"),
    ]

    def __init__(self) -> None:
        super().__init__()
        cfg = load_harness_config()
        creds = resolve_openai_credentials(oauth_env=cfg.oauth_token_env, api_key_env=cfg.api_key_env)
        provider = OpenAICompatProvider(
            api_base=(creds.base_url if creds and creds.base_url else cfg.api_base),
            api_key=(creds.token if creds else cfg.api_key()),
        )
        self.loop = AgentLoop(provider=provider, tools=default_registry(), config=cfg, session=SessionStore(".titan/session.jsonl"))
        self.history = [Message(role=Role.SYSTEM, content="You are Titan. Use tools when needed and finish tasks.")]
        self.auth_source = creds.source if creds else "none"
        self.state = UiState()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="root"):
            yield Static("", id="status")
            with Horizontal(id="main"):
                yield RichLog(id="chat", wrap=True, highlight=False, markup=False, auto_scroll=False)
                with Container(id="right"):
                    yield RichLog(id="trace", wrap=True, highlight=False, markup=False, auto_scroll=False)
                    yield RichLog(id="tools", wrap=True, highlight=False, markup=False, auto_scroll=False)
            yield Input(placeholder="Type message and press Enter…", id="composer")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(RichLog).write("Titan ready.")
        self.query_one("#trace", RichLog).write("trace: ready")
        self.query_one("#tools", RichLog).write("tools: ready")
        self.query_one(Input).focus()
        self.set_interval(0.2, self._tick_status)
        self._tick_status()

    def _verb(self) -> str:
        if not self.state.pending or self.state.run_started_at is None:
            return "ready"
        if self.state.tool_calls > 0:
            tool_verbs = ["reading", "writing", "executing", "inspecting", "patching", "verifying"]
            return tool_verbs[(self.state.tool_calls - 1) % len(tool_verbs)]
        elapsed = int((time.time() - self.state.run_started_at) * 1000)
        phase = (elapsed // 1200) % 4
        return ["thinking", "reasoning", "tooling", "finalizing"][phase]

    def _tick_status(self) -> None:
        elapsed_ms = 0
        if self.state.pending and self.state.run_started_at is not None:
            elapsed_ms = int((time.time() - self.state.run_started_at) * 1000)
        s = (
            f"auth={self.auth_source} | {self._verb()} | iter={self.state.iteration} "
            f"tools={self.state.tool_calls} elapsed_ms={elapsed_ms} | "
            "PgUp/PgDn Shift+↑/↓ Ctrl+Home/End"
        )
        self.query_one("#status", Static).update(s)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.state.pending:
            return
        inp = self.query_one(Input)
        inp.value = ""

        self.query_one(RichLog).write(f"You: {text}")
        self.query_one(RichLog).write("Ferro: thinking…")
        self.action_scroll_bottom()

        self.state.pending = True
        self.state.run_started_at = time.time()
        self.state.iteration = 0
        self.state.tool_calls = 0
        self.state.last_assistant_text = ""

        def _run_blocking() -> None:
            def cb(ev: AgentEvent) -> None:
                self.post_message(LoopEventMsg(ev))

            out = self.loop.run_with_callback(text, self.history, on_event=cb)
            self.post_message(LoopDoneMsg(out))

        self.run_worker(_run_blocking, thread=True)

    def on_loop_event_msg(self, msg: LoopEventMsg) -> None:
        ev = msg.event
        chat = self.query_one(RichLog)
        trace = self.query_one("#trace", RichLog)
        tools = self.query_one("#tools", RichLog)

        if ev.type == "iteration_started":
            self.state.iteration = int(ev.payload.get("iteration", self.state.iteration))
            trace.write(f"iter {self.state.iteration} started")
        elif ev.type == "assistant_message":
            t = (ev.payload.get("text") or "").strip()
            if t:
                self.state.last_assistant_text = t
                trace.write("assistant delta")
        elif ev.type == "tool_call":
            self.state.tool_calls = int(ev.payload.get("count", self.state.tool_calls + 1))
            name = ev.payload.get("name", "tool")
            chat.write(f"Tool→ {name}")
            tools.write(f"→ {name} args={ev.payload.get('arguments', {})}")
        elif ev.type == "tool_result":
            name = ev.payload.get("name", "tool")
            err = bool(ev.payload.get("is_error", False))
            body = (ev.payload.get("content") or "").strip().replace("\n", " ")
            if len(body) > 180:
                body = body[:180] + "…"
            prefix = "Tool✖" if err else "Tool✓"
            chat.write(f"{prefix} {name}: {body}")
            tools.write(f"{prefix} {name}: {body}")
        elif ev.type == "provider_error":
            chat.write(f"Provider error: {ev.payload.get('error', '')}")
            trace.write(f"provider_error {ev.payload.get('error', '')}")

        self.action_scroll_bottom()

    def on_loop_done_msg(self, msg: LoopDoneMsg) -> None:
        out = msg.outcome
        chat = self.query_one(RichLog)
        final_text = out.text.strip()
        if final_text and final_text != self.state.last_assistant_text:
            chat.write(f"Ferro: {final_text}")
        chat.write(
            f"stop={out.stop.reason} iter={out.stop.iterations} tools={out.stop.tool_calls_total} elapsed_ms={out.stop.elapsed_ms}"
        )
        self.state.pending = False
        self.state.last_stop = out.stop.reason.value
        self.action_scroll_bottom()

    def on_key(self, event) -> None:
        inp = self.query_one(Input)
        if event.key == "up" and not inp.value.strip():
            self.action_scroll_up()
            event.prevent_default()
        elif event.key == "down" and not inp.value.strip():
            self.action_scroll_down()
            event.prevent_default()

    def action_clear_chat(self) -> None:
        chat = self.query_one(RichLog)
        chat.clear()
        chat.write("Chat cleared.")

    def action_scroll_bottom(self) -> None:
        self.query_one(RichLog).action_scroll_end()

    def action_scroll_top(self) -> None:
        self.query_one(RichLog).action_scroll_home()

    def action_scroll_up(self) -> None:
        self.query_one(RichLog).action_scroll_up()

    def action_scroll_down(self) -> None:
        self.query_one(RichLog).action_scroll_down()

    def action_scroll_page_up(self) -> None:
        for _ in range(10):
            self.query_one(RichLog).action_scroll_up()

    def action_scroll_page_down(self) -> None:
        for _ in range(10):
            self.query_one(RichLog).action_scroll_down()


def run() -> None:
    FerroclawChatApp().run()
