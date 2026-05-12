from __future__ import annotations
from rich.console import Console
from rich.text import Text
from .config import load_harness_config
from .loop import AgentLoop
from .mock_provider import MockProvider, make_tool_then_final_script
from .tools import default_registry
from .types import Message, Role


def demo_tui() -> None:
    c = Console()
    cfg = load_harness_config()
    cfg.permission_mode = "allow"
    loop = AgentLoop(provider=MockProvider(script=make_tool_then_final_script()), tools=default_registry(), config=cfg)
    history = [Message(role=Role.SYSTEM, content="You are Titan. Use tools and finish.")]

    c.print("titan minimal tui demo (mock provider)")
    while True:
        try:
            user = input("[You] ")
        except (KeyboardInterrupt, EOFError):
            c.print("bye")
            break
        if not user.strip():
            continue
        c.print(Text("● thinking", style="cyan"))
        out = loop.run(user, history)
        c.print(f"[Ferro] {out.text}")
        c.print(f"stop={out.stop.reason} iter={out.stop.iterations} tools={out.stop.tool_calls_total}")
