from __future__ import annotations
from rich.console import Console
from .config import load_harness_config
from .provider import build_provider_from_config
from .tools import default_registry
from .loop import AgentLoop
from .session import SessionStore
from .types import Message, Role


def main() -> None:
    c = Console()
    cfg = load_harness_config()
    provider = build_provider_from_config(cfg)
    loop = AgentLoop(provider=provider, tools=default_registry(), config=cfg, session=SessionStore(".titan/session.jsonl"))

    history = [Message(role=Role.SYSTEM, content="You are Titan. Use tools when needed and finish tasks.")]
    auth_src = f"provider={cfg.provider}"
    c.print(f"Titan. Ctrl+C to exit. auth={auth_src}")
    while True:
        try:
            user = input("\n[You] ")
        except (KeyboardInterrupt, EOFError):
            c.print("\nBye")
            break
        if not user.strip():
            continue
        out = loop.run(user, history)
        c.print(f"[Ferro] {out.text}")
        c.print(f"stop={out.stop.reason} iter={out.stop.iterations} tools={out.stop.tool_calls_total} elapsed_ms={out.stop.elapsed_ms}")
