from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path.home() / ".titan" / "config.json"
RESILIENT_DEFAULT_MAX_ITERATIONS = 75
RESILIENT_DEFAULT_MAX_WALL_CLOCK_MS = 600000
RESILIENT_DEFAULT_MAX_TOOL_CALLS_TOTAL = 256
LEGACY_LOW_MAX_ITERATIONS = 16
LEGACY_LOW_MAX_WALL_CLOCK_MS = 120000
LEGACY_LOW_MAX_TOOL_CALLS_TOTAL = 64


@dataclass
class RetryConfig:
    max_retries: int = 2
    base_delay_ms: int = 200
    max_delay_ms: int = 1500


@dataclass
class HarnessConfig:
    model: str = "gpt-5.4"
    provider: str = "openai-codex"
    api_base: str = ""
    api_keys: dict[str, str] = field(default_factory=dict)
    auth_mode: str = "oauth"
    oauth_token_env: str = "OPENAI_OAUTH_TOKEN"
    api_key_env: str = "OPENAI_API_KEY"
    max_iterations: int = RESILIENT_DEFAULT_MAX_ITERATIONS
    max_wall_clock_ms: int = RESILIENT_DEFAULT_MAX_WALL_CLOCK_MS
    max_tool_calls_per_iteration: int = 8
    max_tool_calls_total: int = RESILIENT_DEFAULT_MAX_TOOL_CALLS_TOTAL
    max_consecutive_empty_turns: int = 3
    chat_recaps_enabled: bool = False
    learning_enabled: bool = False
    permission_mode: str = "allow"
    retry: RetryConfig = field(default_factory=RetryConfig)

    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")


def _deep_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _parse_int(value: str, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _parse_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return fallback


def _load_file_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _migrate_legacy_low_budgets(path: Path, data: dict[str, Any]) -> None:
    """Lift old short-task defaults so Titan keeps pushing through coding work.

    Early local configs used 16 iterations / 2 minutes / 64 total tools. That
    budget is too small for medium TUI coding tasks that inspect, edit, and
    validate a project. Only migrate that exact low-budget profile (or lower)
    so deliberate custom budgets are otherwise preserved.
    """
    if not data:
        return
    try:
        is_legacy_low_profile = (
            int(data.get("max_iterations", RESILIENT_DEFAULT_MAX_ITERATIONS)) <= LEGACY_LOW_MAX_ITERATIONS
            and int(data.get("max_wall_clock_ms", RESILIENT_DEFAULT_MAX_WALL_CLOCK_MS)) <= LEGACY_LOW_MAX_WALL_CLOCK_MS
            and int(data.get("max_tool_calls_total", RESILIENT_DEFAULT_MAX_TOOL_CALLS_TOTAL)) <= LEGACY_LOW_MAX_TOOL_CALLS_TOTAL
        )
    except Exception:
        return
    if not is_legacy_low_profile:
        return
    data["max_iterations"] = RESILIENT_DEFAULT_MAX_ITERATIONS
    data["max_wall_clock_ms"] = RESILIENT_DEFAULT_MAX_WALL_CLOCK_MS
    data["max_tool_calls_total"] = RESILIENT_DEFAULT_MAX_TOOL_CALLS_TOTAL
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
    except Exception:
        pass


def resolve_config_path(path_override: str | None = None) -> Path:
    if path_override:
        return Path(path_override).expanduser().resolve()
    env = os.getenv("TITAN_CONFIG_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def load_harness_config(
    provider_override: str | None = None,
    model_override: str | None = None,
    config_path_override: str | None = None,
) -> HarnessConfig:
    cfg = HarnessConfig()
    path = resolve_config_path(config_path_override)
    data = _load_file_config(path)
    _migrate_legacy_low_budgets(path, data)

    cfg.provider = str(_deep_get(data, "provider", cfg.provider))
    cfg.model = str(_deep_get(data, "model", cfg.model))
    cfg.api_base = str(_deep_get(data, "api_base", cfg.api_base))
    api_keys = _deep_get(data, "api_keys", cfg.api_keys)
    cfg.api_keys = dict(api_keys) if isinstance(api_keys, dict) else {}
    cfg.permission_mode = str(_deep_get(data, "permission_mode", cfg.permission_mode))
    cfg.max_iterations = int(_deep_get(data, "max_iterations", cfg.max_iterations))
    cfg.max_wall_clock_ms = int(_deep_get(data, "max_wall_clock_ms", cfg.max_wall_clock_ms))
    cfg.max_tool_calls_per_iteration = int(
        _deep_get(data, "max_tool_calls_per_iteration", cfg.max_tool_calls_per_iteration)
    )
    cfg.max_tool_calls_total = int(_deep_get(data, "max_tool_calls_total", cfg.max_tool_calls_total))
    cfg.max_consecutive_empty_turns = int(
        _deep_get(data, "max_consecutive_empty_turns", cfg.max_consecutive_empty_turns)
    )
    cfg.chat_recaps_enabled = _parse_bool(_deep_get(data, "chat_recaps_enabled", cfg.chat_recaps_enabled), cfg.chat_recaps_enabled)
    cfg.learning_enabled = _parse_bool(_deep_get(data, "learning_enabled", cfg.learning_enabled), cfg.learning_enabled)
    cfg.retry.max_retries = int(_deep_get(data, "retry.max_retries", cfg.retry.max_retries))
    cfg.retry.base_delay_ms = int(_deep_get(data, "retry.base_delay_ms", cfg.retry.base_delay_ms))
    cfg.retry.max_delay_ms = int(_deep_get(data, "retry.max_delay_ms", cfg.retry.max_delay_ms))

    # env overrides
    cfg.provider = os.getenv("TITAN_PROVIDER", os.getenv("FERRO_PROVIDER", cfg.provider))
    cfg.model = os.getenv("TITAN_MODEL", os.getenv("FERRO_MODEL", cfg.model))
    cfg.api_base = os.getenv("TITAN_API_BASE", os.getenv("FERRO_API_BASE", cfg.api_base))
    cfg.permission_mode = os.getenv("TITAN_PERMISSION_MODE", cfg.permission_mode)
    cfg.max_iterations = _parse_int(os.getenv("TITAN_MAX_ITERATIONS", str(cfg.max_iterations)), cfg.max_iterations)
    cfg.max_wall_clock_ms = _parse_int(os.getenv("TITAN_MAX_WALL_CLOCK_MS", str(cfg.max_wall_clock_ms)), cfg.max_wall_clock_ms)
    cfg.max_tool_calls_per_iteration = _parse_int(
        os.getenv("TITAN_MAX_TOOL_CALLS_PER_ITERATION", str(cfg.max_tool_calls_per_iteration)),
        cfg.max_tool_calls_per_iteration,
    )
    cfg.max_tool_calls_total = _parse_int(
        os.getenv("TITAN_MAX_TOOL_CALLS_TOTAL", str(cfg.max_tool_calls_total)),
        cfg.max_tool_calls_total,
    )
    cfg.max_consecutive_empty_turns = _parse_int(
        os.getenv("TITAN_MAX_CONSECUTIVE_EMPTY_TURNS", str(cfg.max_consecutive_empty_turns)),
        cfg.max_consecutive_empty_turns,
    )
    cfg.chat_recaps_enabled = _parse_bool(
        os.getenv("TITAN_CHAT_RECAPS_ENABLED", str(cfg.chat_recaps_enabled)),
        cfg.chat_recaps_enabled,
    )
    cfg.learning_enabled = _parse_bool(
        os.getenv("TITAN_LEARNING_ENABLED", str(cfg.learning_enabled)),
        cfg.learning_enabled,
    )

    if provider_override:
        cfg.provider = provider_override
    if model_override:
        cfg.model = model_override

    return cfg


def write_default_config(path: Path, force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "provider": "openai-codex",
        "model": "gpt-5.4",
        "api_base": "",
        "api_keys": {},
        "permission_mode": "allow",
        "max_iterations": RESILIENT_DEFAULT_MAX_ITERATIONS,
        "max_wall_clock_ms": RESILIENT_DEFAULT_MAX_WALL_CLOCK_MS,
        "max_tool_calls_per_iteration": 8,
        "max_tool_calls_total": RESILIENT_DEFAULT_MAX_TOOL_CALLS_TOTAL,
        "max_consecutive_empty_turns": 3,
        "chat_recaps_enabled": False,
        "learning_enabled": False,
        "retry": {
            "max_retries": 2,
            "base_delay_ms": 200,
            "max_delay_ms": 1500,
        },
    }
    path.write_text(json.dumps(content, indent=2) + "\n")
    return True


def _parse_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def update_config_key(path: Path, key: str, value: str) -> None:
    data = _load_file_config(path)
    if not data:
        data = {}

    parts = key.split(".")
    cur: dict[str, Any] = data
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt

    cur[parts[-1]] = _parse_value(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def get_config_key(path: Path, key: str) -> Any:
    data = _load_file_config(path)
    return _deep_get(data, key, None)


def unset_config_key(path: Path, key: str) -> bool:
    data = _load_file_config(path)
    if not data:
        return False

    parts = key.split(".")
    cur: Any = data
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]

    last = parts[-1]
    if not isinstance(cur, dict) or last not in cur:
        return False

    del cur[last]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return True
