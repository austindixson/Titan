import json
from pathlib import Path

from titan.config import (
    load_harness_config,
    resolve_config_path,
    update_config_key,
    write_default_config,
)


def test_write_default_and_load(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))

    assert write_default_config(resolve_config_path(), force=False) is True
    cfg = load_harness_config()
    assert cfg.provider == "openai-codex"
    assert cfg.model == "gpt-5.4"
    assert cfg.max_iterations == 75
    assert cfg.max_wall_clock_ms == 600000
    assert cfg.max_tool_calls_total == 256
    assert cfg.max_consecutive_empty_turns == 3
    assert cfg.chat_recaps_enabled is False
    assert cfg.learning_enabled is False


def test_legacy_low_budget_config_is_migrated_for_resilient_tui_tasks(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    cfg_path.write_text(
        json.dumps(
            {
                "provider": "openai-codex",
                "model": "gpt-5.4",
                "max_iterations": 16,
                "max_wall_clock_ms": 120000,
                "max_tool_calls_per_iteration": 8,
                "max_tool_calls_total": 64,
            }
        )
    )

    cfg = load_harness_config()
    data = json.loads(cfg_path.read_text())

    assert cfg.max_iterations == 75
    assert cfg.max_wall_clock_ms == 600000
    assert cfg.max_tool_calls_total == 256
    assert data["max_iterations"] == 75
    assert data["max_wall_clock_ms"] == 600000
    assert data["max_tool_calls_total"] == 256


def test_env_overrides_long_task_budgets(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    write_default_config(resolve_config_path(), force=True)

    monkeypatch.setenv("TITAN_MAX_ITERATIONS", "77")
    monkeypatch.setenv("TITAN_MAX_WALL_CLOCK_MS", "901000")
    monkeypatch.setenv("TITAN_MAX_TOOL_CALLS_PER_ITERATION", "13")
    monkeypatch.setenv("TITAN_MAX_TOOL_CALLS_TOTAL", "999")
    monkeypatch.setenv("TITAN_MAX_CONSECUTIVE_EMPTY_TURNS", "5")
    monkeypatch.setenv("TITAN_CHAT_RECAPS_ENABLED", "true")
    monkeypatch.setenv("TITAN_LEARNING_ENABLED", "true")

    cfg = load_harness_config()
    assert cfg.max_iterations == 77
    assert cfg.max_wall_clock_ms == 901000
    assert cfg.max_tool_calls_per_iteration == 13
    assert cfg.max_tool_calls_total == 999
    assert cfg.max_consecutive_empty_turns == 5
    assert cfg.chat_recaps_enabled is True
    assert cfg.learning_enabled is True


def test_config_set_empty_turn_recovery_key(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    write_default_config(resolve_config_path(), force=True)

    update_config_key(resolve_config_path(), "max_consecutive_empty_turns", "6")
    cfg = load_harness_config()
    assert cfg.max_consecutive_empty_turns == 6

    data = json.loads(cfg_path.read_text())
    assert data["max_consecutive_empty_turns"] == 6


def test_config_chat_recaps_and_learning_can_be_enabled_but_default_off(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    write_default_config(resolve_config_path(), force=True)

    data = json.loads(cfg_path.read_text())
    assert data["chat_recaps_enabled"] is False
    assert data["learning_enabled"] is False

    update_config_key(resolve_config_path(), "chat_recaps_enabled", "true")
    update_config_key(resolve_config_path(), "learning_enabled", "true")
    cfg = load_harness_config()
    assert cfg.chat_recaps_enabled is True
    assert cfg.learning_enabled is True


def test_config_set_nested_key(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    write_default_config(resolve_config_path(), force=True)

    update_config_key(resolve_config_path(), "retry.max_retries", "5")
    data = json.loads(cfg_path.read_text())
    assert data["retry"]["max_retries"] == 5

    cfg = load_harness_config()
    assert cfg.retry.max_retries == 5


def test_config_loads_saved_provider_api_keys(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    write_default_config(resolve_config_path(), force=True)

    update_config_key(resolve_config_path(), "api_keys.xai", "xai-test-key")

    cfg = load_harness_config()
    assert cfg.api_keys["xai"] == "xai-test-key"


def test_env_overrides_file(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    write_default_config(resolve_config_path(), force=True)
    update_config_key(resolve_config_path(), "model", "from-file")

    monkeypatch.setenv("TITAN_MODEL", "from-env")
    cfg = load_harness_config()
    assert cfg.model == "from-env"


def test_get_and_unset_key(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("TITAN_CONFIG_PATH", str(cfg_path))
    write_default_config(resolve_config_path(), force=True)

    update_config_key(resolve_config_path(), "retry.max_retries", "9")
    from titan.config import get_config_key, unset_config_key

    assert get_config_key(resolve_config_path(), "retry.max_retries") == 9
    assert unset_config_key(resolve_config_path(), "retry.max_retries") is True
    assert get_config_key(resolve_config_path(), "retry.max_retries") is None
