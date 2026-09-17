import json
from pathlib import Path

from titan.auth import provider_default_base_url, resolve_openai_credentials, supported_openai_compat_providers


def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN", "tok_env")
    creds = resolve_openai_credentials()
    assert creds is not None
    assert creds.token == "tok_env"
    assert creds.source == "env:OPENAI_OAUTH_TOKEN"


def test_resolve_from_hermes_auth_file(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENAI_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "tok_hermes"}
            }
        },
        "credential_pool": {
            "openai-codex": [{"base_url": "https://chatgpt.com/backend-api/codex"}]
        }
    }))
    monkeypatch.setenv("HERMES_AUTH_PATH", str(p))

    creds = resolve_openai_credentials()
    assert creds is not None
    assert creds.token == "tok_hermes"
    assert creds.base_url == "https://chatgpt.com/backend-api/codex"
    assert creds.source.startswith("hermes:")


def test_litert_is_a_supported_openai_compat_provider():
    assert "litert" in supported_openai_compat_providers()
    assert provider_default_base_url("litert") == "http://ghost32:9379/v1"
