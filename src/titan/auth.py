from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OpenAICredentials:
    token: str
    base_url: Optional[str] = None
    source: str = ""


_OPENAI_COMPAT_PROVIDER_SPECS: dict[str, dict[str, str]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
    },
    "openai-codex": {
        "api_key_env": "OPENAI_OAUTH_TOKEN",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://chatgpt.com/backend-api/codex",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
    "xai": {
        "api_key_env": "XAI_API_KEY",
        "base_url_env": "XAI_BASE_URL",
        "default_base_url": "https://api.x.ai/v1",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url_env": "GROQ_BASE_URL",
        "default_base_url": "https://api.groq.com/openai/v1",
    },
    "cerebras": {
        "api_key_env": "CEREBRAS_API_KEY",
        "base_url_env": "CEREBRAS_BASE_URL",
        "default_base_url": "https://api.cerebras.ai/v1",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
    },
    "mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "base_url_env": "MISTRAL_BASE_URL",
        "default_base_url": "https://api.mistral.ai/v1",
    },
    "zai": {
        "api_key_env": "ZAI_API_KEY",
        "base_url_env": "ZAI_BASE_URL",
        "default_base_url": "https://api.z.ai/api/paas/v4",
    },
    "moonshotai": {
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url_env": "MOONSHOT_BASE_URL",
        "default_base_url": "https://api.moonshot.ai/v1",
    },
}


def _read_hermes_auth(path: Path) -> OpenAICredentials | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    providers = data.get("providers") or {}
    cred_pool = data.get("credential_pool") or {}

    token = (
        (((providers.get("openai-codex") or {}).get("tokens") or {}).get("access_token"))
        or (((cred_pool.get("openai-codex") or [{}])[0]).get("access_token"))
        or ""
    )
    base_url = (((cred_pool.get("openai-codex") or [{}])[0]).get("base_url"))
    if token:
        return OpenAICredentials(token=token, base_url=base_url, source=f"hermes:{path}")
    return None


def _read_pi_auth(path: Path, provider: str, base_url_env: str) -> OpenAICredentials | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    row = data.get(provider)
    if not isinstance(row, dict):
        return None
    key = str(row.get("key", "")).strip()
    if not key:
        return None
    base_url = os.getenv(base_url_env, "").strip() or None
    return OpenAICredentials(token=key, base_url=base_url, source=f"pi:{path}#{provider}")


def supported_openai_compat_providers() -> list[str]:
    return sorted(_OPENAI_COMPAT_PROVIDER_SPECS.keys())


def resolve_provider_credentials(provider: str, api_key_env: str | None = None, base_url: str | None = None) -> OpenAICredentials | None:
    key = provider.strip().lower()
    spec = _OPENAI_COMPAT_PROVIDER_SPECS.get(key)
    if spec is None:
        raise ValueError(f"unsupported provider '{provider}'. supported: {', '.join(supported_openai_compat_providers())}")

    env_key_name = api_key_env or spec["api_key_env"]
    env_base_name = spec["base_url_env"]

    token = os.getenv(env_key_name, "").strip()
    if token:
        resolved_base = (base_url or os.getenv(env_base_name, "").strip() or spec["default_base_url"]).strip()
        return OpenAICredentials(token=token, base_url=resolved_base, source=f"env:{env_key_name}")

    pi_auth_path = Path(os.getenv("PI_AUTH_PATH", str(Path.home() / ".pi" / "agent" / "auth.json")))
    pi_creds = _read_pi_auth(pi_auth_path, key, env_base_name)
    if pi_creds:
        pi_creds.base_url = (base_url or pi_creds.base_url or spec["default_base_url"]).strip()
        return pi_creds

    if key in ("openai", "openai-codex"):
        hermes_auth_path = Path(os.getenv("HERMES_AUTH_PATH", str(Path.home() / ".hermes" / "auth.json")))
        hermes_creds = _read_hermes_auth(hermes_auth_path)
        if hermes_creds:
            hermes_creds.base_url = (base_url or hermes_creds.base_url or spec["default_base_url"]).strip()
            return hermes_creds

    return None


def resolve_openai_credentials(
    oauth_env: str = "OPENAI_OAUTH_TOKEN",
    api_key_env: str = "OPENAI_API_KEY",
) -> OpenAICredentials | None:
    creds = resolve_provider_credentials("openai-codex", api_key_env=oauth_env)
    if creds:
        return creds
    return resolve_provider_credentials("openai", api_key_env=api_key_env)
