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
    "grok": {
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
        "default_base_url": "https://api.z.ai/api/coding/paas/v4",
    },
    "moonshotai": {
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url_env": "MOONSHOT_BASE_URL",
        "default_base_url": "https://api.moonshot.ai/v1",
    },
    "litert": {
        "api_key_env": "LITERT_API_KEY",
        "base_url_env": "LITERT_BASE_URL",
        "default_base_url": "http://ghost32:9379/v1",
    },
    "homebase": {
        "api_key_env": "HOMEBASE_API_KEY",
        "base_url_env": "HOMEBASE_BASE_URL",
        "default_base_url": "http://ghost128s-macbook-pro:9081/v1",
        "description": "Local home base model via Tailscale (auto-detects ghost128s-macbook-pro.ts.net or localhost)",
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


def provider_default_base_url(provider: str) -> str:
    key = provider.strip().lower()
    spec = _OPENAI_COMPAT_PROVIDER_SPECS.get(key)
    if spec is None:
        raise ValueError(f"unsupported provider '{provider}'. supported: {', '.join(supported_openai_compat_providers())}")
    return spec["default_base_url"]


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


_GROK_FAMILY = {"grok", "xai", "xai-oauth"}
_CODEX_FAMILY = {"openai-codex", "codex"}
_TUI_HIDDEN_ALIASES = {"xai", "xai-oauth", "codex"}
_PREFERRED_TUI_PROVIDERS = ("grok", "openai-codex", "homebase", "litert")
_PROVIDER_ALIASES = {
    "codex": "openai-codex",
    "astra": "openai-codex",
    "gpt-6-astra": "openai-codex",
    "xai-oauth": "grok",
}

PROVIDER_DEFAULT_MODELS = {
    "grok": "grok-4.6",
    "xai": "grok-4.6",
    "openai-codex": "gpt-5.6-luna",
    "openai": "gpt-5.4",
    "litert": "gemma4-12b",
    "homebase": "homebase-brain",
    "zai": "glm-5.1",
}

PROVIDER_MODEL_OPTIONS = {
    "grok": ["grok-4.6", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-3-mini"],
    "xai": ["grok-4.6", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-3-mini"],
    "openai-codex": ["gpt-6-astra", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6", "gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini"],
    "openai": ["gpt-6-astra", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6", "gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini"],
    "litert": ["gemma4-12b"],
    "homebase": ["homebase-brain"],
    "zai": ["glm-5.1"],
}


def canonical_provider(provider: str) -> str:
    return _PROVIDER_ALIASES.get(provider.strip().lower(), provider.strip().lower())


def provider_family(provider: str) -> str:
    key = canonical_provider(provider)
    if key in _GROK_FAMILY or key == "grok":
        return "grok"
    if key in _CODEX_FAMILY or key == "openai-codex":
        return "codex"
    return key


def provider_display_name(provider: str) -> str:
    family = provider_family(provider)
    if family == "grok":
        return "Grok"
    if family == "codex":
        return "Codex"
    key = canonical_provider(provider)
    if key == "homebase":
        return "Home Base"
    if key == "litert":
        return "LiteRT"
    return key


def provider_default_model(provider: str) -> str:
    key = canonical_provider(provider)
    return PROVIDER_DEFAULT_MODELS.get(key, PROVIDER_DEFAULT_MODELS.get(provider_family(provider), "gpt-5.4"))


def _is_frontier_provider(provider: str) -> bool:
    """Check if a provider is a frontier provider that needs dynamic model listing."""
    key = canonical_provider(provider)
    family = provider_family(provider)
    return key in ("grok", "xai", "openai-codex", "openai") or family in ("grok", "codex")


def provider_model_options(provider: str) -> list[str]:
    key = canonical_provider(provider)
    
    # For frontier providers, use dynamic listing
    if _is_frontier_provider(provider):
        try:
            from .provider import OpenAICompatProvider, ProviderError
            creds = resolve_provider_credentials(provider)
            if creds and creds.token:
                base_url = creds.base_url or provider_default_base_url(provider)
                prov = OpenAICompatProvider(api_base=base_url, api_key=creds.token)
                dynamic_models = prov.list_models()
                if dynamic_models:
                    return dynamic_models
        except Exception:
            # Fall through to static list if dynamic listing fails
            pass
    
    models = list(PROVIDER_MODEL_OPTIONS.get(key) or PROVIDER_MODEL_OPTIONS.get(provider_family(key), []))
    default = provider_default_model(key)
    if default and default not in models:
        models.insert(0, default)
    return models


def ordered_provider_options(names: list[str] | tuple[str, ...]) -> list[str]:
    seen: list[str] = []
    present: set[str] = set()
    for raw in names:
        name = str(raw).strip().lower()
        if not name or name in present or name in _TUI_HIDDEN_ALIASES:
            continue
        present.add(name)
        seen.append(name)
    head = [name for name in _PREFERRED_TUI_PROVIDERS if name in present]
    tail = [name for name in seen if name not in _PREFERRED_TUI_PROVIDERS]
    return head + tail
