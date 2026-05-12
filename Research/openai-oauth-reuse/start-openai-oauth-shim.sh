#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f "$HOME/.hermes/auth.json" ]]; then
  echo "Missing ~/.hermes/auth.json (Hermes auth not found)" >&2
  exit 1
fi

TOKEN="$(jq -r '.providers["openai-codex"].tokens.access_token // .credential_pool["openai-codex"][0].access_token // empty' "$HOME/.hermes/auth.json")"
BASE_URL="$(jq -r '.credential_pool["openai-codex"][0].base_url // empty' "$HOME/.hermes/auth.json")"

if [[ -z "$TOKEN" || "$TOKEN" == "null" ]]; then
  echo "No OpenAI token found in ~/.hermes/auth.json. Run: hermes login" >&2
  exit 1
fi

export OPENAI_OAUTH_TOKEN="$TOKEN"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${BASE_URL:-https://api.openai.com/v1}}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.4}"
export PORT="${PORT:-8082}"

exec node server.js
