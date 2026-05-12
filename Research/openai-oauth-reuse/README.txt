OpenAI OAuth reuse research + integration notes

Source folder examined:
- /Users/ghost/Desktop/openproxy

Key extraction from source:
1) openproxy/start-openai-oauth-shim.sh
   - Reads ~/.hermes/auth.json
   - Token lookup order:
     a) .providers["openai-codex"].tokens.access_token
     b) .credential_pool["openai-codex"][0].access_token
   - Base URL fallback:
     .credential_pool["openai-codex"][0].base_url -> https://api.openai.com/v1

2) openproxy/server.js
   - Token lookup order inside request handling:
     OPENAI_OAUTH_TOKEN -> OPENAI_API_KEY -> Authorization bearer from request

What was implemented in Titan:
- Added credential resolver:
  /Users/ghost/Desktop/Titan/src/titan/auth.py

Behavior now:
1) OPENAI_OAUTH_TOKEN
2) OPENAI_API_KEY
3) ~/.hermes/auth.json (same lookup semantics as openproxy shim)

Also added:
- Base URL carry-over from Hermes auth credential pool when present.
- Startup prints auth source in CLI: env var name or hermes auth file.
- Non-retryable provider error when no credentials can be found.

Changed files:
- /Users/ghost/Desktop/Titan/src/titan/auth.py
- /Users/ghost/Desktop/Titan/src/titan/config.py
- /Users/ghost/Desktop/Titan/src/titan/provider.py
- /Users/ghost/Desktop/Titan/src/titan/cli.py
- /Users/ghost/Desktop/Titan/tests/test_auth_resolution.py

Validation:
- pytest: 8 passed

Copied source artifacts into this folder:
- start-openai-oauth-shim.sh
- openproxy-server.js
