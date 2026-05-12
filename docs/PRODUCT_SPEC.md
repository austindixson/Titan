# OpenProxy Product Specification

## Product summary
OpenProxy is a local-first GUI application for configuring LLM providers and exposing compatibility proxy endpoints so users can connect any harness to any model backend.

## Product principles
- Local-first and developer-friendly
- Protocol compatibility over vendor lock-in
- Observable request translation
- Minimal configuration for common cases
- Progressive power-user controls

## Users
- AI developers using multiple model vendors
- Prompt engineers switching between providers
- Users of harnesses/tools locked to one API protocol
- Local model experimenters using Ollama or OpenAI-compatible servers

## Jobs to be done
- "Let me store and test all my model providers in one place."
- "Let me expose an Anthropic-compatible endpoint backed by OpenAI."
- "Let me expose an OpenAI-compatible endpoint backed by Ollama."
- "Let me see what request translation happened when something breaks."

## MVP scope
### Included
- Browser UI served from the same local server
- Provider CRUD
- Proxy CRUD
- Local JSON persistence
- Health endpoint
- API endpoints for app config
- Dynamic proxy routing for at least:
  - Anthropic-compatible -> OpenAI provider
- Dashboard summary
- Logs view for recent requests
- Test console with basic request execution

### Deferred
- Multi-user auth
- Cloud sync
- Encrypted secret storage
- Full OpenAI-compatible outbound protocol
- Full Anthropic-compatible outbound protocol for non-OpenAI providers
- Billing/cost analytics
- Desktop packaging

## Key concepts
### Provider
Represents an upstream LLM service.
Fields:
- id
- name
- type
- baseUrl
- apiKey
- defaultModel
- enabled
- metadata

### Proxy
Represents a local compatibility endpoint.
Fields:
- id
- name
- slug
- exposedProtocol
- targetProviderId
- targetModel
- enabled

### Request log
Represents an incoming proxied request and its result.
Fields:
- id
- proxyId
- timestamp
- method
- path
- upstream
- status
- latencyMs
- requestPreview
- error

## Information architecture
- Dashboard
- Providers
- Proxies
- Logs
- Test Console
- Settings

## API surface for the app itself
### App API
- `GET /api/health`
- `GET /api/providers`
- `POST /api/providers`
- `PUT /api/providers/:id`
- `DELETE /api/providers/:id`
- `GET /api/proxies`
- `POST /api/proxies`
- `PUT /api/proxies/:id`
- `DELETE /api/proxies/:id`
- `GET /api/logs`
- `POST /api/test/request`

### Proxy API
- `GET /proxy/:slug/health`
- `GET /proxy/:slug/v1/models`
- `POST /proxy/:slug/v1/messages`
- `POST /proxy/:slug/v1/messages/count_tokens`

## UX requirements
- The app should open in a browser and be understandable without docs.
- Empty states should guide the user to create their first provider and proxy.
- Testing a provider should require one click.
- Creating a proxy should require fewer than 10 fields in MVP.

## Non-functional requirements
- Fast local startup
- No external DB requirement
- Config survives restarts
- Request logs capped to prevent unbounded growth

## Risks
- Protocol translation complexity grows quickly across vendors
- Secret handling needs hardening before production use
- Browser UI needs careful handling of partial or failing upstream responses

## MVP acceptance
- User can open the UI in a browser
- User can create a provider
- User can create a proxy linked to that provider
- User can call the proxy endpoint
- User can see recent logs in the UI
