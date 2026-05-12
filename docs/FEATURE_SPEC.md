# Feature Spec: OpenProxyApp MVP

## Goal
Turn the current single-purpose shim into a GUI-driven local LLM proxy manager.

## MVP Scope

### Included
1. Web UI served from the same Express app
2. Provider CRUD API and UI
3. Proxy CRUD API and UI
4. Local persistence to JSON files
5. Request log capture
6. Dynamic proxy routing for Anthropic-compatible -> OpenAI-compatible upstreams
7. Health/test endpoints for the app shell

### Deferred
1. Encryption at rest
2. Full multi-provider protocol matrix
3. User accounts
4. Team sync
5. Billing analytics
6. Desktop packaging

## Information Architecture

### Navigation
- Dashboard
- Providers
- Proxies
- Logs
- Test Console
- Settings

### Dashboard
Shows:
- total providers
- enabled providers
- total proxies
- recent logs
- quick links to create provider/proxy

### Providers
Fields:
- id
- name
- type
- baseUrl
- apiKey
- defaultModel
- enabled

Supported types in MVP:
- openai
- anthropic
- ollama
- custom-openai

Behavior:
- create provider
- edit provider
- delete provider
- test provider (basic validation only in MVP)

### Proxies
Fields:
- id
- name
- path
- sourceProtocol
- targetProviderId
- targetModel
- enabled

Supported source protocols in MVP:
- anthropic

Supported target provider behavior in MVP:
- openai-like `/responses`

Behavior:
- create proxy
- edit proxy
- delete proxy
- list generated endpoint path

### Logs
Each log entry stores:
- id
- ts
- proxyId
- method
- path
- status
- latencyMs
- requestSummary
- error

### Test Console
MVP capability:
- display example curl snippets
- indicate configured proxy URLs

## Backend API Spec

### UI/API management endpoints
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

### Dynamic proxy surface
Per proxy path:
- `GET /proxy/:proxyId/health`
- `GET /proxy/:proxyId/v1/models`
- `POST /proxy/:proxyId/v1/messages`
- `POST /proxy/:proxyId/v1/messages/count_tokens`

## Validation Rules

### Provider
- `name` required
- `type` required
- `baseUrl` required for non-local types
- `apiKey` optional in storage but required at request time for remote providers unless inherited via env

### Proxy
- `name` required
- `path` required and unique
- `sourceProtocol` must be `anthropic` in MVP
- `targetProviderId` must exist

## UX Notes
- Keep forms simple and inspectable
- Display raw endpoint URL for each proxy
- Mask API keys in lists
- Surface last error in logs table

## Success Criteria
- User can open browser UI at `/`
- User can add provider without editing files
- User can create a proxy that exposes Anthropic-compatible endpoint
- User can send requests to generated proxy URL
- User can see logs in UI
