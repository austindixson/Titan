# OpenProxyApp PRD

## Product
OpenProxyApp is a local-first graphical app for managing multiple LLM providers and creating compatibility proxies so any harness can talk to any LLM backend.

## Problem
Users have many LLM APIs and tools, but harnesses often only support one protocol family such as OpenAI or Anthropic. Users need a single place to manage providers, store endpoints and tokens, define model mappings, create protocol bridges, and inspect requests.

## Vision
A GUI control plane plus local proxy runtime that makes LLM connectivity programmable, observable, and protocol-agnostic.

## Primary users
- Developers using multiple LLM vendors
- Power users running local and hosted models
- Teams testing harnesses against different providers

## Core jobs to be done
- Add and manage LLM providers
- Store auth and endpoint config locally
- Create a proxy exposing one protocol backed by another provider
- Map models and request behavior
- Test requests visually
- Inspect logs and failures

## MVP scope
- Local web GUI
- Provider CRUD
- Proxy CRUD
- JSON file persistence
- Health dashboard
- Request logs
- Anthropic-compatible exposed endpoint backed by OpenAI
- OpenAI-compatible exposed endpoint stub structure
- Request test console

## Out of scope for MVP
- Multi-user auth
- Cloud sync
- Billing analytics
- Full provider matrix
- Enterprise secret management

## Success criteria
- User can add at least one provider in GUI
- User can create at least one proxy in GUI
- Proxy route is mounted and visible in dashboard
- User can test proxy health and inspect logs
- Existing Anthropic-to-OpenAI shim still works

## Risks
- Protocol mismatch complexity
- Streaming compatibility differences
- Secure secret handling beyond MVP

## Release plan
1. Spec and architecture
2. Backend modularization and persistence
3. GUI shell and CRUD flows
4. Dynamic proxy registration
5. Logs and test console
6. Provider expansion
