# SignalCouncil

AI-powered multi-strategy financial decision intelligence.

## Architecture

- **extension/**: Chrome Extension (React, TypeScript, Vite, Manifest V3). Provides Chart + User Interface.
- **backend/**: Cloud Run / API (Python 3.12, FastAPI, Pydantic v2, uv).
- **shared/**: Shared types and utilities.
- **tests/**: E2E and integration tests.
- **infra/**: Infrastructure code (Docker, Cloud Run configs, Firestore adapter).
- **docs/**: Documentation.

## Core Requirements & Principles

- **No Authentication**: The application operates statelessly or relies on local extension state; no centralized user authentication.
- **No Trading Execution**: SignalCouncil provides analysis and decision intelligence only, it does not execute trades.
- **No Broker Integration**: We do not connect to broker APIs for execution or portfolio syncing.
- **Provider Abstraction**: Market data providers are abstracted.
- **Deterministic Strategy Engine**: Employs classical indicators (Trend, Momentum, Mean Reversion) deterministically.
- **ADK Integration Isolated**: Gemini/ADK logic for consensus and explanation is isolated from core domain logic.

## Setup

### Backend
```bash
cd backend
uv run uvicorn main:app --reload
```

### Extension
```bash
cd extension
npm run dev
```
