# OpenRouter Free-Model Reverse Proxy

Local reverse proxy that discovers OpenRouter's free-tier models, ranks them with a deterministic scorer tuned for agentic coding/chat, and exposes a single OpenAI-compatible endpoint with automatic fallback.

## Setup

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY
```

## Run

```bash
uv run uvicorn proxy.main:app --host 127.0.0.1 --port 8787
```

## Verify

```bash
# ranked model list
curl http://127.0.0.1:8787/v1/models

# non-streaming
curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"say hi"}]}'

# streaming
curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"say hi"}],"stream":true}'

# health
curl http://127.0.0.1:8787/health
```

## Config (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | required | OpenRouter bearer token |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base |
| `POLL_INTERVAL_SECONDS` | `86400` | Free model refresh interval (1 day) |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8787` | Bind port |
| `HTTP_REFERER` | `http://localhost` | OpenRouter `HTTP-Referer` header |
| `X_TITLE` | `hermes-free-proxy` | OpenRouter `X-Title` header |

## Ranking

The ranker in `ranker.py` is a pure deterministic scorer:

1. **Hard filters**: drops anything without `tools` in `supported_parameters`; drops `openrouter/free` (meta-router, recursion risk).
2. **Family tier score**: known-good families get explicit scores (Qwen3-Coder > Qwen3-Next > GPT-OSS 120B > GLM-4.5 > Nemotron Super > MiniMax M2 > ...).
3. **Tiebreaker**: longer context wins, then alphabetical.

When new free models appear, they default to a mid-tier score until the list in `ranker.py` is updated. No external calls, no LLM, no extra dependency.

## Failure behaviour

- Discovery fails → keep previous model list
- Model returns 429/5xx or network error → walk to next ranked model
- All models exhausted → `502` with OpenAI-shaped error (Hermes falls back to glm-5.1)
- Streaming: retry is possible only before the first byte reaches the client
