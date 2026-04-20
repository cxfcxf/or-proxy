# OpenRouter Free-Model Reverse Proxy

Local reverse proxy that discovers OpenRouter's free-tier models, ranks them using an LLM, and exposes a single OpenAI-compatible endpoint with automatic fallback and sticky routing.

## Setup

```bash
# Add to ~/.sieg.env or .env
OPENROUTER_API_KEY=sk-or-...
```

## Run

```bash
uv run uvicorn proxy.main:app --host 127.0.0.1 --port 8787
```

Or via Docker:

```bash
docker run --name=or-proxy --user=app --env-file=~/.sieg.env \
  -p 127.0.0.1:8787:8787 --restart=unless-stopped \
  or-proxy:latest uvicorn proxy.main:app --host 0.0.0.0 --port 8787
```

## Verify

```bash
# health + current sticky model
curl http://127.0.0.1:8787/health

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
```

## Config (`.env` / `~/.sieg.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | required | OpenRouter bearer token |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base |
| `RANKER_MODEL` | `anthropic/claude-sonnet-4.6` | Model used to rank free models |
| `POLL_INTERVAL_SECONDS` | `86400` | Free model refresh interval (1 day) |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8787` | Bind port |
| `HTTP_REFERER` | `https://github.com/cxfcxf/or-proxy` | OpenRouter `HTTP-Referer` header |
| `X_TITLE` | `or-proxy` | OpenRouter `X-Title` header |

## Ranking

On startup and every `POLL_INTERVAL_SECONDS`, the proxy:

1. Fetches all free models from OpenRouter
2. Filters out non-tool-capable and non-chat models (embeddings, TTS, etc.)
3. Asks `RANKER_MODEL` to rank them for agentic/coding use (tool calling, reasoning, large context)
4. Falls back to context-length sort if the LLM call fails

The full ranked order is logged on each refresh.

## Routing

- **Sticky routing**: once a model succeeds it becomes sticky — subsequent requests start from that model, skipping rate-limited ones above it
- **1-hour TTL**: sticky resets to #1 every hour so rate limits have time to clear
- **Re-rank reset**: sticky also clears whenever the model list is refreshed
- **Fallback chain**: on 429/5xx, walks down the ranked list until one succeeds
- **All fail**: returns `502` with an OpenAI-shaped error body

## Failure behaviour

- Discovery fails → keep previous model list
- LLM ranker fails → fall back to context-length sort
- Model returns 429/5xx or network error → try next ranked model
- All models exhausted → `502`
- Streaming: retry only possible before first byte reaches the client
