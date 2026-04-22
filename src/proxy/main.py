import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import FastAPI, Request

from .config import settings
from .discovery import fetch_free_models
from .ranker import rank_models
from .router import forward_chat_completion
from .schemas import ModelList, ModelObject
from .state import state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


async def _do_refresh() -> None:
    models = await fetch_free_models(_client)
    if not models:
        log.warning("discovery returned no free models; keeping previous list")
        return

    ranked = await rank_models(models, _client)
    if not ranked:
        log.warning("ranker returned empty list; using discovery order")
        ranked = [{"id": m["id"], "context_length": m.get("context_length") or 0} for m in models]

    async with state.lock:
        state.ranked_models = ranked
        state.last_refresh = datetime.utcnow()
        state.sticky_model = None
        state.sticky_since = None

    log.info("state updated: %d models, top=%s", len(ranked), ranked[0]["id"] if ranked else "—")


async def _refresh_loop() -> None:
    while True:
        try:
            await _do_refresh()
        except Exception as e:
            log.error("refresh cycle error: %s", e)
        await asyncio.sleep(settings.poll_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient()
    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()
    await _client.aclose()


app = FastAPI(title="openrouter-free-proxy", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await forward_chat_completion(request, _client)


def _active_model_info() -> dict | None:
    """Return sticky model info if set, else top ranked model."""
    sticky = state.sticky_model
    models = state.ranked_models
    if sticky:
        info = next((m for m in models if m["id"] == sticky), None)
        if info:
            return info
    return models[0] if models else None


@app.get("/v1/models")
async def list_models():
    async with state.lock:
        info = _active_model_info()
    ctx = info["context_length"] if info else None
    display_id = info["id"] if info else "auto"
    return ModelList(data=[ModelObject(id=display_id, owned_by="proxy", context_length=ctx)])


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    async with state.lock:
        info = _active_model_info()
    ctx = info["context_length"] if info else None
    return ModelObject(id=model_id, owned_by="proxy", context_length=ctx)


@app.get("/health")
async def health():
    async with state.lock:
        return {
            "status": "ok",
            "model_count": len(state.ranked_models),
            "last_refresh": state.last_refresh.isoformat() if state.last_refresh else None,
            "sticky_model": state.sticky_model,
            "ranked_models": [m["id"] for m in state.ranked_models],
        }
