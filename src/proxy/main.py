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

    ranked = rank_models(models)
    if not ranked:
        log.warning("ranker returned empty list; using discovery order")
        ranked = [m["id"] for m in models]

    async with state.lock:
        state.ranked_models = ranked
        state.last_refresh = datetime.utcnow()

    log.info("state updated: %d models, top=%s", len(ranked), ranked[0] if ranked else "—")


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


@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelObject(id="auto", owned_by="proxy")])


@app.get("/health")
async def health():
    async with state.lock:
        return {
            "status": "ok",
            "model_count": len(state.ranked_models),
            "last_refresh": state.last_refresh.isoformat() if state.last_refresh else None,
            "ranked_models": list(state.ranked_models),
        }
