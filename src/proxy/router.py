import logging
from datetime import datetime, timedelta
from typing import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .schemas import ErrorDetail, ErrorResponse
from .state import state

STICKY_TTL = timedelta(hours=1)

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


async def forward_embeddings(request: Request, client: httpx.AsyncClient) -> JSONResponse:
    body = await request.json()
    url = f"{settings.openrouter_base_url}/embeddings"
    try:
        resp = await client.post(url, json=body, headers=_or_headers(), timeout=_NONSTREAM_TIMEOUT)
    except httpx.RequestError as e:
        return JSONResponse(status_code=502, content={"error": str(e)})
    return JSONResponse(status_code=resp.status_code, content=resp.json())

_NONSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


def _or_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": settings.http_referer,
        "X-Title": settings.x_title,
    }


def _error_502(msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(error=ErrorDetail(message=msg)).model_dump(),
    )


def _sticky_order(models: list[str], sticky: str | None) -> list[str]:
    """Reorder models to start from sticky, falling back to ranked order."""
    if sticky and sticky in models:
        idx = models.index(sticky)
        return models[idx:] + models[:idx]
    return models


async def _set_sticky(model_id: str) -> None:
    async with state.lock:
        if state.sticky_model != model_id:
            log.info("sticky model -> %s", model_id)
            state.sticky_model = model_id
            state.sticky_since = datetime.utcnow()


async def forward_chat_completion(request: Request, client: httpx.AsyncClient):
    body = await request.json()
    is_stream = bool(body.get("stream"))

    async with state.lock:
        models = [m["id"] for m in state.ranked_models]
        sticky = state.sticky_model
        sticky_since = state.sticky_since
        if sticky and sticky_since and datetime.utcnow() - sticky_since > STICKY_TTL:
            log.info("sticky TTL expired, resetting to #1")
            state.sticky_model = None
            state.sticky_since = None
            sticky = None

    if not models:
        return _error_502("no free models available yet; wait for first refresh")

    ordered = _sticky_order(models, sticky)
    url = f"{settings.openrouter_base_url}/chat/completions"

    if is_stream:
        return await _try_stream(client, url, body, ordered)
    return await _try_non_stream(client, url, body, ordered)


async def _try_non_stream(
    client: httpx.AsyncClient, url: str, body: dict, models: list[str]
) -> JSONResponse:
    for model_id in models:
        payload = {**body, "model": model_id}
        try:
            resp = await client.post(url, json=payload, headers=_or_headers(), timeout=_NONSTREAM_TIMEOUT)
        except httpx.RequestError as e:
            log.warning("model %s network error: %s", model_id, e)
            continue

        if resp.status_code in RETRY_STATUSES:
            log.warning("model %s returned %d, trying next", model_id, resp.status_code)
            continue

        await _set_sticky(model_id)
        return JSONResponse(status_code=resp.status_code, content=resp.json())

    return _error_502("all free models failed")


async def _try_stream(
    client: httpx.AsyncClient, url: str, body: dict, models: list[str]
) -> StreamingResponse | JSONResponse:
    for model_id in models:
        try:
            result = await _attempt_stream(client, url, {**body, "model": model_id}, model_id)
            if result is not None:
                await _set_sticky(model_id)
                return result
        except httpx.RequestError as e:
            log.warning("model %s stream error: %s", model_id, e)

    return _error_502("all free models failed (stream)")


async def _attempt_stream(
    client: httpx.AsyncClient, url: str, payload: dict, model_id: str
) -> StreamingResponse | None:
    req = client.build_request("POST", url, json=payload, headers=_or_headers(), timeout=_STREAM_TIMEOUT)
    response = await client.send(req, stream=True)

    if response.status_code in RETRY_STATUSES:
        await response.aclose()
        log.warning("model %s stream status %d, trying next", model_id, response.status_code)
        return None

    async def _gen(resp: httpx.Response) -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        _gen(response),
        status_code=response.status_code,
        media_type="text/event-stream",
        headers={"X-Proxy-Model": model_id},
    )
