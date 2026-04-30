import logging

import httpx

from .config import settings

log = logging.getLogger(__name__)


async def fetch_free_models(client: httpx.AsyncClient) -> list[dict]:
    try:
        resp = await client.get(
            f"{settings.openrouter_base_url}/models",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error("discovery failed: %s", e)
        return []

    free = []
    for m in data.get("data", []):
        pricing = m.get("pricing", {})
        if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
            free.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": m.get("context_length"),
                "supported_parameters": m.get("supported_parameters", []),
                "description": (m.get("description") or "")[:200],
                "created": m.get("created") or 0,
            })

    free.sort(key=lambda m: m["id"])
    log.info("discovered %d free models", len(free))
    return free
