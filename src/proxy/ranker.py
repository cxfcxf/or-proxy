"""LLM-based ranker for OpenRouter free models."""

import json
import logging

import httpx

from .config import settings

log = logging.getLogger(__name__)

_EXCLUDE_IDS: set[str] = {"openrouter/free"}

# Non-chat modalities that are useless as auxiliary models
_EXCLUDE_SUBSTRINGS: list[str] = ["embed", "lyria", "tts", "whisper"]


def _is_usable(model: dict) -> bool:
    mid = model["id"]
    if mid in _EXCLUDE_IDS:
        return False
    if any(s in mid for s in _EXCLUDE_SUBSTRINGS):
        return False
    if "tools" not in (model.get("supported_parameters") or []):
        return False
    return True


async def rank_models(
    models: list[dict],
    client: httpx.AsyncClient,
    ranker_model: str | None = None,
) -> list[dict]:
    """Ask an LLM to rank models for auxiliary/agentic use. Returns best-first list of {"id", "context_length"}."""
    ranker_model = ranker_model or settings.ranker_model
    usable = [m for m in models if _is_usable(m)]

    if not usable:
        return []

    model_list = "\n".join(
        f"- {m['id']} | ctx={m.get('context_length') or 0} | {m.get('name', '')} | {m.get('description', '')}"
        for m in usable
    )

    prompt = (
        "Rank these free LLM models for use as an auxiliary model in an AI agent system "
        "(tool calling, agentic subtasks, reasoning, coding). "
        "Prioritize: strong tool use, good reasoning, large context, reputable family. "
        "Use context_length as tiebreaker between equals.\n\n"
        f"{model_list}\n\n"
        "Return ONLY a JSON array of model IDs best-first, no explanation:\n"
        '["model/id-1", "model/id-2", ...]'
    )

    try:
        resp = await client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": settings.http_referer,
                "X-Title": settings.x_title,
            },
            json={
                "model": ranker_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("LLM ranker failed: %s — falling back to context_length sort", e)
        return _fallback_rank(usable)

    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        log.error("LLM ranker returned no JSON array — falling back")
        return _fallback_rank(usable)

    try:
        ranked_ids: list[str] = json.loads(content[start:end])
    except json.JSONDecodeError as e:
        log.error("LLM ranker JSON parse error: %s — falling back", e)
        return _fallback_rank(usable)

    by_id = {m["id"]: m for m in usable}
    seen: set[str] = set()
    result: list[dict] = []

    for mid in ranked_ids:
        if mid in by_id and mid not in seen:
            m = by_id[mid]
            result.append({"id": m["id"], "context_length": m.get("context_length") or 0})
            seen.add(mid)

    # Append any models LLM missed, sorted by context_length desc
    for m in sorted(usable, key=lambda x: -(x.get("context_length") or 0)):
        if m["id"] not in seen:
            result.append({"id": m["id"], "context_length": m.get("context_length") or 0})

    log.info("ranker: %d usable / %d total, top=%s", len(result), len(models), result[0]["id"] if result else "—")
    return result


def _fallback_rank(models: list[dict]) -> list[dict]:
    sorted_models = sorted(models, key=lambda m: -(m.get("context_length") or 0))
    return [{"id": m["id"], "context_length": m.get("context_length") or 0} for m in sorted_models]
