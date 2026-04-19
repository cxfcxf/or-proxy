#!/usr/bin/env python3
"""Fetch free OpenRouter models, ask an LLM to rank them for auxiliary/agentic use."""

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path.home() / ".sieg.env")

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

# Model used to do the ranking — pick something capable and free
RANKER_MODEL = "google/gemini-3.1-flash-lite-preview"


def fetch_free_models() -> list[dict]:
    resp = httpx.get(f"{OPENROUTER_BASE}/models", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    free = []
    for m in resp.json().get("data", []):
        pricing = m.get("pricing", {})
        if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
            free.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": m.get("context_length") or 0,
                "supported_parameters": m.get("supported_parameters") or [],
                "description": (m.get("description") or "")[:300],
            })
    return free


def ask_llm_to_rank(models: list[dict]) -> list[str]:
    model_list = "\n".join(
        f"- {m['id']} | ctx={m['context_length']} | tools={'yes' if 'tools' in m['supported_parameters'] else 'no'} | {m['name']}: {m['description']}"
        for m in models
    )

    prompt = f"""You are ranking free LLM models on OpenRouter for use as an **auxiliary model** in an AI agent system.

Auxiliary use means: tool calling, function use, quick agentic subtasks, reasoning, coding assistance.
Prioritize: strong tool use support, good reasoning, large context, reputable model family.
Deprioritize: embedding-only models, multimodal-only, models with poor reasoning or tiny context.

Here are the available free models:
{model_list}

Return ONLY a JSON array of model IDs in ranked order (best first), no explanation:
["model/id-1", "model/id-2", ...]"""

    resp = httpx.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers=HEADERS,
        json={
            "model": RANKER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    # Extract JSON array from response
    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        print("LLM response did not contain JSON array:", content, file=sys.stderr)
        sys.exit(1)

    ranked_ids: list[str] = json.loads(content[start:end])
    return ranked_ids


def main():
    print("Fetching free models...", file=sys.stderr)
    models = fetch_free_models()
    print(f"Found {len(models)} free models", file=sys.stderr)

    # Build lookup
    by_id = {m["id"]: m for m in models}

    print("Asking LLM to rank...", file=sys.stderr)
    ranked_ids = ask_llm_to_rank(models)

    # Resolve with context_length as tiebreaker for any unknowns
    seen = set()
    result = []
    for mid in ranked_ids:
        if mid in by_id and mid not in seen:
            m = by_id[mid]
            result.append(m)
            seen.add(mid)

    # Append any models LLM missed, sorted by context_length desc
    leftovers = sorted(
        [m for m in models if m["id"] not in seen],
        key=lambda m: -m["context_length"],
    )
    result.extend(leftovers)

    print(json.dumps([{"id": m["id"], "context_length": m["context_length"]} for m in result], indent=2))


if __name__ == "__main__":
    main()
