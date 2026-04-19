"""Deterministic ranker for OpenRouter free models.

Ordering priority:
1. Known-good families get tiered scores (higher = better)
2. Ties broken by context_length (longer wins)
3. Final tiebreak on id (alphabetical) for stability
"""

import logging

log = logging.getLogger(__name__)

# Tiered family preferences. Matched by substring against model id.
# First match wins, so list more specific patterns before generic ones.
_FAMILY_SCORES: list[tuple[str, int]] = [
    ("qwen3-coder", 100),
    ("qwen3-next", 90),
    ("gpt-oss-120b", 85),
    ("glm-4.5", 80),
    ("nemotron-3-super", 75),
    ("minimax-m2", 70),
    ("gpt-oss-20b", 60),
    ("llama-3.3-70b", 55),
    ("gemma-4-31b", 50),
    ("gemma-4", 45),
    ("elephant", 40),
    ("nemotron-3-nano", 35),
    ("trinity", 30),
    ("nemotron-nano", 25),
]

# Hard exclusions (recursion risk, non-chat modality, etc.)
_EXCLUDE_IDS: set[str] = {"openrouter/free"}


def _family_score(model_id: str) -> int:
    for needle, score in _FAMILY_SCORES:
        if needle in model_id:
            return score
    return 10  # unknown but still usable


def _is_usable(model: dict) -> bool:
    if model["id"] in _EXCLUDE_IDS:
        return False
    if "tools" not in (model.get("supported_parameters") or []):
        return False
    return True


def rank_models(models: list[dict]) -> list[dict]:
    """Return models ranked best-first, each as {"id": ..., "context_length": ...}."""
    usable = [m for m in models if _is_usable(m)]

    scored = [
        (_family_score(m["id"]), m.get("context_length") or 0, m["id"], m.get("context_length") or 0)
        for m in usable
    ]
    # Sort: family score desc, context desc, id asc
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

    ranked = [{"id": t[2], "context_length": t[3]} for t in scored]
    log.info(
        "ranker: %d usable / %d total, top=%s",
        len(ranked),
        len(models),
        ranked[0]["id"] if ranked else "—",
    )
    return ranked
