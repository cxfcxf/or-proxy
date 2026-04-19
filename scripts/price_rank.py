"""Rank OpenRouter models by price/performance for agentic coding/chat.

Fetches the live /models catalog, filters to tool-capable chat models, then
scores each model along five dimensions and computes a weighted composite:

    coding     SWE-bench Verified, LiveCodeBench, Aider polyglot
    tools      BFCL v3, tau-bench, nexus function-calling
    finance    FinanceBench, MMLU-Pro (business/econ), FinQA
    reasoning  GPQA Diamond, AIME, MATH, ARC-AGI
    general    MMLU-Pro, IFEval, Arena-Hard

Scores are 0-100 rough benchmarks from public leaderboards through 2026-Q1.
They are approximate — treat as a sortable heuristic, not ground truth.
Unknown models get a mid-pack default (35/dim).

Usage:
    .venv/bin/python scripts/price_rank.py                    # balanced top 30
    .venv/bin/python scripts/price_rank.py --profile coding   # weight coding heavy
    .venv/bin/python scripts/price_rank.py --all --no-free
    .venv/bin/python scripts/price_rank.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import dotenv_values

# Per-family benchmark estimates. Keys are substrings matched against model id
# (first match wins, list specific before generic). Values are 0-100 scores
# approximating public leaderboard performance.
#
# Sources consulted: SWE-bench Verified, LiveCodeBench, Aider polyglot,
# BFCL v3, tau-bench, FinanceBench, FinQA, GPQA Diamond, AIME 2025, MMLU-Pro,
# Arena-Hard. Numbers are rounded to the nearest 5.
PERF_PROFILES: list[tuple[str, dict[str, int]]] = [
    # ---------- Anthropic ----------
    ("claude-opus-4.7",      {"coding": 85, "tools": 92, "finance": 85, "reasoning": 88, "general": 90}),
    ("claude-opus-4.6",      {"coding": 83, "tools": 90, "finance": 83, "reasoning": 86, "general": 88}),
    ("claude-opus-4.5",      {"coding": 80, "tools": 88, "finance": 80, "reasoning": 82, "general": 85}),
    ("claude-opus-4.1",      {"coding": 78, "tools": 85, "finance": 78, "reasoning": 80, "general": 82}),
    ("claude-opus-4",        {"coding": 75, "tools": 82, "finance": 75, "reasoning": 78, "general": 80}),
    ("claude-sonnet-4.6",    {"coding": 82, "tools": 90, "finance": 78, "reasoning": 80, "general": 84}),
    ("claude-sonnet-4.5",    {"coding": 78, "tools": 88, "finance": 75, "reasoning": 77, "general": 82}),
    ("claude-sonnet-4",      {"coding": 72, "tools": 82, "finance": 70, "reasoning": 72, "general": 78}),
    ("claude-haiku-4.5",     {"coding": 60, "tools": 70, "finance": 55, "reasoning": 58, "general": 65}),
    ("claude-3.7-sonnet",    {"coding": 65, "tools": 75, "finance": 65, "reasoning": 65, "general": 72}),
    ("claude-3.5-haiku",     {"coding": 42, "tools": 55, "finance": 40, "reasoning": 40, "general": 50}),
    ("claude-3-haiku",       {"coding": 30, "tools": 38, "finance": 32, "reasoning": 32, "general": 40}),

    # ---------- OpenAI ----------
    ("gpt-5-pro",            {"coding": 85, "tools": 82, "finance": 90, "reasoning": 95, "general": 90}),
    ("gpt-5.4-pro",          {"coding": 84, "tools": 82, "finance": 89, "reasoning": 93, "general": 89}),
    ("gpt-5.2-pro",          {"coding": 82, "tools": 80, "finance": 86, "reasoning": 90, "general": 87}),
    ("gpt-5.4",              {"coding": 80, "tools": 80, "finance": 85, "reasoning": 88, "general": 86}),
    ("gpt-5.3",              {"coding": 78, "tools": 78, "finance": 82, "reasoning": 85, "general": 84}),
    ("gpt-5.2",              {"coding": 76, "tools": 76, "finance": 80, "reasoning": 82, "general": 82}),
    ("gpt-5.1-codex",        {"coding": 82, "tools": 78, "finance": 70, "reasoning": 78, "general": 78}),
    ("gpt-5.1",              {"coding": 74, "tools": 76, "finance": 78, "reasoning": 80, "general": 80}),
    ("gpt-5-codex",          {"coding": 80, "tools": 76, "finance": 68, "reasoning": 76, "general": 76}),
    ("gpt-5-mini",           {"coding": 62, "tools": 68, "finance": 65, "reasoning": 70, "general": 70}),
    ("gpt-5-nano",           {"coding": 50, "tools": 55, "finance": 52, "reasoning": 58, "general": 58}),
    ("gpt-5.4-nano",         {"coding": 52, "tools": 57, "finance": 54, "reasoning": 60, "general": 60}),
    ("gpt-5",                {"coding": 72, "tools": 75, "finance": 76, "reasoning": 78, "general": 78}),
    ("o4-mini",              {"coding": 68, "tools": 60, "finance": 65, "reasoning": 82, "general": 70}),
    ("o3-mini",              {"coding": 62, "tools": 55, "finance": 60, "reasoning": 78, "general": 65}),
    ("o3-pro",               {"coding": 75, "tools": 65, "finance": 72, "reasoning": 90, "general": 78}),
    ("o3",                   {"coding": 70, "tools": 60, "finance": 68, "reasoning": 85, "general": 72}),
    ("o1",                   {"coding": 60, "tools": 50, "finance": 58, "reasoning": 78, "general": 62}),
    ("gpt-4.1-mini",         {"coding": 50, "tools": 58, "finance": 55, "reasoning": 52, "general": 58}),
    ("gpt-4.1-nano",         {"coding": 38, "tools": 45, "finance": 40, "reasoning": 38, "general": 45}),
    ("gpt-4.1",              {"coding": 58, "tools": 65, "finance": 62, "reasoning": 60, "general": 65}),
    ("gpt-4o-mini",          {"coding": 35, "tools": 45, "finance": 38, "reasoning": 35, "general": 45}),
    ("gpt-4o",               {"coding": 52, "tools": 60, "finance": 55, "reasoning": 50, "general": 62}),
    ("gpt-oss-120b",         {"coding": 55, "tools": 60, "finance": 52, "reasoning": 58, "general": 60}),
    ("gpt-oss-20b",          {"coding": 42, "tools": 48, "finance": 40, "reasoning": 44, "general": 48}),

    # ---------- Google ----------
    ("gemini-3.1-pro",       {"coding": 75, "tools": 72, "finance": 78, "reasoning": 82, "general": 82}),
    ("gemini-3-flash",       {"coding": 55, "tools": 55, "finance": 55, "reasoning": 58, "general": 62}),
    ("gemini-3.1-flash",     {"coding": 48, "tools": 50, "finance": 48, "reasoning": 50, "general": 55}),
    ("gemini-2.5-pro",       {"coding": 72, "tools": 70, "finance": 75, "reasoning": 80, "general": 80}),
    ("gemini-2.5-flash-lite",{"coding": 40, "tools": 45, "finance": 42, "reasoning": 42, "general": 50}),
    ("gemini-2.5-flash",     {"coding": 55, "tools": 58, "finance": 55, "reasoning": 58, "general": 65}),
    ("gemini-2.0-flash",     {"coding": 45, "tools": 52, "finance": 48, "reasoning": 48, "general": 55}),
    ("gemma-4-31b",          {"coding": 38, "tools": 40, "finance": 35, "reasoning": 40, "general": 45}),
    ("gemma-4",              {"coding": 35, "tools": 38, "finance": 32, "reasoning": 35, "general": 42}),

    # ---------- xAI ----------
    ("grok-4.20",            {"coding": 78, "tools": 72, "finance": 75, "reasoning": 82, "general": 80}),
    ("grok-4.1-fast",        {"coding": 65, "tools": 68, "finance": 62, "reasoning": 70, "general": 70}),
    ("grok-4-fast",          {"coding": 62, "tools": 65, "finance": 60, "reasoning": 68, "general": 68}),
    ("grok-4",               {"coding": 72, "tools": 68, "finance": 70, "reasoning": 78, "general": 76}),
    ("grok-3-mini",          {"coding": 45, "tools": 48, "finance": 45, "reasoning": 50, "general": 52}),
    ("grok-3",               {"coding": 55, "tools": 55, "finance": 55, "reasoning": 58, "general": 62}),
    ("grok-code-fast",       {"coding": 58, "tools": 50, "finance": 40, "reasoning": 50, "general": 55}),

    # ---------- DeepSeek ----------
    ("deepseek-v3.2",        {"coding": 68, "tools": 60, "finance": 62, "reasoning": 70, "general": 68}),
    ("deepseek-r1-0528",     {"coding": 68, "tools": 55, "finance": 65, "reasoning": 80, "general": 70}),
    ("deepseek-r1t2",        {"coding": 65, "tools": 55, "finance": 62, "reasoning": 75, "general": 68}),
    ("deepseek-r1",          {"coding": 62, "tools": 52, "finance": 60, "reasoning": 78, "general": 65}),
    ("deepseek-v3.1",        {"coding": 62, "tools": 58, "finance": 58, "reasoning": 62, "general": 62}),
    ("deepseek-chat-v3",     {"coding": 58, "tools": 55, "finance": 55, "reasoning": 58, "general": 60}),
    ("deepseek-chat",        {"coding": 55, "tools": 52, "finance": 52, "reasoning": 55, "general": 58}),

    # ---------- Qwen ----------
    ("qwen3-coder-plus",     {"coding": 72, "tools": 68, "finance": 52, "reasoning": 62, "general": 62}),
    ("qwen3-coder-next",     {"coding": 70, "tools": 65, "finance": 50, "reasoning": 60, "general": 60}),
    ("qwen3-coder-flash",    {"coding": 65, "tools": 62, "finance": 48, "reasoning": 55, "general": 58}),
    ("qwen3-coder-30b",      {"coding": 62, "tools": 60, "finance": 45, "reasoning": 52, "general": 55}),
    ("qwen3-coder",          {"coding": 68, "tools": 65, "finance": 50, "reasoning": 58, "general": 60}),
    ("qwen3-next-80b",       {"coding": 58, "tools": 58, "finance": 55, "reasoning": 62, "general": 62}),
    ("qwen3-max",            {"coding": 55, "tools": 55, "finance": 55, "reasoning": 58, "general": 60}),
    ("qwen3-235b",           {"coding": 55, "tools": 55, "finance": 52, "reasoning": 58, "general": 60}),
    ("qwen3.5-flash",        {"coding": 45, "tools": 48, "finance": 45, "reasoning": 48, "general": 52}),
    ("qwen3.5-plus",         {"coding": 52, "tools": 52, "finance": 50, "reasoning": 55, "general": 58}),
    ("qwen3.5-122b",         {"coding": 55, "tools": 55, "finance": 52, "reasoning": 58, "general": 60}),
    ("qwen3.5-397b",         {"coding": 58, "tools": 58, "finance": 55, "reasoning": 62, "general": 62}),
    ("qwen3.5",              {"coding": 48, "tools": 50, "finance": 48, "reasoning": 52, "general": 55}),
    ("qwen3.6-plus",         {"coding": 55, "tools": 55, "finance": 52, "reasoning": 58, "general": 60}),
    ("qwen3-vl",             {"coding": 45, "tools": 50, "finance": 42, "reasoning": 48, "general": 52}),
    ("qwen3-30b-a3b-thinking",{"coding": 50, "tools": 48, "finance": 48, "reasoning": 60, "general": 55}),
    ("qwen3-30b",            {"coding": 48, "tools": 48, "finance": 42, "reasoning": 50, "general": 52}),
    ("qwen3-14b",            {"coding": 42, "tools": 45, "finance": 40, "reasoning": 45, "general": 48}),
    ("qwen3-32b",            {"coding": 50, "tools": 50, "finance": 45, "reasoning": 52, "general": 55}),
    ("qwen3-8b",             {"coding": 38, "tools": 42, "finance": 35, "reasoning": 40, "general": 45}),
    ("qwen-plus",            {"coding": 48, "tools": 50, "finance": 48, "reasoning": 52, "general": 55}),
    ("qwen-max",             {"coding": 52, "tools": 52, "finance": 52, "reasoning": 55, "general": 58}),
    ("qwen-turbo",           {"coding": 35, "tools": 40, "finance": 35, "reasoning": 38, "general": 42}),
    ("qwen-2.5-72b",         {"coding": 48, "tools": 48, "finance": 45, "reasoning": 50, "general": 52}),
    ("qwen-2.5-7b",          {"coding": 30, "tools": 32, "finance": 28, "reasoning": 32, "general": 38}),
    ("qwen-vl",              {"coding": 38, "tools": 42, "finance": 38, "reasoning": 42, "general": 45}),
    ("qwq-32b",              {"coding": 48, "tools": 38, "finance": 45, "reasoning": 60, "general": 52}),

    # ---------- Z.ai GLM ----------
    ("glm-5.1",              {"coding": 52, "tools": 52, "finance": 50, "reasoning": 55, "general": 58}),
    ("glm-5v-turbo",         {"coding": 48, "tools": 50, "finance": 45, "reasoning": 50, "general": 55}),
    ("glm-5-turbo",          {"coding": 50, "tools": 50, "finance": 48, "reasoning": 52, "general": 55}),
    ("glm-5",                {"coding": 55, "tools": 55, "finance": 52, "reasoning": 58, "general": 60}),
    ("glm-4.7-flash",        {"coding": 48, "tools": 48, "finance": 42, "reasoning": 48, "general": 52}),
    ("glm-4.7",              {"coding": 55, "tools": 52, "finance": 50, "reasoning": 55, "general": 58}),
    ("glm-4.6v",             {"coding": 58, "tools": 58, "finance": 55, "reasoning": 60, "general": 62}),
    ("glm-4.6",              {"coding": 60, "tools": 58, "finance": 55, "reasoning": 62, "general": 62}),
    ("glm-4.5v",             {"coding": 55, "tools": 58, "finance": 52, "reasoning": 58, "general": 60}),
    ("glm-4.5-air",          {"coding": 50, "tools": 55, "finance": 48, "reasoning": 52, "general": 55}),
    ("glm-4.5",              {"coding": 58, "tools": 60, "finance": 55, "reasoning": 58, "general": 62}),
    ("glm-4",                {"coding": 42, "tools": 45, "finance": 42, "reasoning": 45, "general": 50}),

    # ---------- Meta Llama ----------
    ("llama-4-scout",        {"coding": 50, "tools": 55, "finance": 48, "reasoning": 52, "general": 58}),
    ("llama-4",              {"coding": 58, "tools": 62, "finance": 55, "reasoning": 58, "general": 65}),
    ("llama-3.3-70b",        {"coding": 45, "tools": 52, "finance": 45, "reasoning": 48, "general": 55}),
    ("llama-3.1-70b",        {"coding": 42, "tools": 48, "finance": 42, "reasoning": 45, "general": 52}),
    ("llama-3.1-8b",         {"coding": 25, "tools": 32, "finance": 25, "reasoning": 28, "general": 35}),
    ("llama-3-8b",           {"coding": 22, "tools": 28, "finance": 22, "reasoning": 25, "general": 32}),

    # ---------- Mistral ----------
    ("mistral-large",        {"coding": 48, "tools": 55, "finance": 50, "reasoning": 52, "general": 60}),
    ("mistral-medium",       {"coding": 45, "tools": 50, "finance": 45, "reasoning": 48, "general": 55}),
    ("mistral-small",        {"coding": 38, "tools": 42, "finance": 38, "reasoning": 40, "general": 48}),
    ("mistral-nemo",         {"coding": 30, "tools": 35, "finance": 30, "reasoning": 32, "general": 42}),
    ("mistral-saba",         {"coding": 28, "tools": 32, "finance": 30, "reasoning": 30, "general": 40}),
    ("ministral-14b",        {"coding": 35, "tools": 38, "finance": 35, "reasoning": 38, "general": 45}),
    ("ministral-8b",         {"coding": 28, "tools": 32, "finance": 28, "reasoning": 30, "general": 40}),
    ("ministral-3b",         {"coding": 22, "tools": 28, "finance": 22, "reasoning": 25, "general": 32}),
    ("codestral",            {"coding": 58, "tools": 42, "finance": 30, "reasoning": 45, "general": 48}),
    ("devstral-medium",      {"coding": 55, "tools": 52, "finance": 35, "reasoning": 48, "general": 50}),
    ("devstral-small",       {"coding": 48, "tools": 45, "finance": 30, "reasoning": 42, "general": 45}),
    ("devstral",             {"coding": 52, "tools": 48, "finance": 32, "reasoning": 45, "general": 48}),
    ("mixtral-8x22b",        {"coding": 42, "tools": 45, "finance": 42, "reasoning": 45, "general": 52}),
    ("mixtral-8x7b",         {"coding": 32, "tools": 38, "finance": 32, "reasoning": 35, "general": 42}),
    ("pixtral",              {"coding": 38, "tools": 42, "finance": 38, "reasoning": 40, "general": 48}),

    # ---------- Misc ----------
    ("minimax-m2.7",         {"coding": 55, "tools": 55, "finance": 50, "reasoning": 58, "general": 58}),
    ("minimax-m2.5",         {"coding": 50, "tools": 52, "finance": 48, "reasoning": 52, "general": 55}),
    ("minimax-m2",           {"coding": 48, "tools": 50, "finance": 45, "reasoning": 50, "general": 52}),
    ("minimax-m1",           {"coding": 42, "tools": 45, "finance": 40, "reasoning": 45, "general": 48}),
    ("kimi-k2.5",            {"coding": 52, "tools": 55, "finance": 48, "reasoning": 55, "general": 58}),
    ("kimi-k2-thinking",     {"coding": 52, "tools": 50, "finance": 50, "reasoning": 62, "general": 58}),
    ("kimi-k2",              {"coding": 48, "tools": 52, "finance": 45, "reasoning": 50, "general": 55}),
    ("nemotron-3-super",     {"coding": 55, "tools": 55, "finance": 50, "reasoning": 58, "general": 60}),
    ("nemotron-3-nano",      {"coding": 38, "tools": 42, "finance": 35, "reasoning": 40, "general": 45}),
    ("nemotron-nano-12b",    {"coding": 32, "tools": 38, "finance": 32, "reasoning": 35, "general": 42}),
    ("nemotron-nano-9b",     {"coding": 30, "tools": 35, "finance": 30, "reasoning": 32, "general": 40}),
    ("nemotron-nano",        {"coding": 28, "tools": 32, "finance": 28, "reasoning": 30, "general": 38}),
    ("nemotron-super",       {"coding": 48, "tools": 50, "finance": 45, "reasoning": 52, "general": 55}),
    ("nova-premier",         {"coding": 55, "tools": 55, "finance": 55, "reasoning": 55, "general": 62}),
    ("nova-pro",             {"coding": 45, "tools": 50, "finance": 48, "reasoning": 48, "general": 55}),
    ("nova-2-lite",          {"coding": 38, "tools": 42, "finance": 38, "reasoning": 40, "general": 48}),
    ("nova-lite",            {"coding": 32, "tools": 38, "finance": 32, "reasoning": 35, "general": 42}),
    ("nova-micro",           {"coding": 22, "tools": 28, "finance": 22, "reasoning": 25, "general": 32}),
    ("command-r-plus",       {"coding": 38, "tools": 48, "finance": 42, "reasoning": 42, "general": 50}),
    ("command-r",            {"coding": 32, "tools": 42, "finance": 35, "reasoning": 35, "general": 45}),
    ("ernie-4.5-vl",         {"coding": 35, "tools": 40, "finance": 38, "reasoning": 38, "general": 45}),
    ("ernie-4.5",            {"coding": 40, "tools": 42, "finance": 42, "reasoning": 42, "general": 48}),
    ("seed-2.0-mini",        {"coding": 32, "tools": 35, "finance": 32, "reasoning": 35, "general": 42}),
    ("seed-2.0-lite",        {"coding": 30, "tools": 32, "finance": 30, "reasoning": 32, "general": 40}),
    ("seed-1.6-flash",       {"coding": 32, "tools": 35, "finance": 32, "reasoning": 35, "general": 42}),
    ("seed-1.6",             {"coding": 42, "tools": 45, "finance": 42, "reasoning": 45, "general": 50}),
    ("mimo-v2-pro",          {"coding": 38, "tools": 40, "finance": 35, "reasoning": 42, "general": 45}),
    ("mimo-v2-omni",         {"coding": 35, "tools": 38, "finance": 32, "reasoning": 38, "general": 42}),
    ("mimo-v2-flash",        {"coding": 28, "tools": 32, "finance": 28, "reasoning": 32, "general": 38}),
    ("step-3.5-flash",       {"coding": 32, "tools": 35, "finance": 30, "reasoning": 35, "general": 42}),
    ("trinity-large",        {"coding": 35, "tools": 38, "finance": 32, "reasoning": 38, "general": 42}),
    ("trinity-mini",         {"coding": 28, "tools": 30, "finance": 25, "reasoning": 30, "general": 35}),
    ("virtuoso-large",       {"coding": 42, "tools": 45, "finance": 42, "reasoning": 45, "general": 50}),
    ("jamba-large",          {"coding": 38, "tools": 42, "finance": 40, "reasoning": 40, "general": 48}),
    ("solar-pro",            {"coding": 38, "tools": 42, "finance": 38, "reasoning": 42, "general": 48}),
    ("olmo-3",               {"coding": 35, "tools": 38, "finance": 32, "reasoning": 38, "general": 42}),
    ("kat-coder",            {"coding": 50, "tools": 42, "finance": 28, "reasoning": 40, "general": 45}),
    ("intellect-3",          {"coding": 38, "tools": 38, "finance": 32, "reasoning": 40, "general": 42}),
    ("tongyi-deepresearch",  {"coding": 38, "tools": 45, "finance": 48, "reasoning": 55, "general": 50}),
    ("elephant",             {"coding": 35, "tools": 38, "finance": 32, "reasoning": 35, "general": 42}),
]

DEFAULT_SCORES = {"coding": 35, "tools": 35, "finance": 35, "reasoning": 35, "general": 40}

# Weighting presets. Each profile sums to 1.0.
PROFILES = {
    "balanced":  {"coding": 0.20, "tools": 0.20, "finance": 0.20, "reasoning": 0.20, "general": 0.20},
    "coding":    {"coding": 0.50, "tools": 0.25, "finance": 0.05, "reasoning": 0.15, "general": 0.05},
    "tools":     {"coding": 0.15, "tools": 0.55, "finance": 0.05, "reasoning": 0.15, "general": 0.10},
    "finance":   {"coding": 0.05, "tools": 0.15, "finance": 0.50, "reasoning": 0.20, "general": 0.10},
    "reasoning": {"coding": 0.15, "tools": 0.10, "finance": 0.15, "reasoning": 0.50, "general": 0.10},
    "agentic":   {"coding": 0.30, "tools": 0.40, "finance": 0.05, "reasoning": 0.15, "general": 0.10},
}

BLEND_PROMPT = 0.75
BLEND_COMPL = 0.25


def lookup_scores(model_id: str) -> dict[str, int]:
    mid = model_id.lower()
    for needle, scores in PERF_PROFILES:
        if needle in mid:
            return scores
    return DEFAULT_SCORES


def composite(scores: dict[str, int], weights: dict[str, float]) -> float:
    return sum(scores[k] * weights[k] for k in weights)


def blended_price_per_mtok(pricing: dict) -> float | None:
    try:
        p = float(pricing.get("prompt", "0") or 0)
        c = float(pricing.get("completion", "0") or 0)
    except (TypeError, ValueError):
        return None
    if p < 0 or c < 0:
        return None
    if p == 0 and c == 0:
        return 0.0
    return (p * BLEND_PROMPT + c * BLEND_COMPL) * 1_000_000


def load_api_key() -> str:
    env_path = Path.home() / ".sieg.env"
    if env_path.exists():
        key = dotenv_values(env_path).get("OPENROUTER_API_KEY")
        if key:
            return key
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    sys.exit("OPENROUTER_API_KEY not found in ~/.sieg.env or environment")


def fetch_models(api_key: str) -> list[dict]:
    resp = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def score_models(models: list[dict], weights: dict[str, float]) -> list[dict]:
    rows = []
    for m in models:
        if "tools" not in (m.get("supported_parameters") or []):
            continue
        price = blended_price_per_mtok(m.get("pricing") or {})
        if price is None:
            continue

        scores = lookup_scores(m["id"])
        perf = composite(scores, weights)

        if price == 0:
            ppd = float("inf")
        else:
            ppd = perf / price

        rows.append({
            "id": m["id"],
            "context": m.get("context_length") or 0,
            "price_per_mtok": price,
            "perf": round(perf, 1),
            "scores": scores,
            "perf_per_dollar": ppd,
        })

    rows.sort(key=lambda r: (-r["perf_per_dollar"], -r["perf"]))
    return rows


def format_table(rows: list[dict], limit: int | None, detailed: bool) -> str:
    if limit:
        rows = rows[:limit]
    if detailed:
        header = f"{'RANK':<5}{'MODEL':<50}{'COD':>5}{'TOL':>5}{'FIN':>5}{'RSN':>5}{'GEN':>5}{'PERF':>7}{'$/MTok':>11}{'PERF/$':>10}"
    else:
        header = f"{'RANK':<5}{'MODEL':<50}{'PERF':>7}{'$/MTok':>11}{'PERF/$':>10}{'CTX':>10}"
    out = [header, "-" * len(header)]
    for i, r in enumerate(rows, 1):
        price = "FREE" if r["price_per_mtok"] == 0 else f"${r['price_per_mtok']:.3f}"
        ppd = "∞" if r["perf_per_dollar"] == float("inf") else f"{r['perf_per_dollar']:.1f}"
        if detailed:
            s = r["scores"]
            out.append(
                f"{i:<5}{r['id']:<50}{s['coding']:>5}{s['tools']:>5}{s['finance']:>5}"
                f"{s['reasoning']:>5}{s['general']:>5}{r['perf']:>7.1f}{price:>11}{ppd:>10}"
            )
        else:
            out.append(
                f"{i:<5}{r['id']:<50}{r['perf']:>7.1f}{price:>11}{ppd:>10}{r['context']:>10,}"
            )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="balanced",
                    help="weighting preset (default: balanced)")
    ap.add_argument("--all", action="store_true", help="show all models (default: top 30)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--no-free", action="store_true", help="hide $0 models")
    ap.add_argument("--detailed", action="store_true", help="show per-dimension scores")
    args = ap.parse_args()

    weights = PROFILES[args.profile]
    models = fetch_models(load_api_key())
    rows = score_models(models, weights)

    if args.no_free:
        rows = [r for r in rows if r["price_per_mtok"] > 0]

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    limit = None if args.all else 30
    print(f"Profile: {args.profile}  weights={weights}")
    print(format_table(rows, limit, args.detailed))
    print(f"\n{len(rows)} tool-capable models scored. Perf 0-100 (approx leaderboard).")


if __name__ == "__main__":
    main()
