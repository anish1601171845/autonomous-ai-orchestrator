"""Minimal OpenAI-compatible chat client (works with OpenAI, Azure-compatible gateways, Ollama, vLLM, LM Studio).

If LLM_API_KEY is empty the agents use their deterministic built-in backends instead; every
output records which backend produced it.
"""
from __future__ import annotations

import json
import re
import time

import httpx

from app.core.config import settings
from app.core.security import get_logger, redact

log = get_logger(__name__)


class LLMUnavailable(RuntimeError):
    pass


def enabled() -> bool:
    return bool(settings.llm_api_key)


USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def chat(messages: list[dict], temperature: float | None = None, retries: int = 2) -> str:
    """Send a chat completion. Temperature is only sent when LLM_TEMPERATURE (or the argument) is set,
    because newer reasoning models (e.g. OpenAI GPT-5/6 families) reject non-default temperatures."""
    if not enabled():
        raise LLMUnavailable("LLM_API_KEY not configured")
    import os

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    body: dict = {"model": settings.llm_model, "messages": messages}
    temp = temperature if temperature is not None else (float(os.environ["LLM_TEMPERATURE"]) if os.getenv("LLM_TEMPERATURE") else None)
    if temp is not None:
        body["temperature"] = temp
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout_s)
            if r.status_code == 400 and "temperature" in r.text and "temperature" in body:
                body.pop("temperature")  # model only supports its default temperature: retry without it
                r = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"retryable {r.status_code}", request=r.request, response=r)
            if r.status_code in (401, 403, 404):
                raise LLMUnavailable(f"LLM request rejected ({r.status_code}): check LLM_API_KEY / LLM_MODEL. {redact(r.text[:200])}")
            r.raise_for_status()
            data = r.json()
            u = data.get("usage") or {}
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += int(u.get("prompt_tokens", 0))
            USAGE["completion_tokens"] += int(u.get("completion_tokens", 0))
            return data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            last = e
            log.warning("LLM call failed (attempt %d): %s", attempt + 1, redact(str(e)))
            time.sleep(min(2 ** attempt, 8))
    raise LLMUnavailable(f"LLM request failed after retries: {redact(str(last))}")


def extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    raw = m.group(1) if m else text
    start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0], default=0)
    return json.loads(raw[start:])


def extract_code_blocks(text: str) -> dict[str, str]:
    """Parse blocks of the form ```python filename.py\n...``` into {filename: code}."""
    out = {}
    for name, code in re.findall(r"```python\s+([\w\-/]+\.py)\s*\n(.*?)```", text, re.S):
        out[name.split("/")[-1]] = code.strip() + "\n"
    return out
