"""How hard each model call thinks, which is the main dial for response time.

Claude Sonnet 5.5 defaults to high effort with adaptive thinking: it may reason privately before writing a single
word, which is right for hard problems but makes chat feel slow. Anthropic recommends medium or low effort for chat,
and `thinking: between_tools` to skip the up-front thinking (it can still think between tool calls). Claude Haiku 5.5
thinks by default too; for our short helper jobs (Assist guesses, commands, summaries, memory, fact-check claims) low
effort with thinking off is fast and plenty.

Kinds:
  chat   - answers in the chat and over MCP: no up-front thinking, medium effort (CHAT_EFFORT / CHAT_THINKING override)
  deep   - PM Toolkit and other long work: medium effort, adaptive thinking
  helper - short structured jobs: low effort, the least thinking the model allows
Unknown models get no extra fields, so switching models by env var never breaks a request.
"""
from __future__ import annotations

import os

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def params(model: str, kind: str = "chat", override: dict | None = None) -> dict:
    """`override` comes from Tune's settings (agent/tuned.json): {"effort": ..., "thinking": ...}. Env vars still win."""
    override = override or {}
    m = (model or "").lower()
    if m.startswith("claude-haiku-5-5"):
        if kind == "helper":
            return {"output_config": {"effort": "low"}, "thinking": {"type": "disabled"}}
        if kind == "chat":
            return {"output_config": {"effort": "low"}}
        return {"output_config": {"effort": "medium"}}
    if m.startswith("claude-sonnet-5-5"):
        if kind == "chat":
            effort = (os.environ.get("CHAT_EFFORT") or override.get("effort") or "medium").lower()
            effort = effort if effort in EFFORTS else "medium"
            thinking = (os.environ.get("CHAT_THINKING") or override.get("thinking") or "between_tools").lower()
            if effort in ("xhigh", "max") or thinking not in ("between_tools", "adaptive"):
                thinking = "adaptive"   # between_tools isn't allowed at xhigh/max
            return {"output_config": {"effort": effort}, "thinking": {"type": thinking}}
        if kind == "helper":
            return {"output_config": {"effort": "low"}, "thinking": {"type": "between_tools"}}
        return {"output_config": {"effort": "medium"}}
    return {}
