"""The agent loop, written by hand on the Claude Messages API (no agent framework).

One user turn:
  1. Send the conversation + tool specs to the model.
  2. If the model stops with stop_reason == "tool_use", run each requested tool and send the
     results back as tool_result blocks in a user message. Repeat.
  3. When it stops for any other reason, the text it produced is the answer.
  4. Run the grounding check on that answer.

The client is injected, so tests can drive the loop with a fake model and no network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import grounding
from .kb import KnowledgeBase
from .prompts import SYSTEM_PROMPT
from .tools import ToolRegistry

DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass
class Reply:
    text: str
    stop_reason: str
    steps: int
    tool_calls: list[dict] = field(default_factory=list)
    report: grounding.GroundingReport | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def _get(block: Any, key: str, default=None):
    """Read a field from an SDK object or a plain dict (tests use dicts)."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


class Agent:
    def __init__(
        self,
        client: Any,
        kb: KnowledgeBase | None = None,
        model: str = DEFAULT_MODEL,
        max_steps: int = 8,
        max_tokens: int = 1500,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.client = client
        self.kb = kb or KnowledgeBase.load()
        self.tools = ToolRegistry(self.kb)
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.on_event = on_event or (lambda kind, data: None)
        self.messages: list[dict] = []  # full conversation, kept across turns

    def reset(self) -> None:
        self.messages = []

    def ask(self, question: str) -> Reply:
        """Answer one question. If the API call fails mid-turn, the history is rolled back and the error re-raised."""
        checkpoint = len(self.messages)
        try:
            return self._ask(question)
        except Exception:
            del self.messages[checkpoint:]
            raise

    def _ask(self, question: str) -> Reply:
        start = time.monotonic()
        self.tools.reset_turn()
        self.messages.append({"role": "user", "content": question})
        reply = Reply(text="", stop_reason="", steps=0)

        for step in range(1, self.max_steps + 1):
            reply.steps = step
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=self.tools.specs(),
                messages=self.messages,
            )
            usage = _get(response, "usage")
            if usage is not None:
                reply.input_tokens += _get(usage, "input_tokens", 0) or 0
                reply.output_tokens += _get(usage, "output_tokens", 0) or 0

            content = list(_get(response, "content", []) or [])
            stop = _get(response, "stop_reason", "")
            # Keep the assistant turn exactly as returned (including any thinking blocks).
            self.messages.append({"role": "assistant", "content": content})

            if stop != "tool_use":
                reply.stop_reason = stop
                reply.text = "".join(_get(b, "text", "") for b in content if _get(b, "type") == "text").strip()
                if stop == "max_tokens":
                    reply.text += "\n\n[Answer cut off at the token limit.]"
                elif stop == "refusal" and not reply.text:
                    reply.text = "I can't help with that one. Please ask about Harsha's professional experience."
                break

            # Run every tool the model asked for, then answer all of them in one user message.
            results = []
            for block in content:
                if _get(block, "type") != "tool_use":
                    continue
                name, args, use_id = _get(block, "name"), _get(block, "input") or {}, _get(block, "id")
                self.on_event("tool_call", {"name": name, "input": args})
                output, is_error = self.tools.call(name, args)
                self.on_event("tool_result", {"name": name, "is_error": is_error, "output": output})
                reply.tool_calls.append({"name": name, "input": args, "is_error": is_error})
                result = {"type": "tool_result", "tool_use_id": use_id, "content": output}
                if is_error:
                    result["is_error"] = True
                results.append(result)
            if not results:  # stop_reason said tool_use but no tool_use blocks: end safely
                reply.stop_reason = "no_tool_blocks"
                break
            self.messages.append({"role": "user", "content": results})
        else:
            reply.stop_reason = "max_steps"
            reply.text = "I couldn't finish looking that up. Please try a narrower question."
            # Leave the history valid: the last assistant turn asked for tools we already answered,
            # so close the turn with a short assistant note.
            self.messages.append({"role": "assistant", "content": [{"type": "text", "text": reply.text}]})

        reply.report = grounding.check(reply.text, set(self.tools.retrieved), self.kb)
        reply.seconds = round(time.monotonic() - start, 2)
        self.on_event("done", {"report": reply.report.summary(), "steps": reply.steps})
        return reply
