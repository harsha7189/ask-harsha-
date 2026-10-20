"""The agent loop, written by hand on the Claude Messages API (no agent framework).

One user turn:
  1. Send the conversation + tool specs to the model.
  2. If the model stops with stop_reason == "tool_use", run each requested client tool and send the
     results back as tool_result blocks in a user message. Repeat.
  3. If it stops with "pause_turn" (a long server-side web search), send the conversation back
     unchanged so the model can continue.
  4. Any other stop reason ends the turn; the text it produced is the answer.
  5. Run the grounding check on that answer.

Server tools such as web_search run on Anthropic's side: their blocks arrive inside the assistant
message and are passed back untouched. The client is injected, so tests can drive the loop with a
fake model and no network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import grounding, speed
from .kb import KnowledgeBase
from .prompts import SYSTEM_PROMPT
from .tools import ToolRegistry

DEFAULT_MODEL = "claude-sonnet-5-5"
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}


@dataclass
class Reply:
    text: str
    stop_reason: str
    steps: int
    tool_calls: list[dict] = field(default_factory=list)
    steps_log: list[dict] = field(default_factory=list)      # what the loop did, in order, for "How I got this"
    report: grounding.GroundingReport | None = None
    web_sources: list[dict] = field(default_factory=list)   # [{"url", "title"}] from web search citations
    web_searches: int = 0
    memory_changes: list[dict] = field(default_factory=list)
    input_tokens: int = 0         # billed-equivalent: cache writes count 1.25x and cache reads 0.1x
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    seconds: float = 0.0
    timing: dict = field(default_factory=dict)   # where the time went: first words, each model call, tools
    fallback: str = ""            # the backup model's name, if the main one was unavailable for this answer


RETRY_PAUSE = 1.5
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def retryable(e: Exception) -> bool:
    """Worth retrying: the model was busy, rate limited or unreachable. Not: a bad request or a bad key."""
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status in RETRY_STATUS
    name = type(e).__name__
    return any(w in name for w in ("Overloaded", "RateLimit", "Connection", "Timeout", "InternalServer", "ServiceUnavailable"))


def strip_thinking(message: dict) -> dict:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    kept = [b for b in content if _get(b, "type") not in ("thinking", "redacted_thinking")]
    return {**message, "content": kept or [{"type": "text", "text": "(continued)"}]}


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
        system: str | None = None,
        server_tools: list[dict] | None = None,
        lookup: Any = None,
        strict_grounding: bool = True,
        stream: bool = False,
        record_tools: bool = True,
        speed: str = "chat",
        container: str | None = None,
        fallback_model: str | None = None,
    ):
        self.client = client
        self.kb = kb or KnowledgeBase.load()
        self.tools = ToolRegistry(self.kb)
        if not record_tools:          # e.g. the PM toolkit: no access to Harsha's record, just the model (and web search)
            self.tools._tools.clear()
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.on_event = on_event or (lambda kind, data: None)
        self.system = system or SYSTEM_PROMPT
        self.context = ""             # per-person text (settings, memory, project, chat summary) after the cached prompt
        self.cache_system = True      # cache tools + system prompt: later calls and steps reuse them at a tenth of the price
        self.server_tools = server_tools or []
        self.lookup = lookup          # object with .by_id covering facts (+ notes); defaults to the KB
        self.strict_grounding = strict_grounding
        self.stream = stream            # stream text deltas through on_event("text", ...)
        self.speed = speed              # how hard the model thinks: chat | deep | helper (see agent/speed.py)
        self.container_id = container   # code execution sandbox to reuse across calls (files carry over)
        self.fallback_model = fallback_model   # used for a call when the main model is overloaded or unreachable
        self.used_fallback = ""
        self.speed_override: dict = {}   # Tune's chat settings (effort, thinking); env vars still win
        self._first_text: float | None = None
        self.messages: list[dict] = []  # full conversation, kept across turns

    def system_param(self):
        """The system prompt as blocks: the stable part marked for caching, then the per-person context."""
        if not self.cache_system:
            return self.system + self.context
        blocks = [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]
        if self.context:
            blocks.append({"type": "text", "text": self.context})
        return blocks

    def reset(self) -> None:
        self.messages = []

    def _call(self, kwargs: dict):
        """One model call, resilient to a busy or unreachable model.

        A transient failure (overloaded, rate limited, 5xx, connection or timeout) is retried once after a short pause.
        If it fails again and a backup model is set, the call is made with the backup model instead, and the reply says
        so. Anything else (a bad request, an auth error) is raised straight away.
        """
        try:
            return self._call_once(kwargs)
        except Exception as e:
            if not retryable(e):
                raise
            first = e
        time.sleep(RETRY_PAUSE)
        self.on_event("text_reset", {})   # anything streamed by the failed attempt isn't part of the answer
        try:
            return self._call_once(kwargs)
        except Exception as e:
            if not retryable(e) or not self.fallback_model or self.fallback_model == kwargs.get("model"):
                raise
        self.on_event("text_reset", {})
        self.on_event("fallback", {"model": self.fallback_model, "reason": type(first).__name__})
        backup = {k: v for k, v in kwargs.items() if k not in ("output_config", "thinking")}
        backup.update(model=self.fallback_model, **speed.params(self.fallback_model, self.speed, self.speed_override))
        backup["messages"] = [strip_thinking(m) for m in kwargs["messages"]]   # thinking blocks belong to the model that wrote them
        self.used_fallback = self.fallback_model
        return self._call_once(backup)

    def _call_once(self, kwargs: dict):
        if not self.stream:
            return self.client.messages.create(**kwargs)
        with self.client.messages.stream(**kwargs) as s:
            for ev in s:
                kind = _get(ev, "type")
                if kind == "text":
                    delta = _get(ev, "text", "")
                    if delta:
                        if self._first_text is None:
                            self._first_text = time.monotonic()
                        self.on_event("text", {"delta": delta})
                elif kind == "content_block_start":
                    block = _get(ev, "content_block")
                    if _get(block, "type") == "server_tool_use" and _get(block, "name") == "web_search":
                        self.on_event("web_search_start", {})
                    elif _get(block, "type") == "tool_use":
                        self.on_event("tool_start", {"name": _get(block, "name")})   # e.g. "Writing your file" while it's written
                    elif _get(block, "type") == "server_tool_use":
                        self.on_event("server_tool_start", {"name": _get(block, "name")})   # e.g. code running in the sandbox
            return s.get_final_message()

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
        self._first_text = None
        model_s, tools_s, calls = 0.0, 0.0, []
        self.tools.reset_turn()
        self.used_fallback = ""
        self.messages.append({"role": "user", "content": question})
        reply = Reply(text="", stop_reason="", steps=0)
        text_parts: list[str] = []   # answer text since the last client-tool round
        seen_urls: set[str] = set()

        for step in range(1, self.max_steps + 1):
            reply.steps = step
            call = dict(model=self.model, max_tokens=self.max_tokens, system=self.system_param(), messages=self.messages,
                        **speed.params(self.model, self.speed, self.speed_override))
            tools = self.tools.specs() + self.server_tools
            if tools:
                call["tools"] = tools
            if self.container_id:
                call["container"] = self.container_id
            t0 = time.monotonic()
            response = self._call(call)
            calls.append(round(time.monotonic() - t0, 2))
            model_s += calls[-1]
            usage = _get(response, "usage")
            if usage is not None:
                cw = _get(usage, "cache_creation_input_tokens", 0) or 0
                cr = _get(usage, "cache_read_input_tokens", 0) or 0
                reply.cache_write_tokens += cw
                reply.cache_read_tokens += cr
                reply.input_tokens += (_get(usage, "input_tokens", 0) or 0) + round(cw * 1.25 + cr * 0.1)
                reply.output_tokens += _get(usage, "output_tokens", 0) or 0
                stu = _get(usage, "server_tool_use")
                if stu is not None:
                    reply.web_searches += _get(stu, "web_search_requests", 0) or 0

            content = list(_get(response, "content", []) or [])
            stop = _get(response, "stop_reason", "")
            box = _get(response, "container")
            if box is not None and _get(box, "id"):
                self.container_id = _get(box, "id")
            self.on_event("response_content", {"content": content})
            # Keep the assistant turn exactly as returned (thinking blocks, server tool blocks, citations).
            self.messages.append({"role": "assistant", "content": content})

            for b in content:
                if _get(b, "type") == "server_tool_use" and _get(b, "name") == "web_search":
                    q = (_get(b, "input") or {}).get("query", "") if isinstance(_get(b, "input"), dict) else ""
                    reply.steps_log.append({"kind": "web_search", "query": q})
                if _get(b, "type") != "text":
                    continue
                text_parts.append(_get(b, "text", ""))
                for c in _get(b, "citations", None) or []:
                    url = _get(c, "url")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        reply.web_sources.append({"url": url, "title": _get(c, "title") or url})

            if stop == "pause_turn":
                continue  # the history now ends with the paused assistant turn; sending it back resumes it

            if stop != "tool_use":
                reply.stop_reason = stop
                reply.text = "".join(text_parts).strip()
                if stop == "max_tokens":
                    reply.text += "\n\n[Answer cut off at the length limit.]"
                elif stop == "refusal" and not reply.text:
                    reply.text = "I can't help with that one."
                break

            # Run every client tool the model asked for, then answer all of them in one user message.
            results = []
            for block in content:
                if _get(block, "type") != "tool_use":
                    continue
                name, args, use_id = _get(block, "name"), _get(block, "input") or {}, _get(block, "id")
                self.on_event("tool_call", {"name": name, "input": args})
                before = set(self.tools.retrieved)
                t1 = time.monotonic()
                output, is_error = self.tools.call(name, args)
                tools_s += time.monotonic() - t1
                self.on_event("tool_result", {"name": name, "is_error": is_error, "output": output})
                reply.tool_calls.append({"name": name, "input": args, "is_error": is_error})
                reply.steps_log.append({
                    "kind": name,
                    "query": str(args.get("query") or args.get("topic") or args.get("text") or args.get("id") or args.get("filename") or "")[:200] if isinstance(args, dict) else "",
                    "found": sorted(self.tools.retrieved - before),
                    "error": output[:200] if is_error else None,
                })
                result = {"type": "tool_result", "tool_use_id": use_id, "content": output}
                if is_error:
                    result["is_error"] = True
                results.append(result)
            if not results:  # stop_reason said tool_use but no client tool blocks: end safely
                reply.stop_reason = "no_tool_blocks"
                reply.text = "".join(text_parts).strip()
                break
            self.messages.append({"role": "user", "content": results})
            if any(t.strip() for t in text_parts):
                self.on_event("text_reset", {})  # streamed narration before a tool call isn't part of the answer
            text_parts = []
        else:
            reply.stop_reason = "max_steps"
            reply.text = "I couldn't finish working that out. Please try a narrower question."
            if self.messages[-1]["role"] == "user":
                self.messages.append({"role": "assistant", "content": [{"type": "text", "text": reply.text}]})

        reply.memory_changes = list(self.tools.changes)
        reply.fallback = self.used_fallback
        reply.report = grounding.check(reply.text, set(self.tools.retrieved), self.lookup or self.kb, strict=self.strict_grounding)
        reply.seconds = round(time.monotonic() - start, 2)
        reply.timing = {"total_s": reply.seconds, "model_s": round(model_s, 2), "tools_s": round(tools_s, 2), "calls_s": calls,
                        "first_text_s": round(self._first_text - start, 2) if self._first_text else None}
        self.on_event("done", {"report": reply.report.summary(), "steps": reply.steps})
        return reply
