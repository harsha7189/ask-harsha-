"""The context layer's three memories, all written by a small, fast model:

- Rolling summary: when a chat gets long, older turns are condensed into a short summary instead of being dropped.
- Personal memory: durable things a signed-in person tells the agent about themselves ("I'm a PM in Austin"),
  shown and editable in Customize, and off with one switch.
- Project memory: each project keeps a running "what we know so far" note that every chat in it reads and updates.

One small model call learns both memories from an exchange; it's skipped when there's clearly nothing to learn.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from . import speed
from .factcheck import FactCheckError, parse_json

SUMMARY_AT = 8        # once a chat has more than this many exchanges in play…
KEEP_RECENT = 4       # …older ones are folded into the summary and only this many stay word for word
MAX_SUMMARY = 1500
MAX_MEMORIES = 60
MAX_MEMORY_CHARS = 200
MAX_PROJECT_NOTES = 1500

SUMMARY_PROMPT = """You keep the running summary of a conversation between a person and the Ask Harsha assistant, so \
the assistant can continue it after older messages are dropped. Merge the previous summary with the messages below. \
Keep what the person wants, facts and constraints they gave, decisions made, open questions, and anything the assistant \
offered or promised. Drop small talk. Plain sentences, under 180 words, no preamble."""

LEARN_PROMPT = """You maintain two small memories for the Ask Harsha app, using one exchange between a person and the \
assistant. Return only JSON: {"remember": [..], "forget": [..], "project_notes": string or null}.

remember: durable facts THE PERSON stated about themselves in their own message: role, goals, where they live, \
ongoing work, interests, how they like answers. Each item is {"text": third person, one short line ("Works as a \
product manager at IBM"), "quote": the exact words from the PERSON's message it comes from, copied character for \
character}. At most 3. Nothing already in MEMORIES. Never anything from the assistant's answer, and never a guess or \
inference: if they didn't say it, leave it out. Skip one-off details, anything about other people, and sensitive \
things: health, money amounts, relationships, religion, politics, IDs, passwords, exact addresses.
forget: ids from MEMORIES that the person just contradicted or asked you to forget.
project_notes: only when a PROJECT is given. Rewrite its notes so they also hold anything new and useful from this \
exchange (decisions, findings, constraints, next steps), as short lines starting with "- ", under 1200 characters. \
Keep earlier notes unless they're now wrong. Record what the person decided or stated as fact; record the \
assistant's findings only as "(from an answer)" and never anything listed under DO NOT RECORD. Use null if nothing new \
is worth keeping.
If nothing applies: {"remember": [], "forget": [], "project_notes": null}"""

FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|i'd|i'll|my|me|mine|we|we're|our|us)\b", re.I)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, resp: Any) -> None:
        u = _get(resp, "usage")
        if u is not None:
            self.input_tokens += _get(u, "input_tokens", 0) or 0
            self.output_tokens += _get(u, "output_tokens", 0) or 0


STOP = set("""a an and are as at be been but by can could did do does for from had has have he her his how i i'm im in \
into is it its just like me my of on or our she so that the their them they this to too was we were what when where \
which who will with would you your works work lives live likes like wants want is are has have going planning""".split())


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) >= 3 and w not in STOP]


def stated_by_person(memory_text: str, quote: str, message: str) -> str | None:
    """The person's own words this memory comes from, or None if it can't be traced to their message.

    A model can't be trusted to only record what was said, so code checks: the quote has to really be in their message
    (when one is given), and the memory has to share its key words with those words. Guesses, things the assistant
    said and invented details fail this check and are never saved.
    """
    from .trust import quote_in, squash
    src = quote if quote and quote_in(quote, message) else message
    mine = _words(memory_text)
    theirs = {w[:5] for w in _words(src)}
    if not mine:
        return None
    shared = [w for w in mine if w[:5] in theirs]
    if not shared or len(shared) / len(mine) < 0.25:
        return None
    if src is message:   # no usable quote: keep the sentence of their message that holds the shared words
        for sent in re.split(r"(?<=[.!?])\s+|\n+", message):
            if any(w[:5] in {x[:5] for x in _words(sent)} for w in shared):
                return " ".join(sent.split())[:200]
        return None
    return " ".join(quote.split())[:200]


@dataclass
class Learned:
    remember: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)   # the person's own words behind each remembered line
    rejected: list[str] = field(default_factory=list)  # what the model proposed that they never said
    forget: list[str] = field(default_factory=list)
    project_notes: str | None = None
    usage: Usage = field(default_factory=Usage)


def _text(resp: Any) -> str:
    return "".join(_get(b, "text", "") for b in (_get(resp, "content", []) or []) if _get(b, "type") == "text").strip()


def _transcript(messages: list[dict], limit: int = 1500) -> str:
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        who = "Person" if m.get("role") == "user" else "Assistant"
        out.append(f"{who}: {str(content or '')[:limit]}")
    return "\n\n".join(out)


def condense(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split alternating history into (older turns to summarize, recent turns to keep). Nothing to fold → ([], all)."""
    if len(messages) // 2 <= SUMMARY_AT:
        return [], messages
    cut = len(messages) - KEEP_RECENT * 2
    return messages[:cut], messages[cut:]


def summarize(client: Any, model: str, previous: str, older: list[dict], usage: Usage | None = None) -> str:
    resp = client.messages.create(
        model=model, max_tokens=400, system=SUMMARY_PROMPT, **speed.params(model, "helper"),
        messages=[{"role": "user", "content": f"PREVIOUS SUMMARY:\n{previous.strip() or '(none)'}\n\nMESSAGES:\n{_transcript(older)}"}])
    if usage is not None:
        usage.add(resp)
    return _text(resp)[:MAX_SUMMARY] or previous


def worth_learning(question: str, project: dict | None, memory_on: bool) -> bool:
    """Skip the model call when there's clearly nothing to learn."""
    return bool(project) or (memory_on and bool(FIRST_PERSON.search(question)))


def learn(client: Any, model: str, question: str, answer: str, memories: list[dict], project: dict | None,
          memory_on: bool = True, avoid: list[str] | None = None) -> Learned:
    mem = "\n".join(f"{m['id']}: {m['text']}" for m in memories) or "(none)"
    proj = (f"PROJECT: {project.get('name', '')}\nINSTRUCTIONS: {str(project.get('instructions') or '')[:600]}\n"
            f"CURRENT NOTES:\n{project.get('notes') or '(none yet)'}") if project else "PROJECT: none"
    if avoid:   # claims the answer's trust check found wrong or couldn't verify
        proj += "\n\nDO NOT RECORD (found wrong or unverified):\n" + "\n".join(f"- {a[:200]}" for a in avoid[:10])
    resp = client.messages.create(
        model=model, max_tokens=700, system=LEARN_PROMPT, **speed.params(model, "helper"),
        messages=[{"role": "user", "content": f"MEMORIES:\n{mem}\n\n{proj}\n\nPERSON: {question[:2000]}\n\nASSISTANT: {answer[:2500]}"}])
    out = Learned()
    out.usage.add(resp)
    try:
        d = parse_json(_text(resp))
    except (FactCheckError, ValueError, json.JSONDecodeError):
        return out
    if not isinstance(d, dict):
        return out
    known = {m["id"] for m in memories}
    existing = {m["text"].strip().lower() for m in memories}
    if memory_on:
        for x in d.get("remember") or []:
            quote = ""
            if isinstance(x, dict):
                x, quote = x.get("text", ""), str(x.get("quote") or "")
            x = " ".join(str(x).split())[:MAX_MEMORY_CHARS]
            if not x or x.lower() in existing or len(out.remember) >= 3:
                continue
            said = stated_by_person(x, quote, question)
            if said is None:
                out.rejected.append(x)
                continue
            out.remember.append(x)
            out.quotes.append(said)
            existing.add(x.lower())
        out.forget = [str(x) for x in (d.get("forget") or []) if str(x) in known]
    notes = d.get("project_notes")
    if project and isinstance(notes, str) and notes.strip():
        out.project_notes = notes.strip()[:MAX_PROJECT_NOTES]
    return out


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
