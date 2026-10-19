"""Assist mode: help someone who can't speak clearly, or can't speak at all, say what they mean.

The phone does the listening (its own speech recognition) and the talking (its own voice). This module
does the part in between: it turns what the phone heard, or a few typed words, into up to three full
sentences the person might mean. The person taps the right one and the phone says it out loud.

Inputs it can use:
  heard    what speech recognition heard; it's often wrong for atypical speech, so several alternatives help
  typed    a few words or shorthand ("water cold"), misspellings welcome
  partner  what the other person just said, so the guesses can be replies
  past     this person's earlier picks (what was heard -> the sentence they chose), kept on their device only
  saved    phrases they saved

Nothing here is stored or logged. Guesses are cleaned up in code: at most three, short, no duplicates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .factcheck import FactCheckError, parse_json
from .loop import _get

MAX_GUESSES = 3
MAX_GUESS_CHARS = 200

ASSIST_PROMPT = """You help a person who can't speak clearly, or can't speak at all, say what they mean. Their phone will say your sentence out loud for them to the people around them.

You get some of these:
- HEARD: what the phone's speech recognition heard. For this person it is often wrong: words that sound alike, missing words, garbled syllables. Several alternatives may be listed.
- TYPED: a few words or shorthand they typed, maybe misspelled.
- PARTNER: what the other person just said to them.
- PAST: sentences this person chose before, with what was heard or typed at the time. These show how their speech gets misheard and what they usually want to say.
- SAVED: phrases they saved.

Give up to {max_guesses} guesses at the full sentence they want to say, most likely first.

Rules:
- Write as them: first person, short, natural, the way they'd say it out loud. Under 20 words.
- Stay close to what was heard or typed. Never add names, places, times, amounts or details they didn't give.
- Speech recognition mishears this person, so think about words that sound similar. If what was heard is close to one of their PAST or SAVED sentences, put that sentence first.
- With only PARTNER and nothing heard or typed, suggest short replies that go in different directions (for example yes, no, and something else).
- If one reading is an urgent need (pain, help, can't breathe, fell, medicine, bathroom, emergency), include it and set urgent to true.
- Use the language they used.
- You only put their words into a sentence. Never answer them, give advice, or add your own comments.
- If nothing sensible can be made from the input, return an empty list.

Reply with only JSON in exactly this shape:
{{"guesses": [{{"text": "...", "urgent": false}}]}}"""


@dataclass
class AssistResult:
    guesses: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def _clip(s: Any, n: int) -> str:
    return " ".join(str(s or "").split())[:n]


def build_message(heard: list[str], typed: str, partner: str, past: list[dict], saved: list[str]) -> str:
    parts = []
    heard = [h for h in (_clip(x, 300) for x in heard) if h][:5]
    if heard:
        parts.append("HEARD:\n" + "\n".join(f"- {h}" for h in heard))
    if _clip(typed, 300):
        parts.append(f"TYPED: {_clip(typed, 300)}")
    if _clip(partner, 500):
        parts.append(f"PARTNER: {_clip(partner, 500)}")
    pairs = [(_clip(p.get("input"), 200), _clip(p.get("said"), MAX_GUESS_CHARS)) for p in past if isinstance(p, dict)]
    pairs = [(i, s) for i, s in pairs if s][-20:]
    if pairs:
        parts.append("PAST:\n" + "\n".join(f'- "{i}" -> "{s}"' if i else f'- "{s}"' for i, s in pairs))
    saved = [s for s in (_clip(x, MAX_GUESS_CHARS) for x in saved) if s][:30]
    if saved:
        parts.append("SAVED:\n" + "\n".join(f"- {s}" for s in saved))
    return "\n\n".join(parts)


def clean_guesses(data: dict) -> list[dict]:
    out, seen = [], set()
    for g in data.get("guesses") or []:
        if isinstance(g, str):
            g = {"text": g}
        if not isinstance(g, dict):
            continue
        text = _clip(g.get("text"), MAX_GUESS_CHARS)
        key = text.lower().rstrip(".!?")
        if not text or key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "urgent": bool(g.get("urgent"))})
        if len(out) == MAX_GUESSES:
            break
    return out


def interpret(client: Any, model: str, heard: list[str] | None = None, typed: str = "", partner: str = "",
              past: list[dict] | None = None, saved: list[str] | None = None) -> AssistResult:
    message = build_message(heard or [], typed, partner, past or [], saved or [])
    if not message or not any(k in message for k in ("HEARD:", "TYPED:", "PARTNER:")):
        return AssistResult()
    resp = client.messages.create(
        model=model, max_tokens=400,
        system=ASSIST_PROMPT.format(max_guesses=MAX_GUESSES),
        messages=[{"role": "user", "content": message}],
    )
    usage = _get(resp, "usage")
    res = AssistResult(input_tokens=(_get(usage, "input_tokens", 0) or 0) if usage is not None else 0,
                       output_tokens=(_get(usage, "output_tokens", 0) or 0) if usage is not None else 0)
    if _get(resp, "stop_reason") == "refusal":
        return res
    text = "".join(_get(b, "text", "") for b in (_get(resp, "content", []) or []) if _get(b, "type") == "text")
    try:
        res.guesses = clean_guesses(parse_json(text))
    except FactCheckError:
        res.guesses = []
    return res
