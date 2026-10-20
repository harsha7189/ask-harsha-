"""Talk to the sidebar: turn a plain-words command into one safe app action.

"open my job search", "answers I saved about pricing", "make my answers shorter", "start a trip project".

Common commands are matched by simple rules first (instant, free). Anything else goes to a small, fast model that
must pick exactly one action from a fixed list by calling a tool; ids it returns are checked against what the app
sent. Nothing here can delete anything or sign anyone out: those stay as buttons with a second tap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import speed

SECTIONS = ("projects", "saved", "tools", "customize", "chats")
TOOLS = ("morning", "meeting", "feedback", "interview", "release", "spec", "assist", "lab")
PREF_VALUES = {
    "length": ("short", "balanced", "detailed"),
    "tone": ("", "friendly", "professional", "direct"),
    "read_aloud": (True, False),
    "jury": (True, False),
    "proof": (True, False),
}
ACTIONS = ("open_section", "open_project", "open_chat", "search_chats", "search_saved", "new_chat", "create_project",
           "set_pref", "open_tool", "ask", "sign_in", "channel_up", "channel_down", "reply")

TOOL_SPEC = {
    "name": "do",
    "description": "Do exactly one thing in the Ask Harsha app for the person.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS), "description": (
                "open_section: show a sidebar section. open_project: tune to one of their projects. open_chat: open one past chat. "
                "search_chats / search_saved: find chats or saved answers about a topic. new_chat: start a chat (optionally in a "
                "project). create_project: make a new project. set_pref: change a setting. open_tool: open a PM tool or Assist mode. "
                "ask: they want to ask the AI a question now. sign_in: they want to sign in or create an account. channel_up / "
                "channel_down: next or previous channel. reply: nothing fits; answer in one short line.")},
            "section": {"type": "string", "enum": list(SECTIONS)},
            "project_id": {"type": "string", "description": "An id from PROJECTS, exactly as given."},
            "chat_id": {"type": "string", "description": "An id from CHATS, exactly as given."},
            "query": {"type": "string", "description": "Search words, or the question for ask."},
            "name": {"type": "string", "description": "Name for a new project."},
            "icon": {"type": "string", "description": "One emoji for a new project."},
            "instructions": {"type": "string", "description": "Instructions for a new project, if they gave any."},
            "pref": {"type": "string", "enum": ["length", "tone", "read_aloud", "jury", "proof", "name", "city", "about"]},
            "value": {"description": "length: short|balanced|detailed. tone: ''|friendly|professional|direct. read_aloud/jury: true|false. name/city/about: text."},
            "tool": {"type": "string", "enum": list(TOOLS)},
            "say": {"type": "string", "description": "One short, friendly line confirming what you did (under 12 words)."},
        },
        "required": ["action", "say"],
    },
}

PROMPT = """You are the remote control inside the Ask Harsha app's sidebar. The person types or says a short command and you \
do exactly one thing by calling the `do` tool. Pick ids only from the lists below and copy them exactly. If they name a \
project or chat loosely ("my job one", "the Lisbon chat"), pick the closest match. If they ask a real question that \
isn't about the app ("what's the capital of Peru"), use ask with their question. If nothing fits, use reply. You can't \
delete things or sign people out; for that, reply that it's under Customize. Signed in: {signed_in}.

PROJECTS (id: name):
{projects}

CHATS (id: title [project]):
{chats}

SAVED ANSWERS (question):
{saved}

SETTINGS NOW: {prefs}"""


@dataclass
class Command:
    action: str = "reply"
    args: dict = field(default_factory=dict)
    say: str = ""
    source: str = "rules"
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {"action": self.action, "args": self.args, "say": self.say, "source": self.source}


def _norm(t: str) -> str:
    return re.sub(r"[^\w\s']", " ", t.lower()).strip()


def quick(text: str, context: dict) -> Command | None:
    """Rules for the commands people use most. Returns None when the model should decide."""
    t = " ".join(_norm(text).split())
    if not t:
        return None
    words = set(t.split())
    if len(t.split()) > 7:   # longer sentences are usually questions or detailed requests: let the model decide
        return None

    def has(*phrases):
        return any(re.search(r"\b" + re.escape(p) + r"\b", t) for p in phrases)

    # settings
    if has("shorter", "short answers", "less detail", "be brief", "briefer", "too long", "concise"):
        return Command("set_pref", {"pref": "length", "value": "short"}, "Answers set to Short.")
    if has("longer", "more detail", "detailed answers", "in depth", "too short", "elaborate"):
        return Command("set_pref", {"pref": "length", "value": "detailed"}, "Answers set to Detailed.")
    if has("normal length", "balanced"):
        return Command("set_pref", {"pref": "length", "value": "balanced"}, "Answers set to Balanced.")
    if has("proof mode off", "turn off proof", "stop proof", "disable proof"):
        return Command("set_pref", {"pref": "proof", "value": False}, "Proof Mode is off. Files are still checked.")
    if has("proof mode", "turn on proof", "prove it", "verify my tasks", "check your work"):
        return Command("set_pref", {"pref": "proof", "value": True}, "Proof Mode is on. I'll check every task against what you asked.")
    if has("stop reading", "don't read", "dont read", "stop talking", "turn off read", "mute", "no voice", "quiet"):
        return Command("set_pref", {"pref": "read_aloud", "value": False}, "I'll stop reading answers aloud.")
    if has("read aloud", "read answers", "read it out", "read out loud", "talk back", "speak answers", "read to me"):
        return Command("set_pref", {"pref": "read_aloud", "value": True}, "I'll read new answers aloud.")
    for tone, phrases in (("friendly", ("friendlier", "more friendly", "be friendly", "warmer", "casual")),
                          ("professional", ("formal", "professional", "more polished")),
                          ("direct", ("to the point", "more direct", "be direct", "no fluff", "blunt"))):
        if has(*phrases):
            return Command("set_pref", {"pref": "tone", "value": tone}, f"Tone set to {'Formal' if tone == 'professional' else tone.title()}.")
    if m := re.fullmatch(r"(?:call me|my name is|i'm|i am)\s+([a-z][a-z' -]{0,30})", t):
        name = m.group(1).strip().title()
        return Command("set_pref", {"pref": "name", "value": name}, f"Nice to meet you, {name}.")

    # tools
    for tool, phrases in (("morning", ("morning brief", "daily brief", "brief me", "news brief")),
                          ("meeting", ("meeting notes", "meeting summary", "record a meeting", "record meeting", "transcript")),
                          ("feedback", ("customer feedback", "survey", "app reviews", "nps")),
                          ("interview", ("customer interview", "interview guide", "user interview")),
                          ("release", ("release notes", "changelog")),
                          ("spec", ("spec review", "review my spec", "prd")),
                          ("assist", ("assist mode", "speak for me", "talk for me", "can't speak", "cant speak")),
                          ("lab", ("discovery lab", "research lab", "open the lab", "run an experiment"))):
        if has(*phrases):
            return Command("open_tool", {"tool": tool}, "Opening it now.")

    # navigation
    if t in ("new chat", "start a new chat", "start new chat", "new conversation", "start over", "clear chat"):
        return Command("new_chat", {}, "Fresh chat, ready.")
    if t in ("next channel", "channel up", "ch up", "next"):
        return Command("channel_up", {}, "")
    if t in ("previous channel", "channel down", "ch down", "previous", "back a channel"):
        return Command("channel_down", {}, "")
    if has("sign in", "log in", "login", "create an account", "make an account", "passkey"):
        return Command("sign_in", {}, "Let's get you signed in.")
    for section, phrases in (("customize", ("settings", "customize", "preferences")),
                             ("tools", ("tools", "toolkit")),
                             ("saved", ("saved", "my list", "starred", "bookmarks")),
                             ("projects", ("projects", "my projects")),
                             ("chats", ("my chats", "all chats", "history", "recent chats"))):
        if t in phrases or t in {"open " + p for p in phrases} or t in {"show " + p for p in phrases} or t in {"show me " + p for p in phrases}:
            return Command("open_section", {"section": section}, "")

    # "open <project>" when the name is clearly there
    projects = context.get("projects") or []
    if words & {"open", "go", "show", "switch", "tune", "watch", "channel"}:
        rest = re.sub(r"^(open|go to|show me|show|switch to|tune to|watch|channel)\s+(my\s+|the\s+)?", "", t)
        rest = re.sub(r"\s+(project|channel)$", "", rest).strip()
        for p in projects:
            if rest and _norm(p.get("name", "")) == rest:
                return Command("open_project", {"project_id": p["id"]}, f"Tuned to {p.get('name')}.")
    return None


def _list(items: list[str], empty: str) -> str:
    return "\n".join(items) if items else empty


def interpret(client: Any, model: str, text: str, context: dict) -> Command:
    cmd = quick(text, context)
    if cmd:
        return cmd
    projects = (context.get("projects") or [])[:50]
    chats = (context.get("chats") or [])[:80]
    saved = (context.get("saved") or [])[:60]
    pname = {p["id"]: p.get("name", "") for p in projects}
    system = PROMPT.format(
        signed_in="yes" if context.get("signed_in") else "no",
        projects=_list([f"{p['id']}: {str(p.get('name', ''))[:60]}" for p in projects], "(none)"),
        chats=_list([f"{c['id']}: {str(c.get('title', ''))[:80]}" + (f" [{pname.get(c.get('project'), '')}]" if c.get("project") in pname else "")
                     for c in chats], "(none)"),
        saved=_list([f"- {str(s.get('question', ''))[:80]}" for s in saved], "(none)"),
        prefs={k: v for k, v in (context.get("prefs") or {}).items() if k in ("length", "tone", "read_aloud", "jury", "name", "city")},
    )
    resp = client.messages.create(model=model, max_tokens=300, system=system, tools=[TOOL_SPEC], **speed.params(model, "helper"),
                                  tool_choice={"type": "tool", "name": "do"},
                                  messages=[{"role": "user", "content": text[:300]}])
    usage = _get(resp, "usage")
    cmd = Command(source="model")
    if usage is not None:
        cmd.input_tokens = _get(usage, "input_tokens", 0) or 0
        cmd.output_tokens = _get(usage, "output_tokens", 0) or 0
    block = next((b for b in (_get(resp, "content", []) or []) if _get(b, "type") == "tool_use"), None)
    raw = dict(_get(block, "input", {}) or {}) if block is not None else {}
    return validate(raw, context, cmd)


def validate(raw: dict, context: dict, cmd: Command | None = None) -> Command:
    """Keep only a known action with arguments that make sense; anything else becomes a polite reply."""
    cmd = cmd or Command(source="model")
    action = raw.get("action")
    say = " ".join(str(raw.get("say") or "").split())[:120]
    ids_p = {p["id"] for p in context.get("projects") or []}
    ids_c = {c["id"] for c in context.get("chats") or []}
    fallback = Command("reply", {}, "I can open projects, chats, saved answers, tools and settings. Try “open my projects”.",
                       cmd.source, cmd.input_tokens, cmd.output_tokens)
    args: dict = {}
    if action == "open_section" and raw.get("section") in SECTIONS:
        args = {"section": raw["section"]}
    elif action == "open_project" and raw.get("project_id") in ids_p:
        args = {"project_id": raw["project_id"]}
    elif action == "open_chat" and raw.get("chat_id") in ids_c:
        args = {"chat_id": raw["chat_id"]}
    elif action in ("search_chats", "search_saved", "ask") and str(raw.get("query") or "").strip():
        args = {"query": str(raw["query"]).strip()[:300]}
    elif action == "new_chat":
        args = {"project_id": raw["project_id"]} if raw.get("project_id") in ids_p else {}
    elif action == "create_project" and str(raw.get("name") or "").strip():
        args = {"name": " ".join(str(raw["name"]).split())[:60], "icon": str(raw.get("icon") or "📁")[:8],
                "instructions": str(raw.get("instructions") or "").strip()[:4000]}
    elif action == "set_pref":
        pref, value = raw.get("pref"), raw.get("value")
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        if pref in PREF_VALUES and value in PREF_VALUES[pref]:
            args = {"pref": pref, "value": value}
        elif pref in ("name", "city", "about") and isinstance(value, str):
            args = {"pref": pref, "value": value.strip()[:1000 if pref == "about" else 80]}
        else:
            return fallback
    elif action == "open_tool" and raw.get("tool") in TOOLS:
        args = {"tool": raw["tool"]}
    elif action in ("sign_in", "channel_up", "channel_down"):
        args = {}
    elif action == "reply" and say:
        args = {}
    else:
        return fallback
    cmd.action, cmd.args, cmd.say = action, args, say
    return cmd


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
