"""Tools the model can call. Each tool = a JSON schema the model sees + a Python function we run.

The registry is plain data, so adding a tool means adding one entry here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .kb import KnowledgeBase


class ToolError(Exception):
    """Raised for bad tool input. The message goes back to the model with is_error=true."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    run: Callable[[dict], Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class ToolRegistry:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        # IDs the model has retrieved during the current user turn; used by the grounding check.
        self.retrieved: set[str] = set()
        # Memory changes made during the current turn, reported back to the page.
        self.changes: list[dict] = []
        # Set by suggest_asking_harsha when the model offers to hand the question to Harsha.
        self.offer: dict | None = None
        # Files made by create_file during the current turn: [{filename, kind, size, ..., spec}]
        self.files: list[dict] = []
        self._tools: dict[str, Tool] = {}
        for tool in self._build():
            self._tools[tool.name] = tool

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    # ---- tool implementations -------------------------------------------------
    def _search_facts(self, args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolError("Missing 'query'. Pass a few keywords, for example 'AI agent adoption'.")
        k = args.get("k", 5)
        if not isinstance(k, int) or not 1 <= k <= 10:
            raise ToolError("'k' must be an integer from 1 to 10.")
        hits = self.kb.search(query, k=k)
        self.retrieved.update(f.id for f, _ in hits)
        if not hits:
            return {"query": query, "results": [], "note": "No matching facts. Try other keywords or get_topic, or tell the user this isn't in Harsha's record."}
        return {"query": query, "results": [dict(f.to_dict(), score=s) for f, s in hits]}

    def _get_topic(self, args: dict) -> dict:
        topic = str(args.get("topic", "")).strip().lower()
        facts = self.kb.by_topic(topic)
        if not facts:
            raise ToolError(f"Unknown topic '{topic}'. Valid topics: {', '.join(sorted(self.kb.topics()))}.")
        self.retrieved.update(f.id for f in facts)
        return {"topic": topic, "results": [f.to_dict() for f in facts]}

    def _list_topics(self, args: dict) -> dict:
        return {"subject": self.kb.subject, "topics": self.kb.topics()}

    def _build(self) -> list[Tool]:
        topics = sorted(self.kb.topics())
        return [
            Tool(
                name="search_facts",
                description=(
                    "Search Harsha's verified career facts by keyword. Returns facts with IDs like F14 that you must cite. "
                    "Use specific keywords from the question (company, skill, metric, tool). Call it again with different "
                    "keywords if the first search misses."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keywords, for example 'Cybage sprints' or 'DRM Widevine'."},
                        "k": {"type": "integer", "description": "How many facts to return, 1 to 10. Default 5.", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
                run=self._search_facts,
            ),
            Tool(
                name="get_topic",
                description="Return every fact in one topic. Use for broad questions such as 'walk me through his IBM work' or 'what certifications does he have'.",
                input_schema={
                    "type": "object",
                    "properties": {"topic": {"type": "string", "enum": topics, "description": "The topic to fetch."}},
                    "required": ["topic"],
                },
                run=self._get_topic,
            ),
            Tool(
                name="list_topics",
                description="List the topics in Harsha's record with fact counts. Use when you are unsure where something might be.",
                input_schema={"type": "object", "properties": {}},
                run=self._list_topics,
            ),
        ]

    # ---- dispatch --------------------------------------------------------------
    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a tool. Returns (content_string, is_error). Never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool '{name}'. Available tools: {', '.join(self._tools)}.", True
        if not isinstance(args, dict):
            return "Tool input must be a JSON object.", True
        try:
            return json.dumps(tool.run(args), ensure_ascii=False), False
        except ToolError as e:
            return str(e), True
        except Exception as e:  # a bug in a tool should not crash the conversation
            return f"Tool '{name}' failed: {type(e).__name__}: {e}", True

    def reset_turn(self) -> None:
        self.retrieved.clear()
        self.changes.clear()
        self.offer = None
        self.files = []


def add_notes_tools(registry: ToolRegistry, index, store, owner: bool) -> None:
    """Give the model search over Harsha's notes; give remember/forget only when Harsha is verified."""
    from .notes import NotesUnavailable  # local import keeps tools.py usable without notes

    def search_notes(args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolError("Missing 'query'.")
        hits = index.search(query, k=5)
        registry.retrieved.update(n.id for n, _ in hits)
        if not hits:
            return {"query": query, "results": [], "note": "No matching notes."}
        return {"query": query, "results": [{"id": n.id, "text": n.text, "score": s} for n, s in hits]}

    registry.add(Tool(
        name="search_notes",
        description="Search notes Harsha has personally taught you (his preferences, opinions, interests, plans, knowledge he wants you to have). Returns notes with IDs like N003 that you must cite when you use them.",
        input_schema={"type": "object", "properties": {"query": {"type": "string", "description": "Keywords."}}, "required": ["query"]},
        run=search_notes,
    ))
    if not owner:
        return

    def remember(args: dict) -> dict:
        try:
            note = store.add(str(args.get("text", "")))
        except ValueError as e:
            raise ToolError(str(e))
        except NotesUnavailable as e:
            raise ToolError(f"Memory storage is unavailable: {e}")
        registry.retrieved.add(note.id)
        registry.changes.append({"action": "added", "id": note.id, "text": note.text})
        return {"saved": note.to_dict()}

    def forget(args: dict) -> dict:
        note_id = str(args.get("id", "")).strip().upper()
        try:
            note = store.delete(note_id)
        except KeyError:
            raise ToolError(f"No note with ID '{note_id}'. Use search_notes to find the right ID.")
        except NotesUnavailable as e:
            raise ToolError(f"Memory storage is unavailable: {e}")
        registry.changes.append({"action": "removed", "id": note.id, "text": note.text})
        return {"removed": note.to_dict()}

    registry.add(Tool(
        name="remember_note",
        description="Save a new note to your long-term memory. Only Harsha can use this. Write one clear, self-contained statement (for example 'Harsha prefers concise answers' or 'Harsha's favorite book is X'). Check search_notes first to avoid duplicates.",
        input_schema={"type": "object", "properties": {"text": {"type": "string", "description": "The note, under 500 characters."}}, "required": ["text"]},
        run=remember,
    ))
    registry.add(Tool(
        name="forget_note",
        description="Delete a note from memory by its ID (find it with search_notes first). Only Harsha can use this.",
        input_schema={"type": "object", "properties": {"id": {"type": "string", "description": "Note ID like N003."}}, "required": ["id"]},
        run=forget,
    ))


def add_file_tool(registry: ToolRegistry, max_files: int = 5) -> None:
    """Let the model make downloadable files (Excel, CSV, PDF, Word, HTML, code). See agent/makefiles.py."""
    from . import makefiles

    def make(args: dict) -> dict:
        if len(registry.files) >= max_files:
            raise ToolError(f"That's {max_files} files in one answer, the most allowed. Finish the answer.")
        try:
            spec = makefiles.check(args)
            built = makefiles.build(spec)
        except makefiles.FileError as e:
            raise ToolError(str(e)) from e
        info = makefiles.summary(spec, built)
        registry.files.append({**info, "spec": spec})
        return {"ok": True, "file": info,
                "note": "The person now sees a card with Download (and Preview where it fits). In your reply, say in a sentence "
                        "or two what's in the file and how to use it. Don't paste the file's contents."}

    t = makefiles.TOOL_SPEC
    registry.add(Tool(name=t["name"], description=t["description"], input_schema=t["input_schema"], run=make))


def add_ask_harsha_tool(registry: ToolRegistry) -> None:
    """Let the model offer the visitor a way to send the question to the real Harsha.

    The tool only records the offer; the page shows the pop-up and the visitor decides whether to send anything.
    """
    registry.offer = None

    def offer(args: dict) -> dict:
        reason = " ".join(str(args.get("reason", "")).split())[:200]
        kind = args.get("kind") if args.get("kind") in ("about_harsha", "complex", "uncertain", "judgment") else "complex"
        if not reason:
            raise ToolError("Give a short reason, for example 'His record doesn't say whether he mentors students.'")
        registry.offer = {"reason": reason, "kind": kind}
        return {"ok": True, "note": "The app will show the visitor a button to send this question to Harsha. Finish your answer normally, "
                "and mention the option in one short sentence at the end. Don't promise when or whether he'll reply."}

    registry.add(Tool(
        name="suggest_asking_harsha",
        description=(
            "Offer the visitor a button to send their question to the real Harsha, who answers personally later. Use it when "
            "(1) they ask about Harsha and his record and notes don't cover it, or (2) the question is genuinely complex or "
            "uncertain and a real person's judgment or experience would add something your answer can't: a hard trade-off with "
            "no clear answer, sources that disagree, or you aren't confident. Still give your best answer. Don't use it for "
            "simple questions you can answer well, more than once per conversation, or for emergencies, self-harm, or medical, "
            "legal or financial decisions; for those, point to the right professional or service instead. Harsha is a product "
            "manager, so don't present him as an expert outside his experience."
        ),
        input_schema={"type": "object", "properties": {
            "reason": {"type": "string", "description": "One short sentence the visitor will see, saying why a human could help, e.g. 'This depends on your situation, and Harsha has made a similar career move.'"},
            "kind": {"type": "string", "enum": ["about_harsha", "complex", "uncertain", "judgment"]},
        }, "required": ["reason", "kind"]},
        run=offer,
    ))
