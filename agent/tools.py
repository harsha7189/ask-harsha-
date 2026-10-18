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
