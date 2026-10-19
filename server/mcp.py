"""MCP endpoint: lets other AI assistants (Claude, or any MCP client) use Ask Harsha as a tool.

The Model Context Protocol is JSON-RPC 2.0. This is the "Streamable HTTP" transport in its simplest form: the
client POSTs a message to /mcp and gets a JSON reply. It's stateless, so no session ids are needed, and there's
no sign-in: everything it exposes is public anyway.

Tools:
  ask_harsha_agent     ask the full agent (web search, Harsha's record, citations); costs a model call
  search_harsha_facts  search Harsha's verified career facts directly (free)
  get_harsha_topic     every fact on one topic (free)
  verify_answer        check an Ask Harsha verification link or receipt id (free)
"""
from __future__ import annotations

import json
from typing import Any, Callable

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "ask-harsha", "title": "Ask Harsha", "version": "1.0.0"}
INSTRUCTIONS = (
    "Ask Harsha is an AI agent Harsha Reddy built from scratch. Use search_harsha_facts or get_harsha_topic for "
    "verified facts about Harsha's career (each has an ID like F14; cite it). Use ask_harsha_agent for a full answer "
    "in the agent's own words, which comes with a verification link proving it's genuine. Use verify_answer to check "
    "an Ask Harsha verification link someone shared."
)

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


class ToolFailure(Exception):
    """Reported back to the client as a tool result with isError, so the calling model can see what went wrong."""


def tool_specs(topics: list[str]) -> list[dict]:
    ro = {"readOnlyHint": True, "openWorldHint": False}
    return [
        {"name": "ask_harsha_agent", "title": "Ask the Ask Harsha agent",
         "description": "Ask Harsha Reddy's AI agent a question: about Harsha's career (answered only from his verified record, with citations) or anything general (it can search the web). Returns the answer, its sources, and a link that proves the answer is genuine.",
         "inputSchema": {"type": "object", "properties": {"question": {"type": "string", "description": "The question, up to 2000 characters."}}, "required": ["question"]},
         "annotations": {"readOnlyHint": True, "openWorldHint": True}},
        {"name": "search_harsha_facts", "title": "Search Harsha's verified facts",
         "description": "Keyword search over Harsha Reddy's verified career facts (roles, products, metrics, skills, education, certifications). Each result has an ID like F14 to cite.",
         "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}}, "required": ["query"]},
         "annotations": ro},
        {"name": "get_harsha_topic", "title": "Get a topic from Harsha's record",
         "description": f"Every verified fact on one topic. Topics: {', '.join(topics)}.",
         "inputSchema": {"type": "object", "properties": {"topic": {"type": "string", "enum": topics}}, "required": ["topic"]},
         "annotations": ro},
        {"name": "verify_answer", "title": "Verify an Ask Harsha answer",
         "description": "Check whether an Ask Harsha verification link (…/v/<id>) or receipt id is genuine, and get the exact original question and answer.",
         "inputSchema": {"type": "object", "properties": {"link": {"type": "string", "description": "The verification link or just the id."}}, "required": ["link"]},
         "annotations": ro},
    ]


def handle(message: Any, tools: dict[str, Callable[[dict], str]], topics: list[str]) -> dict | None:
    """Answer one JSON-RPC message. Returns None for notifications (no reply)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return _error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST, "Invalid JSON-RPC request.")
    mid, method, params = message.get("id"), message["method"], message.get("params") or {}
    if "id" not in message:
        return None  # notifications such as notifications/initialized need no reply
    if method == "initialize":
        asked = params.get("protocolVersion") if isinstance(params, dict) else None
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[1]
        return _result(mid, {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS})
    if method == "ping":
        return _result(mid, {})
    if method == "tools/list":
        return _result(mid, {"tools": tool_specs(topics)})
    if method == "tools/call":
        if not isinstance(params, dict) or params.get("name") not in tools:
            return _error(mid, INVALID_PARAMS, f"Unknown tool: {params.get('name') if isinstance(params, dict) else params}")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return _error(mid, INVALID_PARAMS, "arguments must be an object.")
        try:
            text = tools[params["name"]](args)
            return _result(mid, {"content": [{"type": "text", "text": text}], "isError": False})
        except ToolFailure as e:
            return _result(mid, {"content": [{"type": "text", "text": str(e)}], "isError": True})
        except Exception:
            return _result(mid, {"content": [{"type": "text", "text": "The tool failed unexpectedly. Please try again."}], "isError": True})
    return _error(mid, METHOD_NOT_FOUND, f"Method not found: {method}")


def _result(mid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def parse(body: bytes) -> tuple[Any, dict | None]:
    try:
        return json.loads(body.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, _error(None, PARSE_ERROR, "Parse error.")
