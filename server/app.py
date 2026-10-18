"""Ask Harsha web server: a general AI assistant that also answers, with citations, about Harsha's career.

    uvicorn server.app:app --reload
    POST /api/chat          {"question": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}
    POST /api/owner/check   header X-Owner-Passcode -> {"owner": true}
    GET  /api/notes         owner only: list memory
    DELETE /api/notes/{id}  owner only: remove a note

Everyone can chat. Only requests carrying the owner passcode can change the assistant's memory.
The server is stateless: the page sends recent turns with each question and nothing is stored
except notes Harsha explicitly asks it to remember.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.kb import KnowledgeBase  # noqa: E402
from agent.loop import DEFAULT_MODEL, WEB_SEARCH_TOOL, Agent  # noqa: E402
from agent.notes import FileNotesStore, GitHubNotesStore, NotesIndex, NotesStore, NotesUnavailable  # noqa: E402
from agent.prompts import build_general_prompt  # noqa: E402
from agent.tools import add_notes_tools  # noqa: E402
from cli import load_dotenv  # noqa: E402
from server.guard import DailyBudget, RateLimiter  # noqa: E402

load_dotenv(ROOT / ".env")  # local runs; on Render the variables come from the dashboard

STATIC = Path(__file__).resolve().parent / "static"
log = logging.getLogger("ask_harsha")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DECLINE = re.compile(r"isn't in|is not in|not in (?:harsha's|his) record|ask him directly|best discussed", re.I)
MEMORY_REQUEST = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:remember|forget|save|note)\b", re.I)


def answer_kind(report, owner: bool, question: str, reply) -> str:
    """What the page labels the answer as: record, memory, mixed, web, general, or none (visitor memory refusal)."""
    if not owner and MEMORY_REQUEST.search(question) and not report.cited:
        return "none"
    facts = any(c.startswith("F") for c in report.cited)
    notes = any(c.startswith("N") for c in report.cited)
    if facts and notes:
        return "mixed"
    if facts:
        return "record"
    if notes:
        return "memory"
    return "web" if reply.web_sources else "general"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("ASK_HARSHA_MODEL", DEFAULT_MODEL))
    daily_budget_usd: float = field(default_factory=lambda: _env_float("DAILY_BUDGET_USD", 2.0))
    # Prices per million tokens for the model above (Claude Sonnet 5.5 list prices) and per web search.
    price_in: float = field(default_factory=lambda: _env_float("PRICE_IN_PER_MTOK", 2.0))
    price_out: float = field(default_factory=lambda: _env_float("PRICE_OUT_PER_MTOK", 10.0))
    price_per_search: float = field(default_factory=lambda: _env_float("PRICE_PER_SEARCH", 0.01))
    rate_limit: int = field(default_factory=lambda: int(_env_float("RATE_LIMIT_PER_WINDOW", 12)))
    rate_window: float = field(default_factory=lambda: _env_float("RATE_WINDOW_SECONDS", 600))
    allowed_origins: list[str] = field(default_factory=lambda: [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()])
    web_search: bool = field(default_factory=lambda: os.environ.get("WEB_SEARCH", "true").lower() == "true")
    owner_passcode: str = field(default_factory=lambda: os.environ.get("OWNER_PASSCODE", ""))
    timezone: str = field(default_factory=lambda: os.environ.get("ASSISTANT_TIMEZONE", "America/Chicago"))
    max_question_chars: int = 2000
    max_history_turns: int = 6
    log_questions: bool = field(default_factory=lambda: os.environ.get("LOG_QUESTIONS", "false").lower() == "true")
    contact_email: str = "harshareddy7189@gmail.com"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


def clean_history(history: list[Turn], max_turns: int) -> list[dict]:
    """Keep the most recent alternating user/assistant pairs, starting with a user turn."""
    pairs, i = [], 0
    while i + 1 < len(history):
        if history[i].role == "user" and history[i + 1].role == "assistant":
            pairs.append(({"role": "user", "content": history[i].content[:2000]},
                          {"role": "assistant", "content": history[i + 1].content[:4000]}))
            i += 2
        else:
            i += 1
    out = []
    for u, a in pairs[-max_turns:]:
        out += [u, a]
    return out


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def default_notes_store() -> NotesStore:
    token, repo = os.environ.get("GITHUB_TOKEN", ""), os.environ.get("NOTES_REPO", "")
    if token and repo:
        return GitHubNotesStore(repo, token, branch=os.environ.get("NOTES_BRANCH", "notes"))
    return FileNotesStore(ROOT / "knowledge" / "notes.local.json")


class _Lookup:
    """Facts and notes behind one `by_id` mapping, for the grounding check.

    Built on access, so a note saved earlier in the same reply can be cited in that reply.
    """

    def __init__(self, kb: KnowledgeBase, store: NotesStore):
        self.kb, self.store = kb, store

    @property
    def by_id(self) -> dict:
        return {**self.kb.by_id, **{n.id: n for n in self.store.all()}}


def create_app(client=None, kb: KnowledgeBase | None = None, settings: Settings | None = None, notes: NotesStore | None = None) -> FastAPI:
    settings = settings or Settings()
    kb = kb or KnowledgeBase.load()
    store = notes if notes is not None else default_notes_store()
    index = NotesIndex(store)
    limiter = RateLimiter(settings.rate_limit, settings.rate_window)
    passcode_failures = RateLimiter(5, 3600)  # 5 wrong passcodes per hour per visitor
    budget = DailyBudget(settings.daily_budget_usd, settings.price_in, settings.price_out)
    state = {"client": client}

    def get_client():
        if state["client"] is None:
            import anthropic  # imported lazily so tests run without it
            state["client"] = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return state["client"]

    app = FastAPI(title="Ask Harsha", docs_url=None, redoc_url=None)
    app.state.budget, app.state.limiter, app.state.settings, app.state.notes = budget, limiter, settings, store
    if settings.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["POST", "GET", "DELETE"], allow_headers=["Content-Type", "X-Owner-Passcode"])

    def error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
        return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers)

    def owner_status(request: Request) -> str:
        """'owner', 'visitor' (no passcode sent) or 'denied' (wrong passcode)."""
        given = request.headers.get("x-owner-passcode", "")
        if not given:
            return "visitor"
        if settings.owner_passcode and hmac.compare_digest(given.encode(), settings.owner_passcode.encode()):
            return "owner"
        return "denied"

    def too_many_wrong(ip: str) -> bool:
        return not passcode_failures.allow(ip)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index_page():
        return FileResponse(STATIC / "index.html")

    @app.post("/api/owner/check")
    def owner_check(request: Request):
        ip = client_ip(request)
        status = owner_status(request)
        if status == "owner":
            return {"owner": True}
        if not settings.owner_passcode:
            return error(503, "not_configured", "Owner mode isn't set up yet. Add OWNER_PASSCODE in Render.")
        if too_many_wrong(ip):
            return error(429, "locked", "Too many wrong passcodes. Try again in an hour.")
        return error(401, "wrong_passcode", "That passcode isn't right.")

    def require_owner(request: Request):
        status = owner_status(request)
        if status != "owner":
            if status == "denied" and too_many_wrong(client_ip(request)):
                return error(429, "locked", "Too many wrong passcodes. Try again in an hour.")
            return error(401, "owner_only", "Only Harsha can manage memory.")
        return None

    @app.get("/api/notes")
    def list_notes(request: Request):
        denied = require_owner(request)
        if denied:
            return denied
        return {"notes": [n.to_dict() for n in store.all()]}

    @app.delete("/api/notes/{note_id}")
    def delete_note(note_id: str, request: Request):
        denied = require_owner(request)
        if denied:
            return denied
        try:
            note = store.delete(note_id.upper())
        except KeyError:
            return error(404, "not_found", "That note doesn't exist.")
        except NotesUnavailable as e:
            return error(503, "storage", f"Memory storage is unavailable: {e}")
        log.info(json.dumps({"event": "note_removed", "id": note.id}))
        return {"removed": note.to_dict()}

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request):
        question = req.question.strip()
        if not question:
            return error(400, "empty", "Type a question first.")
        if len(question) > settings.max_question_chars:
            return error(400, "too_long", f"Please keep messages under {settings.max_question_chars} characters.")
        ip = client_ip(request)
        status = owner_status(request)
        if status == "denied" and too_many_wrong(ip):
            return error(429, "locked", "Too many wrong passcodes. Try again in an hour.")
        owner = status == "owner"
        if not owner and not limiter.allow(ip):
            wait = limiter.retry_after(ip)
            return error(429, "rate_limited", f"That's a lot of questions at once. Try again in about {max(1, wait // 60)} minute(s).", {"Retry-After": str(wait)})
        if budget.exhausted():
            return error(503, "resting", f"Ask Harsha is resting for today. You can reach Harsha directly at {settings.contact_email}.")

        today = datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %B %-d, %Y")
        agent = Agent(
            get_client(), kb=kb, model=settings.model, max_steps=8, max_tokens=1600,
            system=build_general_prompt(today, owner=owner, web=settings.web_search),
            server_tools=[WEB_SEARCH_TOOL] if settings.web_search else [],
            lookup=_Lookup(kb, store), strict_grounding=False,
        )
        add_notes_tools(agent.tools, index, store, owner=owner)
        agent.messages = clean_history(req.history, settings.max_history_turns)
        try:
            reply = agent.ask(question)
        except Exception as e:  # API outage, bad key, network: log the cause, show a friendly message
            log.error(json.dumps({"event": "upstream_error", "type": type(e).__name__, "detail": str(e)[:300]}))
            return error(502, "upstream", "Ask Harsha couldn't reach its model just now. Please try again in a moment.")

        cost = budget.add(reply.input_tokens, reply.output_tokens) + budget.add_flat(reply.web_searches * settings.price_per_search)
        report = reply.report
        lookup = _Lookup(kb, store).by_id
        cited = [{"id": c, "text": lookup[c].text, "kind": "note" if c.startswith("N") else "fact"} for c in report.cited if c in lookup]
        record = {"event": "answer", "owner": owner, "kind": report.kind, "grounded": report.ok, "steps": reply.steps,
                  "web_searches": reply.web_searches, "memory_changes": len(reply.memory_changes),
                  "tokens_in": reply.input_tokens, "tokens_out": reply.output_tokens, "cost_usd": round(cost, 5),
                  "declined": bool(DECLINE.search(reply.text)), "spent_today_usd": round(budget.spent, 4)}
        if settings.log_questions:
            record["question"] = question[:200]
        log.info(json.dumps(record))
        return {
            "answer": reply.text,
            "kind": report.kind,
            "label": answer_kind(report, owner, question, reply),
            "trace": reply.steps_log,
            "grounded": report.ok,
            "check": report.summary(),
            "citations": cited,
            "web_sources": reply.web_sources[:8],
            "memory_changes": reply.memory_changes,
            "owner": owner,
            "retrieved": sorted(agent.tools.retrieved),
            "sources": {i: {"text": lookup[i].text, "kind": "note" if i.startswith("N") else "fact"} for i in agent.tools.retrieved if i in lookup},
            "steps": reply.steps,
        }

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
