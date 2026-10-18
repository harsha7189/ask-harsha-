"""Ask Harsha web server: the same agent as the CLI, behind a small JSON API and a phone-friendly chat page.

    uvicorn server.app:app --reload           # local
    POST /api/chat  {"question": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}

The server is stateless: the page sends the recent conversation with each question, and the
server never stores it. Your API key stays on the server.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.kb import KnowledgeBase  # noqa: E402
from agent.loop import DEFAULT_MODEL, Agent  # noqa: E402
from cli import load_dotenv  # noqa: E402
from server.guard import DailyBudget, RateLimiter  # noqa: E402

load_dotenv(ROOT / ".env")  # local runs; on Render the variables come from the dashboard

STATIC = Path(__file__).resolve().parent / "static"
log = logging.getLogger("ask_harsha")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DECLINE = re.compile(r"isn't in|is not in|not in (?:harsha's|his) record|ask him directly|best discussed", re.I)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("ASK_HARSHA_MODEL", DEFAULT_MODEL))
    daily_budget_usd: float = field(default_factory=lambda: _env_float("DAILY_BUDGET_USD", 2.0))
    # Prices per million tokens for the model above. Defaults are Claude Sonnet 5.5 list prices;
    # update them if you change model (see platform.claude.com/docs/en/about-claude/pricing).
    price_in: float = field(default_factory=lambda: _env_float("PRICE_IN_PER_MTOK", 2.0))
    price_out: float = field(default_factory=lambda: _env_float("PRICE_OUT_PER_MTOK", 10.0))
    rate_limit: int = field(default_factory=lambda: int(_env_float("RATE_LIMIT_PER_WINDOW", 12)))
    rate_window: float = field(default_factory=lambda: _env_float("RATE_WINDOW_SECONDS", 600))
    allowed_origins: list[str] = field(default_factory=lambda: [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()])
    max_question_chars: int = 500
    max_history_turns: int = 6        # question/answer pairs sent back as context
    log_questions: bool = field(default_factory=lambda: os.environ.get("LOG_QUESTIONS", "false").lower() == "true")
    contact_email: str = "harshareddy7189@gmail.com"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


def clean_history(history: list[Turn], max_turns: int) -> list[dict]:
    """Keep the most recent alternating user/assistant pairs, starting with a user turn."""
    pairs, i = [], 0
    while i + 1 < len(history):
        if history[i].role == "user" and history[i + 1].role == "assistant":
            pairs.append(({"role": "user", "content": history[i].content[:1000]},
                          {"role": "assistant", "content": history[i + 1].content[:2000]}))
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


def create_app(client=None, kb: KnowledgeBase | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    kb = kb or KnowledgeBase.load()
    limiter = RateLimiter(settings.rate_limit, settings.rate_window)
    budget = DailyBudget(settings.daily_budget_usd, settings.price_in, settings.price_out)
    state = {"client": client}

    def get_client():
        if state["client"] is None:
            import anthropic  # imported lazily so tests run without it
            state["client"] = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return state["client"]

    app = FastAPI(title="Ask Harsha", docs_url=None, redoc_url=None)
    app.state.budget, app.state.limiter, app.state.settings = budget, limiter, settings
    if settings.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["POST", "GET"], allow_headers=["Content-Type"])

    def error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
        return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request):
        question = req.question.strip()
        if not question:
            return error(400, "empty", "Type a question first.")
        if len(question) > settings.max_question_chars:
            return error(400, "too_long", f"Please keep questions under {settings.max_question_chars} characters.")
        ip = client_ip(request)
        if not limiter.allow(ip):
            wait = limiter.retry_after(ip)
            return error(429, "rate_limited", f"That's a lot of questions at once. Try again in about {max(1, wait // 60)} minute(s).", {"Retry-After": str(wait)})
        if budget.exhausted():
            return error(503, "resting", f"Ask Harsha is resting for today. You can reach Harsha directly at {settings.contact_email}.")

        agent = Agent(get_client(), kb=kb, model=settings.model, max_steps=6, max_tokens=800)
        agent.messages = clean_history(req.history, settings.max_history_turns)
        try:
            reply = agent.ask(question)
        except Exception as e:  # API outage, bad key, network: log the cause, show a friendly message
            log.error(json.dumps({"event": "upstream_error", "type": type(e).__name__, "detail": str(e)[:300]}))
            return error(502, "upstream", "Ask Harsha couldn't reach its model just now. Please try again in a moment.")

        cost = budget.add(reply.input_tokens, reply.output_tokens)
        report = reply.report
        cited = [{"id": c, "text": kb.by_id[c].text} for c in report.cited if c in kb.by_id]
        record = {"event": "answer", "grounded": report.ok, "steps": reply.steps, "tokens_in": reply.input_tokens,
                  "tokens_out": reply.output_tokens, "cost_usd": round(cost, 5), "declined": bool(DECLINE.search(reply.text)),
                  "spent_today_usd": round(budget.spent, 4)}
        if settings.log_questions:
            record["question"] = question[:200]
        log.info(json.dumps(record))
        return {
            "answer": reply.text,
            "grounded": report.ok,
            "check": report.summary(),
            "citations": cited,
            "retrieved": sorted(agent.tools.retrieved),
            "steps": reply.steps,
        }

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
