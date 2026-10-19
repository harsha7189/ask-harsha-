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
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import assist, attachments as files, factcheck, inbox as inboxmod, receipts as rcpt, toolkit  # noqa: E402
from server import mcp  # noqa: E402
from agent.evals import load_cases, score as score_case  # noqa: E402
from agent.kb import KnowledgeBase  # noqa: E402
from agent.loop import DEFAULT_MODEL, WEB_SEARCH_TOOL, Agent  # noqa: E402

FACTCHECK_SEARCH_TOOL = {**WEB_SEARCH_TOOL, "max_uses": 5}
from agent.notes import FileNotesStore, GitHubNotesStore, NotesIndex, NotesStore, NotesUnavailable  # noqa: E402
from agent.prompts import build_general_prompt  # noqa: E402
from agent.tools import add_ask_harsha_tool, add_notes_tools  # noqa: E402
from cli import load_dotenv  # noqa: E402
from server.guard import DailyBudget, RateLimiter  # noqa: E402

load_dotenv(ROOT / ".env")  # local runs; on Render the variables come from the dashboard

STATIC = Path(__file__).resolve().parent / "static"
log = logging.getLogger("ask_harsha")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DECLINE = re.compile(r"isn't in|is not in|not in (?:harsha's|his) record|ask him directly|best discussed", re.I)
MEMORY_REQUEST = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:remember|forget|save|note)\b", re.I)


def answer_kind(report, owner: bool, question: str, reply, has_files: bool = False) -> str:
    """What the page labels the answer as: record, memory, mixed, file, web, general, or none (visitor memory refusal)."""
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
    if has_files:
        return "file"
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
    # Assist mode is someone's voice, so it gets a fast model and a looser per-visitor limit.
    assist_model: str = field(default_factory=lambda: os.environ.get("ASSIST_MODEL", "claude-haiku-5-5"))
    assist_rate_limit: int = field(default_factory=lambda: int(_env_float("ASSIST_RATE_LIMIT_PER_WINDOW", 60)))


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ToolRequest(BaseModel):
    text: str = Field(default="", max_length=150_000)
    attachments: list[Attachment] = Field(default_factory=list, max_length=files.MAX_FILES)
    options: dict = Field(default_factory=dict)


class ShareRequest(BaseModel):
    receipt: dict
    signature: str = Field(min_length=10, max_length=200)


class InboxRequest(BaseModel):
    question: str = Field(min_length=1, max_length=inboxmod.MAX_QUESTION)
    reason: str = Field(default="", max_length=300)
    kind: str = Field(default="complex", max_length=20)
    context: str = Field(default="", max_length=6000)
    note: str = Field(default="", max_length=600)


class InboxAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=inboxmod.MAX_ANSWER)
    remember: bool = False


class FactCheckRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=6000)


class PastPick(BaseModel):
    input: str = Field(default="", max_length=300)
    said: str = Field(min_length=1, max_length=300)


class AssistRequest(BaseModel):
    heard: list[str] = Field(default_factory=list, max_length=5)
    typed: str = Field(default="", max_length=300)
    partner: str = Field(default="", max_length=600)
    past: list[PastPick] = Field(default_factory=list, max_length=20)
    saved: list[str] = Field(default_factory=list, max_length=30)


class Attachment(BaseModel):
    name: str = Field(default="file", max_length=200)
    kind: Literal["image", "pdf", "docx", "text"]
    media_type: str = Field(default="", max_length=100)
    data: str = Field(min_length=1, max_length=14_000_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=40)
    attachments: list[Attachment] = Field(default_factory=list, max_length=files.MAX_FILES)


FILES_INTRO = "Here are the files I've shared in this chat. Use them to answer my questions."
FILES_ACK = "I have your files and will use them to answer."


def file_turns(blocks: list[dict]) -> list[dict]:
    """Attached files go first, as their own exchange, so they sit at a stable spot the prompt cache can reuse
    for every follow-up question about them."""
    if not blocks:
        return []
    return [{"role": "user", "content": [{"type": "text", "text": FILES_INTRO}, *blocks]},
            {"role": "assistant", "content": FILES_ACK}]


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


def default_inbox_store() -> "inboxmod.InboxStore":
    key = inboxmod.make_key(os.environ.get("INBOX_KEY", ""), os.environ.get("OWNER_PASSCODE", ""))
    token, repo = os.environ.get("GITHUB_TOKEN", ""), os.environ.get("NOTES_REPO", "")
    if token and repo:
        return inboxmod.GitHubInboxStore(repo, token, key, branch=os.environ.get("NOTES_BRANCH", "notes"))
    return inboxmod.FileInboxStore(ROOT / "knowledge" / "inbox.local.json", key)


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


def create_app(client=None, kb: KnowledgeBase | None = None, settings: Settings | None = None, notes: NotesStore | None = None,
               inbox: "inboxmod.InboxStore | None" = None, receipts: "rcpt.ReceiptStore | None" = None) -> FastAPI:
    settings = settings or Settings()
    kb = kb or KnowledgeBase.load()
    store = notes if notes is not None else default_notes_store()
    index = NotesIndex(store)
    inbox = inbox if inbox is not None else default_inbox_store()
    inbox_limiter = RateLimiter(5, 86400)        # questions sent to Harsha: 5 a day per visitor
    ticket_limiter = RateLimiter(120, 600)
    signer = rcpt.Signer.from_secrets(os.environ.get("SIGNING_KEY", ""), os.environ.get("INBOX_KEY", ""), settings.owner_passcode)
    if receipts is None:
        fkey = inboxmod.make_key(os.environ.get("INBOX_KEY", ""), settings.owner_passcode)
        token, repo = os.environ.get("GITHUB_TOKEN", ""), os.environ.get("NOTES_REPO", "")
        receipts = (rcpt.GitHubReceiptStore(repo, token, fkey, branch=os.environ.get("NOTES_BRANCH", "notes"))
                    if token and repo else rcpt.ReceiptStore(fkey))
    share_limiter = RateLimiter(20, 86400)       # shared receipts: 20 a day per visitor
    mcp_limiter = RateLimiter(30, 600)           # agent questions over MCP, per client address
    limiter = RateLimiter(settings.rate_limit, settings.rate_window)
    assist_limiter = RateLimiter(settings.assist_rate_limit, settings.rate_window)
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

    # ---------- PM Toolkit ----------
    brief_cache: dict = {}   # (date, topics) -> final payload; one morning brief per topic set per day

    @app.get("/tools")
    def tools_page():
        return FileResponse(STATIC / "tools.html", headers={"Cache-Control": "no-cache"})

    @app.post("/api/tools/{tool_id}/stream")
    def run_tool(tool_id: str, req: ToolRequest, request: Request):
        tool = toolkit.TOOLS.get(tool_id)
        if tool is None:
            return error(404, "not_found", "That tool doesn't exist.")
        ip = client_ip(request)
        status = owner_status(request)
        if status == "denied" and too_many_wrong(ip):
            return error(429, "locked", "Too many wrong passcodes. Try again in an hour.")
        owner = status == "owner"
        limit = 150_000 if owner else 60_000
        text = req.text.strip()
        if len(text) > limit:
            return error(400, "too_long", f"That's too long. Please keep it under {limit:,} characters, or split it in two.")
        options = {k: v for k, v in (req.options or {}).items() if k in ("topics", "history", "mode", "audience", "version")}
        if isinstance(options.get("history"), list):
            options["history"] = [" ".join(str(h).split())[:120] for h in options["history"][-10:]]
        if tool_id != "morning" and not text and not req.attachments:
            return error(400, "empty", "Paste some text or attach a file first.")
        today = datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %B %-d, %Y")
        cache_key = None
        if tool_id == "morning" and not text:
            cache_key = (today, tuple(t.lower() for t in sorted(toolkit._clean_topics(options.get("topics")), key=str.lower)))
            if cache_key in brief_cache:
                q: queue.Queue = queue.Queue()
                q.put(("final", {**brief_cache[cache_key], "cached": True, "cost_usd": 0}))
                q.put(None)
                return StreamingResponse(stream_events(q), media_type="text/event-stream", headers=SSE_HEADERS)
        if not owner and not limiter.allow(ip):
            wait = limiter.retry_after(ip)
            return error(429, "rate_limited", f"That's a lot at once. Try again in about {max(1, wait // 60)} minute(s).", {"Retry-After": str(wait)})
        if budget.exhausted():
            return error(503, "resting", f"Ask Harsha is resting for today. You can reach Harsha directly at {settings.contact_email}.")
        try:
            prepared = files.prepare(req.attachments, max_pdf_pages=100 if owner else 30)
        except files.AttachmentError as e:
            return error(400, "attachment", str(e))
        q = queue.Queue()

        def on_event(kind: str, data: dict) -> None:
            if kind == "text":
                q.put(("text", {"delta": data["delta"]}))
            elif kind == "text_reset":
                q.put(("text_reset", {}))
            elif kind == "web_search_start":
                q.put(("status", {"label": "Searching the web"}))

        agent = Agent(get_client(), kb=kb, model=settings.model, max_steps=4, max_tokens=tool.max_tokens,
                      system=toolkit.system_prompt(tool_id, today, options), on_event=on_event,
                      server_tools=[{**WEB_SEARCH_TOOL, "max_uses": 5}] if (tool.web and settings.web_search) else [],
                      strict_grounding=False, stream=True, record_tools=False)
        agent.messages = file_turns(prepared.blocks)

        def work():
            started = time.monotonic()
            q.put(("status", {"label": "Reading your files" if prepared.blocks else ("Finding today's news" if tool.web else "Working on it")}))
            try:
                reply = agent.ask(toolkit.user_message(tool_id, text, options))
            except Exception as e:
                upstream_failed(e)
                q.put(("error", {"message": UPSTREAM_MSG}))
                return
            cost = budget.add(reply.input_tokens, reply.output_tokens) + budget.add_flat(reply.web_searches * settings.price_per_search)
            body, quiz = toolkit.split_quiz(reply.text) if tool_id == "morning" else (reply.text, None)
            payload = {"tool": tool_id, "text": body, "quiz": quiz, "lesson": toolkit.lesson_title(body) if tool_id == "morning" else "",
                       "web_sources": reply.web_sources[:10], "cost_usd": round(cost, 4),
                       "seconds": round(time.monotonic() - started, 1), "cut_off": reply.stop_reason == "max_tokens"}
            if cache_key and body and reply.stop_reason == "end_turn":
                if len(brief_cache) > 30:
                    brief_cache.clear()
                brief_cache[cache_key] = payload
            log.info(json.dumps({"event": "tool", "tool": tool_id, "owner": owner, "chars": len(text),
                                 "attachments": [a["kind"] for a in prepared.summary], "web_searches": reply.web_searches,
                                 "tokens_in": reply.input_tokens, "tokens_out": reply.output_tokens,
                                 "cost_usd": round(cost, 5), "spent_today_usd": round(budget.spent, 4)}))
            q.put(("final", payload))

        return StreamingResponse(stream_events(q, work), media_type="text/event-stream", headers=SSE_HEADERS)

    # ---------- Verified answers ----------
    def origin(request: Request) -> str:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
        return f"{proto}://{request.headers.get('host', request.url.netloc)}"

    RID = re.compile(r"[A-Za-z0-9_-]{8,32}")

    @app.post("/api/receipts")
    def share_receipt(req: ShareRequest, request: Request):
        if signer is None or not receipts.enabled:
            return error(503, "receipts", "Verified sharing isn't set up on this server.")
        if len(rcpt.canonical(req.receipt)) > rcpt.MAX_RECEIPT_BYTES:
            return error(400, "too_large", "That answer is too long to share as a verified link.")
        if not signer.verify(req.receipt, req.signature):
            return error(400, "not_verified", "This answer couldn't be verified as coming from Ask Harsha, so it can't be shared as verified.")
        if owner_status(request) != "owner" and not share_limiter.allow(client_ip(request)):
            return error(429, "rate_limited", "You've shared a lot today. Please try again tomorrow.")
        try:
            rid = receipts.put(req.receipt, req.signature)
        except rcpt.ReceiptError as e:
            return error(503, "receipts", str(e))
        log.info(json.dumps({"event": "receipt_shared"}))
        return {"id": rid, "url": f"{origin(request)}/v/{rid}"}

    def load_receipt(rid: str):
        if not RID.fullmatch(rid):
            return None
        return receipts.get(rid)

    @app.get("/api/receipts/{rid}")
    def get_receipt(rid: str, request: Request):
        if not ticket_limiter.allow(client_ip(request)):
            return error(429, "rate_limited", "Too many checks at once. Try again in a few minutes.")
        try:
            data = load_receipt(rid)
        except rcpt.ReceiptError as e:
            return error(503, "receipts", str(e))
        if data is None:
            return error(404, "not_found", "There's no verified answer at this link.")
        valid = signer is not None and signer.verify(data["receipt"], data["sig"])
        return {"receipt": data["receipt"], "signature": data["sig"], "valid": valid,
                "public_key": signer.public_b64 if signer else "", "fingerprint": signer.fingerprint if signer else ""}

    @app.get("/api/receipts/{rid}/qr.svg")
    def receipt_qr(rid: str, request: Request):
        if not RID.fullmatch(rid):
            return error(404, "not_found", "Not found.")
        import io
        import qrcode
        import qrcode.image.svg
        img = qrcode.make(f"{origin(request)}/v/{rid}", image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
        buf = io.BytesIO()
        img.save(buf)
        return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/v/{rid}")
    def verify_page(rid: str):
        return FileResponse(STATIC / "verify.html", headers={"Cache-Control": "no-cache"})

    @app.get("/.well-known/ask-harsha-key.json")
    def public_key():
        if signer is None:
            return error(404, "not_found", "Signing isn't set up on this server.")
        return {"alg": "Ed25519", "public_key": signer.public_b64, "fingerprint": signer.fingerprint,
                "signed_bytes": "JSON of the receipt with sorted keys, no spaces (separators ',' and ':'), UTF-8",
                "encoding": "base64url without padding, for both the key and the signature"}

    # ---------- MCP: other AIs can use this agent ----------
    def mcp_ask(args: dict, ip: str) -> str:
        question = str(args.get("question", "")).strip()
        if not question:
            raise mcp.ToolFailure("Pass a question.")
        if len(question) > settings.max_question_chars:
            raise mcp.ToolFailure(f"Keep the question under {settings.max_question_chars} characters.")
        if not mcp_limiter.allow(ip):
            raise mcp.ToolFailure("Too many questions from this client right now. Try again in a few minutes, or use search_harsha_facts, which is free.")
        if budget.exhausted():
            raise mcp.ToolFailure("Ask Harsha is resting for today (daily budget reached). search_harsha_facts and get_harsha_topic still work.")
        agent = build_agent(owner=False)
        try:
            reply = agent.ask(question)
        except Exception as e:
            upstream_failed(e)
            raise mcp.ToolFailure("The agent couldn't reach its model just now. Please try again.")
        data = finish(agent, reply, False, question, event="mcp_answer")
        out = [data["answer"]]
        if data["citations"]:
            out.append("Sources from Harsha's record:\n" + "\n".join(f"- [{c['id']}] {c['text']}" for c in data["citations"]))
        if data["web_sources"]:
            out.append("Web sources:\n" + "\n".join(f"- {w['title']}: {w['url']}" for w in data["web_sources"]))
        if data.get("receipt") and receipts.enabled:
            try:
                rid = receipts.put(data["receipt"], data["signature"])
                out.append(f"Verify this answer (signed by Ask Harsha): {settings_origin['url']}/v/{rid}")
            except rcpt.ReceiptError:
                pass
        return "\n\n".join(out)

    def mcp_search(args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            raise mcp.ToolFailure("Pass a query, for example 'AI agent adoption'.")
        k = args.get("k", 5)
        k = k if isinstance(k, int) and 1 <= k <= 10 else 5
        hits = kb.search(query, k=k)
        if not hits:
            return json.dumps({"query": query, "results": [], "note": "No matching facts. Try other keywords or get_harsha_topic."})
        return json.dumps({"query": query, "results": [f.to_dict() for f, _ in hits]}, ensure_ascii=False)

    def mcp_topic(args: dict) -> str:
        topic = str(args.get("topic", "")).strip().lower()
        facts = kb.by_topic(topic)
        if not facts:
            raise mcp.ToolFailure(f"Unknown topic. Topics: {', '.join(sorted(kb.topics()))}.")
        return json.dumps({"topic": topic, "results": [f.to_dict() for f in facts]}, ensure_ascii=False)

    def mcp_verify(args: dict) -> str:
        link = str(args.get("link", "")).strip().rstrip("/")
        rid = link.rsplit("/v/", 1)[-1].split("?")[0].split("#")[0]
        try:
            data = load_receipt(rid)
        except rcpt.ReceiptError as e:
            raise mcp.ToolFailure(str(e))
        if data is None:
            return "Not verified: there's no Ask Harsha answer at that link."
        r = data["receipt"]
        ok = signer is not None and signer.verify(r, data["sig"])
        status = "Verified: the signature matches, so this is exactly what Ask Harsha said." if ok else "Not verified: the signature doesn't match."
        return f"{status}\nAsked: {r['question']}\nAnswered at {r['issued_at']} by {r['model']}:\n{r['answer']}"

    settings_origin = {"url": ""}

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        settings_origin["url"] = origin(request)
        message, err = mcp.parse(await request.body())
        if err:
            return JSONResponse(err, status_code=400)
        ip = client_ip(request)
        tools = {"ask_harsha_agent": lambda a: mcp_ask(a, ip), "search_harsha_facts": mcp_search,
                 "get_harsha_topic": mcp_topic, "verify_answer": mcp_verify}
        topics = sorted(kb.topics())
        from starlette.concurrency import run_in_threadpool  # the agent call blocks; keep the event loop free
        if isinstance(message, list):  # JSON-RPC batch (older protocol versions)
            replies = [r for r in [await run_in_threadpool(mcp.handle, m, tools, topics) for m in message] if r is not None]
            return JSONResponse(replies) if replies else Response(status_code=202)
        reply = await run_in_threadpool(mcp.handle, message, tools, topics)
        if reply is None:
            return Response(status_code=202)
        if isinstance(message, dict) and message.get("method") == "tools/call":
            log.info(json.dumps({"event": "mcp_call", "tool": (message.get("params") or {}).get("name")}))
        return JSONResponse(reply)

    @app.get("/mcp")
    def mcp_get():
        return Response(status_code=405, headers={"Allow": "POST"})

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index_page():
        return FileResponse(STATIC / "index.html")

    @app.get("/assist")
    def assist_page():
        return FileResponse(STATIC / "assist.html", headers={"Cache-Control": "no-cache"})

    @app.post("/api/assist/interpret")
    def assist_interpret(req: AssistRequest, request: Request):
        """Turn what was heard or typed into up to three sentences. Nothing about the content is stored or logged."""
        if not any(h.strip() for h in req.heard) and not req.typed.strip() and not req.partner.strip():
            return error(400, "empty", "I didn't get anything to work with. Try speaking again or type a few words.")
        ip = client_ip(request)
        status = owner_status(request)
        if status == "denied" and too_many_wrong(ip):
            return error(429, "locked", "Too many wrong passcodes. Try again in an hour.")
        owner = status == "owner"
        if not owner and not assist_limiter.allow(ip):
            wait = assist_limiter.retry_after(ip)
            return error(429, "rate_limited", f"Too many at once. Quick phrases still work. Try again in about {max(1, wait // 60)} minute(s).", {"Retry-After": str(wait)})
        if budget.exhausted():
            return error(503, "resting", "Guessing is resting for today, but quick phrases, saved phrases and typing a full sentence still work.")
        started = time.monotonic()
        try:
            res = assist.interpret(get_client(), settings.assist_model, heard=req.heard, typed=req.typed,
                                   partner=req.partner, past=[p.model_dump() for p in req.past], saved=req.saved)
        except Exception as e:
            upstream_failed(e)
            return error(502, "upstream", "I couldn't reach the model just now. Quick phrases and typing still work.")
        cost = budget.add(res.input_tokens, res.output_tokens)
        mode = "partner" if req.partner.strip() and not (req.typed.strip() or any(h.strip() for h in req.heard)) else ("heard" if any(h.strip() for h in req.heard) else "typed")
        log.info(json.dumps({"event": "assist", "owner": owner, "mode": mode, "guesses": len(res.guesses),
                             "tokens_in": res.input_tokens, "tokens_out": res.output_tokens,
                             "cost_usd": round(cost, 5), "spent_today_usd": round(budget.spent, 4)}))
        return {"guesses": res.guesses, "cost_usd": round(cost, 5), "seconds": round(time.monotonic() - started, 2)}

    # ---------- Ask the real Harsha ----------
    def inbox_error(e: Exception) -> JSONResponse:
        log.error(json.dumps({"event": "inbox_error", "detail": str(e)[:200]}))
        return error(503, "inbox", str(e) if isinstance(e, inboxmod.InboxUnavailable) else "The inbox is unavailable right now. Please try again later.")

    @app.post("/api/inbox")
    def inbox_submit(req: InboxRequest, request: Request):
        if not inbox.enabled:
            return error(503, "inbox", "Ask the real Harsha isn't set up yet.")
        ip = client_ip(request)
        if owner_status(request) != "owner" and not inbox_limiter.allow(ip):
            return error(429, "rate_limited", "You've sent a few questions to Harsha today. Please try again tomorrow.")
        try:
            item = inbox.submit(req.question, req.reason, req.kind, req.context, req.note)
        except ValueError as e:
            return error(400, "empty", str(e))
        except Exception as e:
            return inbox_error(e)
        log.info(json.dumps({"event": "inbox_submit", "kind": item.kind}))
        return {"id": item.id, "status": "waiting"}

    @app.get("/api/inbox/{item_id}")
    def inbox_ticket(item_id: str, request: Request):
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,32}", item_id):
            return error(404, "not_found", "That link isn't valid.")
        if not ticket_limiter.allow(client_ip(request)):
            return error(429, "rate_limited", "Too many checks at once. Try again in a few minutes.")
        try:
            item = inbox.get(item_id)
        except Exception as e:
            return inbox_error(e)
        if item is None or item.status == "dismissed":
            return error(404, "not_found", "That question isn't in Harsha's inbox.")
        return item.public()

    @app.get("/api/owner/inbox")
    def inbox_list(request: Request):
        denied = require_owner(request)
        if denied:
            return denied
        try:
            items = inbox.all()
        except Exception as e:
            return inbox_error(e)
        order = {"open": 0, "answered": 1, "dismissed": 2}
        items.sort(key=lambda i: (order.get(i.status, 3), i.created if i.status != "open" else ""), reverse=False)
        open_items = [i for i in items if i.status == "open"]
        done = sorted([i for i in items if i.status == "answered"], key=lambda i: i.answered_at, reverse=True)[:30]
        from dataclasses import asdict
        return {"open": [asdict(i) for i in sorted(open_items, key=lambda i: i.created)], "answered": [asdict(i) for i in done]}

    @app.post("/api/owner/inbox/{item_id}/answer")
    def inbox_answer(item_id: str, req: InboxAnswer, request: Request):
        denied = require_owner(request)
        if denied:
            return denied
        try:
            item = inbox.get(item_id)
            if item is None:
                return error(404, "not_found", "That question isn't in the inbox.")
            note_id = ""
            if req.remember:
                text = f'Asked "{item.question[:200]}", Harsha answered: {req.answer.strip()}'
                try:
                    note_id = store.add(text[:500]).id
                except (ValueError, NotesUnavailable) as e:
                    return error(503, "storage", f"Your answer wasn't sent because saving it to memory failed: {e}")
            item = inbox.answer(item_id, req.answer, note_id=note_id)
        except ValueError as e:
            return error(400, "empty", str(e))
        except Exception as e:
            return inbox_error(e)
        log.info(json.dumps({"event": "inbox_answer", "remembered": bool(note_id)}))
        return {"item": item.public(), "note_id": note_id}

    @app.post("/api/owner/inbox/{item_id}/dismiss")
    def inbox_dismiss(item_id: str, request: Request):
        denied = require_owner(request)
        if denied:
            return denied
        try:
            inbox.dismiss(item_id)
        except KeyError:
            return error(404, "not_found", "That question isn't in the inbox.")
        except Exception as e:
            return inbox_error(e)
        return {"ok": True}

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

    def prepare(req: ChatRequest, request: Request, stream: bool):
        """Validate and build the agent for one question. Returns (agent, owner, question) or an error response."""
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
        try:
            prepared = files.prepare(req.attachments, max_pdf_pages=100 if owner else 30)
        except files.AttachmentError as e:
            return error(400, "attachment", str(e))
        agent = build_agent(owner, stream)
        agent.messages = file_turns(prepared.blocks) + clean_history(req.history, settings.max_history_turns)
        agent.attached = prepared.summary
        return agent, owner, question

    def build_agent(owner: bool, stream: bool = False) -> Agent:
        today = datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %B %-d, %Y")
        agent = Agent(
            get_client(), kb=kb, model=settings.model, max_steps=8, max_tokens=1600,
            system=build_general_prompt(today, owner=owner, web=settings.web_search),
            server_tools=[WEB_SEARCH_TOOL] if settings.web_search else [],
            lookup=_Lookup(kb, store), strict_grounding=False, stream=stream,
        )
        add_notes_tools(agent.tools, index, store, owner=owner)
        if inbox.enabled:
            add_ask_harsha_tool(agent.tools)
        return agent

    def finish(agent: Agent, reply, owner: bool, question: str, event: str = "answer") -> dict:
        """Charge the budget, log one line, and build the response payload."""
        cost = budget.add(reply.input_tokens, reply.output_tokens) + budget.add_flat(reply.web_searches * settings.price_per_search)
        report = reply.report
        lookup = _Lookup(kb, store).by_id
        cited = [{"id": c, "text": lookup[c].text, "kind": "note" if c.startswith("N") else "fact"} for c in report.cited if c in lookup]
        label = answer_kind(report, owner, question, reply, has_files=bool(getattr(agent, "attached", None)))
        record = {"event": event, "owner": owner, "kind": report.kind, "grounded": report.ok, "steps": reply.steps,
                  "web_searches": reply.web_searches, "memory_changes": len(reply.memory_changes),
                  "tokens_in": reply.input_tokens, "tokens_out": reply.output_tokens, "cost_usd": round(cost, 5),
                  "declined": bool(DECLINE.search(reply.text)), "spent_today_usd": round(budget.spent, 4),
                  "attachments": [a["kind"] for a in getattr(agent, "attached", [])],
                  "cache_read": reply.cache_read_tokens, "cache_write": reply.cache_write_tokens}
        if settings.log_questions or event == "eval_case":  # eval questions are our own test cases
            record["question"] = question[:200]
        log.info(json.dumps(record))
        return {
            "answer": reply.text,
            "kind": report.kind,
            "label": label,
            "trace": reply.steps_log,
            "grounded": report.ok,
            "check": report.summary(),
            "citations": cited,
            "web_sources": reply.web_sources[:8],
            "memory_changes": reply.memory_changes,
            "ask_harsha": agent.tools.offer,
            "owner": owner,
            "retrieved": sorted(agent.tools.retrieved),
            "sources": {i: {"text": lookup[i].text, "kind": "note" if i.startswith("N") else "fact"} for i in agent.tools.retrieved if i in lookup},
            "steps": reply.steps,
            "cost_usd": round(cost, 5),
            **sign_answer(question, reply.text, label, report.ok, cited, reply.web_sources[:8],
                          [a["kind"] for a in getattr(agent, "attached", [])]),
        }

    def sign_answer(question, answer, label, grounded, cited, web_sources, attachments) -> dict:
        if signer is None or not answer:
            return {}
        receipt = rcpt.build(question, answer, settings.model, label, grounded, cited, web_sources, attachments, signer.fingerprint)
        return {"receipt": receipt, "signature": signer.sign(receipt)}

    def upstream_failed(e: Exception) -> None:
        log.error(json.dumps({"event": "upstream_error", "type": type(e).__name__, "detail": str(e)[:300]}))

    UPSTREAM_MSG = "Ask Harsha couldn't reach its model just now. Please try again in a moment."

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request):
        prep = prepare(req, request, stream=False)
        if isinstance(prep, JSONResponse):
            return prep
        agent, owner, question = prep
        try:
            reply = agent.ask(question)
        except Exception as e:  # API outage, bad key, network: log the cause, show a friendly message
            upstream_failed(e)
            return error(502, "upstream", UPSTREAM_MSG)
        return finish(agent, reply, owner, question)

    STATUS = {
        "search_facts": "Looking up Harsha's career facts",
        "get_topic": "Opening Harsha's record",
        "list_topics": "Checking what's in Harsha's record",
        "search_notes": "Searching Harsha's memory",
        "remember_note": "Saving to memory",
        "forget_note": "Removing from memory",
        "suggest_asking_harsha": "Flagging this for Harsha",
    }

    def sse(kind: str, data: dict) -> str:
        return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def start_worker(work, q: "queue.Queue") -> None:
        def runner():
            try:
                work()
            finally:
                q.put(None)
        threading.Thread(target=runner, daemon=True).start()

    def stream_events(q: "queue.Queue", work=None):
        """Yield queued events as Server-Sent Events, with keep-alive pings.

        With `work`, the worker starts only once the client begins reading, so an abandoned request costs nothing.
        """
        if work is not None:
            start_worker(work, q)
        yield ": connected\n\n"
        while True:
            try:
                item = q.get(timeout=15)
            except queue.Empty:
                yield ": ping\n\n"  # keeps proxies from closing a quiet connection while the model thinks
                continue
            if item is None:
                return
            yield sse(*item)

    SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest, request: Request):
        prep = prepare(req, request, stream=True)
        if isinstance(prep, JSONResponse):
            return prep
        agent, owner, question = prep
        q: queue.Queue = queue.Queue()

        def on_event(kind: str, data: dict) -> None:
            if kind == "text":
                q.put(("text", {"delta": data["delta"]}))
            elif kind == "text_reset":
                q.put(("text_reset", {}))
            elif kind == "web_search_start":
                q.put(("status", {"label": "Searching the web"}))
            elif kind == "tool_call":
                q.put(("status", {"label": STATUS.get(data["name"], "Working")}))

        agent.on_event = on_event

        def work():
            try:
                n = len(agent.attached)
                if n:
                    q.put(("status", {"label": "Reading your file" if n == 1 else f"Reading your {n} files"}))
                reply = agent.ask(question)
                q.put(("final", finish(agent, reply, owner, question)))
            except Exception as e:
                upstream_failed(e)
                q.put(("error", {"message": UPSTREAM_MSG}))

        return StreamingResponse(stream_events(q, work), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.post("/api/factcheck/stream")
    def factcheck_stream(req: FactCheckRequest, request: Request):
        answer = req.answer.strip()
        if not answer:
            return error(400, "empty", "There's nothing to check.")
        ip = client_ip(request)
        status = owner_status(request)
        if status == "denied" and too_many_wrong(ip):
            return error(429, "locked", "Too many wrong passcodes. Try again in an hour.")
        owner = status == "owner"
        if not owner and not limiter.allow(ip):
            wait = limiter.retry_after(ip)
            return error(429, "rate_limited", f"That's a lot at once. Try again in about {max(1, wait // 60)} minute(s).", {"Retry-After": str(wait)})
        if budget.exhausted():
            return error(503, "resting", f"Ask Harsha is resting for today. You can reach Harsha directly at {settings.contact_email}.")
        q: queue.Queue = queue.Queue()
        today = datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %B %-d, %Y")

        def work():
            started = time.monotonic()
            try:
                res = factcheck.run(get_client(), settings.model, answer, _Lookup(kb, store).by_id, today,
                                    settings.web_search, FACTCHECK_SEARCH_TOOL, lambda k, d: q.put((k, d)), kb=kb)
            except factcheck.FactCheckError as e:
                q.put(("error", {"message": f"The fact check couldn't finish: {e} Please try again."}))
                return
            except Exception as e:
                upstream_failed(e)
                q.put(("error", {"message": UPSTREAM_MSG}))
                return
            cost = budget.add(res.input_tokens, res.output_tokens) + budget.add_flat(res.web_searches * settings.price_per_search)
            lookup = _Lookup(kb, store).by_id
            texts = {s["id"]: lookup[s["id"]].text for v in res.verdicts for s in v["sources"] if s["type"] == "record" and s["id"] in lookup}
            counts = res.counts()
            log.info(json.dumps({"event": "factcheck", "owner": owner, "claims": len(res.claims), **counts,
                                 "downgraded": res.downgraded, "web_searches": res.web_searches,
                                 "tokens_in": res.input_tokens, "tokens_out": res.output_tokens,
                                 "cost_usd": round(cost, 5), "spent_today_usd": round(budget.spent, 4)}))
            q.put(("verdicts", {"verdicts": res.verdicts, "record_texts": texts}))
            q.put(("final", {"counts": counts, "claims": len(res.claims), "cost_usd": round(cost, 4),
                             "seconds": round(time.monotonic() - started, 1), "downgraded": res.downgraded}))

        return StreamingResponse(stream_events(q, work), media_type="text/event-stream", headers=SSE_HEADERS)

    eval_lock = threading.Lock()

    @app.post("/api/owner/evals")
    def run_evals(request: Request):
        denied = require_owner(request)
        if denied:
            return denied
        if budget.exhausted():
            return error(503, "resting", "Today's budget is used up, so evals can't run until tomorrow.")
        if not eval_lock.acquire(blocking=False):
            return error(409, "busy", "An eval run is already in progress.")
        cases = load_cases(mode="general", web=settings.web_search)
        q: queue.Queue = queue.Queue()

        def work():
            started, passed, cost, rows = time.monotonic(), 0, 0.0, []
            try:
                q.put(("start", {"total": len(cases)}))
                for i, case in enumerate(cases, 1):
                    if budget.exhausted():
                        q.put(("stopped", {"message": "Stopped early: today's budget is used up.", "done": i - 1}))
                        break
                    agent = build_agent(owner=False)  # evals test what visitors get
                    t0 = time.monotonic()
                    try:
                        reply = agent.ask(case["question"])
                        payload = finish(agent, reply, False, case["question"], event="eval_case")
                        failures = score_case(case, reply.text, reply.report, payload["label"], reply.memory_changes)
                        answer, case_cost = reply.text, payload["cost_usd"]
                    except Exception as e:
                        upstream_failed(e)
                        failures, answer, case_cost = ["model unreachable"], "", 0.0
                    ok = not failures
                    passed += ok
                    cost += case_cost
                    row = {"n": i, "id": case["id"], "question": case["question"], "pass": ok, "failures": failures,
                           "answer": answer[:1500], "seconds": round(time.monotonic() - t0, 1), "cost_usd": case_cost}
                    rows.append(row)
                    q.put(("case", row))
                summary = {"passed": passed, "total": len(rows), "cost_usd": round(cost, 4),
                           "seconds": round(time.monotonic() - started, 1), "model": settings.model}
                log.info(json.dumps({"event": "eval_run", **summary, "failed": [r["id"] for r in rows if not r["pass"]]}))
                q.put(("summary", summary))
            finally:
                eval_lock.release()

        start_worker(work, q)  # started now, so the lock is always released even if the page goes away
        return StreamingResponse(stream_events(q), media_type="text/event-stream", headers=SSE_HEADERS)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
