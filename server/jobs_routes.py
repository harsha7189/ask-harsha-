"""Job Pilot's API: settings, the nightly run, the tracker, and the endpoints the laptop runner calls.

Access: Job Pilot spends real money and acts under one person's name, so it's limited to the owner (signed in with a
passkey and the owner passcode). The runner authenticates with its own token, made in the settings and revocable.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import queue
import secrets
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import zipfile

from agent import attachments as files
from agent import jobsearch as js
from agent import learning as lr
from agent import pilot
from agent import resume as rs

SHEET_FIELDS = ["first_name", "last_name", "email", "phone", "address", "city", "state", "zip", "country", "linkedin", "website",
                "authorized", "sponsorship", "relocate", "salary", "start_date", "notice", "adult", "previous_employee", "noncompete",
                "how_heard", "gender", "race", "veteran", "disability", "pronouns", "remote_ok", "certify"]


class ResumeUpload(BaseModel):
    name: str = Field(default="resume", max_length=200)
    data: str = Field(min_length=10, max_length=15_000_000)     # base64 of a .docx, .pdf or .txt
    add: bool = False                                            # add bullets to the bank instead of replacing the base


class ProfileUpdate(BaseModel):
    enabled: bool | None = None
    rules: dict | None = None
    resume_rules: dict | None = None
    evidence_notes: str | None = Field(default=None, max_length=20000)


class OutcomeUpdate(BaseModel):
    outcome: str = Field(pattern="^(recruiter_reply|interview|offer|rejected|clear)$")


class RunnerOutcomes(BaseModel):
    items: list[dict] = Field(max_length=200)


class SheetUpdate(BaseModel):
    sheet: dict


class AppAction(BaseModel):
    action: str = Field(pattern="^(skip|requeue|mark_submitted|retry)$")


class RunnerFields(BaseModel):
    app_id: str
    fields: list[dict] = Field(max_length=80)


class RunnerReport(BaseModel):
    app_id: str
    status: str = Field(pattern="^(applying|filled|submitted|needs_you|failed)$")
    step: str = Field(default="", max_length=200)
    detail: str = Field(default="", max_length=2000)
    screenshots: list[str] = Field(default_factory=list, max_length=6)    # base64 JPEG/PNG
    confirmation: str = Field(default="", max_length=4000)


def install(app, *, accounts, telemetry, budget, get_client, settings, require_owner, current_user, error, stream_events, start_worker,
            sse_headers, upstream_failed, log, fernet, static_dir, web_search: bool):

    def owner_user(request: Request):
        denied = require_owner(request)
        if denied:
            return denied
        if accounts is None:
            return error(503, "accounts", "Job Pilot needs the database (DATABASE_URL).")
        user = current_user(request)
        if not user:
            return error(401, "signed_out", "Sign in with your passkey to use Job Pilot.")
        return user

    def profile_of(uid: str) -> dict:
        p = accounts.job_profile(uid) or {"user_id": uid, "enabled": False, "rules": {}, "resume_rules": {}}
        p["sheet"] = decrypt(p.get("sheet_enc"))
        p["evidence_notes"] = decrypt(p.get("notes_enc")).get("text", "")
        return p

    def decrypt(token: str | None) -> dict:
        if not token or fernet is None:
            return {}
        try:
            return json.loads(fernet.decrypt(token.encode()).decode())
        except Exception:
            return {}

    def today() -> str:
        return datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %B %-d, %Y")

    def public(p: dict) -> dict:
        base = p.get("base") or {}
        return {"enabled": bool(p.get("enabled")), "rules": pilot.rules_of(p), "resume_rules": {**rs.DEFAULT_RULES, **(p.get("resume_rules") or {})},
                "resume": {"name": base.get("name"), "jobs": [{"company": e["company"], "title": e["title"], "start": e["start"], "end": e["end"],
                                                              "bullets": len(e["bullets"])} for e in base.get("experience", [])],
                           "bank": len(base.get("bank", [])), "skills": len(base.get("skills", [])), "dropped": base.get("parse_dropped", 0)} if base else None,
                "sheet_fields": {k: bool(p["sheet"].get(k)) for k in SHEET_FIELDS}, "runner": bool(p.get("runner_hash")),
                "companies": accounts.company_count(), "sponsor_index": bool(telemetry.get("sponsors:meta")),
                "encryption": fernet is not None, "last_run": telemetry.get(f"jobs:last_run:{p['user_id']}"),
                "evidence_notes": p.get("evidence_notes", "")}

    @app.get("/jobs")
    def jobs_page():
        return FileResponse(static_dir / "jobs.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/jobs/profile")
    def get_profile(request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        return public(profile_of(user.id))

    @app.put("/api/jobs/profile")
    def put_profile(req: ProfileUpdate, request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        p = profile_of(user.id)
        if req.enabled is not None:
            if req.enabled and not p.get("base"):
                return error(400, "no_resume", "Upload your résumé before turning Job Pilot on.")
            p["enabled"] = req.enabled
        if req.rules is not None:
            p["rules"] = {k: v for k, v in req.rules.items() if k in pilot.DEFAULT_RULES}
        if req.resume_rules is not None:
            p["resume_rules"] = {k: v for k, v in req.resume_rules.items() if k in rs.DEFAULT_RULES}
        if req.evidence_notes is not None:
            if fernet is None:
                return error(503, "encryption", "Set INBOX_KEY (or OWNER_PASSCODE) on the server so your notes can be encrypted.")
            p["notes_enc"] = fernet.encrypt(json.dumps({"text": req.evidence_notes.strip()}).encode()).decode()
        accounts.save_job_profile(user.id, p)
        return public(profile_of(user.id))

    @app.post("/api/jobs/resume")
    def upload_resume(req: ResumeUpload, request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        raw = base64.b64decode(req.data.split(",")[-1])
        name = req.name.lower()
        try:
            if name.endswith(".docx"):
                text = files.docx_text(raw, req.name)
            elif name.endswith(".pdf"):
                import pypdf
                text = "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(io.BytesIO(raw)).pages)
            else:
                text = raw.decode("utf-8", "replace")
        except Exception as e:
            return error(400, "resume", f"Couldn't read that file: {e}")
        if len(text.strip()) < 200:
            return error(400, "resume", "That file has too little text to be a résumé.")
        usage = rs.Usage()
        try:
            parsed = rs.parse_resume(get_client(), settings.assist_model, text, usage)
        except Exception as e:
            upstream_failed(e)
            return error(400, "resume", f"Couldn't read the résumé's structure ({type(e).__name__}). Try a .docx.")
        budget.add(usage.input_tokens, usage.output_tokens)
        p = profile_of(user.id)
        if req.add and p.get("base"):
            added = rs.add_to_bank(p["base"], parsed)
            msg = f"Added {added} new true bullet{'s' if added != 1 else ''} to your bank."
        else:
            p["base"] = parsed
            msg = f"Read {sum(len(e['bullets']) for e in parsed['experience'])} bullets across {len(parsed['experience'])} jobs."
            if parsed.get("parse_dropped"):
                msg += f" {parsed['parse_dropped']} line(s) weren't copied exactly, so they were left out."
        accounts.save_job_profile(user.id, p)
        return {"message": msg, **public(profile_of(user.id))}

    @app.get("/api/jobs/sheet")
    def get_sheet(request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        return {"sheet": profile_of(user.id)["sheet"], "fields": SHEET_FIELDS}

    @app.put("/api/jobs/sheet")
    def put_sheet(req: SheetUpdate, request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        if fernet is None:
            return error(503, "encryption", "Set INBOX_KEY (or OWNER_PASSCODE) on the server so the answer sheet can be encrypted.")
        sheet = {k: str(v)[:500] for k, v in req.sheet.items() if k in SHEET_FIELDS and v not in (None, "")}
        custom = [{"question": str(c.get("question", ""))[:300], "answer": str(c.get("answer", ""))[:1000]}
                  for c in (req.sheet.get("custom") or [])[:40] if isinstance(c, dict) and c.get("question")]
        sheet["custom"] = custom
        p = profile_of(user.id)
        accounts.save_job_profile(user.id, p, sheet_enc=fernet.encrypt(json.dumps(sheet).encode()).decode())
        return {"ok": True, "saved": sorted(k for k in sheet if k != "custom"), "custom": len(custom)}

    @app.post("/api/jobs/runner-token")
    def new_runner_token(request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        token = "jp_" + secrets.token_urlsafe(24)
        p = profile_of(user.id)
        accounts.save_job_profile(user.id, p, runner_hash=hashlib.sha256(token.encode()).hexdigest())
        return {"token": token, "note": "Shown once. Put it in the runner's settings. Making a new one turns this one off."}

    @app.post("/api/jobs/sponsors")
    async def upload_sponsors(request: Request):
        """The compact H-1B index built by runner/build_sponsor_index.py from the Department of Labor's public data.
        Accepts the owner's session or the runner token (the script runs on the laptop)."""
        user = runner(request) if request.headers.get("X-Runner-Token") else owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        body = await request.json()
        index = body.get("index") or {}
        if not isinstance(index, dict) or not index:
            return error(400, "index", "That index is empty.")
        clean = {js.norm_company(k): {"pm": int(v.get("pm", 0)), "total": int(v.get("total", 0)), "years": v.get("years", [])[:4]}
                 for k, v in list(index.items())[:300_000] if isinstance(v, dict) and js.norm_company(k)}
        telemetry.put("sponsors:index", clean)
        telemetry.put("sponsors:meta", {"employers": len(clean), "source": str(body.get("source", ""))[:300], "uploaded": int(time.time() * 1000)})
        return {"employers": len(clean)}

    run_lock = threading.Lock()

    def night(uid: str, emit) -> dict:
        p = profile_of(uid)
        p["user_id"] = uid
        res = pilot.run_night(get_client(), settings.model, settings.assist_model, today(), p, accounts, emit,
                              sponsor_index=telemetry.get("sponsors:index") or {}, web=web_search, budget_ok=lambda: not budget.exhausted())
        if res.get("ok"):
            budget.add(res["input_tokens"], res["output_tokens"])
            budget.add_flat(res["web_searches"] * settings.price_per_search)
        telemetry.put(f"jobs:last_run:{uid}", {**{k: v for k, v in res.items() if k != "errors"}, "errors": res.get("errors", [])[:10],
                                                "at": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds")})
        telemetry.add("jobs_run", {k: res.get(k) for k in ("jobs_seen", "screened_in", "prepared", "ready", "needs_you", "seconds")})
        log.info(json.dumps({"event": "job_pilot_run", **{k: res.get(k) for k in ("jobs_seen", "screened_in", "prepared", "ready", "needs_you", "seconds")}}))
        return res

    @app.post("/api/jobs/run")
    def run_now(request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        if budget.exhausted():
            return error(503, "resting", "Today's budget is used up.")
        if not run_lock.acquire(blocking=False):
            return error(409, "busy", "Job Pilot is already running.")
        q: queue.Queue = queue.Queue()

        def work():
            try:
                q.put(("final", night(user.id, lambda k, d: q.put((k, d)))))
            except Exception as e:
                upstream_failed(e)
                q.put(("error", {"message": f"The run stopped ({type(e).__name__})."}))
            finally:
                run_lock.release()

        start_worker(work, q)
        return StreamingResponse(stream_events(q), media_type="text/event-stream", headers=sse_headers)

    def nightly():
        """Each enabled profile runs once a night at its run hour (server time zone)."""
        while True:
            time.sleep(900)
            try:
                now = datetime.now(ZoneInfo(settings.timezone))
                for uid in accounts.job_profiles_enabled():
                    hour = int(pilot.rules_of(accounts.job_profile(uid) or {}).get("run_hour", 22))
                    if now.hour >= hour and not budget.exhausted() and telemetry.claim(f"jobs:night:{uid}", now.date().isoformat()):
                        with run_lock:
                            night(uid, lambda k, d: None)
            except Exception as e:
                log.error(json.dumps({"event": "job_pilot_nightly_error", "type": type(e).__name__, "detail": str(e)[:200]}))

    if accounts is not None and settings_flag("JOB_PILOT_NIGHTLY"):
        threading.Thread(target=nightly, daemon=True, name="job-pilot-nightly").start()

    @app.get("/api/jobs/apps")
    def list_apps(request: Request, status: str = ""):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        return {"apps": accounts.apps(user.id, status or None)}

    @app.get("/api/jobs/apps/{app_id}")
    def get_app(app_id: str, request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        a = accounts.app(user.id, app_id)
        if not a:
            return error(404, "not_found", "That application doesn't exist.")
        a.pop("files_b64", None)
        return a

    @app.get("/api/jobs/apps/{app_id}/file/{kind}")
    def app_file(app_id: str, kind: str, request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        return send_file(user.id, app_id, kind)

    def send_file(uid: str, app_id: str, kind: str):
        if kind not in ("pdf", "docx"):
            return error(404, "not_found", "No such file.")
        data = accounts.app_file(uid, app_id, kind)
        if not data:
            return error(404, "not_found", "That file isn't ready.")
        a = accounts.app(uid, app_id) or {}
        name = f"{((a.get('resume') or {}).get('cv') or {}).get('name', 'Resume').replace(' ', '_')}_{(a.get('job') or {}).get('company', '')}".strip("_")
        name = "".join(ch for ch in name if ch.isalnum() or ch in "_-")[:80] or "Resume"
        mime = "application/pdf" if kind == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return Response(data, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}.{kind}"'})

    @app.post("/api/jobs/apps/{app_id}/action")
    def app_action(app_id: str, req: AppAction, request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        new = {"skip": "skipped", "requeue": "ready", "retry": "found", "mark_submitted": "submitted"}[req.action]
        extra = {"submitted_at": int(time.time() * 1000)} if new == "submitted" else {}
        try:
            accounts.set_app_status(user.id, app_id, new, detail=f"by you: {req.action}", **extra)
        except ValueError as e:
            return error(404, "not_found", str(e))
        return {"ok": True, "status": new}

    # ---------- outcomes, learning and the morning report ----------
    RANK = {"rejected": 1, "recruiter_reply": 2, "interview": 3, "offer": 4}

    def record_outcome(uid: str, app_id: str, outcome: str, source: str, at: int | None = None) -> dict:
        a = accounts.app(uid, app_id)
        if not a:
            raise ValueError("That application doesn't exist.")
        at = at or int(time.time() * 1000)
        log_ = (a.get("outcome_log") or []) + [{"outcome": outcome, "source": source, "at": at}]
        if outcome == "clear":
            best = None
        else:   # the furthest it got is what counts (an interview followed by a rejection still got an interview)
            best = max([outcome] + ([a["outcome"]] if a.get("outcome") in RANK else []), key=lambda o: RANK[o])
        accounts.save_app(uid, {"key": a["key"], "outcome": best, "outcome_at": at, "outcome_log": log_[-20:], "force": True,
                                "status": a["status"]})
        return {"outcome": best}

    @app.post("/api/jobs/apps/{app_id}/outcome")
    def set_outcome(app_id: str, req: OutcomeUpdate, request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        try:
            return record_outcome(user.id, app_id, req.outcome, "you")
        except ValueError as e:
            return error(404, "not_found", str(e))

    @app.get("/api/jobs/learning")
    def learning(request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        return lr.analyze(accounts.app_bodies(user.id, None, 1000))

    def report_for(uid: str, hours: int) -> dict:
        since = int(time.time() * 1000) - hours * 3600 * 1000
        bodies = [b for b in accounts.app_bodies(uid, None, 600) if (b.get("updated") or 0) >= since]
        def card(b):
            ch = b.get("check") or {}
            job = b.get("job") or {}
            return {"id": b["id"], "company": job.get("company"), "title": job.get("title"), "location": job.get("location"), "url": job.get("url"),
                    "ats": job.get("ats"), "status": b["status"], "submitted_at": b.get("submitted_at"),
                    "strategy": lr.STRATEGIES.get(b.get("strategy"), {}).get("label"), "match": (b.get("qualify") or {}).get("match"),
                    "sponsor": ((b.get("screen") or {}).get("sponsor") or {}).get("level"), "problem": (b.get("brief") or {}).get("problem", ""),
                    "first_impression": (ch.get("recruiter") or {}).get("first_impression", ""), "defensibility": ch.get("defensibility"),
                    "ats_ok": (ch.get("ats") or {}).get("passed"), "replaced": len(ch.get("replaced") or []), "why": (b.get("why") or [])[-2:],
                    "prep": len(b.get("prep") or []), "has_files": bool((b.get("resume") or {}).get("text"))}
        out = {"since": since, "hours": hours, "submitted": [], "needs_you": [], "failed": [], "ready": [], "skipped": 0, "skip_reasons": {}}
        for b in bodies:
            st = b["status"]
            if st == "submitted" and (b.get("submitted_at") or 0) >= since:
                out["submitted"].append(card(b))
            elif st in ("needs_you", "failed", "ready"):
                out[st].append(card(b))
            elif st == "skipped":
                out["skipped"] += 1
                r = ((b.get("why") or ["other"])[0] or "other").split(":")[0][:60]
                out["skip_reasons"][r] = out["skip_reasons"].get(r, 0) + 1
        out["submitted"].sort(key=lambda c: c["submitted_at"] or 0)
        out["last_run"] = telemetry.get(f"jobs:last_run:{uid}")
        out["learning"] = lr.analyze(accounts.app_bodies(uid, None, 1000))["verdict"]
        return out

    @app.get("/api/jobs/report")
    def report(request: Request, hours: int = 24):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        return report_for(user.id, max(1, min(hours, 24 * 14)))

    def prep_markdown(b: dict) -> str:
        job, brief = b.get("job") or {}, b.get("brief") or {}
        lines = [f"# {job.get('title')} at {job.get('company')}", "", f"Posting: {job.get('url')}", "",
                 "## What they're hiring to solve", brief.get("problem", ""), "", "## Interview prep: questions a hiring manager will ask", ""]
        for p in b.get("prep") or []:
            lines.append(f"**{p['claim']}**")
            lines += [f"- {q}" for q in p.get("questions", [])]
            if p.get("prepare"):
                lines.append(f"- Have ready: {p['prepare']}")
            lines.append("")
        musts = (b.get("qualify") or {}).get("musts") or []
        if musts:
            lines += ["## Must-haves and your evidence", ""] + [f"- {m['must_have']}: {m['status']}" + (f" (“{m['quote']}”)" if m.get("quote") else "") for m in musts]
        return "\n".join(lines) + "\n"

    @app.get("/api/jobs/report/zip")
    def report_zip(request: Request, hours: int = 24):
        """Every résumé sent in the window (PDF and Word) plus an interview prep sheet for each, in one download."""
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        rep = report_for(user.id, max(1, min(hours, 24 * 14)))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            summary = [f"Job Pilot: {len(rep['submitted'])} application(s) submitted in the last {rep['hours']} hours", ""]
            for c in rep["submitted"]:
                folder = "".join(ch for ch in f"{c['company']}_{c['title']}" if ch.isalnum() or ch in " _-").strip().replace(" ", "_")[:70] or c["id"]
                for kind in ("pdf", "docx"):
                    data = accounts.app_file(user.id, c["id"], kind)
                    if data:
                        z.writestr(f"{folder}/Resume.{kind}", data)
                b = accounts.app(user.id, c["id"]) or {}
                b.pop("files_b64", None)
                z.writestr(f"{folder}/Interview_prep.md", prep_markdown(b))
                summary.append(f"- {c['title']} at {c['company']} ({c['location']}): {c['url']}")
            z.writestr("Summary.txt", "\n".join(summary) + "\n")
        stamp = datetime.now(ZoneInfo(settings.timezone)).strftime("%Y-%m-%d")
        return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="Job_Pilot_{stamp}.zip"'})

    @app.get("/api/jobs/apps/{app_id}/prep")
    def app_prep(app_id: str, request: Request):
        user = owner_user(request)
        if isinstance(user, JSONResponse):
            return user
        a = accounts.app(user.id, app_id)
        if not a:
            return error(404, "not_found", "That application doesn't exist.")
        return Response(prep_markdown(a), media_type="text/markdown; charset=utf-8")

    # ---------- the laptop runner ----------
    def runner(request: Request):
        token = request.headers.get("X-Runner-Token", "")
        p = accounts.job_profile_by_runner(hashlib.sha256(token.encode()).hexdigest()) if (accounts is not None and token.startswith("jp_")) else None
        if not p:
            return error(401, "runner", "This runner token isn't valid. Make a new one in Job Pilot's settings.")
        p["sheet"] = decrypt(p.get("sheet_enc"))
        return p

    @app.get("/api/jobs/runner/next")
    def runner_next(request: Request):
        p = runner(request)
        if isinstance(p, JSONResponse):
            return p
        rules = pilot.rules_of(p)
        since = int(datetime.now(ZoneInfo(settings.timezone)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000) - 6 * 3600 * 1000
        done = accounts.apps_since(p["user_id"], "submitted", since) + accounts.apps_since(p["user_id"], "filled", since)
        if done >= int(rules.get("nightly_cap", 10)):
            return {"app": None, "reason": "Tonight's cap is reached."}
        ready = accounts.apps(p["user_id"], "ready", 50)
        if not ready:
            return {"app": None, "reason": "Nothing ready."}
        ready.sort(key=lambda a: -(a.get("priority") or 0))
        a = accounts.app(p["user_id"], ready[0]["id"])
        accounts.set_app_status(p["user_id"], a["id"], "applying", detail="runner picked it up")
        sheet = p["sheet"]
        return {"app": {"id": a["id"], "company": a["job"]["company"], "title": a["job"]["title"], "url": a["job"]["url"], "ats": a["job"]["ats"],
                        "answers": a.get("answers", [])}, "dry_run": bool(rules.get("dry_run", True)),
                "profile": {k: sheet.get(k, "") for k in ("first_name", "last_name", "email", "phone", "city", "state", "zip", "country", "address", "linkedin", "website")}}

    @app.get("/api/jobs/runner/file/{app_id}/{kind}")
    def runner_file(app_id: str, kind: str, request: Request):
        p = runner(request)
        if isinstance(p, JSONResponse):
            return p
        return send_file(p["user_id"], app_id, kind)

    @app.post("/api/jobs/runner/answers")
    def runner_answers(req: RunnerFields, request: Request):
        """Answers for the fields the runner found on a form page. Nothing unsupported is ever guessed."""
        p = runner(request)
        if isinstance(p, JSONResponse):
            return p
        a = accounts.app(p["user_id"], req.app_id)
        if not a:
            return error(404, "not_found", "That application doesn't exist.")
        usage = rs.Usage()
        cache = {x["label"]: x for x in a.get("answers", []) if x.get("value") is not None}
        out = []
        for f in req.fields[:80]:
            label = str(f.get("label", ""))[:400]
            if not label:
                continue
            if label in cache:
                out.append({"label": label, **{k: cache[label][k] for k in ("value", "source", "quote", "needs_you", "why")}})
                continue
            q = {"label": label, "options": [str(o)[:200] for o in (f.get("options") or [])[:60]], "type": str(f.get("type", ""))[:30],
                 "required": bool(f.get("required"))}
            ans = rs.answer_question(get_client(), settings.assist_model, q, p["sheet"], (a.get("resume") or {}).get("text", ""),
                                     (p.get("base") or {}).get("source_text", ""), usage)
            out.append({"label": label, **ans})
        budget.add(usage.input_tokens, usage.output_tokens)
        a.setdefault("answers", [])
        known = {x["label"] for x in a["answers"]}
        a["answers"] += [x for x in out if x["label"] not in known]
        accounts.save_app(p["user_id"], {**a, "force": True})
        return {"answers": out}

    @app.get("/api/jobs/runner/watch")
    def runner_watch(request: Request):
        """Submitted applications from the last 90 days, so the laptop can match replies in his inbox to them locally."""
        p = runner(request)
        if isinstance(p, JSONResponse):
            return p
        cutoff = int(time.time() * 1000) - 90 * 86_400_000
        return {"apps": [{"id": b["id"], "company": (b.get("job") or {}).get("company", ""), "ats": (b.get("job") or {}).get("ats", ""),
                          "submitted_at": b.get("submitted_at"), "outcome": b.get("outcome")}
                         for b in accounts.app_bodies(p["user_id"], "submitted", 500) if (b.get("submitted_at") or 0) >= cutoff]}

    @app.post("/api/jobs/runner/outcomes")
    def runner_outcomes(req: RunnerOutcomes, request: Request):
        """Only {app_id, outcome, at} arrives: the email itself never leaves the laptop."""
        p = runner(request)
        if isinstance(p, JSONResponse):
            return p
        n = 0
        for it in req.items:
            o = str(it.get("outcome", ""))
            if o not in RANK:
                continue
            try:
                record_outcome(p["user_id"], str(it.get("app_id", "")), o, "email", int(it.get("at") or 0) or None)
                n += 1
            except (ValueError, TypeError):
                continue
        return {"recorded": n}

    @app.post("/api/jobs/runner/report")
    def runner_report(req: RunnerReport, request: Request):
        p = runner(request)
        if isinstance(p, JSONResponse):
            return p
        a = accounts.app(p["user_id"], req.app_id)
        if not a:
            return error(404, "not_found", "That application doesn't exist.")
        shots = [s.split(",")[-1] for s in req.screenshots if len(s) < 3_000_000][:6]
        extra = {"step": req.step, "detail": req.detail}
        if shots:
            extra["screenshots"] = shots
        if req.confirmation:
            extra["confirmation"] = req.confirmation
        if req.status == "submitted":
            extra["submitted_at"] = int(time.time() * 1000)
        if req.status == "needs_you":
            extra["why"] = (a.get("why") or []) + [req.detail or req.step]
        accounts.set_app_status(p["user_id"], req.app_id, req.status, **extra)
        log.info(json.dumps({"event": "job_pilot_report", "status": req.status, "ats": (a.get("job") or {}).get("ats")}))
        return {"ok": True}


def settings_flag(name: str) -> bool:
    import os
    return os.environ.get(name, "0") == "1"
