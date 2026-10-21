"""Job Pilot, part 3: the nightly run that ties finding, screening, research, tailoring and checking together.

The server prepares applications (no browser needed). A small runner on Harsha's laptop then fills in and submits
them (see runner/job_pilot.py), asking the server for answers to each form's questions.

Application states:
  found       screened in, waiting for a night with room under the cap
  ready       researched, tailored and passed both checks; waiting for the runner
  needs_you   something only Harsha can do: a check that didn't pass, a question nothing supports, a CAPTCHA
  filled      dry run: every page filled, stopped before Submit (screenshot attached)
  submitted   the runner clicked Submit and saw a confirmation
  skipped     screened out, or he skipped it
  failed      the runner hit an error it couldn't recover from
"""
from __future__ import annotations

import time
from typing import Any, Callable

from . import jobsearch as js
from . import resume as rs

DEFAULT_RULES = {
    "titles": ["Product Manager", "Product Owner", "Technical Product Manager"],
    "min_years_required": 2, "max_years_required": 5,
    "preferred_locations": ["Austin", "Dallas"], "open_to_relocate": True,
    "exclude_companies": ["IBM"],
    "needs_sponsorship": True, "sponsor_filter": "likely",       # likely: skip employers unlikely to sponsor
    "nightly_cap": 10, "dry_run": True, "boards_per_night": 60, "web_discovery": True, "run_hour": 22,
}


def rules_of(profile: dict) -> dict:
    return {**DEFAULT_RULES, **(profile.get("rules") or {})}


def run_night(client, model: str, helper: str, today: str, profile: dict, store, emit: Callable[[str, dict], None] | None = None,
              fetcher: js.Fetcher | None = None, sponsor_index: dict | None = None, web: bool = True, budget_ok: Callable[[], bool] = lambda: True) -> dict:
    """One night's work for one person. `store` saves apps and the company registry (accounts.Accounts or a test double)."""
    emit = emit or (lambda k, d: None)
    rules = rules_of(profile)
    base = profile.get("base")
    if not base:
        return {"ok": False, "message": "Upload your résumé first."}
    f = fetcher or js.Fetcher()
    usage = rs.Usage()
    started = time.monotonic()
    summary = {"discovered_boards": 0, "jobs_seen": 0, "screened_in": 0, "prepared": 0, "ready": 0, "needs_you": 0, "errors": []}

    # 1. widen the company list with web search, then read every board's feed
    if web and rules.get("web_discovery"):
        emit("status", {"label": "Searching for new companies hiring product people"})
        try:
            u = {}
            for b in js.discover(client, helper, rules, u):
                if store.add_company(b["ats"], b["board"], b.get("job_id", ""), b.get("url", "")):
                    summary["discovered_boards"] += 1
            usage.input_tokens += u.get("input_tokens", 0)
            usage.output_tokens += u.get("output_tokens", 0)
            usage.web_searches += u.get("web_searches", 0)
        except Exception as e:
            summary["errors"].append(f"discovery: {type(e).__name__}")
    boards = store.companies_to_check(int(rules.get("boards_per_night", 60)))
    seen = store.seen_keys(profile["user_id"])
    candidates = []
    for n, b in enumerate(boards, 1):
        emit("status", {"label": f"Reading job feeds ({n} of {len(boards)})"})
        try:
            jobs = js.board_jobs(f, b["ats"], b["board"], b.get("job_id", ""), b.get("url", ""))
            store.mark_checked(b["ats"], b["board"], jobs[0].company if jobs else "")
        except Exception as e:
            summary["errors"].append(f"{b['ats']}:{b['board']}: {type(e).__name__}")
            store.mark_checked(b["ats"], b["board"], "", failed=True)
            continue
        for j in jobs:
            summary["jobs_seen"] += 1
            if not js.TITLE_IN.search(j.title):
                continue
            s = js.screen(j, rules, sponsor_index or {}, seen)
            seen.add(j.key)
            if not s["ok"]:
                if "Already seen" not in s["reasons"]:
                    store.save_app(profile["user_id"], {"key": j.key, "job": j.to_dict(), "screen": s, "status": "skipped", "why": s["reasons"]})
                continue
            candidates.append((js.priority(j, s), j, s))
    candidates.sort(key=lambda x: -x[0])
    summary["screened_in"] = len(candidates)
    cap = int(rules.get("nightly_cap", 10))

    # 2. research, tailor and check the best ones; the rest wait for another night
    for rank, (score, j, s) in enumerate(candidates):
        app = {"key": j.key, "job": j.to_dict(), "screen": s, "priority": score, "status": "found"}
        if rank >= cap or not budget_ok():
            store.save_app(profile["user_id"], app)
            continue
        emit("status", {"label": f"Preparing {j.title} at {j.company} ({rank + 1} of {min(cap, len(candidates))})"})
        try:
            if j.ats == "greenhouse":
                try:
                    app["questions"] = js.greenhouse_questions(f, j.board, j.job_id)
                except Exception:
                    app["questions"] = []
            brief = rs.role_brief(client, model, today, j.to_dict(), usage, web=web)
            out = rs.tailor_and_check(client, model, helper, base, brief, j.to_dict(), {**profile.get("resume_rules", {}),
                                      "allowed_titles": rules.get("titles")}, usage, emit=emit)
            app.update(brief=brief, resume={"cv": out["cv"], "text": out["text"], "pages": out["pages"]}, check=out["check"],
                       files={"pdf": out["pdf"], "docx": out["docx"]}, status="ready" if out["status"] in ("ready", "ready_with_notes") else "needs_you")
            if out["status"] == "needs_you":
                app["why"] = ["The résumé didn't pass its checks: " + "; ".join(i["detail"] for i in (out["check"]["truth"]["issues"] + out["check"]["fit"]["issues"])[:3])]
            if app.get("questions"):
                app["answers"] = prefill(client, helper, app, profile, usage)
                if any(a["needs_you"] for a in app["answers"] if a.get("required")):
                    app["status"] = "needs_you"
                    app["why"] = app.get("why", []) + ["A required question has no supported answer."]
            summary["prepared"] += 1
            summary["ready" if app["status"] == "ready" else "needs_you"] += 1
        except Exception as e:
            summary["errors"].append(f"{j.company}: {type(e).__name__}: {str(e)[:120]}")
            app.update(status="found", why=[f"Preparation failed ({type(e).__name__}); it'll be retried."])
        store.save_app(profile["user_id"], app)
    summary.update(seconds=round(time.monotonic() - started, 1), input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                   web_searches=usage.web_searches, ok=True)
    return summary


def prefill(client, helper: str, app: dict, profile: dict, usage: rs.Usage) -> list[dict]:
    """Answers for the questions the job feed lists in advance (Greenhouse)."""
    sheet = profile.get("sheet") or {}
    base_text = (profile.get("base") or {}).get("source_text", "")
    out = []
    for q in app.get("questions", [])[:40]:
        if q.get("type") in ("input_file",) or not q.get("label"):
            continue
        a = rs.answer_question(client, helper, q, sheet, app["resume"]["text"], base_text, usage)
        out.append({"label": q["label"], "required": q.get("required", False), **a})
    return out
