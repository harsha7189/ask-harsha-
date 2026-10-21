"""Job Pilot, part 3: the nightly run that ties finding, screening, research, tailoring and checking together.

The server prepares applications (no browser needed). A small runner on Harsha's laptop then fills in and submits
them (see runner/job_pilot.py), asking the server for answers to each form's questions.

Application states:
  found       screened in, waiting for a night with room under the cap
  ready       researched, qualified, tailored, and passed every check (truth, fit, ATS parsing, defensibility);
              the runner submits it on its own
  skipped     screened out, not a strong enough match, or he skipped it
  needs_you   something only Harsha can do: a check that didn't pass, a question nothing supports, a CAPTCHA
  filled      dry run: every page filled, stopped before Submit (screenshot attached)
  submitted   the runner clicked Submit and saw a confirmation
  failed      the runner hit an error it couldn't recover from
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

from . import credibility as cr
from . import jobsearch as js
from . import learning as lr
from . import resume as rs

DEFAULT_RULES = {
    "titles": ["Product Manager", "Product Owner", "Technical Product Manager"],
    "min_years_required": 2, "max_years_required": 4,          # 3-5 years means: skip anything asking for 5 or more
    "abroad_countries": ["Germany", "Netherlands", "Ireland"],  # only when the posting says it sponsors
    "preferred_locations": ["Austin", "Dallas"], "open_to_relocate": True,
    "exclude_companies": ["IBM"],
    "needs_sponsorship": True, "sponsor_filter": "likely",       # likely: skip employers unlikely to sponsor
    "nightly_cap": 10, "dry_run": False, "qualify_pool": 20, "boards_per_night": 60, "web_discovery": True, "run_hour": 22,
}


def rules_of(profile: dict) -> dict:
    r = {**DEFAULT_RULES, **(profile.get("rules") or {})}
    if int(r.get("max_years_required") or 4) >= 5:      # the old default let 5+ through; he wants 3-5 years, so 5 and up are skipped
        r["max_years_required"] = 4
    return r


def still_eligible(app: dict, rules: dict) -> str | None:
    """Checked again right before applying, so a rule changed after preparation still holds. Returns why not, or None."""
    job = app.get("job") or {}
    skip = [js.norm_company(x) for x in rules.get("exclude_companies", []) if js.norm_company(x)]
    if any(x in js.norm_company(job.get("company", "")) for x in skip):
        return "Company is on your skip list"
    yrs = [y for y in ((app.get("screen") or {}).get("years"), (app.get("brief") or {}).get("years_required"),
                       js.years_required(job.get("description", ""))) if isinstance(y, int)]
    if yrs and max(yrs) > int(rules.get("max_years_required", 4)):
        return f"Asks for {max(yrs)}+ years"
    if job:
        j = js.Job(**{k: v for k, v in job.items() if k in js.Job.__dataclass_fields__})
        s = js.screen(j, rules, {}, set())
        bad = [r for r in s["reasons"] if r.startswith(("Outside the US", "Remote, but", "In ", "Asks for")) or r.endswith("isn't on your list")]
        if bad:
            return bad[0]
    return None


def run_night(client, model: str, helper: str, today: str, profile: dict, store, emit: Callable[[str, dict], None] | None = None,
              fetcher: js.Fetcher | None = None, sponsor_index: dict | None = None, web: bool = True, budget_ok: Callable[[], bool] = lambda: True) -> dict:
    """One night's work for one person. `store` saves apps and the company registry (accounts.Accounts or a test double)."""
    emit = emit or (lambda k, d: None)
    rules = rules_of(profile)
    base = profile.get("base")
    if not base:
        return {"ok": False, "message": "Upload your résumé first."}
    base = rs.with_notes(base, profile.get("evidence_notes", ""))      # his notes are true material for tailoring too
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
    # earlier nights' over-the-cap jobs come back into the running (for up to 10 days)
    carried = 0
    for a in store.app_bodies(profile["user_id"], "found", 60) if hasattr(store, "app_bodies") else []:
        if time.time() * 1000 - int(a.get("created") or 0) > 10 * 86_400_000 or not a.get("job"):
            continue
        jd = {k: v for k, v in a["job"].items() if k in js.Job.__dataclass_fields__}
        j = js.Job(**jd)
        candidates.append((a.get("priority") or js.priority(j, a["screen"]), j, a["screen"]))
        carried += 1
    # applications that waited only on a question get another try with the current answer sheet and rules
    summary["unblocked"] = 0
    for a in store.app_bodies(profile["user_id"], "needs_you", 60) if hasattr(store, "app_bodies") else []:
        why = " ".join(a.get("why") or [])
        if "question" not in why or "didn't pass its checks" in why or not a.get("resume") or still_eligible(a, rules):
            continue
        if a.get("questions"):
            a["answers"] = prefill(client, helper, a, profile, usage)
            if any(x["needs_you"] for x in a["answers"] if x.get("required")):
                continue
        store.save_app(profile["user_id"], {"key": a["key"], "answers": a.get("answers", []), "status": "ready",
                                            "why": ["Re-answered with your updated answer sheet."], "force": True})
        summary["unblocked"] += 1
    candidates.sort(key=lambda x: -x[0])
    summary["screened_in"] = len(candidates) - carried
    summary["carried_over"] = carried
    cap = int(rules.get("nightly_cap", 10))
    pool = int(rules.get("qualify_pool") or cap * 2)
    history = store.app_bodies(profile["user_id"], None, 400) if hasattr(store, "app_bodies") else []
    learned = lr.analyze(history)
    winner = learned["verdict"]["winner"]
    summary["not_qualified"] = 0

    # 2. research the best candidates and check he's genuinely qualified before spending a tailored résumé on them
    qualified = []
    for rank, (score, j, s) in enumerate(candidates):
        app = {"key": j.key, "job": j.to_dict(), "screen": s, "priority": score, "status": "found", "force": True}
        if rank >= pool or not budget_ok():
            store.save_app(profile["user_id"], app)
            continue
        emit("status", {"label": f"Researching {j.title} at {j.company} ({rank + 1} of {min(pool, len(candidates))})"})
        try:
            brief = rs.role_brief(client, model, today, j.to_dict(), usage, web=web)
            q = cr.qualify(client, helper, base, brief, j.to_dict(), usage, max_years=int(rules.get("max_years_required", 4)))
        except Exception as e:
            summary["errors"].append(f"{j.company}: {type(e).__name__}: {str(e)[:120]}")
            store.save_app(profile["user_id"], {**app, "why": [f"Research failed ({type(e).__name__}); it'll be retried."]})
            continue
        app.update(brief=brief, qualify=q)
        if q["decision"] == "skip":
            summary["not_qualified"] += 1
            store.save_app(profile["user_id"], {**app, "status": "skipped", "why": [f"Not a strong enough match ({q['match']})"] + q["reasons"]})
            continue
        app["priority"] = score + cr.MATCH_BONUS[q["match"]]
        qualified.append(app)
    qualified.sort(key=lambda a: -a["priority"])

    # 3. tailor, check and prepare the best ones; the rest wait for another night
    for n, app in enumerate(qualified):
        j = js.Job(**{k: v for k, v in app["job"].items() if k in js.Job.__dataclass_fields__})
        if n >= cap or not budget_ok():
            store.save_app(profile["user_id"], app)
            continue
        emit("status", {"label": f"Preparing {j.title} at {j.company} ({n + 1} of {min(cap, len(qualified))})"})
        try:
            if j.ats == "greenhouse":
                try:
                    app["questions"] = js.greenhouse_questions(f, j.board, j.job_id)
                except Exception:
                    app["questions"] = []
            app["strategy"] = lr.assign(history, app, winner)
            out = rs.tailor_and_check(client, model, helper, base, app["brief"], j.to_dict(),
                                      {**profile.get("resume_rules", {}), "allowed_titles": rules.get("titles"),
                                       "strategy_text": lr.STRATEGIES[app["strategy"]]["text"]},
                                      usage, emit=emit, notes=profile.get("evidence_notes", ""))
            app.update(resume={"cv": out["cv"], "text": out["text"], "pages": out["pages"]}, check=out["check"], prep=out["prep"],
                       version=hashlib.sha256(out["text"].encode()).hexdigest()[:12],
                       files={"pdf": out["pdf"], "docx": out["docx"]}, status="ready" if out["status"] in ("ready", "ready_with_notes") else "needs_you")
            if out["status"] == "needs_you":
                ch = out["check"]
                app["why"] = ["The résumé didn't pass its checks: " + "; ".join(
                    [i["detail"] for i in ch["truth"]["issues"] + ch["fit"]["issues"]][:2] + ch["ats"]["issues"][:1])]
            if app.get("questions"):
                app["answers"] = prefill(client, helper, app, profile, usage)
                if any(a["needs_you"] for a in app["answers"] if a.get("required")):
                    app["status"] = "needs_you"
                    app["why"] = app.get("why", []) + ["A required question has no supported answer."]
            history.append(app)
            summary["prepared"] += 1
            summary["ready" if app["status"] == "ready" else "needs_you"] += 1
        except Exception as e:
            summary["errors"].append(f"{j.company}: {type(e).__name__}: {str(e)[:120]}")
            app.update(status="found", why=[f"Preparation failed ({type(e).__name__}); it'll be retried."])
        store.save_app(profile["user_id"], app)
    summary.update(seconds=round(time.monotonic() - started, 1), input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                   web_searches=usage.web_searches, ok=True, learning=learned["verdict"]["message"])
    return summary


def prefill(client, helper: str, app: dict, profile: dict, usage: rs.Usage) -> list[dict]:
    """Answers for the questions the job feed lists in advance (Greenhouse)."""
    sheet = profile.get("sheet") or {}
    base = rs.with_notes(profile.get("base") or {"experience": []}, profile.get("evidence_notes", ""))
    out = []
    for q in app.get("questions", [])[:40]:
        if q.get("type") in ("input_file",) or not q.get("label"):
            continue
        a = rs.answer_question(client, helper, q, sheet, app["resume"]["text"], base.get("source_text", ""), usage,
                               ctx={"base": base, "job": app.get("job") or {}, "brief": app.get("brief") or {}})
        out.append({"label": q["label"], "required": q.get("required", False), **a})
    return out
