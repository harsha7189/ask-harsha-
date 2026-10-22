"""Job Pilot's credibility layer and learning engine: natural writing, ATS parsing, the recruiter's view, interview
defensibility, opportunity selection, outcome learning, inbox matching, and the morning report."""
import base64
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import pytest

from agent import credibility as cr, learning as lr, pilot, resume as rs
from fakes import response, tool_use
from test_jobs import GH_FEED, RESUME_TEXT, RULES, JobsClient, Opener, base, job, setup_store, tailored

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
import job_pilot as jp  # noqa: E402


def cv_of(bullets_ibm=None):
    b = base()
    t = tailored()
    if bullets_ibm:
        t["experience"][0]["bullets"] = [{"text": x, "sources": ["E1B1"], "covers": []} for x in bullets_ibm]
    cv = {**t, "experience": [{**e, "company": b["experience"][i]["company"], "location": b["experience"][i]["location"],
                               "start": b["experience"][i]["start"], "end": b["experience"][i]["end"]} for i, e in enumerate(t["experience"])]}
    return b, rs.full_cv(b, cv)


# ---------------------------------------------------------------- natural writing
def test_style_flags_ai_sounding_writing_and_passes_plain_writing():
    _, clean = cv_of()
    assert cr.style_check(clean, {"keywords": ["Service Cloud"]}).issues == []
    _, bad = cv_of(["Spearheaded a cutting-edge synergy initiative leveraging Salesforce to protect retention",
                    "Reduced costs 150% by cutting tickets",
                    "Did stuff",
                    "Reduced defects 35% through acceptance criteria",
                    "Reduced escalations 40% by launching self-service onboarding with Salesforce Service Cloud",
                    ])
    bad["skills"] = ["Service Cloud"] * 3 + bad["skills"]
    bad["summary"] += " Service Cloud expert. Service Cloud lead."
    rules = {i["rule"] for i in cr.style_check(bad, {"keywords": ["Service Cloud"]}).issues}
    assert {"hype", "implausible", "too_short", "repetitive", "stuffing"} <= rules


# ---------------------------------------------------------------- ATS parsing
def test_ats_reads_back_every_section_job_and_bullet():
    _, cv = cv_of()
    out = cr.ats_check(cv, rs.render_pdf(cv), rs.render_docx(cv))
    assert out["passed"], out["issues"]
    assert set(out["parsers"]) == {"PDF (pypdf)", "Word (raw XML)", "Word (python-docx)"}


def test_ats_catches_a_file_that_lost_content_or_uses_tables():
    _, cv = cv_of()
    short = json.loads(json.dumps(cv))
    short["experience"][0]["bullets"] = short["experience"][0]["bullets"][:-1]
    out = cr.ats_check(cv, rs.render_pdf(short), rs.render_docx(short))
    assert not out["passed"] and any("IBM bullet 5" in i for i in out["issues"])
    from docx import Document
    d = Document(io.BytesIO(rs.render_docx(cv)))
    d.add_table(rows=1, cols=2)
    buf = io.BytesIO()
    d.save(buf)
    assert any("tables" in i for i in cr.ats_check(cv, rs.render_pdf(cv), buf.getvalue())["issues"])


# ---------------------------------------------------------------- recruiter view and defensibility
class Scripted:
    def __init__(self, payload, name):
        self.payload, self.name, self.requests = payload, name, []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        return response([tool_use("x", self.name, self.payload)], "tool_use")


def test_recruiter_weak_spots_need_real_quotes():
    _, cv = cv_of()
    c = Scripted({"first_impression": "Support automation PM.", "dimensions": [
        {"dimension": "credibility", "level": "weak", "quote": "Drove 35% adoption of an AI agent", "why": "Vague how.", "suggestion": "Say how."},
        {"dimension": "clarity", "level": "weak", "quote": "words that are not in the résumé", "why": "x"}]}, "review")
    r = cr.recruiter_review(c, "helper", cv, {"seniority": "mid"}, {"title": "PM", "company": "Acme"}, rs.Usage())
    assert r["dimensions"]["credibility"]["level"] == "weak" and len(r["weak"]) == 1
    assert r["dimensions"]["clarity"]["level"] == "adequate"          # an unverified "weak" doesn't count


def test_overreach_goes_back_to_his_words_and_prep_is_written():
    out = rs.tailor_and_check(JobsClient(bad_first=False, overreach=True), "main", "helper", base(),
                              {"must_haves": ["Salesforce Service Cloud experience"]}, job().to_dict(), RULES, rs.Usage(), notes="")
    d = out["check"]["defensibility"]
    assert d["overreach"] == 1 and any("goes beyond what you can back up" in r for r in out["check"]["replaced"])
    assert out["cv"]["experience"][0]["bullets"][0].get("original")
    assert out["prep"] and all(p["questions"] for p in out["prep"])
    assert out["check"]["ats"]["passed"] and out["check"]["recruiter"]["dimensions"]["relevance"]["level"] == "strong"
    assert out["status"] == "ready"


def test_unverified_overreach_becomes_prep_not_a_removal():
    b, cv = cv_of()
    c = Scripted({"claims": [{"n": 1, "questions": ["How measured?"], "status": "overreach", "evidence": "made-up words here"}]}, "defend")
    d = cr.defensibility(c, "helper", b, cv, "", {"title": "PM", "company": "Acme"}, rs.Usage())
    assert not d["overreach"] and d["claims"][0]["status"] == "prepare"
    assert "HIS NOTES" in c.requests[0]["messages"][0]["content"]


# ---------------------------------------------------------------- opportunity selection
def test_qualify_skips_weak_matches_and_too_senior_roles():
    b = base()
    musts = ["Salesforce Service Cloud", "Healthcare claims", "Kubernetes", "Ads auctions"]
    weak = Scripted({"items": [{"n": 1, "status": "has", "quote": "Salesforce Service Cloud"},
                               {"n": 2, "status": "has", "quote": "invented healthcare claims work"}]}, "qualify")
    q = cr.qualify(weak, "helper", b, {"must_haves": musts, "years_required": 3}, {}, rs.Usage())
    assert q["decision"] == "skip" and q["match"] == "weak" and q["musts"][1]["status"] == "missing"     # no proof, no credit
    strong = Scripted({"items": [{"n": i, "status": "has", "quote": "Salesforce Service Cloud"} for i in range(1, 5)]}, "qualify")
    assert cr.qualify(strong, "helper", b, {"must_haves": musts}, {}, rs.Usage())["match"] == "strong"
    assert cr.qualify(strong, "helper", b, {"must_haves": musts, "years_required": 10}, {}, rs.Usage())["decision"] == "skip"


def test_a_night_skips_weak_matches_and_assigns_strategies():
    from agent import jobsearch as js
    store, u = setup_store()
    store.add_company("greenhouse", "acme")
    f = js.Fetcher(Opener({"boards-api.greenhouse.io/v1/boards/acme/jobs?content": GH_FEED, "boards-api.greenhouse.io/v1/boards/acme": {"name": "Acme"}}))
    profile = {"user_id": u.id, "base": base(), "rules": {"nightly_cap": 5}, "sheet": {}}
    res = pilot.run_night(JobsClient(bad_first=False, match="missing"), "main", "helper", "today", profile, store, fetcher=f, web=False)
    assert res["not_qualified"] == 2 and res["prepared"] == 0
    assert all("Not a strong enough match" in a["why"][0] for a in store.apps(u.id, "skipped") if a["title"] != "Product Manager")
    store2, u2 = setup_store()
    store2.add_company("greenhouse", "acme")
    profile["user_id"] = u2.id
    res = pilot.run_night(JobsClient(bad_first=False), "main", "helper", "today", profile, store2, fetcher=f, web=False)
    ready = store2.app_bodies(u2.id, "ready")
    assert res["prepared"] == 2 and sorted(a["strategy"] for a in ready) == sorted(lr.STRATEGIES)   # balanced in comparable jobs
    assert all(a["qualify"]["match"] == "strong" and a["prep"] and a["version"] for a in ready)


def test_jobs_over_the_cap_come_back_the_next_night():
    from agent import jobsearch as js
    store, u = setup_store()
    store.add_company("greenhouse", "acme")
    f = js.Fetcher(Opener({"boards-api.greenhouse.io/v1/boards/acme/jobs?content": GH_FEED, "boards-api.greenhouse.io/v1/boards/acme": {"name": "Acme"}}))
    profile = {"user_id": u.id, "base": base(), "rules": {"nightly_cap": 1}, "sheet": {}}
    pilot.run_night(JobsClient(bad_first=False), "main", "helper", "today", profile, store, fetcher=f, web=False)
    again = pilot.run_night(JobsClient(bad_first=False), "main", "helper", "today", profile, store, fetcher=f, web=False)
    assert again["carried_over"] == 1 and again["prepared"] == 1 and len(store.apps(u.id, "ready")) == 2


# ---------------------------------------------------------------- learning
def apps_for(strategy, positive, total, match="strong", sponsor="high"):
    out = []
    for i in range(total):
        out.append({"status": "submitted", "strategy": strategy, "submitted_at": 1, "outcome": "interview" if i < positive else "rejected",
                    "qualify": {"match": match}, "screen": {"sponsor": {"level": sponsor}, "location_rank": 0}, "job": {"ats": "greenhouse"}})
    return out


def test_no_winner_without_enough_evidence():
    a = lr.analyze(apps_for("problem_first", 8, 10) + apps_for("requirements_first", 1, 10), now_ms=10)
    assert a["verdict"]["winner"] is None and "Not enough evidence yet" in a["verdict"]["message"]


def test_a_real_difference_with_enough_data_wins_and_exploration_continues():
    a = lr.analyze(apps_for("problem_first", 18, 40) + apps_for("requirements_first", 6, 40), now_ms=10)
    assert a["verdict"]["winner"] == "problem_first" and a["verdict"]["p_value"] < 0.05
    lo, hi = a["arms"]["problem_first"]["response_ci"]
    assert lo < 0.45 < hi
    picks = [lr.assign([], {}, "problem_first", random.Random(i)) for i in range(400)]
    assert 0.7 < picks.count("problem_first") / 400 < 0.9


def test_stratification_stops_a_strategy_taking_credit_for_easier_jobs():
    # problem_first only drew strong matches (which reply more); within each match level the strategies do the same
    a = lr.analyze(apps_for("problem_first", 20, 40, "strong") + apps_for("requirements_first", 20, 40, "strong")
                   + apps_for("requirements_first", 2, 40, "stretch") + apps_for("problem_first", 2, 40, "stretch"), now_ms=10)
    assert a["verdict"]["winner"] is None
    skew = lr.analyze(apps_for("problem_first", 20, 40, "strong") + apps_for("requirements_first", 2, 40, "stretch"), now_ms=10)
    assert skew["verdict"]["winner"] is None          # never compared on comparable jobs, so no verdict


def test_silence_counts_after_21_days_and_balance_within_groups():
    now = 30 * lr.DAY_MS
    assert lr.outcome_of({"submitted_at": now - 22 * lr.DAY_MS}, now) == "no_response"
    assert lr.outcome_of({"submitted_at": now - 5 * lr.DAY_MS}, now) is None
    hist = [{"strategy": "problem_first", "qualify": {"match": "strong"}, "screen": {"sponsor": {"level": "high"}}}]
    assert lr.assign(hist, {"qualify": {"match": "strong"}, "screen": {"sponsor": {"level": "high"}}}) == "requirements_first"


# ---------------------------------------------------------------- reading replies (on the laptop)
def test_inbox_replies_are_matched_and_classified_locally():
    watch = [{"id": "J1", "company": "Acme, Inc.", "submitted_at": 100}, {"id": "J2", "company": "Globex", "submitted_at": 100},
             {"id": "J3", "company": "Initech", "submitted_at": 100}, {"id": "J4", "company": "Umbrella", "submitted_at": 100, "outcome": "interview"}]
    msgs = [
        {"from": "Acme Talent <no-reply@myworkday.com>", "subject": "Your application to Acme", "body": "Thank you for applying!", "at": 200},
        {"from": "Acme Talent <no-reply@myworkday.com>", "subject": "Acme: an update", "body": "Unfortunately we have decided to move forward with other candidates.", "at": 300},
        {"from": "Jane Doe <jane@globex.com>", "subject": "Your application", "body": "Your background looks great, could we chat about the role?", "at": 300},
        {"from": "Globex News <news@globex.com>", "subject": "Globex quarterly", "body": "Our new product", "at": 310},
        {"from": "Recruiting <recruiting@initech.com>", "subject": "Initech interview invitation", "body": "Please book a time: calendly.com/x", "at": 300},
        {"from": "Umbrella <no-reply@icims.com>", "subject": "Umbrella update", "body": "Unfortunately the position has been filled.", "at": 300},
        {"from": "Acme <x@acme.com>", "subject": "Acme", "body": "Unfortunately...", "at": 50},   # before he applied
    ]
    got = {o["app_id"]: o["outcome"] for o in jp.match_outcomes(msgs, watch)}
    assert got == {"J1": "rejected", "J2": "recruiter_reply", "J3": "interview"}     # J4 already further along


# ---------------------------------------------------------------- the server: report, outcomes, zip
def test_morning_report_outcomes_zip_and_private_notes():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from agent import inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from server.app import Settings, create_app
    store, u = setup_store()
    app = create_app(client=JobsClient(bad_first=False), settings=Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="pw", web_search=False),
                     notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(), receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=store)
    c = TestClient(app)
    c.cookies.set("askh_session", store.start_session(u))
    h = {"X-Owner-Passcode": "pw"}
    c.post("/api/jobs/resume", headers=h, json={"name": "resume.txt", "data": base64.b64encode(RESUME_TEXT.encode()).decode()})
    assert c.put("/api/jobs/profile", headers=h, json={"evidence_notes": "The $8M was 14 accounts in the FY24 renewal review."}).json()["evidence_notes"].startswith("The $8M")
    raw = store.db.one("SELECT body FROM job_profiles WHERE user_id = %s", (u.id,))[0]
    assert "14 accounts" not in raw                                                   # encrypted at rest
    aid = store.save_app(u.id, {"key": "greenhouse:acme:9", "status": "ready", "priority": 50, "job": job(key="9").to_dict(), "strategy": "problem_first",
                                "qualify": {"match": "strong", "musts": [{"must_have": "Service Cloud", "status": "has", "quote": "Salesforce Service Cloud"}]},
                                "brief": {"problem": "Scale support automation."}, "prep": [{"claim": "Cut case time 11→4 minutes", "questions": ["How measured?"],
                                "status": "prepare", "prepare": "The baseline period", "evidence": "x"}],
                                "check": {"recruiter": {"first_impression": "Support automation PM."}, "defensibility": {"defensible": 3, "prepare": 1, "overreach": 0},
                                          "ats": {"passed": True}}, "resume": {"text": "Skills: Salesforce Service Cloud"},
                                "files": {"pdf": b"%PDF-1.4 x", "docx": b"PK.."}})
    tok = c.post("/api/jobs/runner-token", headers=h).json()["token"]
    r = {"X-Runner-Token": tok}
    assert c.get("/api/jobs/runner/next", headers=r).json()["dry_run"] is False                         # applies on its own
    c.post("/api/jobs/runner/report", headers=r, json={"app_id": aid, "status": "submitted", "confirmation": "Thank you for applying"})
    rep = c.get("/api/jobs/report", headers=h).json()
    assert [x["id"] for x in rep["submitted"]] == [aid] and rep["submitted"][0]["first_impression"] == "Support automation PM."
    z = zipfile.ZipFile(io.BytesIO(c.get("/api/jobs/report/zip", headers=h).content))
    names = z.namelist()
    assert "Summary.txt" in names and any(n.endswith("/Resume.pdf") for n in names) and any(n.endswith("/Resume.docx") for n in names)
    prep = z.read(next(n for n in names if n.endswith("Interview_prep.md"))).decode()
    assert "How measured?" in prep and "Have ready: The baseline period" in prep and "Scale support automation" in prep
    assert c.get("/api/jobs/runner/watch", headers=r).json()["apps"][0]["id"] == aid
    assert c.post("/api/jobs/runner/outcomes", headers=r, json={"items": [{"app_id": aid, "outcome": "interview", "at": 5}]}).json()["recorded"] == 1
    assert c.post(f"/api/jobs/apps/{aid}/outcome", headers=h, json={"outcome": "rejected"}).json()["outcome"] == "interview"   # furthest it got
    L = c.get("/api/jobs/learning", headers=h).json()
    assert L["arms"]["problem_first"]["positive"] == 1 and L["decided"] == 1
    assert c.get("/api/jobs/learning").status_code in (401, 403)


# ---------------------------------------------------------------- evidence notes as true material
NOTES = """IBM — Product Manager, Salesforce enterprise platform
IBM Video Streaming
- I defined FairPlay and Widevine DRM entitlement rules for protected playback.
- short
CYBAGE SOFTWARE — Associate Product Manager
- I coordinated launches across marketing and sales while managing an $18M budget, eliminating $1.2M in avoidable spend."""


def test_notes_become_citable_sources_under_the_right_job():
    b = rs.with_notes(base(), NOTES)
    bank = b["notes_bank"]
    assert [(x["job"], x["text"][:12]) for x in bank] == [("E1", "I defined Fa"), ("E2", "I coordinate")]
    assert bank[0]["id"] in rs.sources_for(b, "E1") and bank[1]["id"] not in rs.sources_for(b, "E1")
    cv = tailored()
    cv["experience"][0]["bullets"][4] = {"text": "Defined FairPlay and Widevine DRM entitlement rules for protected playback", "sources": [bank[0]["id"]], "covers": []}
    cv["experience"][1]["bullets"][3] = {"text": "Eliminated $1.2M in avoidable spend while managing an $18M launch budget", "sources": [bank[1]["id"]], "covers": []}
    full = {**cv, "experience": [{**e, "company": b["experience"][i]["company"]} for i, e in enumerate(cv["experience"])]}
    c = rs.check_truth(None, "helper", b, full, RULES, rs.Usage(), judge=False)
    assert c.passed, c.issues
    plain = rs.check_truth(None, "helper", base(), full, RULES, rs.Usage(), judge=False)    # without the notes, the same bullets fail
    assert not plain.passed


# ---------------------------------------------------------------- filling the form without him
from test_jobs import JobsClient as _JC  # noqa: E402

SHEET = {"first_name": "Harsha", "city": "Austin", "state": "TX", "authorized": "Yes", "sponsorship": "Yes", "authorized_eu": "No",
         "visa_status": "F-1 STEM OPT"}


def ask(label, options=None, typ="", ctx=None, sheet=SHEET):
    b = base()
    return rs.answer_question(_JC(), "helper", {"label": label, "options": options or [], "type": typ, "required": True}, sheet,
                              "Skills: Salesforce Service Cloud, API design", RESUME_TEXT, rs.Usage(),
                              ctx or {"base": b, "job": {"title": "PM", "company": "Acme", "description": "Own support automation."}, "brief": {"problem": "Scale support."}})


def test_status_questions_with_any_wording_or_options():
    assert ask("Will you now or in the future require sponsorship for employment visa status (e.g., H-1B)?", ["Yes", "No"])["value"] == "Yes"
    assert ask("Are you legally authorized to work in the United States?", ["Yes", "No"])["value"] == "Yes"
    assert ask("Are you authorized to work in the US without employer sponsorship?", ["Yes", "No"])["value"] == "No"
    opts = ["I am a U.S. citizen", "I am a lawful permanent resident (green card)", "I am authorized to work but will require sponsorship in the future",
            "I am authorized and will not require sponsorship"]
    got = ask("Which of the following describes your work authorization in the US?", opts)
    assert got["value"] == opts[2] and not got["needs_you"]                                   # never citizen or green card
    assert ask("Do you require visa sponsorship to work in Germany?", ["Yes", "No"])["value"] == "Yes"   # abroad, from the EU answer
    assert ask("Are you eligible to work in the European Union?", ["Yes", "No"])["value"] == "No"
    no_eu = {k: v for k, v in SHEET.items() if k != "authorized_eu"}
    assert ask("Do you require visa sponsorship to work in Germany?", ["Yes", "No"], sheet=no_eu)["needs_you"]


def test_answers_from_his_record_not_from_guessing():
    assert ask("Current employer")["value"] == "IBM"
    assert ask("How many years of product management experience do you have?", ["0-2 years", "3-5 years", "5+ years"])["value"] == "5+ years" \
        or ask("How many years of product management experience do you have?", ["0-2 years", "3-5 years", "5+ years"])["value"] == "3-5 years"
    assert ask("Location (City, State)")["value"] == "Austin, TX"
    assert ask("University")["value"] == "JNTU"


class Writer(_JC):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def create(self, **kw):
        if (kw.get("tool_choice") or {}).get("name") == "write":
            self.requests.append(kw)
            return response([tool_use("w", "write", {"text": self.text})], "tool_use")
        return super().create(**kw)


def test_essay_questions_are_written_from_his_record_and_checked():
    ok = "I've spent the last few years on Salesforce Service Cloud automation, which is the work this role needs to scale support."
    r = rs.answer_question(Writer(ok), "helper", {"label": "Why do you want to work at Acme?", "type": "textarea", "required": True}, SHEET,
                           "Salesforce Service Cloud automation", RESUME_TEXT, rs.Usage(),
                           {"base": base(), "job": {"title": "PM", "company": "Acme", "description": "Scale support"}, "brief": {}})
    assert r["source"] == "written" and r["value"] == ok
    bad = "I grew revenue 300% using Snowflake, and I'm a US citizen."
    r = rs.answer_question(Writer(bad), "helper", {"label": "Why do you want to work at Acme?", "type": "textarea", "required": True}, SHEET,
                           "Salesforce Service Cloud automation", RESUME_TEXT, rs.Usage(), {"base": base(), "job": {}, "brief": {}})
    assert r["needs_you"] and r["value"] is None


def test_locations_and_years_rules():
    from agent import jobsearch as js
    J = lambda loc, d="Requirements: 3+ years of product management experience. " * 8: js.Job("greenhouse", "b", "1", "Co", "Product Manager", loc, "u", d)
    rules = pilot.rules_of({"rules": {"max_years_required": 5}})           # an old saved 5 still means "skip 5+"
    assert rules["max_years_required"] == 4
    assert js.screen(J("Kuala Lumpur, Malaysia"), rules, {}, set())["reasons"][0].startswith("Outside the US")
    assert js.screen(J("Berlin, Germany"), rules, {}, set())["reasons"] == ["In Germany, and the posting doesn't say it sponsors visas"]
    assert js.screen(J("Berlin, Germany", "We offer visa sponsorship and relocation support. 3+ years of product experience. " * 6), rules, {}, set())["ok"]
    assert not js.screen(J("London, UK", "We offer visa sponsorship. " * 20), rules, {}, set())["ok"]
    assert js.screen(J("Austin, TX", "5+ years of product management experience. " * 8), rules, {}, set())["reasons"] == ["Asks for 5+ years"]
    assert js.screen(J("Remote"), rules, {}, set())["ok"]
    assert not js.screen(J("Remote", "Remote - Europe. Requirements: 3+ years of product experience. " * 6), rules, {}, set())["ok"]
    stale = {"job": {**J("Austin, TX").to_dict(), "description": "5+ years of product management experience. " * 8}, "screen": {"years": 3}}
    assert pilot.still_eligible(stale, rules) == "Asks for 5+ years"


def test_waiting_on_a_question_gets_retried_with_the_new_sheet():
    from agent import jobsearch as js
    store, u = setup_store()
    j = js.Job("greenhouse", "acme", "7", "Acme", "Product Manager", "Austin, TX", "u", "Requirements: 3+ years of product management experience. " * 8)
    store.save_app(u.id, {"key": j.key, "status": "needs_you", "job": j.to_dict(), "screen": {"years": 3}, "resume": {"text": "Skills: Salesforce Service Cloud"},
                          "questions": [{"label": "Will you now or in the future require sponsorship?", "required": True, "options": ["Yes", "No"]}],
                          "why": ["A required question has no supported answer."]})
    profile = {"user_id": u.id, "base": base(), "rules": {"nightly_cap": 1}, "sheet": {"authorized": "Yes", "sponsorship": "Yes"}}
    res = pilot.run_night(JobsClient(bad_first=False), "main", "helper", "today", profile, store, fetcher=js.Fetcher(Opener({})), web=False)
    assert res["unblocked"] == 1
    a = store.app_bodies(u.id, "ready")[0]
    assert a["answers"][0]["value"] == "Yes"


def test_small_format_problems_are_repaired_not_blocking():
    b = base()
    cv = tailored()
    cv["experience"][0]["bullets"][1]["text"] = "Owned the TTL requirement for three releases of Redis caching, cutting AI-agent latency 40%"
    cv["experience"][0]["bullets"][2]["text"] = "Cut case resolution time 11→4 minutes and reduced escalations 40% with triage rules"
    cv["experience"][1]["bullets"] = cv["experience"][1]["bullets"][:2]
    cv = {**cv, "experience": [{**e, "company": b["experience"][i]["company"]} for i, e in enumerate(cv["experience"])]}
    replaced = []
    assert rs.repair_format(b, cv, RULES, replaced)
    ibm = cv["experience"][0]["bullets"]
    assert "3 releases" in ibm[1]["text"] and ibm[2].get("original") and len(cv["experience"][1]["bullets"]) == 4
    assert all(rs.metric_count(x["text"]) <= 1 for x in ibm)


def test_out_of_credit_stops_the_run_with_a_clear_message():
    class Broke(JobsClient):
        def create(self, **kw):
            raise Exception("Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API.'}}")
    from agent import jobsearch as js
    store, u = setup_store()
    res = pilot.run_night(Broke(), "main", "helper", "today", {"user_id": u.id, "base": base(), "rules": {}, "sheet": {}}, store,
                          fetcher=js.Fetcher(Opener({})), web=True)
    assert "out of API credit" in res["stopped"] and len(res["errors"]) == 1


def test_untrue_rewrites_fall_back_to_his_own_words_instead_of_blocking():
    b = base()
    cv = tailored()
    cv["summary"] = "BS-trained product manager with 9+ years of experience."
    cv["skills"] = cv["skills"] + ["Kubernetes"]
    cv["experience"][0]["bullets"][2]["text"] = "Cut case resolution time 11→3 minutes across 5 teams with triage rules"
    cv["experience"][1]["bullets"][1]["text"] = "Accelerated release cadence 25% across five releases by restructuring the backlog"
    cv = {**cv, "experience": [{**e, **{k: b["experience"][i][k] for k in ("company", "location", "start", "end")}} for i, e in enumerate(cv["experience"])]}
    saved = rs.full_cv(b, cv)
    out = rs.finalize_saved(b, saved, RULES)
    assert out["ok"] and out["truth"]["passed"] and out["ats"]["passed"] and out["pages"] == 1
    assert out["cv"]["summary"] == b["summary"] and "Kubernetes" not in out["cv"]["skills"]
    assert "11→3" not in out["text"] and "five releases" not in out["text"]


def test_held_back_resumes_are_rescued_on_the_next_run_without_model_calls():
    from agent import jobsearch as js
    store, u = setup_store()
    b = base()
    cv = tailored()
    cv["summary"] = "BS in product with 9+ years."
    cv = {**cv, "experience": [{**e, **{k: b["experience"][i][k] for k in ("company", "location", "start", "end")}} for i, e in enumerate(cv["experience"])]}
    j = js.Job("ashby", "ocra", "1", "Ocra", "Technical Product Owner", "Austin, TX", "u", "Requirements: 3+ years of product experience. " * 8)
    store.save_app(u.id, {"key": j.key, "status": "needs_you", "job": j.to_dict(), "screen": {"years": 3}, "brief": {"years_required": 3},
                          "resume": {"cv": rs.full_cv(b, cv), "text": "x"}, "check": {"truth": {"passed": False}},
                          "why": ["The résumé didn't pass its checks: Not in your résumé: BS."]})
    client = JobsClient(bad_first=False)
    res = pilot.run_night(client, "main", "helper", "today", {"user_id": u.id, "base": b, "rules": {"nightly_cap": 1}, "sheet": {}}, store,
                          fetcher=js.Fetcher(Opener({})), web=False)
    assert res["rescued"] == 1
    a = store.app_bodies(u.id, "ready")[0]
    assert a["resume"]["cv"]["summary"] == b["summary"] and store.app_file(u.id, a["id"], "pdf")[:5] == b"%PDF-"
