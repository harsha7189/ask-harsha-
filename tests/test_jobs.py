"""Job Pilot: finding and screening roles, sponsorship signals, the tailored résumé's two-step check, honest answers,
a full night's run, and the runner's API."""
import base64
import json
import re

import pytest

from agent import accounts as acct, jobsearch as js, pilot, resume as rs
from fakes import FakeClient, response, text, tool_use

RESUME_TEXT = """Harsha Reddy
Austin, TX | 512-555-0100 | harsha@example.com
Product Manager
Summary: Technical product manager with 5+ years in B2B SaaS and Salesforce Service Cloud automation.
Skills: Salesforce Service Cloud, MuleSoft, API design, Agile, SQL, Jira
Product Manager, IBM, Austin, TX, March 2023 - Present
- Protected $8M+ in at-risk retention by automating diagnostic context capture for enterprise accounts
- Owned the TTL requirement and API specs for Redis caching that engineering built, cutting AI-agent latency 40%
- Cut case resolution time 11→4 minutes by defining automated triage rules in Salesforce Service Cloud
- Reduced post-release defects 35% through acceptance criteria and release readiness reviews
- Drove 35% adoption of an AI agent with a go-to-market plan
Associate Product Manager, Cybage Software, Pune, India, December 2020 - March 2023
- Aligned 5 teams to deliver 26 consecutive on-time sprints
- Accelerated release cadence 25% by restructuring the backlog around outcomes
- Reduced escalations 40% by launching self-service onboarding
- Shipped zero critical incidents across 3 releases with a risk-based test plan
Education: B.Tech, Computer Science, JNTU
Certifications: CSPO, SAFe POPM, Salesforce Certified Administrator"""

BULLETS_IBM = RESUME_TEXT.split("Austin, TX, March 2023 - Present\n")[1].split("\nAssociate")[0].replace("- ", "").split("\n")
BULLETS_CY = RESUME_TEXT.split("December 2020 - March 2023\n")[1].split("\nEducation")[0].replace("- ", "").split("\n")
PARSED = {"name": "Harsha Reddy", "contact": {"email": "harsha@example.com", "phone": "512-555-0100", "location": "Austin, TX"},
          "headline": "Product Manager", "summary": "Technical product manager with 5+ years in B2B SaaS and Salesforce Service Cloud automation.",
          "skills": ["Salesforce Service Cloud", "MuleSoft", "API design", "Agile", "SQL", "Jira"],
          "experience": [{"company": "IBM", "title": "Product Manager", "location": "Austin, TX", "start": "Mar 2023", "end": "Present",
                          "bullets": BULLETS_IBM + ["Invented a bullet that isn't in the résumé"]},
                         {"company": "Cybage Software", "title": "Associate Product Manager", "location": "Pune, India", "start": "12/2020", "end": "03/2023", "bullets": BULLETS_CY}],
          "education": [{"school": "JNTU", "degree": "B.Tech", "field": "Computer Science"}], "certifications": ["CSPO", "SAFe POPM", "Salesforce Certified Administrator"]}


def base():
    return rs.parse_resume(FakeClient([response([tool_use("r", "submit_resume", PARSED)], "tool_use")]), "helper", RESUME_TEXT, rs.Usage())


def test_parsing_never_invents_and_formats_dates():
    b = base()
    assert b["parse_dropped"] == 1 and len(b["experience"][0]["bullets"]) == 5
    assert b["experience"][1]["start"] == "December 2020" and b["experience"][0]["start"] == "March 2023"
    assert rs.years_of_experience(b) >= 5.5


# ---------------------------------------------------------------- screening
def job(title="Product Manager", loc="Austin, TX", desc=None, company="Acme", key="1"):
    d = desc if desc is not None else ("We need a PM. Requirements: 3+ years of product management experience with B2B SaaS. " * 6)
    return js.Job("greenhouse", "acme", key, company, title, loc, f"https://boards.greenhouse.io/acme/jobs/{key}", d)


def test_screening_rules():
    rules = pilot.rules_of({"rules": {"exclude_companies": ["IBM", "Amazon"]}})
    ok = js.screen(job(), rules, {}, set())
    assert ok["ok"] and ok["location_rank"] == 0 and ok["years"] == 3
    assert not js.screen(job(title="Director of Product"), rules, {}, set())["ok"]
    assert "Asks for 7+ years" in js.screen(job(desc="Requirements: 7+ years of product management experience. " * 8), rules, {}, set())["reasons"]
    no = js.screen(job(desc="3+ years of product experience. We are unable to sponsor visas for this role. " * 5), rules, {}, set())
    assert "Says it won't sponsor" in no["reasons"] and no["sponsor"]["level"] == "excluded"
    assert "Company excluded" in js.screen(job(company="Amazon.com Services LLC"), rules, {}, set())["reasons"]
    assert js.screen(job(loc="Bengaluru, India"), rules, {}, set())["location_rank"] is None
    assert "Already seen" in js.screen(job(), rules, {}, {"greenhouse:acme:1"})["reasons"]
    index = {"acme": {"pm": 7, "total": 40}}
    assert js.sponsor_likelihood(job(company="Acme, Inc."), index)["level"] == "high"
    assert js.sponsor_likelihood(job(company="Nobody Corp"), index)["level"] == "unknown"
    austin, remote, ny = (js.screen(job(loc=l), rules, index, set()) for l in ("Dallas, TX", "Remote - US", "New York, NY"))
    assert js.priority(job(), austin) > js.priority(job(), remote) > js.priority(job(), ny)


class Opener:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, req, timeout=0):
        self.calls.append(req.full_url)
        body = next((v for k, v in self.routes.items() if k in req.full_url), None)
        if body is None:
            import urllib.error
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)

        class R:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def read(s):
                return (body if isinstance(body, str) else json.dumps(body)).encode()
        return R()


def test_reading_each_hiring_system():
    f = js.Fetcher(Opener({
        "boards-api.greenhouse.io/v1/boards/acme/jobs?content": {"jobs": [{"id": 11, "title": "Product Manager", "location": {"name": "Austin, TX"},
                                                                           "absolute_url": "https://boards.greenhouse.io/acme/jobs/11", "content": "&lt;p&gt;Own the &lt;b&gt;roadmap&lt;/b&gt;&lt;/p&gt;"}]},
        "boards-api.greenhouse.io/v1/boards/acme": {"name": "Acme"},
        "api.lever.co/v0/postings/plaid": [{"id": "u1", "text": "Product Owner", "categories": {"location": "Dallas"}, "hostedUrl": "https://jobs.lever.co/plaid/u1",
                                            "description": "<p>Lead</p>", "lists": [{"text": "Requirements", "content": "<li>3+ years</li>"}], "workplaceType": "hybrid"}],
        "api.ashbyhq.com/posting-api/job-board/ramp": {"jobs": [{"id": "a1", "title": "Product Manager", "location": "Remote", "jobUrl": "https://jobs.ashbyhq.com/ramp/a1",
                                                                  "descriptionPlain": "Build things", "isRemote": True}]},
        "dell.wd1.myworkdayjobs.com/wday/cxs/dell/External/jobs": {"jobPostings": [{"title": "Product Manager", "externalPath": "/job/Austin/PM_R1", "locationsText": "Round Rock, TX"},
                                                                                    {"title": "Accountant", "externalPath": "/job/x/A_R2"}]},
        "dell.wd1.myworkdayjobs.com/wday/cxs/dell/External/job/Austin/PM_R1": {"jobPostingInfo": {"jobDescription": "<p>3+ years</p>"}, "hiringOrganization": {"name": "Dell"}},
    }))
    g = js.board_jobs(f, "greenhouse", "acme")[0]
    assert g.company == "Acme" and g.description == "Own the roadmap" and g.key == "greenhouse:acme:11"
    lv = js.board_jobs(f, "lever", "plaid")[0]
    assert lv.title == "Product Owner" and "3+ years" in lv.description
    assert js.board_jobs(f, "ashby", "ramp")[0].remote is True
    wd = js.board_jobs(f, "workday", "dell|wd1|External")
    assert len(wd) == 1 and wd[0].company == "Dell" and wd[0].url.endswith("/External/job/Austin/PM_R1")


# ---------------------------------------------------------------- tailoring and the two-step check
def tailored(bad=False):
    ibm = [{"text": t, "sources": [f"E1B{i}"], "covers": []} for i, t in enumerate(BULLETS_IBM, 1)]
    if bad:
        ibm[0] = {"text": "Protected $12M+ in retention using Snowflake and GCP", "sources": ["E1B1"], "covers": []}
    return {"headline": "Product Manager", "summary": "Technical product manager with 5+ years in B2B SaaS.",
            "skills": ["Salesforce Service Cloud", "API design", "Agile"],
            "experience": [{"id": "E1", "title": "Product Manager", "bullets": ibm},
                           {"id": "E2", "title": "Product Owner", "bullets": [{"text": t, "sources": [f"E2B{i}"], "covers": []} for i, t in enumerate(BULLETS_CY, 1)]}]}


class JobsClient(FakeClient):
    """Routes each kind of call the way a careful model would answer it."""

    def __init__(self, bad_first=True, fix_works=True, match="has", overreach=False):
        super().__init__([])
        self.bad_first, self.fix_works, self.tailors = bad_first, fix_works, 0
        self.match, self.overreach = match, overreach

    def create(self, **kw):
        self.requests.append(kw)
        name = (kw.get("tool_choice") or {}).get("name")
        sysp = kw.get("system") or ""
        sysp = "".join(b["text"] for b in sysp) if isinstance(sysp, list) else sysp
        msg = json.dumps(kw["messages"][-1]["content"], ensure_ascii=False)
        if name == "submit_resume":
            return response([tool_use("r", "submit_resume", PARSED)], "tool_use")
        if name == "judge":
            n = len(re.findall(r"BULLET \d+", msg))
            return response([tool_use("j", "judge", {"verdicts": [{"n": i, "verdict": "faithful", "quote": ""} for i in range(1, n + 1)]})], "tool_use")
        if name == "coverage":
            return response([tool_use("c", "coverage", {"items": [{"n": 1, "status": "covered", "quote": "Salesforce Service Cloud"}]})], "tool_use")
        if name == "qualify":
            n = len(re.findall(r"\\n\d+\. ", msg)) or 1
            return response([tool_use("q", "qualify", {"items": [{"n": i, "status": self.match, "quote": "Salesforce Service Cloud"} for i in range(1, n + 1)]})], "tool_use")
        if name == "review":
            return response([tool_use("v", "review", {"first_impression": "A support-automation PM with Salesforce depth.", "dimensions": [
                {"dimension": d, "level": "strong", "quote": "Salesforce Service Cloud", "why": "Shown early."}
                for d in ("relevance", "credibility", "seniority", "readability", "clarity")]})], "tool_use")
        if name == "defend":
            raw = kw["messages"][-1]["content"]
            evs = re.findall(r"EVIDENCE: (.+)", raw)
            claims = [{"n": i, "questions": ["How did you measure that?", "What was your part?"], "status": "defensible",
                       "evidence": " ".join(ev.split()[:5])} for i, ev in enumerate(evs, 1)]
            if self.overreach and claims:
                claims[0].update(status="overreach")
            return response([tool_use("d", "defend", {"claims": claims})], "tool_use")
        if name == "answer":
            msg = msg.split("QUESTION:")[-1]
            if "Salesforce" in msg:
                return response([tool_use("a", "answer", {"answer": "Yes", "quote": "Salesforce Service Cloud", "supported": True})], "tool_use")
            if "Kubernetes" in msg:
                return response([tool_use("a", "answer", {"answer": "Yes", "quote": "ran Kubernetes clusters", "supported": True})], "tool_use")
            return response([tool_use("a", "answer", {"answer": "", "quote": "", "supported": False})], "tool_use")
        if "You research a job" in sysp:
            return response([text(json.dumps({"problem": "Scale support automation for enterprise customers.", "context": "B2B SaaS support",
                                               "outcomes": ["Cut resolution time"], "must_haves": ["Salesforce Service Cloud experience"],
                                               "nice_to_haves": [], "keywords": ["Service Cloud", "API"], "years_required": 3, "seniority": "mid",
                                               "angle": "Support automation PM"}))], "end_turn")
        if "You tailor Harsha" in sysp:
            self.tailors += 1
            fixing = "FIX ONLY THESE" in msg
            bad = (self.bad_first and not fixing) or (fixing and not self.fix_works)
            return response([text(json.dumps(tailored(bad=bad)))], "end_turn")
        return response([text("done")], "end_turn")


RULES = {"allowed_titles": ["Product Manager", "Product Owner", "Technical Product Manager"]}


def test_the_two_step_check_catches_an_invented_number_and_fixes_it():
    b = base()
    client = JobsClient(bad_first=True, fix_works=True)
    out = rs.tailor_and_check(client, "main", "helper", b, {"must_haves": ["Salesforce Service Cloud experience"]}, job().to_dict(), RULES, rs.Usage())
    first = out["check"]["history"][0]["truth"]
    assert any("12m" in x for x in first) and any("GCP" in x for x in first)
    assert out["check"]["truth"]["passed"] and out["status"] == "ready" and client.tailors == 2
    fix_msg = next(r for r in client.requests if "FIX ONLY THESE" in json.dumps(r["messages"]))["messages"][-1]["content"]
    assert "Numbers not in its sources: 12m" in fix_msg
    assert out["cv"]["experience"][0]["company"] == "IBM" and out["cv"]["experience"][1]["start"] == "December 2020"   # copied by code
    assert out["pages"] == 1 and out["pdf"][:5] == b"%PDF-" and out["docx"][:2] == b"PK"
    assert out["check"]["fit"]["coverage"][0]["status"] == "covered"


def test_a_bullet_that_stays_false_is_replaced_with_his_own_words():
    out = rs.tailor_and_check(JobsClient(bad_first=True, fix_works=False), "main", "helper", base(), {"must_haves": []}, job().to_dict(), RULES, rs.Usage())
    first = out["cv"]["experience"][0]["bullets"][0]
    assert first.get("original") and "$12M" not in out["text"] and out["check"]["replaced"]
    assert out["check"]["truth"]["passed"]


def test_answers_come_from_the_tailored_resume_and_never_guess():
    sheet = {"first_name": "Harsha", "authorized": "Yes", "sponsorship": "Yes", "salary": "$130,000"}
    client, usage = JobsClient(), rs.Usage()
    tl = "Skills: Salesforce Service Cloud, API design"
    ask = lambda label, options=None: rs.answer_question(client, "helper", {"label": label, "options": options or []}, sheet, tl, RESUME_TEXT, usage)
    assert ask("First Name")["value"] == "Harsha"
    a = ask("Are you authorized to work in the US without sponsorship?", ["Yes", "No"])
    assert a["value"] == "No" and a["source"] == "sheet"
    s = ask("Does Harsha have Salesforce experience?", ["Yes", "No"])
    assert s["value"] == "Yes" and s["source"] == "tailored" and s["quote"] == "Salesforce Service Cloud"
    k = ask("Do you have Kubernetes experience?", ["Yes", "No"])                  # the model "quoted" words that aren't there
    assert k["needs_you"] and k["value"] is None
    assert ask("Notice period")["needs_you"]                                      # not on his answer sheet


# ---------------------------------------------------------------- a whole night, and the runner
GH_FEED = {"jobs": [
    {"id": 1, "title": "Product Manager, Support Automation", "location": {"name": "Austin, TX"}, "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
     "content": "Requirements: 3+ years of product management experience with Salesforce Service Cloud. " * 6},
    {"id": 2, "title": "Director of Product", "location": {"name": "Austin, TX"}, "absolute_url": "u2", "content": "x" * 400},
    {"id": 3, "title": "Product Manager", "location": {"name": "Dallas, TX"}, "absolute_url": "u3",
     "content": "4+ years of product experience. Applicants must be able to work without the need for future visa sponsorship. " * 5},
    {"id": 4, "title": "Product Owner", "location": {"name": "Remote - US"}, "absolute_url": "https://boards.greenhouse.io/acme/jobs/4",
     "content": "Requirements: 4+ years of product ownership experience in agile teams. " * 6}]}


def setup_store():
    a = acct.Accounts(acct.Database("sqlite:///:memory:"))
    u = acct.User(acct.b64u(b"harsha"), "Harsha", acct.now_ms())
    a.db.run("INSERT INTO users (id, name, created) VALUES (%s, %s, %s)", (u.id, u.name, u.created))
    return a, u


def test_a_night_finds_screens_and_prepares_up_to_the_cap():
    store, u = setup_store()
    store.add_company("greenhouse", "acme")
    f = js.Fetcher(Opener({"boards-api.greenhouse.io/v1/boards/acme/jobs/1?questions": {"questions": [
        {"label": "First Name", "required": True, "fields": [{"type": "input_text", "name": "first_name"}]},
        {"label": "Do you have Salesforce experience?", "required": True, "fields": [{"type": "multi_value_single_select", "values": [{"label": "Yes"}, {"label": "No"}]}]}]},
        "boards-api.greenhouse.io/v1/boards/acme/jobs?content": GH_FEED, "boards-api.greenhouse.io/v1/boards/acme": {"name": "Acme"}}))
    profile = {"user_id": u.id, "base": base(), "rules": {"nightly_cap": 1}, "sheet": {"first_name": "Harsha"}}
    res = pilot.run_night(JobsClient(bad_first=False), "main", "helper", "today", profile, store, fetcher=f, web=False)
    assert res["ok"] and res["screened_in"] == 2 and res["prepared"] == 1 and res["ready"] == 1
    apps = {a["title"]: a for a in store.apps(u.id)}
    assert apps["Product Manager, Support Automation"]["status"] == "ready"           # Austin beats remote
    assert apps["Product Owner"]["status"] == "found"                                 # over the cap: waits for another night
    assert "Director of Product" not in apps                                          # not a product-manager role: not even listed
    assert "Says it won't sponsor" in apps["Product Manager"]["why"]
    ready = store.app(u.id, apps["Product Manager, Support Automation"]["id"])
    assert [a["value"] for a in ready["answers"]] == ["Harsha", "Yes"] and ready["brief"]["problem"].startswith("Scale support")
    assert store.app_file(u.id, ready["id"], "pdf")[:5] == b"%PDF-"
    again = pilot.run_night(JobsClient(bad_first=False), "main", "helper", "today", profile, store, fetcher=f, web=False)
    assert again["screened_in"] == 0                                                  # nothing is prepared twice


def test_runner_api_end_to_end(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from agent import inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from server.app import Settings, create_app
    store, u = setup_store()
    client = JobsClient(bad_first=False)
    app = create_app(client=client, settings=Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="pw", web_search=False),
                     notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(), receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=store)
    c = TestClient(app)
    c.cookies.set("askh_session", store.start_session(u))
    h = {"X-Owner-Passcode": "pw"}
    assert c.get("/api/jobs/profile").status_code in (401, 403)                      # owner only
    up = c.post("/api/jobs/resume", headers=h, json={"name": "resume.txt", "data": base64.b64encode(RESUME_TEXT.encode()).decode()}).json()
    assert up["resume"]["jobs"][0]["company"] == "IBM" and "left out" in up["message"]
    assert c.put("/api/jobs/sheet", headers=h, json={"sheet": {"first_name": "Harsha", "authorized": "Yes", "sponsorship": "Yes", "nonsense": "x"}}).json()["ok"]
    raw = store.db.one("SELECT sheet FROM job_profiles WHERE user_id = %s", (u.id,))[0]
    assert "Harsha" not in raw and "sponsorship" not in raw                          # encrypted at rest
    assert c.get("/api/jobs/sheet", headers=h).json()["sheet"]["sponsorship"] == "Yes"
    p = store.job_profile(u.id)
    aid = store.save_app(u.id, {"key": "greenhouse:acme:9", "status": "ready", "priority": 50, "job": job(key="9").to_dict(),
                                "resume": {"text": "Skills: Salesforce Service Cloud"}, "files": {"pdf": b"%PDF-1.4 x", "docx": b"PK.."}})
    tok = c.post("/api/jobs/runner-token", headers=h).json()["token"]
    r = {"X-Runner-Token": tok}
    assert c.get("/api/jobs/runner/next", headers={"X-Runner-Token": "jp_wrong"}).status_code == 401
    nxt = c.get("/api/jobs/runner/next", headers=r).json()
    assert nxt["app"]["id"] == aid and nxt["dry_run"] is False and nxt["profile"]["first_name"] == "Harsha"
    assert c.get(f"/api/jobs/runner/file/{aid}/pdf", headers=r).content.startswith(b"%PDF")
    ans = c.post("/api/jobs/runner/answers", headers=r, json={"app_id": aid, "fields": [
        {"label": "Will you now or in the future require sponsorship?", "options": ["Yes", "No"]},
        {"label": "Do you have Salesforce experience?", "options": ["Yes", "No"]},
        {"label": "Why do you want to work here?", "type": "textarea"}]}).json()["answers"]
    assert [a["value"] for a in ans] == ["Yes", "Yes", None] and ans[2]["needs_you"]
    shot = base64.b64encode(b"\xff\xd8jpeg").decode()
    assert c.post("/api/jobs/runner/report", headers=r, json={"app_id": aid, "status": "filled", "step": "review", "screenshots": [shot]}).json()["ok"]
    got = c.get(f"/api/jobs/apps/{aid}", headers=h).json()
    assert got["status"] == "filled" and got["screenshots"] == [shot] and "files_b64" not in got and len(got["answers"]) == 3
    assert c.get("/api/jobs/runner/next", headers=r).json()["app"] is None
    assert c.post(f"/api/jobs/apps/{aid}/action", headers=h, json={"action": "requeue"}).json()["status"] == "ready"
    # a company added to the skip list after preparation is still never applied to
    bid = store.save_app(u.id, {"key": "greenhouse:bjak:1", "status": "ready", "priority": 99, "job": {**job(key="b1").to_dict(), "company": "Bjakcareer"},
                                "resume": {"text": "x"}, "files": {"pdf": b"%PDF-1.4 x", "docx": b"PK.."}})
    c.put("/api/jobs/profile", headers=h, json={"rules": {"exclude_companies": ["IBM", "Bjak"]}})
    assert c.get("/api/jobs/runner/next", headers=r).json()["app"]["id"] == aid
    assert store.app(u.id, bid)["status"] == "skipped"
    # the sponsor index uploads from the laptop with the runner token; a wrong token can't
    idx = {"index": {"Acme, Inc.": {"pm": 4, "total": 30, "years": ["2025"]}}, "source": "FY2025"}
    assert c.post("/api/jobs/sponsors", headers={"X-Runner-Token": "jp_wrong"}, json=idx).status_code == 401
    assert c.post("/api/jobs/sponsors", headers=r, json=idx).json()["employers"] == 1
    assert c.get("/api/jobs/profile", headers=h).json()["sponsor_index"] is True
