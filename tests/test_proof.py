import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import inbox as inboxmod, proof, receipts as rcpt  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import RoutingClient, response, text, tool_use  # noqa: E402
from server.app import Settings, create_app  # noqa: E402

PLAN = {"expects_file": True, "file_types": ["xlsx"],
        "checks": [{"type": "columns", "value": ["Company", "Role", "Link"], "label": "Has Company, Role and Link"},
                   {"type": "min_rows", "value": 5, "label": "Has 5 jobs"}],
        "cannot_check": ["The jobs are real and still open"]}


def sheet(rows):
    return {"filename": "jobs.xlsx", "sheets": [{"name": "Jobs", "rows": rows}]}


BAD = sheet([["Company", "Role"], ["Acme", "PM"], ["Beta", "PM"]])
GOOD = sheet([["Company", "Role", "Link"]] + [[f"Co {i}", "PM", f"https://jobs.example/{i}"] for i in range(5)])


def app_with(script, plan=PLAN):
    client = RoutingClient(script, plan)
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=client, settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=False)
    return TestClient(app), client


def test_a_failed_check_is_fixed_then_verified_and_signed():
    c, client = app_with([response([tool_use("t1", "create_file", BAD)], "tool_use"),
                          response([text("Here's jobs.xlsx with five PM jobs.")], "end_turn"),           # claims success
                          response([tool_use("t2", "create_file", GOOD)], "tool_use"),
                          response([text("Your spreadsheet has 5 jobs with links.")], "end_turn")])
    d = c.post("/api/chat", json={"question": "Find five PM jobs and put them in a spreadsheet with company, role and link", "proof": True}).json()
    p = d["proof"]
    assert p["status"] == "verified" and p["attempts"] == 2 and p["caught"] is True     # a false success claim was caught
    assert p["history"][0]["status"] == "partial" and "Has 5 jobs" in p["history"][0]["problems"]
    fix = client.requests[2]["messages"][-1]["content"]
    assert "Proof Mode check" in fix and "Missing Link" in fix and "Counted 2 data rows" in fix
    assert d["answer"] == "Your spreadsheet has 5 jobs with links." and d["files"][0]["sheets"][0]["rows"] == 6
    assert [s["state"] for s in p["states"]][:3] == ["pending", "in_progress", "verification"] and p["states"][-1]["state"] == "verified"
    review = [ch for ch in p["checks"] if ch["result"] == "unverified"]
    assert review and review[0]["label"] == "The jobs are real and still open"
    assert "Proof Mode is on" in client.requests[0]["system"]
    ok = c.post("/api/proof/verify", json={"receipt": p["receipt"], "signature": p["signature"]}).json()
    assert ok["valid"] is True and "isn't proof of events outside" in ok["meaning"]
    tampered = {**p["receipt"], "status": "failed"}
    assert c.post("/api/proof/verify", json={"receipt": tampered, "signature": p["signature"]}).json()["valid"] is False
    stats = c.get("/api/owner/proof-stats", headers={"X-Owner-Passcode": "open-sesame"}).json()
    assert stats["tasks"] == 1 and stats["false_success_caught"] == 1 and stats["retry_success_rate"] == 1.0
    assert c.get("/api/owner/proof-stats").status_code in (401, 403)


def test_retries_are_bounded_and_the_report_is_honest():
    script = []
    for _ in range(proof.MAX_ATTEMPTS):
        script += [response([tool_use("t", "create_file", BAD)], "tool_use"), response([text("Done!")], "end_turn")]
    c, client = app_with(script)
    p = c.post("/api/chat", json={"question": "Make a spreadsheet of five PM jobs with company, role and link", "proof": True}).json()["proof"]
    assert p["attempts"] == proof.MAX_ATTEMPTS and p["status"] == "partial"
    assert {ch["label"] for ch in p["checks"] if ch["result"] == "fail"} >= {"Has 5 jobs", "Has Company, Role and Link"}


def test_plain_questions_skip_proof_and_reports_are_checked_in_the_reply():
    c, client = app_with([response([text("Paris.")], "end_turn")])
    d = c.post("/api/chat", json={"question": "What's the capital of France?"}).json()
    assert "proof" not in d and client.plan_requests == []
    plan = {"expects_file": False, "file_types": [], "checks": [{"type": "sections", "value": ["Summary", "Risks"], "label": "Has Summary and Risks"}], "cannot_check": []}
    c2, _ = app_with([response([text("## Summary\nOn track.\n\n## Risks\n- Hiring")], "end_turn")], plan)
    p = c2.post("/api/chat", json={"question": "Write a short status report with a summary and risks", "proof": True}).json()["proof"]
    assert p["status"] == "verified" and p["checks"][0]["evidence"].startswith("Found in the reply")


def test_inspector_catches_broken_code_formulas_and_placeholders():
    plan = proof.Plan()
    r = proof.verify(plan, [{"filename": "a.py", "spec": {"filename": "a.py", "content": "def f(:\n  pass\n"}},
                            {"filename": "b.xlsx", "spec": sheet([["A", "B"], ["$1,200", "=SUM(A2:A2)"], ["x", "=NOPE(A9)"]]) | {"filename": "b.xlsx"}},
                            {"filename": "c.md", "spec": {"filename": "c.md", "content": "Dear [Insert name], lorem ipsum"}}], "")
    ev = " ".join(c["evidence"] for c in r.checks if c["result"] == "fail")
    assert "Python syntax error on line 1" in ev and "unknown function NOPE" in ev and "is the text '$1,200'" in ev
    assert "lorem ipsum" in ev.lower() and r.status == "partial"
