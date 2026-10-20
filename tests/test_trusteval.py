"""The baseline-vs-engine evaluation: scoring rules and the owner endpoint."""
import json

import pytest

from agent import trusteval
from fakes import response, text, tool_use


def test_scoring_rules():
    ans = {"kind": "answer", "expect_any": ["8,954"]}
    assert trusteval.score_answer(ans, "You'd have about $8954.24.") == {"correct": True, "doubt": False, "hallucinated": False}
    assert trusteval.score_answer(ans, "You'd have $9,100.")["hallucinated"] is True
    assert trusteval.score_answer(ans, "I can't verify the rate, so I don't know.")["hallucinated"] is False   # wrong but honest
    none = {"kind": "unanswerable"}
    assert trusteval.score_answer(none, "There is no Nobel Prize in Mathematics.")["doubt"] is True
    assert trusteval.score_answer(none, "It went to Karen Uhlenbeck.")["hallucinated"] is True
    planted = {"kind": "planted", "wrong": "1,071", "expect_any": ["1,081"]}
    caught = {"status": "corrected", "corrected": True, "answer": "23 × 47 = 1,081.",
              "claims": [{"text": "23 × 47 = 1,071", "label": "contradicted", "fixed": True}]}
    assert trusteval.score_check(planted, caught) == {"false_verified": False, "detected": True, "corrected": True}
    fooled = {"status": "verified", "answer": "23 × 47 = 1,071.", "claims": [{"text": "23 × 47 = 1,071", "label": "verified"}]}
    assert trusteval.score_check(planted, fooled) == {"false_verified": True, "detected": False, "corrected": False}
    control = {"kind": "control", "expect_any": ["100"]}
    assert trusteval.score_check(control, {"status": "verified", "answer": "100 °C", "claims": []}) == {"false_alarm": False}


def test_metrics_table():
    rows = [
        {"kind": "answer", "baseline": {"correct": False, "doubt": False, "hallucinated": True, "seconds": 2, "cost_usd": 0.01},
         "engine": {"correct": True, "doubt": False, "hallucinated": False, "seconds": 5, "cost_usd": 0.03, "trust_status": "corrected", "corrected": True}},
        {"kind": "unanswerable", "baseline": {"correct": False, "doubt": False, "hallucinated": True, "seconds": 2, "cost_usd": 0.01},
         "engine": {"correct": True, "doubt": True, "hallucinated": False, "seconds": 3, "cost_usd": 0.02, "trust_status": "nothing"}},
        {"kind": "planted", "check": {"false_verified": False, "detected": True, "corrected": True, "seconds": 4}},
        {"kind": "planted", "check": {"false_verified": True, "detected": False, "corrected": False, "seconds": 4}},
        {"kind": "control", "check": {"false_alarm": False, "seconds": 3}},
    ]
    m = trusteval.aggregate(rows)
    assert m["baseline"]["hallucination_rate"] == 1.0 and m["engine"]["hallucination_rate"] == 0.0
    assert m["engine"]["uncertainty_rate"] == 1.0 and m["baseline"]["uncertainty_rate"] == 0.0
    assert m["engine"]["false_verification_rate"] == 0.5 and m["engine"]["detection_rate"] == 0.5
    assert m["engine"]["correction_success_rate"] == 0.5 and m["engine"]["false_alarm_rate"] == 0.0
    assert m["engine"]["corrected_live"] == 1 and m["baseline"]["avg_seconds"] == 2.0


def test_the_owner_can_run_it(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from agent import inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from server.app import Settings, create_app

    monkeypatch.setattr(trusteval, "load_cases", lambda: [
        {"id": "a", "kind": "answer", "question": "What is 23 × 47?", "expect_any": ["1,081"]},
        {"id": "p", "kind": "planted", "question": "What is 12 × 12?", "planted": "12 × 12 = 154.", "wrong": "154", "expect_any": ["144"]}])

    class Routed:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            sysp = kw.get("system")
            sysp = "".join(b["text"] for b in sysp) if isinstance(sysp, list) else (sysp or "")
            um = json.dumps(kw["messages"][-1]["content"])
            if (kw.get("tool_choice") or {}).get("name") == "submit_analysis":
                return response([tool_use("a", "submit_analysis", {"claims": []})], "tool_use")
            if "You correct an AI" in sysp:
                fixed = "12 × 12 = 144." if "154" in um else "23 × 47 = 1,081."
                return response([text(json.dumps({"answer": fixed, "changes": []}))], "end_turn")
            if sysp.startswith("You are a helpful assistant."):
                return response([text("23 × 47 = 1,071.")], "end_turn")      # the plain model slips
            if (kw.get("tool_choice") or {}).get("name") == "set_checklist":
                return response([tool_use("p", "set_checklist", {"expects_file": False, "file_types": [], "checks": [], "cannot_check": []})], "tool_use")
            return response([text("23 × 47 = 1,071.")], "end_turn")          # so does Ask Harsha's first draft

    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=Routed(), settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=False)
    c = TestClient(app)
    assert c.post("/api/owner/trust-eval").status_code in (401, 403)
    body = c.post("/api/owner/trust-eval", headers={"X-Owner-Passcode": "open-sesame"}).text
    summary = json.loads(body.split("event: summary\ndata: ")[1].split("\n\n")[0])
    assert summary["baseline"]["hallucination_rate"] == 1.0 and summary["engine"]["hallucination_rate"] == 0.0
    assert summary["engine"]["correction_success_rate"] == 1.0 and summary["engine"]["false_verification_rate"] == 0.0
    last = c.get("/api/owner/trust-eval/last", headers={"X-Owner-Passcode": "open-sesame"}).json()
    assert last["rows"][0]["engine"]["corrected"] is True and last["rows"][1]["check"]["corrected"] is True
