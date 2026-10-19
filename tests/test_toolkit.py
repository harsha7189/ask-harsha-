import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import inbox as inboxmod, receipts as rcpt, toolkit  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient, response, text  # noqa: E402
from server.app import Settings, create_app  # noqa: E402

BRIEF = """## Today in AI agents, product management
- **A launch** happened. Why it matters: it does.

## Today's lesson: Opportunity solution trees
Some lesson text.

## Try this today
Draw one.

```quiz
{"questions": [{"q": "What is the root?", "options": ["Outcome", "Feature", "Bug", "Team"], "answer": 0, "why": "Start from the outcome."},
               {"q": "Bad", "options": ["only one"], "answer": 0}]}
```"""


def make(script=(), **kw):
    s = Settings(**{"daily_budget_usd": 5.0, "rate_limit": 50, "allowed_origins": [], "owner_passcode": "open-sesame", **kw})
    fake = FakeClient(list(script))
    app = create_app(client=fake, settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")))
    return TestClient(app), fake


def final(r):
    block = [b for b in r.text.split("\n\n") if b.startswith("event: final")][0]
    return json.loads(block.split("data: ", 1)[1])


def test_quiz_is_split_out_and_validated():
    body, quiz = toolkit.split_quiz(BRIEF)
    assert "```quiz" not in body and len(quiz["questions"]) == 1 and quiz["questions"][0]["answer"] == 0
    assert toolkit.lesson_title(body) == "Opportunity solution trees"
    assert toolkit.split_quiz("no quiz here") == ("no quiz here", None)


def test_every_tool_has_a_prompt_and_input_is_wrapped_as_data():
    for tid in toolkit.TOOLS:
        assert len(toolkit.system_prompt(tid, "Monday", {})) > 300
    assert "<input>\nhello\n</input>" in toolkit.user_message("meeting", "hello", {})
    assert "Interview transcript" in toolkit.user_message("interview", "x", {"mode": "analyze"})
    assert "non-leading" in toolkit.system_prompt("interview", "Monday", {"mode": "prepare"})
    assert "Internal" in toolkit.system_prompt("release", "Monday", {"audience": "internal"})
    assert "Leave out ticket numbers" in toolkit.system_prompt("release", "Monday", {"audience": "customers"})


def test_meeting_runs_without_record_tools_or_web_search():
    c, fake = make([response([text("## Summary\n- Ship CSV export")], "end_turn")])
    r = c.post("/api/tools/meeting/stream", json={"text": "Priya: let's ship CSV."})
    d = final(r)
    assert d["text"].startswith("## Summary") and d["tool"] == "meeting"
    req = fake.requests[0]
    assert "tools" not in req and "meeting transcript" in req["system"]
    assert req["messages"][-1]["content"].endswith("</input>")


def test_morning_brief_uses_web_search_and_is_cached_per_day():
    c, fake = make([response([text(BRIEF)], "end_turn")])
    body = {"options": {"topics": ["AI agents", "Product management"]}}
    d = final(c.post("/api/tools/morning/stream", json=body))
    assert d["quiz"]["questions"][0]["q"] == "What is the root?" and d["lesson"] == "Opportunity solution trees"
    assert [t["name"] for t in fake.requests[0]["tools"]] == ["web_search"]
    again = final(c.post("/api/tools/morning/stream", json={"options": {"topics": ["product management", "ai agents"]}}))
    assert again["cached"] is True and again["cost_usd"] == 0 and len(fake.requests) == 1


def test_tool_validation():
    c, _ = make()
    assert c.post("/api/tools/nope/stream", json={"text": "x"}).status_code == 404
    assert c.post("/api/tools/spec/stream", json={"text": "  "}).status_code == 400
    assert c.post("/api/tools/spec/stream", json={"text": "x" * 70_000}).status_code == 400
    bad = c.post("/api/tools/spec/stream", json={"text": "x", "attachments": [{"name": "a.pdf", "kind": "pdf", "data": "aGVsbG8="}]})
    assert bad.status_code == 400


def test_owner_gets_a_higher_input_limit_and_files_work():
    c, fake = make([response([text("## Verdict\nNeeds work")], "end_turn")])
    r = c.post("/api/tools/spec/stream", json={"text": "x" * 70_000, "attachments": [{"name": "s.txt", "kind": "text", "data": "Spec body"}]},
               headers={"x-owner-passcode": "open-sesame"})
    assert final(r)["text"].startswith("## Verdict")
    assert fake.requests[0]["messages"][0]["content"][1]["text"].startswith('<file name="s.txt">')


def test_toolkit_page_is_served():
    c, _ = make()
    assert "PM Toolkit" in c.get("/tools").text
