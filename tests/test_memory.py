import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import accounts as acct, inbox as inboxmod, memory, receipts as rcpt  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient, response, text  # noqa: E402
from server.app import Settings, create_app  # noqa: E402


def turns(n):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    return out


def test_condense_keeps_recent_turns_word_for_word():
    assert memory.condense(turns(8)) == ([], turns(8))
    older, recent = memory.condense(turns(11))
    assert len(older) == 14 and recent == turns(11)[-8:] and recent[0]["content"] == "question 7"


def test_learn_filters_what_the_model_returns():
    reply = {"remember": ["Works as a PM at IBM", "works as a pm at ibm", "Lives in Austin", "Likes tea", "extra"],
             "forget": ["M1", "M999"], "project_notes": "- Budget is $2k"}
    client = FakeClient([response([text(json.dumps(reply))], "end_turn")])
    out = memory.learn(client, "m", "I'm a PM at IBM in Austin", "Nice!", [{"id": "M1", "text": "Works as a PM at IBM"}],
                       {"id": "p1", "name": "Trip"})
    assert out.remember == ["Lives in Austin", "Likes tea", "extra"]   # max 3, no repeats of what it already knows
    assert out.forget == ["M1"] and out.project_notes == "- Budget is $2k"
    off = memory.learn(FakeClient([response([text(json.dumps(reply))], "end_turn")]), "m", "I'm a PM", "ok", [], None, memory_on=False)
    assert off.remember == [] and off.project_notes is None
    assert memory.worth_learning("What's the capital of Peru?", None, True) is False
    assert memory.worth_learning("I'm moving to Denver", None, True) is True
    assert memory.worth_learning("I'm moving to Denver", None, False) is False
    assert memory.worth_learning("Capital of Peru?", {"id": "p"}, False) is True


def setup(script):
    a = acct.Accounts(acct.Database("sqlite:///:memory:"))
    u = acct.User(acct.b64u(b"user-sam"), "Sam", acct.now_ms())
    a.db.run("INSERT INTO users (id, name, created) VALUES (%s, %s, %s)", (u.id, u.name, u.created))
    client = FakeClient(script)
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=client, settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=a)
    c = TestClient(app)
    c.cookies.set("askh_session", a.start_session(u))
    return a, u, c, client


def test_long_chats_are_summarized_not_dropped():
    a, u, c, client = setup([response([text("They're planning a Lisbon trip in May on a $2k budget.")], "end_turn"),
                             response([text("Try the Alfama walk.")], "end_turn")])
    r = c.post("/api/chat", json={"question": "Ideas for day 3?", "history": turns(10), "summary": "Old summary."}).json()
    summarize_req, answer_req = client.requests
    assert "Old summary." in summarize_req["messages"][0]["content"] and "question 0" in summarize_req["messages"][0]["content"]
    assert "Lisbon trip in May" in answer_req["system"] and len(answer_req["messages"]) == 9   # 4 recent turns + the question
    assert r["summarized"] == 12 and r["summary"].startswith("They're planning")
    assert answer_req["system_blocks"][0]["cache_control"] == {"type": "ephemeral"}             # stable prompt cached
    assert "Lisbon" not in answer_req["system_blocks"][0]["text"]                                # personal text after the cache


def test_memories_and_project_notes_are_learned_and_used():
    learned = {"remember": ["Is planning a trip to Lisbon"], "forget": [], "project_notes": "- Trip is in May"}
    a, u, c, client = setup([response([text("Great choice.")], "end_turn"), response([text(json.dumps(learned))], "end_turn"),
                             response([text("Pack light layers.")], "end_turn"), response([text('{"remember":[],"forget":[],"project_notes":null}')], "end_turn")])
    p = a.save_project(u.id, {"name": "Trip", "instructions": "Mid-range budget."})
    c.post("/api/chat", json={"question": "I'm going to Lisbon in May", "project": p["id"]})
    import time
    for _ in range(50):   # learning runs in the background after the answer
        if a.list_memories(u.id):
            break
        time.sleep(0.02)
    assert [m["text"] for m in a.list_memories(u.id)] == ["Is planning a trip to Lisbon"]
    assert a.get_project(u.id, p["id"])["notes"] == "- Trip is in May"
    c.post("/api/chat", json={"question": "What should I pack?", "project": p["id"]})
    sysp = client.requests[2]["system"]
    assert "Is planning a trip to Lisbon" in sysp and "- Trip is in May" in sysp
    mid = a.list_memories(u.id)[0]["id"]
    assert c.patch(f"/api/memory/{mid}", json={"text": "Trip to Lisbon in May"}).json()["memories"][0]["text"] == "Trip to Lisbon in May"
    assert c.delete(f"/api/memory/{mid}").json() == {"memories": []}
    c.post("/api/chat", json={"question": "I love museums", "prefs": {"memory": False}})   # memory off: nothing used or learned
    assert len(client.requests) == 4 or "what_you_remember" not in client.requests[-1]["system"]


def test_stream_reports_what_was_remembered():
    learned = {"remember": ["Works as a product manager"], "forget": [], "project_notes": None}
    a, u, c, client = setup([response([text("Nice to meet you.")], "end_turn"), response([text(json.dumps(learned))], "end_turn")])
    body = c.post("/api/chat/stream", json={"question": "I'm a product manager"}).text
    assert "event: memory" in body and "Works as a product manager" in body
    assert body.index("event: final") < body.index("event: memory")
