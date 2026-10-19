import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import inbox as inboxmod  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient, response, text, tool_use  # noqa: E402
from server.app import Settings, create_app  # noqa: E402

OWNER = {"x-owner-passcode": "open-sesame"}


def make(script=(), inbox=None, notes=None):
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    fake = FakeClient(list(script))
    box = inbox if inbox is not None else inboxmod.MemoryInboxStore()
    notes = notes if notes is not None else MemoryNotesStore()
    return TestClient(create_app(client=fake, settings=s, notes=notes, inbox=box)), fake, box, notes


def test_store_round_trip_is_encrypted():
    key = inboxmod.make_key("test key")
    box = inboxmod.MemoryInboxStore(key)
    item = box.submit("Would Harsha mentor a student?", "Not in his record", "about_harsha", "I don't know.")
    assert len(item.id) >= 12 and "mentor" not in box.text          # stored text is ciphertext
    fresh = inboxmod.MemoryInboxStore(key)
    fresh.text = box.text
    assert fresh.get(item.id).question == "Would Harsha mentor a student?"
    box.answer(item.id, "Yes, happy to.")
    assert box.get(item.id).public() == {"id": item.id, "status": "answered", "question": item.question,
                                         "answer": "Yes, happy to.", "answered_at": box.get(item.id).answered_at}


def test_wrong_key_refuses_instead_of_overwriting():
    box = inboxmod.MemoryInboxStore()
    box.submit("q")
    other = inboxmod.MemoryInboxStore(inboxmod.make_key("a different key"))
    other.text = box.text
    with pytest.raises(inboxmod.InboxUnavailable, match="can't be unlocked"):
        other.submit("new")
    assert other.text == box.text


def test_key_comes_from_inbox_key_or_passcode():
    assert inboxmod.make_key() is None
    assert inboxmod.make_key("k") != inboxmod.make_key(owner_passcode="k")
    assert inboxmod.make_key(owner_passcode="pw") == inboxmod.make_key(owner_passcode="pw")


def test_waiting_ticket_hides_nothing_private():
    box = inboxmod.MemoryInboxStore()
    item = box.submit("q", context="agent said x", note="my note")
    pub = item.public()
    assert pub["status"] == "waiting" and "context" not in pub and "note" not in pub and pub["answer"] == ""


def test_agent_offer_reaches_the_page():
    script = [response([tool_use("t1", "suggest_asking_harsha", {"reason": "His record doesn't say.", "kind": "about_harsha"})], "tool_use"),
              response([text("His record doesn't say whether he mentors students. You can ask him directly.")], "end_turn")]
    c, fake, _, _ = make(script)
    r = c.post("/api/chat", json={"question": "Would Harsha mentor a student?"})
    assert r.json()["ask_harsha"] == {"reason": "His record doesn't say.", "kind": "about_harsha"}
    assert "suggest_asking_harsha" in [t["name"] for t in fake.requests[0]["tools"]]


def test_no_offer_means_null_and_no_tool_without_a_key():
    c, _, _, _ = make([response([text("Canberra.")], "end_turn")])
    assert c.post("/api/chat", json={"question": "Capital of Australia?"}).json()["ask_harsha"] is None
    # an inbox without a key is disabled: the model never sees the tool
    inbox = inboxmod.InboxStore(None)
    c3, fake3, _, _ = make([response([text("ok")], "end_turn")], inbox=inbox)
    c3.post("/api/chat", json={"question": "hi"})
    assert "suggest_asking_harsha" not in [t["name"] for t in fake3.requests[0]["tools"]]
    assert c3.post("/api/inbox", json={"question": "x"}).status_code == 503


def test_visitor_submits_owner_answers_and_teaches_the_agent():
    c, _, box, notes = make()
    r = c.post("/api/inbox", json={"question": "Would Harsha mentor a student?", "reason": "Not in his record", "kind": "about_harsha", "context": "I don't know."})
    tid = r.json()["id"]
    assert c.get(f"/api/inbox/{tid}").json()["status"] == "waiting"
    assert c.get("/api/owner/inbox").status_code == 401
    listed = c.get("/api/owner/inbox", headers=OWNER).json()
    assert listed["open"][0]["question"] == "Would Harsha mentor a student?" and listed["open"][0]["context"] == "I don't know."
    a = c.post(f"/api/owner/inbox/{tid}/answer", json={"answer": "Yes, I enjoy mentoring.", "remember": True}, headers=OWNER).json()
    assert a["note_id"] == "N001" and "Yes, I enjoy mentoring." in notes.get("N001").text
    t = c.get(f"/api/inbox/{tid}").json()
    assert t["status"] == "answered" and t["answer"] == "Yes, I enjoy mentoring."
    assert c.get("/api/owner/inbox", headers=OWNER).json()["open"] == []


def test_dismissed_and_unknown_tickets_are_404():
    c, _, box, _ = make()
    tid = c.post("/api/inbox", json={"question": "spam"}).json()["id"]
    assert c.post(f"/api/owner/inbox/{tid}/dismiss", headers=OWNER).status_code == 200
    assert c.get(f"/api/inbox/{tid}").status_code == 404
    assert c.get("/api/inbox/doesnotexist123").status_code == 404
    assert c.get("/api/inbox/bad!id").status_code == 404


def test_visitors_can_send_five_a_day():
    c, _, _, _ = make()
    hdr = {"x-forwarded-for": "5.5.5.5"}
    codes = [c.post("/api/inbox", json={"question": f"q{i}"}, headers=hdr).status_code for i in range(6)]
    assert codes == [200] * 5 + [429]


def test_nothing_about_the_question_is_logged(caplog):
    c, _, _, _ = make()
    with caplog.at_level("INFO", logger="ask_harsha"):
        c.post("/api/inbox", json={"question": "my secret plan", "note": "private"})
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "inbox_submit" in logged and "secret" not in logged and "private" not in logged


def test_github_store_writes_ciphertext_to_the_notes_branch():
    import base64
    import io
    sent = []

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def opener(req, timeout=10):
        if req.get_method() == "GET":
            import urllib.error
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
        sent.append(json.loads(req.data))
        return Resp(json.dumps({"content": {"sha": "abc"}}).encode())

    box = inboxmod.GitHubInboxStore("me/repo", "tok", inboxmod.make_key("k"), opener=opener)
    box.submit("Would Harsha mentor?")
    body = sent[0]
    assert body["branch"] == "notes" and body["message"] == "Inbox update"
    stored = base64.b64decode(body["content"]).decode()
    assert "mentor" not in stored and json.loads(stored)["v"] == 1
