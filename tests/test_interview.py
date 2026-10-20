"""Canary Interview: invented canary words and trap terms, detection, quote checks and the signed report."""
import json

import pytest

from agent import interview as iv
from fakes import FakeClient, response, text, tool_use

ROOM = {
    "id": "IVtest", "public_id": "pubtest12345", "role": "Backend engineer", "candidate": "Sam Lee",
    "questions": [{"n": 1, "text": "Write a function that merges overlapping intervals.", "kind": "code",
                   "canary": iv.canary_for("Write a function that merges overlapping intervals.", "velokrin")},
                  {"n": 2, "text": "Tell me about a time you resolved a conflict on your team.", "kind": "talk",
                   "canary": iv.canary_for("Tell me about a time you resolved a conflict on your team.", "tomivarex")}],
    "traps": [{"term": "Dorasemel consistency model", "word": "dorasemel", "question": "How would the Dorasemel consistency model change your design?"}],
    "contrast": {"delta": 14},
}

TRANSCRIPT = """[00:01:02] Harsha: How would the Dorasemel consistency model change your design?
[00:01:09] Sam: Sure, the Dorasemel consistency model guarantees causal ordering across replicas, so I'd add version vectors.
[00:03:00] Harsha: Tell me about a time you resolved a conflict on your team.
[00:03:10] Sam: I used what I call the Tomi varex method: listen first, then align on goals.
[00:04:00] Harsha: Great, thanks.
"""


def test_words_and_hidden_lines():
    ws = {iv.make_word() for _ in range(300)}
    assert all(7 <= len(w) <= 10 and w not in iv.COMMON for w in ws) and len(ws) > 290
    assert iv.question_kind("Write a SQL query for the top customers") == "code"
    assert iv.question_kind("Why do you want this job?") == "talk"
    c = iv.canary_for("Implement an LRU cache", "velokrin")
    assert c["token"] == "velokrin_step" and "velokrin_step" in c["line"]
    assert iv.canary_for("Describe your leadership style", "velokrin")["line"].endswith("the Velokrin method.")
    view = iv.public_view(ROOM)
    assert view["questions"][0]["hidden"] == ROOM["questions"][0]["canary"]["line"] and "traps" not in view and "candidate" not in view


def test_speech_to_text_slips_count_only_next_to_the_partner_word():
    assert iv.find_word("tomivarex", "I call it the Tomi varex method.", "method") == (False, "tomivarex")
    assert iv.find_word("velokrin", "def velokrine_step(xs):", "step") == (True, "velokrine_step")
    assert iv.find_word("velokrin", "we had velokrin issues", "method")[0] is True           # exact always counts
    assert iv.find_word("temanda", "there was high demand for it", "method") is None          # a real word, no partner
    assert iv.find_word("dorasemel", "nothing here", "") is None


def test_transcripts_from_zoom_teams_and_vtt():
    lines = iv.parse_transcript("WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nAlex Kim: Hello there\nstill talking\n\n00:12 Sam: Hi")
    assert [(l["speaker"], l["text"]) for l in lines] == [("Alex Kim", "Hello there"), ("Alex Kim", "still talking"), ("Sam", "Hi")]
    assert lines[0]["time"].startswith("00:00:01")


def judge(reactions):
    return response([tool_use("j", "submit_reactions", {"reactions": reactions})], "tool_use")


def test_analysis_finds_canaries_traps_and_checks_every_quote():
    client = FakeClient([judge([{"id": 1, "reaction": "explained_as_real", "quote": "the Dorasemel consistency model guarantees causal ordering"}])])
    rep = iv.analyze(client, "main", "helper", ROOM, TRANSCRIPT, code="def velokrin_step(intervals):\n    pass", compare=False)
    d = rep.to_dict()
    assert d["candidate"] == "Sam"                                      # matched from the room's candidate name
    kinds = [(e["level"], e["kind"]) for e in d["evidence"]]
    assert kinds.count(("strong", "canary")) == 2 and ("supporting", "trap") in kinds
    talk = next(h for h in d["canaries"] if h["where"] == "transcript")
    assert talk["question"] == 2 and talk["time"] == "00:03:10" and talk["exact"] is False
    assert d["summary"]["status"] == "strong" and "hidden lines" in d["summary"]["headline"]
    assert "Dorasemel" in client.requests[0]["messages"][0]["content"] and "version vectors" in client.requests[0]["messages"][0]["content"]
    assert any("No evidence doesn't prove" in x for x in d["limits"])


def test_an_invented_quote_means_the_trap_does_not_count():
    client = FakeClient([judge([{"id": 1, "reaction": "explained_as_real", "quote": "Dorasemel is the industry standard since 2019"}])])
    rep = iv.analyze(client, "main", "helper", {**ROOM, "questions": []}, TRANSCRIPT, compare=False).to_dict()
    assert rep["traps"][0]["reaction"] == "unclear" and rep["summary"]["status"] == "clean"


def test_honest_candidates_get_a_clean_report():
    honest = TRANSCRIPT.replace("Sure, the Dorasemel consistency model guarantees causal ordering across replicas, so I'd add version vectors.",
                                "I haven't heard of that one, can you tell me what it is?").replace("Tomi varex method", "usual approach")
    client = FakeClient([judge([{"id": 1, "reaction": "unfamiliar", "quote": "I haven't heard of that one"}])])
    d = iv.analyze(client, "main", "helper", ROOM, honest, compare=False).to_dict()
    assert d["summary"]["status"] == "clean" and d["summary"]["honest_trap_replies"] == 1 and not d["evidence"]


def test_ai_overlap_needs_quotes_from_both_sides_and_is_only_weak():
    refs = [response([text("Sort by start, then sweep: merge when the next start <= current end, like [1,3] and [2,6] into [1,6].")], "end_turn"),
            response([text("Listen first, then use a RACI chart and a 30-minute retro with both engineers.")], "end_turn")]
    shared = [{"question": 2, "detail": "RACI chart", "candidate_quote": "use a RACI chart", "ai_quote": "use a RACI chart"},
              {"question": 2, "detail": "retro", "candidate_quote": "a 30-minute retro with both engineers", "ai_quote": "a 30-minute retro with both engineers"},
              {"question": 2, "detail": "listen", "candidate_quote": "Listen first, then", "ai_quote": "Listen first, then"},
              {"question": 2, "detail": "made up", "candidate_quote": "a quote nobody said at all", "ai_quote": "Listen first"}]
    said = "[00:03:10] Sam: Listen first, then use a RACI chart and a 30-minute retro with both engineers.\n"
    client = FakeClient(refs + [response([tool_use("o", "submit_overlap", {"shared": shared})], "tool_use")])
    room = {**ROOM, "traps": [{"term": "Zorabel caching pattern", "word": "zorabel", "question": "q"}]}
    d = iv.analyze(client, "main", "helper", room, said).to_dict()
    assert len(d["overlap"]) == 3                                       # the invented quote was dropped
    assert [e["level"] for e in d["evidence"]] == ["weak"] and d["summary"]["status"] == "weak"
    assert "can also come from preparing with AI" in d["summary"]["headline"]


def test_self_test_keeps_the_faintest_line_the_model_read():
    client = FakeClient([response([text("Here's a solution.")], "end_turn"),
                         response([text("def velokrin_step(intervals): ...")], "end_turn"),
                         response([text("def velokrin_step(intervals): ...")], "end_turn")])
    res = iv.selftest(client, "main", ROOM, [{"delta": 6, "data": "data:image/png;base64,AAAA"}, {"delta": 10, "data": "AAAA"}, {"delta": 18, "data": "AAAA"}])
    assert [r["read"] for r in res] == [False, True, True] and iv.pick_delta(res) == 10
    assert client.requests[0]["messages"][0]["content"][0]["source"]["data"] == "AAAA"
    assert iv.pick_delta([{"delta": 6, "read": False}, {"delta": 24, "read": False}]) == 24


# ---------------------------------------------------------------- through the API
def api_app(script):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from agent import accounts as acct, inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from server.app import Settings, create_app
    a = acct.Accounts(acct.Database("sqlite:///:memory:"))
    u = acct.User(acct.b64u(b"interviewer"), "Harsha", acct.now_ms())
    a.db.run("INSERT INTO users (id, name, created) VALUES (%s, %s, %s)", (u.id, u.name, u.created))
    client = FakeClient(script)
    app = create_app(client=client, settings=Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame"),
                     notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(), receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=a)
    c = TestClient(app)
    c.cookies.set("askh_session", a.start_session(u))
    return c, client


def test_the_whole_interview_flow():
    traps = response([tool_use("t", "submit_traps", {"traps": []})], "tool_use")
    c, client = api_app([traps])
    assert c.post("/api/interviews", json={"questions": []}).status_code == 422
    room = c.post("/api/interviews", json={"role": "Backend engineer", "candidate": "Sam Lee",
                                           "questions": ["Write a function that merges intervals.", "Tell me about a conflict."]}).json()
    assert room["public_id"] and len(room["questions"]) == 2 and len(room["traps"]) == 2
    assert all(t["term"].split()[0].lower() == t["word"] and t["term"].lower() in t["question"].lower() for t in room["traps"])
    view = c.get(f"/api/i/{room['public_id']}").json()
    assert view["questions"][0]["hidden"] == room["questions"][0]["canary"]["line"] and "traps" not in view
    assert c.get("/api/i/nope-not-real").status_code == 404
    word = room["questions"][0]["canary"]["word"]
    trap = room["traps"][0]
    transcript = f"Harsha: {trap['question']}\nSam: Never heard of it, what is it?\nHarsha: Fine. Merge intervals now.\n"
    client.script += [judge([{"id": 1, "reaction": "unfamiliar", "quote": "Never heard of it"}])]
    body = c.post(f"/api/interviews/{room['id']}/analyze", json={"transcript": transcript, "code": f"def {word}_step(x): pass", "compare": False}).text
    final = json.loads(body.split("event: final\ndata: ")[1].split("\n\n")[0])
    assert final["summary"]["status"] == "strong" and final["receipt"]["type"] == "interview"
    assert "not a finding that anyone cheated" in final["receipt"]["scope"]
    ok = c.post("/api/proof/verify", json={"receipt": final["receipt"], "signature": final["signature"]}).json()
    assert ok["valid"] is True
    listed = c.get("/api/interviews").json()["interviews"]
    assert listed[0]["status"] == "strong" and listed[0]["candidate"] == "Sam Lee"
    assert c.get(f"/api/interviews/{room['id']}").json()["report"]["summary"]["status"] == "strong"
    assert c.put(f"/api/interviews/{room['id']}/contrast", json={"delta": 9}).json()["delta"] == 9
    assert c.get(f"/api/i/{room['public_id']}").json()["delta"] == 9
    assert c.delete(f"/api/interviews/{room['id']}").json() == {"ok": True}
    assert c.get(f"/api/i/{room['public_id']}").status_code == 404


def test_interviews_need_an_account():
    c, _ = api_app([])
    c.cookies.clear()
    assert c.post("/api/interviews", json={"questions": ["q"]}).status_code == 401
    assert c.get("/api/interviews/status").json()["signed_in"] is False
