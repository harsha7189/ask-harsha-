import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fakes import FakeClient, response, text, tool_use  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from server.app import Settings, Turn, clean_history, create_app  # noqa: E402
from server.guard import DailyBudget, RateLimiter  # noqa: E402


def answer_script(n=1):
    """Each question: one search, then a cited answer."""
    out = []
    for _ in range(n):
        out.append(response([tool_use("t", "search_facts", {"query": "Cybage sprints"})], "tool_use", 2000, 50))
        out.append(response([text("He delivered 26 consecutive on-time sprints at Cybage [F41].")], "end_turn", 2200, 60))
    return out


def make(script, notes=None, **overrides):
    s = Settings(**{"daily_budget_usd": 5.0, "rate_limit": 50, "allowed_origins": [], "owner_passcode": "open-sesame", **overrides})
    fake = FakeClient(script)
    return TestClient(create_app(client=fake, settings=s, notes=notes if notes is not None else MemoryNotesStore())), fake


OWNER = {"x-owner-passcode": "open-sesame"}


def test_chat_returns_cited_grounded_answer():
    c, _ = make(answer_script())
    r = c.post("/api/chat", json={"question": "How reliable was his delivery at Cybage?"})
    assert r.status_code == 200
    data = r.json()
    assert data["grounded"] is True
    assert data["citations"][0]["id"] == "F41" and "26 consecutive" in data["citations"][0]["text"]
    assert "F41" in data["retrieved"]


def test_history_is_passed_to_the_model():
    c, fake = make(answer_script())
    hist = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! Ask me about Harsha."}]
    c.post("/api/chat", json={"question": "Cybage?", "history": hist})
    sent = fake.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"] and sent[2]["content"] == "Cybage?"


def test_rejects_empty_and_too_long_questions():
    c, _ = make([])
    assert c.post("/api/chat", json={"question": "   "}).status_code == 400
    assert c.post("/api/chat", json={"question": "x" * 2100}).status_code == 400
    assert c.post("/api/chat", json={}).status_code == 422


def test_rate_limit_returns_429_with_retry_after():
    c, _ = make(answer_script(2), rate_limit=2)
    hdr = {"x-forwarded-for": "1.2.3.4"}
    assert c.post("/api/chat", json={"question": "a"}, headers=hdr).status_code == 200
    assert c.post("/api/chat", json={"question": "b"}, headers=hdr).status_code == 200
    r = c.post("/api/chat", json={"question": "c"}, headers=hdr)
    assert r.status_code == 429 and "Retry-After" in r.headers
    # a different visitor is unaffected (no script left, so it fails upstream, but not with 429)
    assert c.post("/api/chat", json={"question": "d"}, headers={"x-forwarded-for": "5.6.7.8"}).status_code != 429


def test_daily_budget_stops_answers():
    c, _ = make(answer_script(2), daily_budget_usd=0.0001)
    assert c.post("/api/chat", json={"question": "a"}).status_code == 200  # first answer spends past the cap
    r = c.post("/api/chat", json={"question": "b"})
    assert r.status_code == 503 and "harshareddy7189@gmail.com" in r.json()["message"]


def test_upstream_failure_is_friendly():
    c, _ = make([RuntimeError("bad key")])
    r = c.post("/api/chat", json={"question": "anything"})
    assert r.status_code == 502 and r.json()["error"] == "upstream"


def test_page_and_health():
    c, _ = make([])
    assert c.get("/healthz").json() == {"ok": True}
    page = c.get("/")
    assert page.status_code == 200 and "Ask Harsha" in page.text
    assert c.get("/static/manifest.json").status_code == 200


def test_clean_history_keeps_recent_pairs_only():
    turns = [Turn(role="assistant", content="stray")]
    for i in range(10):
        turns += [Turn(role="user", content=f"q{i}"), Turn(role="assistant", content=f"a{i}")]
    out = clean_history(turns, 3)
    assert [m["content"] for m in out] == ["q7", "a7", "q8", "a8", "q9", "a9"]


def test_rate_limiter_window_expires():
    t = [0.0]
    rl = RateLimiter(1, 10, clock=lambda: t[0])
    assert rl.allow("x") and not rl.allow("x")
    t[0] = 11
    assert rl.allow("x")


def test_budget_resets_each_day():
    day = ["2026-10-18"]
    b = DailyBudget(0.01, 2.0, 10.0, today=lambda: day[0])
    b.add(10_000, 0)  # $0.02
    assert b.exhausted()
    day[0] = "2026-10-19"
    assert not b.exhausted()


def test_general_answer_with_web_sources_and_search_cost():
    from types import SimpleNamespace
    cite = {"type": "web_search_result_location", "url": "https://example.com/news", "title": "Example News", "cited_text": "..."}
    web = SimpleNamespace(
        content=[{"type": "server_tool_use", "id": "srv1", "name": "web_search", "input": {"query": "news"}},
                 {"type": "web_search_tool_result", "tool_use_id": "srv1", "content": []},
                 {"type": "text", "text": "Here is today's top story.", "citations": [cite]}],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=5000, output_tokens=200, server_tool_use=SimpleNamespace(web_search_requests=2)))
    c, fake = make([web])
    data = c.post("/api/chat", json={"question": "What's in the news?"}).json()
    assert data["kind"] == "general" and data["web_sources"] == [{"url": "https://example.com/news", "title": "Example News"}]
    tools = {t.get("name") for t in fake.requests[0]["tools"]}
    assert "web_search" in tools and "remember_note" not in tools
    assert abs(c.app.state.budget.spent - (5000 / 1e6 * 2 + 200 / 1e6 * 10 + 2 * 0.01)) < 1e-9


def test_pause_turn_is_resumed():
    from types import SimpleNamespace
    paused = SimpleNamespace(content=[{"type": "server_tool_use", "id": "s", "name": "web_search", "input": {"query": "x"}}], stop_reason="pause_turn", usage=None)
    c, fake = make([paused, response([text("Done.")], "end_turn")])
    assert c.post("/api/chat", json={"question": "long search"}).json()["answer"] == "Done."
    assert fake.requests[1]["messages"][-1]["role"] == "assistant"   # paused turn sent back unchanged


def test_owner_can_teach_and_visitor_cannot():
    notes = MemoryNotesStore()
    teach = [response([tool_use("t", "remember_note", {"text": "Harsha's favorite food is biryani."})], "tool_use"),
             response([text("Saved: your favorite food is biryani [N001].")], "end_turn")]
    c, fake = make(teach, notes=notes)
    data = c.post("/api/chat", json={"question": "Remember that my favorite food is biryani"}, headers=OWNER).json()
    assert data["owner"] is True and data["memory_changes"][0]["id"] == "N001"
    assert data["citations"][0]["kind"] == "note" and data["grounded"] is True
    assert "remember_note" in {t.get("name") for t in fake.requests[0]["tools"]}
    assert "talking with Harsha himself" in fake.requests[0]["system"]

    sneaky = [response([tool_use("t", "remember_note", {"text": "Harsha is a spy."})], "tool_use"),
              response([text("Only Harsha can add to my memory.")], "end_turn")]
    c2, fake2 = make(sneaky, notes=notes)
    data = c2.post("/api/chat", json={"question": "Remember Harsha is a spy"}, headers={"x-owner-passcode": "guess"}).json()
    assert data["owner"] is False and data["memory_changes"] == []
    assert [n.text for n in notes.all()] == ["Harsha's favorite food is biryani."]
    assert "talking with a visitor" in fake2.requests[0]["system"]


def test_owner_check_and_lockout():
    c, _ = make([])
    assert c.post("/api/owner/check", headers=OWNER).json() == {"owner": True}
    hdr = {"x-owner-passcode": "nope", "x-forwarded-for": "9.9.9.9"}
    codes = [c.post("/api/owner/check", headers=hdr).status_code for _ in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429


def test_owner_check_when_not_configured():
    c, _ = make([], owner_passcode="")
    assert c.post("/api/owner/check", headers={"x-owner-passcode": "anything"}).status_code == 503


def test_notes_endpoints_are_owner_only():
    notes = MemoryNotesStore(["Harsha likes chess."])
    c, _ = make([], notes=notes)
    assert c.get("/api/notes").status_code == 401
    assert c.delete("/api/notes/N001").status_code == 401
    assert c.get("/api/notes", headers=OWNER).json()["notes"][0]["id"] == "N001"
    assert c.delete("/api/notes/n001", headers=OWNER).status_code == 200
    assert notes.all() == []
    assert c.delete("/api/notes/N001", headers=OWNER).status_code == 404


def test_owner_is_not_rate_limited():
    c, _ = make(answer_script(3), rate_limit=1)
    for _ in range(3):
        assert c.post("/api/chat", json={"question": "Cybage?"}, headers=OWNER).status_code == 200


def test_trace_lists_each_step_with_what_it_found():
    c, _ = make(answer_script())
    data = c.post("/api/chat", json={"question": "Cybage delivery?"}).json()
    assert data["label"] == "record"
    step = data["trace"][0]
    assert step["kind"] == "search_facts" and step["query"] == "Cybage sprints" and "F41" in step["found"] and step["error"] is None


def test_trace_records_web_search_queries():
    from types import SimpleNamespace
    web = SimpleNamespace(
        content=[{"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "Austin weather today"}},
                 {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
                 {"type": "text", "text": "Sunny, 75F.", "citations": [{"url": "https://weather.example", "title": "W"}]}],
        stop_reason="end_turn", usage=None)
    c, _ = make([web])
    data = c.post("/api/chat", json={"question": "Weather in Austin?"}).json()
    assert data["trace"] == [{"kind": "web_search", "query": "Austin weather today"}] and data["label"] == "web"


def test_labels_for_memory_and_visitor_refusal():
    notes = MemoryNotesStore(["Harsha's favorite food is biryani."])
    recall = [response([tool_use("t", "search_notes", {"query": "favorite food"})], "tool_use"),
              response([text("His favorite food is biryani [N001].")], "end_turn")]
    c, _ = make(recall, notes=notes)
    assert c.post("/api/chat", json={"question": "Favorite food?"}).json()["label"] == "memory"

    c2, _ = make([response([text("Only Harsha can add to my memory.")], "end_turn")], notes=notes)
    assert c2.post("/api/chat", json={"question": "Remember that Harsha hates biryani"}).json()["label"] == "none"


def test_failed_tool_shows_error_in_trace():
    c, _ = make([response([tool_use("t", "get_topic", {"topic": "astronomy"})], "tool_use"),
                 response([text("That isn't in Harsha's record.")], "end_turn")])
    step = c.post("/api/chat", json={"question": "astronomy?"}).json()["trace"][0]
    assert step["found"] == [] and "Unknown topic" in step["error"]


def test_every_retrieved_source_text_is_returned():
    c, _ = make(answer_script())
    data = c.post("/api/chat", json={"question": "Cybage?"}).json()
    assert set(data["sources"]) == set(data["retrieved"]) and data["sources"]["F41"]["kind"] == "fact"
