import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fakes import FakeClient, response, text, tool_use  # noqa: E402
from server.app import Settings, Turn, clean_history, create_app  # noqa: E402
from server.guard import DailyBudget, RateLimiter  # noqa: E402


def answer_script(n=1):
    """Each question: one search, then a cited answer."""
    out = []
    for _ in range(n):
        out.append(response([tool_use("t", "search_facts", {"query": "Cybage sprints"})], "tool_use", 2000, 50))
        out.append(response([text("He delivered 26 consecutive on-time sprints at Cybage [F41].")], "end_turn", 2200, 60))
    return out


def make(script, **overrides):
    s = Settings(**{"daily_budget_usd": 5.0, "rate_limit": 50, "allowed_origins": [], **overrides})
    fake = FakeClient(script)
    return TestClient(create_app(client=fake, settings=s)), fake


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
    assert c.post("/api/chat", json={"question": "x" * 600}).status_code == 400
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
