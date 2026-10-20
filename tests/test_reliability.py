"""Reliability: the backup model, durable state across restarts, performance history and the health check."""
import json
import re

import pytest

from agent import accounts as acct, telemetry as tele
from agent.loop import Agent, retryable
from fakes import FakeClient, response, text, tool_use


class Busy(Exception):
    status_code = 529


class BadRequest(Exception):
    status_code = 400


class Flaky(FakeClient):
    """Raises the scripted exceptions first, then answers."""

    def __init__(self, failures, script):
        super().__init__(script)
        self.failures = list(failures)
        self.models = []

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        if self.failures:
            self.requests.append(kwargs)
            raise self.failures.pop(0)
        return super().create(**kwargs)


def test_busy_model_is_retried_then_the_backup_answers(monkeypatch):
    monkeypatch.setattr("agent.loop.RETRY_PAUSE", 0)
    client = Flaky([Busy(), Busy()], [response([text("Paris.")], "end_turn")])
    agent = Agent(client, model="claude-sonnet-5-5", fallback_model="claude-haiku-5-5")
    agent.messages = [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "s"}, text("Hello!")]}]
    reply = agent.ask("Capital of France?")
    assert reply.text == "Paris." and reply.fallback == "claude-haiku-5-5"
    assert client.models == ["claude-sonnet-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    last = client.requests[-1]
    assert all(b.get("type") != "thinking" for m in last["messages"] if isinstance(m["content"], list) for b in m["content"])
    assert "thinking" not in last and last["output_config"] == {"effort": "low"}   # the backup's own settings, not Sonnet's


def test_one_retry_is_enough_when_the_blip_passes(monkeypatch):
    monkeypatch.setattr("agent.loop.RETRY_PAUSE", 0)
    client = Flaky([Busy()], [response([text("ok")], "end_turn")])
    reply = Agent(client, model="claude-sonnet-5-5", fallback_model="claude-haiku-5-5").ask("q")
    assert reply.text == "ok" and reply.fallback == "" and client.models == ["claude-sonnet-5-5", "claude-sonnet-5-5"]


def test_bad_requests_are_not_retried():
    client = Flaky([BadRequest()], [])
    with pytest.raises(BadRequest):
        Agent(client, model="claude-sonnet-5-5", fallback_model="claude-haiku-5-5").ask("q")
    assert client.models == ["claude-sonnet-5-5"]
    assert retryable(Busy()) and not retryable(BadRequest()) and retryable(type("APIConnectionError", (Exception,), {})())


def test_a_sign_in_started_before_a_restart_still_works(tmp_path):
    url = f"sqlite:///{tmp_path}/a.db"
    before = acct.Accounts(acct.Database(url))
    cid, _ = before.authentication_options("localhost")
    after = acct.Accounts(acct.Database(url))                     # a new server process, same database
    challenge, data = after._take(cid)
    assert data == {"rp_id": "localhost"} and len(challenge) >= 16
    with pytest.raises(ValueError):
        after._take(cid)                                          # used once


@pytest.mark.parametrize("durable", [False, True])
def test_telemetry_keeps_history_and_state(durable):
    t = tele.Telemetry(acct.Database("sqlite:///:memory:") if durable else None)
    for s in (1.0, 2.0, 3.0, 9.0):
        t.add("answer", {"seconds": s, "first_text_s": s / 2, "cost_usd": 0.01, "label": "general", "cache_read": 10 if s < 3 else 0})
    t.put("health:last", {"ok": True})
    assert t.get("health:last") == {"ok": True} and t.get("missing", 5) == 5
    assert t.claim("health:last_date", "2026-10-21") is True and t.claim("health:last_date", "2026-10-21") is False
    p = tele.performance(t.recent("answer"))
    assert p["answers"] == 4 and p["p50_seconds"] == 3.0 and p["p90_seconds"] == 9.0 and p["cache_hit_share"] == 0.5
    board = tele.Scoreboard(t, "proof", lambda runs: {"tasks": len(runs)})
    board.add({"status": "verified"})
    assert board.summary()["tasks"] == 1 and board.summary()["durable"] is durable


class HealthClient(FakeClient):
    """Answers each health-check request the way a working model would."""

    def __init__(self):
        super().__init__([])
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        sysp = kw.get("system")
        sysp = "".join(b["text"] for b in sysp) if isinstance(sysp, list) else (sysp or "")
        last = kw["messages"][-1]["content"]
        flat = json.dumps(last, ensure_ascii=False)
        if "You correct an AI" in sysp:
            return response([text(json.dumps({"answer": "23 × 47 = 1,081.", "changes": []}))], "end_turn")
        if "12 × 12" in flat:
            return response([text("144")], "end_turn")
        if "job title" in flat:
            return response([tool_use("t1", "search_facts", {"query": "current role title"})], "tool_use")
        if "tool_result" in flat:
            fid = re.search(r"\[(F\d{2,3})\]|\b(F\d{2,3})\b", flat)
            return response([text(f"He's a technical product manager [{fid.group(1) or fid.group(2)}].")], "end_turn")
        if "6 * 7" in flat:
            return response([{"type": "server_tool_use", "id": "b1", "name": "bash_code_execution", "input": {"command": "python -c 'print(6 * 7)'"}},
                             {"type": "bash_code_execution_tool_result", "tool_use_id": "b1",
                              "content": {"type": "bash_code_execution_result", "stdout": "42\n", "stderr": "", "return_code": 0}},
                             text("It printed 42.")], "end_turn")
        return response([text("ok")], "end_turn")


def test_health_check_runs_every_part_and_reports_publicly():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from agent import inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from server.app import Settings, create_app
    a = acct.Accounts(acct.Database("sqlite:///:memory:"))
    app = create_app(client=HealthClient(), settings=Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame"),
                     notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(), receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=a)
    c = TestClient(app)
    assert c.get("/api/health/status").json()["status"] == "unknown"
    assert c.post("/api/owner/health").status_code in (401, 403)
    d = c.post("/api/owner/health", headers={"X-Owner-Passcode": "open-sesame"}).json()
    by = {x["name"]: x for x in d["checks"]}
    assert set(by) == {"database", "signing", "chat", "record", "trust check", "sandbox"}
    assert by["database"]["ok"] and by["chat"]["ok"] and by["record"]["ok"] and by["trust check"]["ok"] and by["sandbox"]["ok"], d
    assert by["signing"]["ok"] in (True, None)
    st = c.get("/api/health/status").json()
    assert st["status"] == "ok" and st["failing"] == [] and "detail" not in st["checks"][0]
    h = c.get("/api/owner/history", headers={"X-Owner-Passcode": "open-sesame"}).json()
    assert h["durable"] is True and h["health"][-1]["ok"] is True
    assert h["performance"]["answers"] == 0                       # health-check answers don't count as real traffic
