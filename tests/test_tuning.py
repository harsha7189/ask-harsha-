"""Tune: faster settings only when they're measured to cost no more and answer just as well, proposed as a pull request."""
import json
import time
import urllib.error

import pytest

from agent import speed, tuning
from fakes import FakeClient, response, text, tool_use


def test_tuned_settings_reach_the_model_but_env_vars_win(monkeypatch):
    monkeypatch.delenv("CHAT_EFFORT", raising=False)
    monkeypatch.delenv("CHAT_THINKING", raising=False)
    p = speed.params("claude-sonnet-5-5", "chat", {"effort": "low", "thinking": "adaptive"})
    assert p == {"output_config": {"effort": "low"}, "thinking": {"type": "adaptive"}}
    assert speed.params("claude-sonnet-5-5", "chat")["output_config"] == {"effort": "medium"}
    monkeypatch.setenv("CHAT_EFFORT", "high")
    assert speed.params("claude-sonnet-5-5", "chat", {"effort": "low"})["output_config"] == {"effort": "high"}
    assert tuning.env_locked() == ["chat_effort"] and "effort" not in tuning.speed_override(tuning.defaults())


def test_settings_are_cleaned_and_simple_questions_detected(tmp_path):
    assert tuning.clean({"chat_effort": "turbo", "web_max_searches": 2, "unknown": 1}) == {**tuning.defaults(), "web_max_searches": 2}
    f = tmp_path / "tuned.json"
    f.write_text(tuning.tuned_file({"chat_effort": "low"}, {"x": 1}))
    assert tuning.load_tuned(f)["chat_effort"] == "low" and tuning.load_tuned(tmp_path / "missing.json") == tuning.defaults()
    assert tuning.is_simple("What's the capital of Peru?")
    for q in ("What does Harsha do?", "How much interest on a $20k loan?", "What's the latest iPhone?", "Fix this code please"):
        assert not tuning.is_simple(q), q


def run(cfg, p50, cost, right):
    rows = [{"id": i, "seconds": p50, "first_text_s": p50 / 2, "cost_usd": cost / 4, "correct": i in right} for i in ("a", "b", "c", "d")]
    return {"config": cfg, "rows": rows, "summary": tuning.summarize(rows)}


def test_a_faster_setting_wins_only_without_losing_quality_or_costing_more():
    base = run(tuning.defaults(), 4.0, 0.04, {"a", "b", "c"})
    worse = run({**tuning.defaults(), "simple_to_fast_model": True}, 2.0, 0.02, {"a", "b"})
    slow = run({**tuning.defaults(), "web_max_searches": 2}, 3.8, 0.04, {"a", "b", "c"})
    pricey = run({**tuning.defaults(), "chat_thinking": "adaptive"}, 3.0, 0.06, {"a", "b", "c"})
    good = run({**tuning.defaults(), "chat_effort": "low"}, 3.0, 0.03, {"a", "b", "c", "d"})
    winner, verdicts = tuning.decide(base, [worse, slow, pricey, good])
    why = {json.dumps(v["config"], sort_keys=True): v["why"] for v in verdicts}
    assert winner["config"]["chat_effort"] == "low"
    assert "Answered worse: got c wrong" in why[json.dumps(worse["config"], sort_keys=True)]
    assert "Not at least 10% faster" in why[json.dumps(slow["config"], sort_keys=True)]
    assert "Costs more" in why[json.dumps(pricey["config"], sort_keys=True)]
    twice = tuning.merge_runs(good, run(good["config"], 3.2, 0.03, {"a", "b"}))
    assert twice["summary"]["right"] == ["a", "b"]                              # right only if right both times


def test_proposals_skip_what_already_lost():
    already = [{"config": {**tuning.defaults(), "chat_effort": "low"}, "why": "lost"}]
    client = FakeClient([response([tool_use("p", "propose", {"candidates": [{"settings": {"chat_effort": "low"}, "why": "dup"},
                                                                         {"settings": {"web_max_searches": 2}, "why": "fewer searches"}]})], "tool_use")])
    cands, _, _ = tuning.propose(client, "helper", tuning.defaults(), {"answers": 0}, already)
    assert [c["config"]["web_max_searches"] for c in cands] == [2] and cands[0]["why"] == "fewer searches"
    assert "lost" in client.requests[0]["messages"][0]["content"]


class FakeGitHub:
    def __init__(self, pulls_allowed=True):
        self.calls, self.pulls_allowed = [], pulls_allowed

    def __call__(self, req, timeout=0):
        url, method = req.full_url, req.get_method()
        self.calls.append((method, url.split("/repos/o/r")[1], json.loads(req.data) if req.data else None))
        path = url.split("/repos/o/r")[1]
        if path == "":
            body = {"default_branch": "main"}
        elif path.startswith("/git/ref/heads/main"):
            body = {"object": {"sha": "abc"}}
        elif path.startswith("/contents/") and method == "GET":
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        elif path == "/pulls":
            if not self.pulls_allowed:
                raise urllib.error.HTTPError(url, 403, "no", {}, None)
            body = {"html_url": "https://github.com/o/r/pull/7"}
        else:
            body = {}

        class R:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def read(s):
                return json.dumps(body).encode()
        return R()


def test_the_winner_becomes_a_pull_request():
    gh = FakeGitHub()
    pr = tuning.open_pull_request("o/r", "tok", {"chat_effort": "low"}, {"baseline": {}}, "body", opener=gh, stamp="20261021-0900")
    assert pr == {"branch": "tune/20261021-0900", "url": "https://github.com/o/r/pull/7", "pull_request": True}
    put = next(c for c in gh.calls if c[0] == "PUT")
    import base64
    assert put[1] == "/contents/agent/tuned.json" and json.loads(base64.b64decode(put[2]["content"]))["settings"]["chat_effort"] == "low"
    no_pr = tuning.open_pull_request("o/r", "tok", {"chat_effort": "low"}, {}, "body", opener=FakeGitHub(pulls_allowed=False), stamp="x")
    assert no_pr["pull_request"] is False and no_pr["url"] == "https://github.com/o/r/compare/main...tune/x?expand=1"
    with pytest.raises(tuning.TuneError):
        tuning.open_pull_request("", "", {}, {}, "")


class TimedClient(FakeClient):
    """Answers the benchmark correctly; low effort answers faster, like the real model would."""

    def __init__(self):
        super().__init__([])

    def create(self, **kw):
        name = (kw.get("tool_choice") or {}).get("name")
        if name == "propose":
            return response([tool_use("p", "propose", {"candidates": [{"settings": {"chat_effort": "low"}, "why": "most time is the model"}]})], "tool_use")
        if name == "submit_analysis":
            return response([tool_use("a", "submit_analysis", {"claims": []})], "tool_use")
        if name == "set_checklist":
            return response([tool_use("c", "set_checklist", {"expects_file": False, "file_types": [], "checks": [], "cannot_check": []})], "tool_use")
        time.sleep(0.03 if (kw.get("output_config") or {}).get("effort") == "low" else 0.12)
        last = json.dumps(kw["messages"][-1]["content"], ensure_ascii=False)
        if "tool_result" in last:
            import re
            fid = re.search(r"\b(F\d{2,3})\b", last).group(1)   # cite a fact the search really returned
            return response([text(f"He's a technical product manager with CSPO, SAFe and Salesforce certifications [{fid}].")], "end_turn")
        answers = {"Australia": "Canberra.", "boil": "100 °C.", "15% of 240": "15% of 240 is 36.", "hours": "There are 168 hours in a week.",
                   "DNS": "DNS turns a domain name into an IP address.", "git": "Use git reset --soft HEAD~1.", "subject": "Lunch is on us!"}
        for k, v in answers.items():
            if k in last:
                return response([text(v)], "end_turn")
        if "Harsha" in last:
            return response([tool_use("t", "search_facts", {"query": "role certifications"})], "tool_use")
        return response([text("ok")], "end_turn")


def test_tune_measures_confirms_and_reports(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from agent import inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from server.app import Settings, create_app
    for k in ("CHAT_EFFORT", "CHAT_THINKING", "GITHUB_TOKEN", "NOTES_REPO", "TUNE_REPO"):
        monkeypatch.delenv(k, raising=False)
    app = create_app(client=TimedClient(), settings=Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="pw", web_search=False),
                     notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(), receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=False)
    c = TestClient(app)
    h = {"X-Owner-Passcode": "pw"}
    assert c.post("/api/owner/tune").status_code in (401, 403)
    body = c.post("/api/owner/tune", headers=h).text
    final = json.loads(body.split("event: final\ndata: ")[1].split("\n\n")[0])
    assert "Confirming" in body and final["winner"]["config"]["chat_effort"] == "low"
    assert final["baseline"]["summary"]["correct"] == 9 and final["winner"]["summary"]["correct"] == 9
    assert final["winner"]["summary"]["p50_seconds"] < final["baseline"]["summary"]["p50_seconds"] and final["receipt"]["type"] == "tune"
    assert c.post("/api/proof/verify", json={"receipt": final["receipt"], "signature": final["signature"]}).json()["valid"]
    st = c.get("/api/owner/tune", headers=h).json()
    assert st["last"]["id"] == final["id"] and st["github"] is False
    pr = c.post("/api/owner/tune/pr", headers=h)
    assert pr.status_code == 400 and "GitHub isn't connected" in pr.json()["message"]
    hist = c.get("/api/owner/history", headers=h).json()
    assert hist["tunes"][-1]["winner"] is True and hist["performance"]["answers"] == 0   # benchmark answers aren't real traffic
