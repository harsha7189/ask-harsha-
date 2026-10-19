import json
import urllib.error

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import inbox as inboxmod, receipts as rcpt  # noqa: E402
from agent.jury import Jury  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient, response, text  # noqa: E402
from server.app import Settings, create_app  # noqa: E402

COMPARE = {"agreement": "partial", "summary": "Two agree; one dates it differently.",
           "agree": ["Canberra is the capital"], "disagree": [{"topic": "Year", "positions": {"Claude": "1913", "Gemini": "1927", "Llama": "x"}}],
           "unique": [{"model": "ChatGPT", "point": "Parliament moved in 1927"}, {"model": "Nobody", "point": "dropped"}], "check": ["The year"]}


def provider(calls, openai_status=None, retired=None):
    def http(method, url, headers, body=None, timeout=60):
        calls.append((method, url, body and body.get("model")))
        if "openai.com/v1/models" in url:
            return {"data": [{"id": "gpt-6-luna"}, {"id": "gpt-6.2-sol"}, {"id": "gpt-6.1-sol"}, {"id": "gpt-realtime-6"}]}
        if "openai.com" in url:
            if openai_status:
                raise urllib.error.HTTPError(url, openai_status, "err", {}, None)
            if retired and body["model"] == retired:
                raise urllib.error.HTTPError(url, 404, "model_not_found", {}, None)
            return {"output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "Canberra, since 1913."}]}],
                    "usage": {"input_tokens": 40, "output_tokens": 12}}
        return {"candidates": [{"content": {"parts": [{"text": "Thinking…", "thought": True}, {"text": "Canberra; Parliament moved there in 1927."}]}}],
                "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 15}}
    return http


def test_both_jurors_answer_in_parallel_and_claude_compares():
    calls, events = [], []
    j = Jury("sk-o", "", "g-key", "", http=provider(calls))
    fake = FakeClient([response([text(json.dumps(COMPARE))], "end_turn", 500, 120)])
    v = j.run(fake, "claude", "Monday", "Capital of Australia?", "Canberra.", lambda k, d: events.append((k, d)))
    texts = {o.id: o.text for o in v.opinions}
    assert texts == {"chatgpt": "Canberra, since 1913.", "gemini": "Canberra; Parliament moved there in 1927."}
    assert v.comparison["agreement"] == "partial"
    assert v.comparison["disagree"][0]["positions"] == {"Claude": "1913", "Gemini": "1927"}   # unknown model names dropped
    assert [u["model"] for u in v.comparison["unique"]] == ["ChatGPT"]
    body = fake.requests[0]["messages"][0]["content"]
    assert '<answer model="Claude">' in body and '<answer model="ChatGPT">' in body and '<answer model="Gemini">' in body
    assert [k for k, _ in events].count("opinion") == 2


def test_retired_model_is_replaced_automatically_and_remembered():
    calls = []
    j = Jury("sk-o", "gpt-5-old", "", "", http=provider(calls, retired="gpt-5-old"))
    op = j.ask_one(j.jurors[0], "Monday", "q")
    assert op.text and op.model == "gpt-6.2-sol"
    j.ask_one(j.jurors[0], "Monday", "q")
    assert [c for c in calls if "v1/models" in c[1]].__len__() == 1 and calls[-1][2] == "gpt-6.2-sol"


def test_missing_key_and_provider_errors_are_explained():
    j = Jury("", "", "g", "", http=provider([], openai_status=401))
    assert [x.id for x in j.ready] == ["gemini"]
    assert j.ask_one(j.jurors[0], "Monday", "q").error == "Not connected"
    j2 = Jury("sk", "", "", "", http=provider([], openai_status=429))
    assert "rate limited" in j2.ask_one(j2.jurors[0], "Monday", "q").error


def make(jury, script=()):
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=FakeClient(list(script)), settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), jury=jury)
    return TestClient(app)


def test_endpoint_streams_opinions_then_comparison():
    c = make(Jury("sk", "", "g", "", http=provider([])), [response([text(json.dumps(COMPARE))], "end_turn")])
    assert c.get("/api/jury").json()["enabled"] is True
    r = c.post("/api/jury/stream", json={"question": "Capital of Australia?", "answer": "Canberra."})
    events = [b.split("\n")[0][7:] for b in r.text.split("\n\n") if b.startswith("event: ")]
    assert events.count("opinion") == 2 and events[-1] == "final"
    final = json.loads([b for b in r.text.split("\n\n") if b.startswith("event: final")][0].split("data: ", 1)[1])
    assert final["comparison"]["agreement"] == "partial" and final["cost_usd"] >= 0.02


def test_endpoint_off_without_keys_and_visitors_are_limited():
    assert make(Jury()).post("/api/jury/stream", json={"question": "q"}).status_code == 503
    c = make(Jury("sk", "", "", "", http=provider([])), [response([text(json.dumps(COMPARE))], "end_turn")] * 6)
    hdr = {"x-forwarded-for": "7.7.7.7"}
    codes = [c.post("/api/jury/stream", json={"question": "q", "answer": "a"}, headers=hdr).status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]
