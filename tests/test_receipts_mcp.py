import base64
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import inbox as inboxmod, receipts as rcpt  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient, response, text, tool_use  # noqa: E402
from server.app import Settings, create_app  # noqa: E402

OWNER = {"x-owner-passcode": "open-sesame"}
CITED = [response([tool_use("t", "search_facts", {"query": "Cybage sprints"})], "tool_use", 2000, 50),
         response([text("He delivered 26 consecutive on-time sprints at Cybage [F41].")], "end_turn", 2200, 60)]


def make(script=()):
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    fake = FakeClient(list(script))
    store = rcpt.ReceiptStore(inboxmod.make_key("k"))
    app = create_app(client=fake, settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(), receipts=store)
    return TestClient(app), fake, store


def test_every_answer_is_signed_and_the_signature_covers_the_content():
    c, _, _ = make(CITED)
    d = c.post("/api/chat", json={"question": "Cybage delivery?"}).json()
    r, sig = d["receipt"], d["signature"]
    assert r["answer"] == d["answer"] and r["question"] == "Cybage delivery?" and r["citations"][0]["id"] == "F41"
    key = c.get("/.well-known/ask-harsha-key.json").json()
    assert key["alg"] == "Ed25519" and rcpt.verify_with_public_key(key["public_key"], r, sig)
    assert not rcpt.verify_with_public_key(key["public_key"], {**r, "answer": r["answer"].replace("26", "62")}, sig)


def test_share_then_anyone_can_verify():
    c, _, store = make(CITED)
    d = c.post("/api/chat", json={"question": "Cybage delivery?"}).json()
    s = c.post("/api/receipts", json={"receipt": d["receipt"], "signature": d["signature"]}).json()
    assert s["url"].endswith("/v/" + s["id"])
    v = c.get(f"/api/receipts/{s['id']}").json()
    assert v["valid"] is True and v["receipt"]["answer"] == d["answer"]
    assert "26 consecutive" not in json.dumps(store._files)          # stored encrypted
    assert c.get(f"/v/{s['id']}").status_code == 200 and "Verified" in c.get(f"/v/{s['id']}").text
    assert c.get(f"/api/receipts/{s['id']}/qr.svg").headers["content-type"].startswith("image/svg")


def test_edited_answers_cannot_be_shared_as_verified():
    c, _, _ = make(CITED)
    d = c.post("/api/chat", json={"question": "Cybage delivery?"}).json()
    forged = {**d["receipt"], "answer": "He was fired from Cybage."}
    r = c.post("/api/receipts", json={"receipt": forged, "signature": d["signature"]})
    assert r.status_code == 400 and "couldn't be verified" in r.json()["message"]


def test_unknown_links_are_not_verified():
    c, _, _ = make()
    assert c.get("/api/receipts/nothinghere1").status_code == 404
    assert c.get("/api/receipts/bad!!id").status_code == 404


def rpc(c, method, params=None, mid=1):
    body = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params is not None:
        body["params"] = params
    return c.post("/mcp", json=body, headers={"accept": "application/json, text/event-stream"}).json()


def test_mcp_handshake_and_tool_list():
    c, _, _ = make()
    init = rpc(c, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
    assert init["result"]["protocolVersion"] == "2025-06-18" and init["result"]["serverInfo"]["name"] == "ask-harsha"
    assert rpc(c, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == "2025-06-18"
    n = c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert n.status_code == 202
    names = [t["name"] for t in rpc(c, "tools/list")["result"]["tools"]]
    assert names == ["ask_harsha_agent", "search_harsha_facts", "get_harsha_topic", "verify_answer"]
    assert rpc(c, "ping")["result"] == {}
    assert rpc(c, "nope")["error"]["code"] == -32601
    assert c.post("/mcp", content=b"{not json").json()["error"]["code"] == -32700
    assert c.get("/mcp").status_code == 405


def test_mcp_free_tools_need_no_model():
    c, fake, _ = make()
    res = rpc(c, "tools/call", {"name": "search_harsha_facts", "arguments": {"query": "Cybage sprints"}})["result"]
    data = json.loads(res["content"][0]["text"])
    assert res["isError"] is False and any(r["id"] == "F41" for r in data["results"])
    bad = rpc(c, "tools/call", {"name": "get_harsha_topic", "arguments": {"topic": "nope"}})["result"]
    assert bad["isError"] is True and "Unknown topic" in bad["content"][0]["text"]
    assert fake.requests == []


def test_mcp_ask_returns_answer_sources_and_a_verify_link():
    c, _, _ = make(CITED)
    res = rpc(c, "tools/call", {"name": "ask_harsha_agent", "arguments": {"question": "Cybage delivery?"}})["result"]
    out = res["content"][0]["text"]
    assert "26 consecutive" in out and "[F41]" in out and "/v/" in out
    link = out.rsplit("Verify this answer (signed by Ask Harsha): ", 1)[1].strip()
    v = rpc(c, "tools/call", {"name": "verify_answer", "arguments": {"link": link}})["result"]["content"][0]["text"]
    assert v.startswith("Verified")
    assert rpc(c, "tools/call", {"name": "verify_answer", "arguments": {"link": "https://x/v/missing12345"}})["result"]["content"][0]["text"].startswith("Not verified")


def test_mcp_agent_has_no_popup_tool_and_knows_it_is_a_tool():
    c, fake, _ = make(CITED)
    rpc(c, "tools/call", {"name": "ask_harsha_agent", "arguments": {"question": "What has Harsha shipped with AI?"}})
    req = fake.requests[0]
    assert "suggest_asking_harsha" not in [t["name"] for t in req["tools"]]
    assert "called as a tool by another AI assistant" in req["system"]
