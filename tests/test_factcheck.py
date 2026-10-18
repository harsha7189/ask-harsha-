import json
from types import SimpleNamespace

import pytest

from agent import factcheck
from agent.kb import KnowledgeBase
from fakes import FakeClient, response, text

KB = KnowledgeBase.load()
LOOKUP = KB.by_id
ANSWER = "Canberra is the capital of Australia. Harsha drove 35% adoption across 200+ users [F14]. You should visit in spring."


def claims_resp(claims):
    return response([text(json.dumps({"claims": claims}))], "end_turn", 300, 80)


def verify_resp(verdicts, urls=("https://en.wikipedia.org/wiki/Canberra",)):
    """A verify turn: one web search returning `urls`, then the JSON verdicts."""
    return SimpleNamespace(
        content=[{"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "capital of Australia"}},
                 {"type": "web_search_tool_result", "tool_use_id": "s1",
                  "content": [{"type": "web_search_result", "url": u, "title": "t", "encrypted_content": "x"} for u in urls]},
                 {"type": "text", "text": json.dumps({"verdicts": verdicts})}],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=4000, output_tokens=200, server_tool_use=SimpleNamespace(web_search_requests=1)))


CLAIMS = [{"id": 1, "text": "Canberra is the capital of Australia.", "kind": "world", "cites": []},
          {"id": 2, "text": "Harsha drove 35% adoption across 200+ users.", "kind": "harsha", "cites": ["F14"]},
          {"id": 3, "text": "You should visit in spring.", "kind": "none", "cites": []}]


def run(script, answer=ANSWER, web=True):
    events = []
    res = factcheck.run(FakeClient(script), "m", answer, LOOKUP, "Sunday, October 18, 2026", web,
                        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}, lambda k, d: events.append((k, d)), kb=KB)
    return res, events


def test_full_check_with_valid_sources():
    res, events = run([claims_resp(CLAIMS), verify_resp([
        {"id": 1, "verdict": "confirmed", "explanation": "Wikipedia confirms it.", "sources": ["https://en.wikipedia.org/wiki/Canberra/"]},
        {"id": 2, "verdict": "confirmed", "explanation": "Matches F14.", "sources": ["F14"]},
        {"id": 3, "verdict": "confirmed", "explanation": "x", "sources": []}])])
    v = {x["id"]: x for x in res.verdicts}
    assert v[1]["verdict"] == "confirmed" and v[1]["sources"][0]["type"] == "web"
    assert v[2]["verdict"] == "confirmed" and v[2]["sources"] == [{"type": "record", "id": "F14"}]
    assert v[3]["verdict"] == "not_checkable"           # opinions can't be "confirmed"
    assert res.counts()["confirmed"] == 2 and res.web_searches == 1
    assert [k for k, _ in events][:2] == ["status", "claims"]


def test_made_up_sources_are_dropped_and_verdict_downgraded():
    res, _ = run([claims_resp(CLAIMS[:2]), verify_resp([
        {"id": 1, "verdict": "confirmed", "sources": ["https://not-actually-searched.example/page"]},
        {"id": 2, "verdict": "confirmed", "sources": ["F99"]}])])
    for v in res.verdicts:
        assert v["verdict"] == "unverified" and v["downgraded"] and v["dropped_sources"] == 1
    assert res.downgraded == 2


def test_record_source_must_be_provided_or_looked_up():
    # F50 exists in the record but wasn't cited in the answer or looked up during the check
    res, _ = run([claims_resp(CLAIMS[1:2]), verify_resp([{"id": 2, "verdict": "confirmed", "sources": ["F50"]}])])
    assert res.verdicts[0]["verdict"] == "unverified"


def test_contradicted_needs_a_source_too():
    res, _ = run([claims_resp(CLAIMS[:1]), verify_resp([
        {"id": 1, "verdict": "contradicted", "explanation": "It's Sydney.", "sources": ["https://en.wikipedia.org/wiki/Canberra"]}])])
    assert res.verdicts[0]["verdict"] == "contradicted"


def test_cites_not_in_the_answer_are_ignored():
    claims, _, _ = factcheck.extract_claims(FakeClient([claims_resp([{"id": 1, "text": "x", "kind": "harsha", "cites": ["F14", "F41"]}])]), "m", ANSWER)
    assert claims[0]["cites"] == ["F14"]


def test_only_opinions_means_no_verify_call():
    client = FakeClient([claims_resp([{"id": 1, "text": "Spring is lovely.", "kind": "none", "cites": []}])])
    res = factcheck.run(client, "m", "Spring is lovely.", LOOKUP, "today", True, None, lambda k, d: None, kb=KB)
    assert res.verdicts[0]["verdict"] == "not_checkable" and len(client.requests) == 1


def test_unreadable_output_raises():
    with pytest.raises(factcheck.FactCheckError):
        run([response([text("Sorry, I can't do that.")], "end_turn")])


def test_claims_are_capped():
    many = [{"id": i, "text": f"claim {i}", "kind": "world", "cites": []} for i in range(1, 20)]
    claims, _, _ = factcheck.extract_claims(FakeClient([claims_resp(many)]), "m", "x")
    assert len(claims) == factcheck.MAX_CLAIMS
