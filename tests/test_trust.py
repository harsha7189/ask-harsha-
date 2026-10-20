"""Hallucination defense: calculations recomputed in code, quotes and URLs checked, wrong answers corrected and re-checked."""
import json

import pytest

from agent import trust
from fakes import FakeClient, response, text, tool_use

TODAY = "Tuesday, October 20, 2026"


def analysis(claims):
    return response([tool_use("a1", "submit_analysis", {"claims": claims})], "tool_use")


def fixed(answer, changes=()):
    return response([text(json.dumps({"answer": answer, "changes": list(changes)}))], "end_turn")


def test_calculations_are_recomputed_in_code():
    found = {c["written"]: c["ok"] for c in trust.find_calculations(
        "23 × 47 = 1,071. So 5,000 × 1.06^10 = $8,954.24, 100/3 ≈ 33.3 and 50/200 = 25%. A 15% tip on $86 is $12.40. "
        "15% of $86 is $12.90. The 2023-24 = season.")}
    assert found == {"23 × 47 = 1,071": False, "5,000 × 1.06^10 = $8,954.24": True, "100/3 ≈ 33.3": True,
                     "50/200 = 25%": True, "15% tip on $86 is $12.40": False, "15% of $86 is $12.90": True}


@pytest.mark.parametrize("expr", ['__import__("os").system("ls")', "(1).real", "2**10000", "open('x')", "[1]*9", "10/0"])
def test_the_evaluator_only_does_arithmetic(expr):
    with pytest.raises(trust.CalcError):
        trust.safe_eval(trust.to_python(expr))
    assert trust.safe_eval("sqrt(7569)") == 87


def test_routing_keeps_ordinary_chat_light():
    assert trust.route("hi", "Hello! What can I help with?", False) == "light"
    assert trust.route("Write a haiku about rain", "Soft rain on the roof", False) == "light"
    assert trust.route("What is 17.5% of 2,480?", "434", False) == "deep"
    assert trust.route("How much interest will I pay on a $20,000 loan at 7%?", "About $4,000 over 5 years " * 30, False) == "deep"
    assert trust.route("What does Harsha do?", "He's a PM at IBM [F01].", True) == "cited"
    assert trust.route("hi", "Hello there", False, setting="deep") == "light"      # small talk never pays for a deep check
    assert trust.route("Who won the 1998 World Cup?", "France won it.", False, setting="deep") == "deep"


def test_light_check_catches_wrong_math_fixes_it_and_rechecks():
    client = FakeClient([fixed("23 × 47 = 1,081.", [{"claim_id": 1, "before": "1,071", "after": "1,081", "why": "code computed 1081"}])])
    rep = trust.run(client, "main", "helper", TODAY, "What is 23 × 47?", "23 × 47 = 1,071.", level="light")
    assert rep.corrected and rep.fix_verified and rep.answer == "23 × 47 = 1,081."
    assert rep.status == "corrected" and rep.changes[0]["after"] == "1,081"
    c = rep.claims[0]
    assert c["label"] == "contradicted" and c["fixed"] and "1,081" in c["why"]
    assert "code" in client.requests[0]["messages"][0]["content"]          # the fix was told what code computed


def test_a_fix_that_is_still_wrong_is_not_called_verified():
    client = FakeClient([fixed("23 × 47 = 1,071, as I said."), fixed("23 × 47 = 1,071 still.")])
    rep = trust.run(client, "main", "helper", TODAY, "What is 23 × 47?", "23 × 47 = 1,071.", level="light")
    assert rep.fix_attempts == 2 and not rep.fix_verified and rep.status == "contradicted"
    assert any("still" in n for n in rep.notes)


def test_light_check_makes_no_model_calls_when_everything_adds_up():
    client = FakeClient([])
    rep = trust.run(client, "main", "helper", TODAY, "q", "12 × 12 = 144, so a gross is 144.", level="light")
    assert rep.status == "verified" and not client.requests and rep.claims[0]["how"] == "calculation"


def test_citations_count_only_when_the_quote_is_really_in_the_source():
    record = {"F01": "Harsha is a Product Manager at IBM in Austin, TX, owning API-backed automation for 500+ enterprise accounts."}
    client = FakeClient([analysis([
        {"text": "Harsha is a PM at IBM", "type": "harsha", "risk": "normal", "sources": ["F01"], "support": "supports",
         "quote": "Harsha is a Product Manager at IBM", "reason": "The record says so."},
        {"text": "Harsha serves 900 accounts", "type": "harsha", "risk": "normal", "sources": ["F01"], "support": "supports",
         "quote": "serves 900 enterprise accounts", "reason": "Looks right."},        # invented quote
        {"text": "Harsha lives in Seattle", "type": "harsha", "risk": "normal", "sources": ["F01"], "support": "not_supported",
         "quote": "", "reason": "F01 says Austin."},
        {"text": "Austin is a great city", "type": "opinion", "risk": "normal", "support": "no_source"},
    ])])
    rep = trust.run(client, "main", "helper", TODAY, "Who is Harsha?", "Harsha is a PM at IBM [F01]...", level="cited", record=record)
    labels = {c["text"]: c["label"] for c in rep.claims}
    assert labels == {"Harsha is a PM at IBM": "verified", "Harsha serves 900 accounts": "unverified", "Harsha lives in Seattle": "unverified"}
    seattle = next(c for c in rep.claims if "Seattle" in c["text"])
    assert seattle["misleading_citation"] and "doesn't say this" in seattle["why"]
    assert "F01 (Harsha's record)" in client.requests[0]["messages"][0]["content"]
    assert not rep.corrected and len(client.requests) == 1       # unverified is shown honestly, not "fixed" by guesswork


def test_web_citations_are_compared_with_the_cited_words():
    passages = [{"url": "https://ex.com/a", "title": "Ex", "quote": "The bridge opened in 1937 and is 2,737 m long.", "text": "It opened in 1937."}]
    client = FakeClient([analysis([{"text": "The bridge opened in 1937", "type": "number", "risk": "normal", "sources": ["W1"],
                                    "support": "supports", "quote": "The bridge opened in 1937"}])])
    rep = trust.run(client, "main", "helper", TODAY, "When did it open?", "It opened in 1937.", level="cited", passages=passages)
    c = rep.claims[0]
    assert c["label"] == "verified" and c["evidence"][0]["url"] == "https://ex.com/a" and c["quote"] == "The bridge opened in 1937"


def skeptic(results, urls):
    blocks = [{"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "check"}},
              {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [{"type": "web_search_result", "url": u, "title": "Src"} for u in urls]},
              text(json.dumps({"results": results}))]
    return response(blocks, "end_turn")


def test_the_skeptic_needs_a_real_source_to_contradict():
    claims = [{"text": "The Berlin Wall fell in 1991", "type": "fact", "risk": "normal", "support": "no_source"},
              {"text": "Gold's symbol is Au", "type": "fact", "risk": "normal", "support": "no_source"},
              {"text": "Paris has 30 million people", "type": "number", "risk": "normal", "support": "no_source"}]
    results = [{"id": 1, "verdict": "wrong", "sources": ["https://history.example/wall"], "note": "It fell on November 9, 1989."},
               {"id": 2, "verdict": "holds", "sources": ["https://chem.example/au"], "note": "Au, from Latin aurum."},
               {"id": 3, "verdict": "wrong", "sources": ["https://made.up/never-searched"], "note": "About 2.1 million."}]
    answer = "The Berlin Wall fell in 1991. Gold's symbol is Au. Paris has 30 million people."
    client = FakeClient([analysis(claims), skeptic(results, ["https://history.example/wall", "https://chem.example/au"]),
                         fixed("The Berlin Wall fell in 1989. Gold's symbol is Au. Paris has about 2.1 million people (city proper), though I couldn't verify that here.",
                               [{"claim_id": 1, "before": "1991", "after": "1989", "why": "history.example"}])])
    rep = trust.run(client, "main", "helper", TODAY, "Facts?", answer, level="deep", web_tool={"type": "web_search_20250305", "name": "web_search"})
    by = {c["id"]: c for c in rep.claims}
    assert by[1]["label"] == "contradicted" and by[1]["fixed"]
    assert by[2]["label"] == "verified" and by[2]["how"] == "web"
    assert by[3]["label"] == "unverified" and by[3]["suspect"]         # disagreement without a source isn't a contradiction
    assert by[3]["challenge"]["dropped"] == 1                           # the URL it never searched was thrown out
    assert rep.corrected and rep.fix_verified and "1989" in rep.answer and rep.challenges == 3
    assert "Trying to disprove" not in json.dumps(client.requests[1]["system"])   # the skeptic got its own prompt
    assert "prove each claim WRONG" in client.requests[1]["system"]


def test_uncertain_claims_are_listed_in_plain_words():
    client = FakeClient([analysis([{"text": "The study had 4,000 participants", "type": "number", "risk": "high", "support": "no_source"}]),
                         skeptic([{"id": 1, "verdict": "unclear", "sources": [], "note": "No record of this study found."}], [])])
    rep = trust.run(client, "main", "helper", TODAY, "What did the study find?", "The study had 4,000 participants.", level="deep",
                    web_tool={"type": "web_search_20250305", "name": "web_search"})
    assert rep.status == "mixed" and rep.uncertain == ["The study had 4,000 participants"] and not rep.corrected


def test_a_failed_check_never_breaks_the_answer():
    rep = trust.run(FakeClient([]), "main", "helper", TODAY, "q", "Some answer about 1,234 things.", level="deep")
    assert rep.answer == "Some answer about 1,234 things." and rep.error and not rep.corrected


# ---------------------------------------------------------------- in the chat
def chat_app(script, web=False):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from agent import inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from fakes import RoutingClient
    from server.app import Settings, create_app
    client = RoutingClient(script)
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame", web_search=web)
    app = create_app(client=client, settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=False)
    return TestClient(app), client


def test_a_wrong_answer_is_corrected_before_it_is_final_and_signed():
    c, client = chat_app([response([text("23 × 47 = 1,071.")], "end_turn"),
                          analysis([{"text": "23 × 47 = 1,071", "type": "calculation", "risk": "normal", "support": "no_source",
                                     "expression": "23*47", "claimed": "1,071"}]),
                          fixed("23 × 47 = 1,081.", [{"claim_id": 1, "before": "1,071", "after": "1,081", "why": "Code computed 23 × 47 = 1,081."}])])
    body = c.post("/api/chat/stream", json={"question": "What is 23 × 47?"}).text
    assert "event: checking" in body and body.index("event: checking") < body.index("event: final")
    assert "Fixing 1 problem" in body and "Re-checking the fix" in body
    final = json.loads(body.split("event: final\ndata: ")[1].split("\n\n")[0])
    t = final["trust"]
    assert final["answer"] == "23 × 47 = 1,081." and t["status"] == "corrected" and t["original"] == "23 × 47 = 1,071."
    assert t["counts"]["corrected"] == 1 and t["changes"][0]["why"].startswith("Code computed")
    assert final["receipt"]["answer"] == "23 × 47 = 1,081." and final["receipt"]["trust"]["status"] == "corrected"
    assert "Accuracy" in client.requests[0]["system"] and "never as a made-up percentage" in client.requests[0]["system"]
    stats = c.get("/api/owner/trust-stats", headers={"X-Owner-Passcode": "open-sesame"}).json()
    assert stats["answers"] == 1 and stats["corrected"] == 1 and stats["levels"]["deep"] == 1


def test_ordinary_chat_gets_the_free_check_only():
    c, client = chat_app([response([text("Hi! What can I help you with today?")], "end_turn")])
    d = c.post("/api/chat", json={"question": "hello"}).json()
    assert d["trust"]["level"] == "light" and d["trust"]["status"] == "nothing" and len(client.requests) == 1


def test_a_deep_check_can_be_run_on_demand():
    c, client = chat_app([analysis([{"text": "Mount Everest is 7,849 m tall", "type": "number", "risk": "normal", "support": "no_source"}]),
                          skeptic([{"id": 1, "verdict": "wrong", "sources": ["https://geo.example/everest"], "note": "It's 8,849 m (2020 survey)."}],
                                  ["https://geo.example/everest"]),
                          fixed("Mount Everest is 8,849 m tall.")], web=True)
    body = c.post("/api/trust/check", json={"question": "How tall is Everest?", "answer": "Mount Everest is 7,849 m tall."}).text
    final = json.loads(body.split("event: final\ndata: ")[1].split("\n\n")[0])
    assert final["on_demand"] and final["status"] == "corrected" and final["answer"] == "Mount Everest is 8,849 m tall."
    assert final["claims"][0]["evidence"][0]["url"] == "https://geo.example/everest"
