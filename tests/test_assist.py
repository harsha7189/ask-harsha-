import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import assist  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient, response, text  # noqa: E402
from server.app import Settings, create_app  # noqa: E402


def guesses_resp(guesses, inp=400, out=60):
    return response([text(json.dumps({"guesses": guesses}))], "end_turn", inp, out)


def make(script, **overrides):
    s = Settings(**{"daily_budget_usd": 5.0, "rate_limit": 2, "assist_rate_limit": 50, "allowed_origins": [],
                    "owner_passcode": "open-sesame", "assist_model": "fast-model", **overrides})
    fake = FakeClient(script)
    return TestClient(create_app(client=fake, settings=s, notes=MemoryNotesStore())), fake


def test_message_lists_every_input_and_past_choices():
    msg = assist.build_message(["wa da co", "water cold"], "", "Do you want anything?",
                               [{"input": "wa da", "said": "Can I have some water, please?"}, {"input": "", "said": "Thank you."}],
                               ["I'd like to sit down."])
    assert "HEARD:\n- wa da co\n- water cold" in msg
    assert "PARTNER: Do you want anything?" in msg
    assert '- "wa da" -> "Can I have some water, please?"' in msg and '- "Thank you."' in msg
    assert "SAVED:\n- I'd like to sit down." in msg


def test_guesses_are_capped_cleaned_and_deduplicated():
    out = assist.clean_guesses({"guesses": [{"text": " Can I have  water? "}, {"text": "can i have water"}, "I'm cold.",
                                            {"text": ""}, {"text": "Help me, please.", "urgent": True}, {"text": "fourth"}]})
    assert [g["text"] for g in out] == ["Can I have water?", "I'm cold.", "Help me, please."]
    assert out[2]["urgent"] is True and out[0]["urgent"] is False


def test_interpret_calls_the_model_once_and_reads_json():
    fake = FakeClient([guesses_resp([{"text": "Can I have some cold water?"}])])
    res = assist.interpret(fake, "m", typed="water cold")
    assert res.guesses[0]["text"] == "Can I have some cold water?" and res.input_tokens == 400
    assert fake.requests[0]["model"] == "m" and "TYPED: water cold" in fake.requests[0]["messages"][0]["content"]


def test_interpret_with_no_input_skips_the_model():
    fake = FakeClient([])
    assert assist.interpret(fake, "m", heard=["  "], past=[{"input": "x", "said": "y"}]).guesses == []
    assert fake.requests == []


def test_unreadable_or_refused_output_gives_no_guesses():
    assert assist.interpret(FakeClient([response([text("Sorry")], "end_turn")]), "m", typed="x").guesses == []
    assert assist.interpret(FakeClient([response([], "refusal")]), "m", typed="x").guesses == []


def test_endpoint_returns_guesses_with_the_fast_model():
    c, fake = make([guesses_resp([{"text": "I need help.", "urgent": True}])])
    r = c.post("/api/assist/interpret", json={"heard": ["I nee hep"]})
    assert r.status_code == 200 and r.json()["guesses"] == [{"text": "I need help.", "urgent": True}]
    assert fake.requests[0]["model"] == "fast-model"


def test_endpoint_rejects_empty_input():
    c, _ = make([])
    assert c.post("/api/assist/interpret", json={"heard": [" "], "typed": ""}).status_code == 400
    assert c.post("/api/assist/interpret", json={"typed": "x" * 400}).status_code == 422


def test_assist_has_its_own_rate_limit_separate_from_chat():
    c, _ = make([guesses_resp([{"text": "Yes."}])] * 3, assist_rate_limit=2)
    hdr = {"x-forwarded-for": "9.9.9.9"}
    assert c.post("/api/assist/interpret", json={"typed": "a"}, headers=hdr).status_code == 200
    assert c.post("/api/assist/interpret", json={"typed": "b"}, headers=hdr).status_code == 200
    assert c.post("/api/assist/interpret", json={"typed": "c"}, headers=hdr).status_code == 429
    assert c.post("/api/assist/interpret", json={"typed": "c"}, headers={**hdr, "x-owner-passcode": "open-sesame"}).status_code == 200


def test_endpoint_respects_the_daily_budget():
    c, _ = make([], daily_budget_usd=0.0)
    r = c.post("/api/assist/interpret", json={"typed": "water"})
    assert r.status_code == 503 and "quick phrases" in r.json()["message"]


def test_nothing_said_is_logged(caplog):
    c, _ = make([guesses_resp([{"text": "My knee hurts."}])])
    with caplog.at_level("INFO", logger="ask_harsha"):
        c.post("/api/assist/interpret", json={"heard": ["my nee hurs"], "partner": "How are you feeling?"})
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert '"event": "assist"' in logged
    assert "knee" not in logged and "nee" not in logged and "feeling" not in logged


def test_upstream_failure_is_a_friendly_502():
    c, _ = make([RuntimeError("boom")])
    r = c.post("/api/assist/interpret", json={"typed": "water"})
    assert r.status_code == 502 and "Quick phrases" in r.json()["message"]


def test_assist_page_is_served():
    c, _ = make([])
    r = c.get("/assist")
    assert r.status_code == 200 and "Hear them" in r.text
