"""Check any AI: the page, the deep check on a pasted answer, and owner stats by source."""
import json

from fakes import response
from test_trust import analysis, chat_app, fixed, skeptic


def test_checking_another_ais_answer():
    c, _ = chat_app([analysis([{"text": "The Eiffel Tower is 450 m tall", "type": "number", "risk": "normal", "support": "no_source"}]),
                     skeptic([{"id": 1, "verdict": "wrong", "sources": ["https://www.toureiffel.paris/en"], "note": "It's 330 m."}], ["https://www.toureiffel.paris/en"]),
                     fixed("The Eiffel Tower is 330 m tall.", [{"claim_id": 1, "before": "450 m", "after": "330 m", "why": "Official site: 330 m."}])], web=True)
    page = c.get("/check")
    assert page.status_code == 200 and "Check Any AI" in page.text
    body = c.post("/api/trust/check", json={"answer": "The Eiffel Tower is 450 m tall.", "question": "How tall is the Eiffel Tower?", "source": "ChatGPT"}).text
    final = json.loads(body.split("event: final\ndata: ")[1].split("\n\n")[0])
    assert final["status"] == "corrected" and final["answer"] == "The Eiffel Tower is 330 m tall."
    stats = c.get("/api/owner/history", headers={"X-Owner-Passcode": "open-sesame"}).json()["checks"]
    assert stats == {"chatgpt": {"checked": 1, "with_errors": 1, "claims": 1}}
    manifest = json.loads(open("server/static/manifest.json").read())
    assert manifest["share_target"]["action"] == "/check"
