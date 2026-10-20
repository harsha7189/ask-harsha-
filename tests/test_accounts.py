import pytest

pytest.importorskip("fastapi")
pytest.importorskip("webauthn")
from fastapi.testclient import TestClient  # noqa: E402

from agent import accounts as acct, inbox as inboxmod, receipts as rcpt  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient  # noqa: E402
from server.app import Settings, create_app  # noqa: E402


def store():
    return acct.Accounts(acct.Database("sqlite:///:memory:"))


def user(a, name="Sam"):
    u = acct.User(acct.b64u(b"user-" + name.encode()), name, acct.now_ms())
    a.db.run("INSERT INTO users (id, name, created) VALUES (%s, %s, %s)", (u.id, u.name, u.created))
    return u


def chat(i, title="Trip", answer="Go to the Alamo.", label="web"):
    return {"id": f"c{i}", "title": title, "created": 1000 + i, "updated": 2000 + i, "wx": {"icon": "☁️"},
            "messages": [{"role": "user", "content": title}, {"role": "assistant", "data": {"answer": answer, "label": label, "grounded": False}}]}


def test_sessions_are_hashed_and_expire():
    a = store()
    u = user(a)
    token = a.start_session(u)
    assert a.session_user(token).id == u.id
    assert a.db.one("SELECT token_hash FROM sessions")[0] != token          # only the hash is stored
    a.db.run("UPDATE sessions SET expires = 0")
    assert a.session_user(token) is None
    assert a.session_user("") is None and a.session_user("made-up") is None


def test_chats_are_private_to_each_user_and_searchable():
    a = store()
    sam, kim = user(a, "Sam"), user(a, "Kim")
    a.save_chat(sam.id, chat(1, "Weekend in San Antonio", "Walk the River Walk and visit the Alamo."))
    a.save_chat(sam.id, chat(2, "Dinner ideas", "Try sheet-pan fajitas."))
    a.save_chat(kim.id, chat(3, "Kim's secret"))
    titles = [c["title"] for c in a.list_chats(sam.id)]
    assert titles == ["Dinner ideas", "Weekend in San Antonio"]
    assert [c["title"] for c in a.list_chats(sam.id, "alamo")] == ["Weekend in San Antonio"]   # searches inside answers
    assert a.get_chat(kim.id, "c1") is None and a.get_chat(sam.id, "c3") is None
    s = a.list_chats(sam.id)[0]
    assert s["questions"] == 1 and s["last_answer"] == "Try sheet-pan fajitas." and s["labels"] == [["web", False]]


def test_save_updates_rename_delete_and_cap():
    a = store()
    u = user(a)
    a.save_chat(u.id, chat(1))
    c = chat(1)
    c["messages"] += c["messages"]
    c["updated"] = 9999
    a.save_chat(u.id, c)
    assert len(a.get_chat(u.id, "c1")["messages"]) == 4 and a.list_chats(u.id)[0]["updated"] == 9999
    a.rename_chat(u.id, "c1", "  New   name ")
    assert a.get_chat(u.id, "c1")["title"] == "New name"
    a.delete_chat(u.id, "c1")
    assert a.list_chats(u.id) == []
    big = chat(9)
    big["messages"] = [{"role": "user", "content": "x" * 20000}] * 100
    with pytest.raises(ValueError, match="too long"):
        a.save_chat(u.id, big)


def test_deleting_an_account_removes_everything():
    a = store()
    u = user(a)
    a.save_chat(u.id, chat(1))
    a.start_session(u)
    a.delete_user(u.id)
    for table in ("users", "sessions", "chats", "credentials"):
        assert a.db.one(f"SELECT COUNT(*) FROM {table}")[0] == 0


def test_challenges_are_single_use():
    a = store()
    cid, options = a.authentication_options("localhost")
    assert '"challenge"' in options
    a._take(cid)
    with pytest.raises(ValueError, match="already used"):
        a._take(cid)


def make(accounts):
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=FakeClient([]), settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=accounts)
    return TestClient(app)


def test_signed_out_people_can_not_read_or_save_chats():
    c = make(store())
    assert c.get("/api/me").json() == {"enabled": True, "user": None}
    assert c.get("/api/chats").status_code == 401
    assert c.put("/api/chats/c1", json=chat(1)).status_code == 401
    off = make(False)
    assert off.get("/api/me").json()["enabled"] is False
    assert off.post("/api/auth/login/options").status_code == 503


def test_signed_in_chat_api_round_trip():
    a = store()
    u = user(a)
    c = make(a)
    c.cookies.set("askh_session", a.start_session(u))
    assert c.get("/api/me").json()["user"]["name"] == "Sam"
    assert c.put("/api/chats/c1", json=chat(1)).status_code == 200
    assert c.post("/api/chats/import", json={"chats": [chat(2), {"bad": True}]}).json() == {"saved": 1}
    assert [x["id"] for x in c.get("/api/chats").json()["chats"]] == ["c2", "c1"]
    assert c.get("/api/chats/c1").json()["messages"][1]["data"]["answer"] == "Go to the Alamo."
    assert c.patch("/api/chats/c1", json={"title": "Renamed"}).status_code == 200
    assert len(c.get("/api/chats/export").json()["chats"]) == 2
    assert c.delete("/api/chats/c1").status_code == 200 and c.get("/api/chats/c1").status_code == 404
    assert c.delete("/api/me").status_code == 200
    assert c.get("/api/me").json()["user"] is None


def test_passkey_options_use_this_site():
    c = make(store())
    d = c.post("/api/auth/register/options", json={"name": "Sam"}).json()
    assert d["options"]["rp"]["id"] == "testserver" and d["options"]["user"]["name"] == "Sam"
    assert d["options"]["authenticatorSelection"]["residentKey"] == "required"
    bad = c.post("/api/auth/register/verify", json={"cid": "x" * 20, "credential": {}})
    assert bad.status_code == 400 and "try again" in bad.json()["message"].lower()
