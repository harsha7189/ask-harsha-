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


def test_database_url_passwords_with_symbols_are_encoded():
    u = "postgresql://postgres.abc:p@ss#w/rd?%@aws-0-us-west-2.pooler.supabase.com:5432/postgres"
    assert acct.fix_password(u) == "postgresql://postgres.abc:p%40ss%23w%2Frd%3F%25@aws-0-us-west-2.pooler.supabase.com:5432/postgres"
    plain = "postgresql://postgres.abc:Simple123@h:5432/postgres"
    assert acct.fix_password(plain) == plain
    with pytest.raises(ValueError, match="YOUR-PASSWORD"):
        acct.connect("postgresql://postgres.abc:[YOUR-PASSWORD]@h:5432/postgres")


def test_recovery_code_is_hashed_single_use_and_signs_out_other_devices():
    a = store()
    u = user(a)
    old_session = a.start_session(u)
    code = a.make_recovery_code(u.id)
    assert len(acct.normalize_code(code)) == 20 and code.count("-") == 4
    assert set(acct.normalize_code(code)) <= set(acct.CODE_ALPHABET)
    assert code not in str(a.db.all("SELECT * FROM recovery"))           # only the hash is stored
    assert a.recovery_status(u.id) is not None
    assert a.recover(" " + code.lower().replace("-", " ") + " ").id == u.id   # forgiving about case and spacing
    assert a.session_user(old_session) is None                            # every other device signed out
    assert a.recovery_status(u.id) is None
    with pytest.raises(ValueError, match="didn't work"):
        a.recover(code)                                                   # used up
    with pytest.raises(ValueError, match="20 letters"):
        a.recover("short")


def test_new_recovery_code_replaces_the_old_one():
    a = store()
    u = user(a)
    first = a.make_recovery_code(u.id)
    second = a.make_recovery_code(u.id)
    with pytest.raises(ValueError):
        a.recover(first)
    assert a.recover(second).id == u.id


def test_passkeys_list_and_the_last_one_can_not_be_removed():
    a = store()
    u = user(a)
    for i, label in enumerate(["iPhone", "Windows"]):
        a.db.run("INSERT INTO credentials (id, user_id, public_key, sign_count, created, label) VALUES (%s, %s, %s, 0, %s, %s)",
                 (f"cred{i}", u.id, "pk", acct.now_ms() + i, label))
    assert [k["label"] for k in a.passkeys(u.id)] == ["iPhone", "Windows"]
    a.remove_passkey(u.id, "cred0")
    with pytest.raises(ValueError, match="only passkey"):
        a.remove_passkey(u.id, "cred1")
    kim = user(a, "Kim")
    with pytest.raises(ValueError, match="isn't on your account"):
        a.remove_passkey(kim.id, "cred1")
    assert acct.device_label("Mozilla/5.0 (iPhone; CPU iPhone OS 19_0 like Mac OS X)") == "iPhone"
    assert acct.device_label("Mozilla/5.0 (Windows NT 10.0; Win64; x64)") == "Windows"


def test_recovery_api_round_trip():
    a = store()
    u = user(a)
    code = a.make_recovery_code(u.id)
    c = make(a)
    assert c.post("/api/auth/recover", json={"code": "ABCD-EFGH-JKMN-PQRS-TVWX"}).status_code == 400
    d = c.post("/api/auth/recover", json={"code": code}).json()
    assert d["recovered"] is True and d["user"]["recovery"] is None
    assert c.get("/api/me").json()["user"]["name"] == "Sam"               # signed in by the cookie
    assert c.get("/api/account/passkeys").json() == {"passkeys": [], "recovery": None}
    opts = c.post("/api/account/passkeys/options").json()["options"]
    assert opts["user"]["id"] == u.id and opts["rp"]["id"] == "testserver"
    fresh = c.post("/api/account/recovery").json()
    assert len(acct.normalize_code(fresh["recovery_code"])) == 20 and c.get("/api/me").json()["user"]["recovery"]
    signed_out = make(a)
    assert signed_out.post("/api/account/recovery").status_code == 401
    assert signed_out.get("/api/account/passkeys").status_code == 401


def test_projects_group_chats_and_feed_the_prompt():
    from agent import personal
    a = store()
    u = user(a)
    p = a.save_project(u.id, {"name": " Job  hunt ", "icon": "💼", "instructions": "I'm targeting senior PM roles in Austin."})
    assert p["name"] == "Job hunt" and p["chats"] == 0
    a.save_chat(u.id, {**chat(1), "project": p["id"]})
    a.save_chat(u.id, {**chat(2), "project": "made-up"})                 # unknown project: saved as a regular chat
    assert a.get_chat(u.id, "c1")["project"] == p["id"] and a.get_chat(u.id, "c2")["project"] is None
    a.save_chat(u.id, chat(1))                                            # no project sent: keeps the one it had
    assert a.get_chat(u.id, "c1")["project"] == p["id"]
    a.set_chat_project(u.id, "c2", p["id"])
    assert a.list_projects(u.id)[0]["chats"] == 2
    assert {c["project"] for c in a.list_chats(u.id)} == {p["id"]}
    block = personal.prompt_block({"name": "Harsha", "length": "short", "tone": "direct"}, a.get_project(u.id, p["id"]))
    assert "Harsha" in block and "senior PM roles" in block and "never change your rules" in block
    assert personal.prompt_block({}) == "" and personal.prompt_block(None, None) == ""
    kim = user(a, "Kim")
    assert a.get_project(kim.id, p["id"]) is None                         # projects are private
    with pytest.raises(ValueError):
        a.set_chat_project(kim.id, "c1", p["id"])
    a.delete_project(u.id, p["id"])
    assert a.get_chat(u.id, "c1")["project"] is None and a.list_projects(u.id) == []   # chats are kept


def test_saved_answers_and_prefs():
    a = store()
    u = user(a)
    a.save_item(u.id, {"id": "c1:1", "kind": "star", "chat_id": "c1", "question": "Dinner?", "data": {"answer": "Fajitas."}})
    a.save_item(u.id, {"id": "card:abc", "kind": "card", "question": "Shipped?", "data": {"answer": "Yes.", "receipt": {"id": "abc"}}})
    assert sorted(x["kind"] for x in a.list_saved(u.id)) == ["card", "star"]
    with pytest.raises(ValueError):
        a.save_item(u.id, {"id": "x", "data": {}})
    a.remove_saved(u.id, "c1:1")
    assert [x["id"] for x in a.list_saved(u.id)] == ["card:abc"]
    assert a.get_prefs(u.id) is None
    saved = a.save_prefs(u.id, {"name": "Harsha", "length": "huge", "rate": 9, "evil": "x"})
    assert saved["length"] == "balanced" and saved["rate"] == 1.6 and "evil" not in saved
    assert a.get_prefs(u.id)["name"] == "Harsha"
    a.delete_user(u.id)
    for table in ("projects", "saved", "prefs"):
        assert a.db.one(f"SELECT COUNT(*) FROM {table}")[0] == 0


def test_project_instructions_reach_the_model():
    a = store()
    u = user(a)
    p = a.save_project(u.id, {"name": "Trip", "instructions": "We travel with a toddler."})
    from fakes import response, text
    calls = []

    class Spy:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            calls.append(kw)
            return response([text("Here you go.")], "end_turn")

    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=Spy(), settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=a)
    c = TestClient(app)
    c.cookies.set("askh_session", a.start_session(u))
    assert c.post("/api/chat", json={"question": "Ideas for Saturday?", "project": p["id"],
                                     "prefs": {"name": "Sam", "tone": "friendly"}}).status_code == 200
    assert "toddler" in calls[-1]["system"] and "Sam" in calls[-1]["system"]
    out = TestClient(app)   # signed out: prefs still apply, project instructions don't
    out.post("/api/chat", json={"question": "Ideas?", "project": p["id"], "prefs": {"name": "Lee"}})
    assert "Lee" in calls[-1]["system"] and "toddler" not in calls[-1]["system"]
    assert c.post("/api/projects", json={"name": "Work"}).json()["name"] == "Work"
    assert len(c.get("/api/projects").json()["projects"]) == 2
    assert c.post("/api/saved", json={"id": "x1", "question": "q", "data": {"answer": "a"}}).status_code == 200
    assert c.get("/api/saved").json()["saved"][0]["id"] == "x1"
    assert c.put("/api/account/prefs", json={"name": "Sam"}).json()["prefs"]["name"] == "Sam"
    assert out.get("/api/projects").status_code == 401 and out.get("/api/saved").status_code == 401
