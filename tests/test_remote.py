import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import inbox as inboxmod, receipts as rcpt, remote  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from fakes import FakeClient, response, tool_use  # noqa: E402
from server.app import Settings, create_app  # noqa: E402

CTX = {"signed_in": True, "projects": [{"id": "p1", "name": "Job search"}, {"id": "p2", "name": "Trip to Lisbon"}],
       "chats": [{"id": "c1", "title": "Lisbon food tips", "project": "p2"}], "saved": [{"question": "How to price SaaS?"}]}


@pytest.mark.parametrize("text,action,args", [
    ("make my answers shorter", "set_pref", {"pref": "length", "value": "short"}),
    ("More detail please", "set_pref", {"pref": "length", "value": "detailed"}),
    ("read answers aloud", "set_pref", {"pref": "read_aloud", "value": True}),
    ("stop reading", "set_pref", {"pref": "read_aloud", "value": False}),
    ("be more formal", "set_pref", {"pref": "tone", "value": "professional"}),
    ("call me Harsha", "set_pref", {"pref": "name", "value": "Harsha"}),
    ("morning brief", "open_tool", {"tool": "morning"}),
    ("new chat", "new_chat", {}),
    ("open settings", "open_section", {"section": "customize"}),
    ("show my list", "open_section", {"section": "saved"}),
    ("open my job search", "open_project", {"project_id": "p1"}),
    ("sign in", "sign_in", {}),
])
def test_common_commands_need_no_model(text, action, args):
    cmd = remote.quick(text, CTX)
    assert cmd and cmd.action == action and cmd.args == args


def test_questions_go_to_the_model():
    assert remote.quick("what's a good shorter route to the airport from downtown Austin", CTX) is None
    assert remote.quick("the lisbon one", CTX) is None


def test_model_answers_are_checked_against_the_context():
    def run(inp):
        client = FakeClient([response([tool_use("t1", "do", inp)], "tool_use")])
        cmd = remote.interpret(client, "m", "the lisbon one", CTX)
        req = client.requests[0]
        assert req["tool_choice"] == {"type": "tool", "name": "do"} and "Trip to Lisbon" in req["system"]
        return cmd
    assert run({"action": "open_chat", "chat_id": "c1", "say": "Opening it."}).args == {"chat_id": "c1"}
    assert run({"action": "open_chat", "chat_id": "c999", "say": "x"}).action == "reply"       # made-up id
    assert run({"action": "delete_everything", "say": "x"}).action == "reply"                  # not an action
    assert run({"action": "set_pref", "pref": "length", "value": "huge", "say": "x"}).action == "reply"
    assert run({"action": "set_pref", "pref": "jury", "value": "false", "say": "Off."}).args == {"pref": "jury", "value": False}
    p = run({"action": "create_project", "name": "  Q4   roadmap ", "icon": "📈", "say": "Made it."})
    assert p.args["name"] == "Q4 roadmap" and p.args["icon"] == "📈"
    assert run({"action": "ask", "query": "capital of Peru?", "say": "Asking."}).args == {"query": "capital of Peru?"}


def test_remote_endpoint():
    client = FakeClient([response([tool_use("t1", "do", {"action": "search_saved", "query": "pricing", "say": "Here's what you saved."})], "tool_use")])
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=client, settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=False)
    c = TestClient(app)
    assert c.post("/api/remote", json={"text": "shorter", "context": CTX}).json()["source"] == "rules"
    assert len(client.requests) == 0
    d = c.post("/api/remote", json={"text": "what did I save about pricing", "context": CTX}).json()
    assert d == {"action": "search_saved", "args": {"query": "pricing"}, "say": "Here's what you saved.", "source": "model"}
