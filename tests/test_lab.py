import json

import pytest

from agent import lab
from fakes import FakeClient, response, text, tool_use


def code_blocks(n, path, code, command, stdout, rc=0):
    return [
        {"type": "server_tool_use", "id": f"f{n}", "name": "text_editor_code_execution", "input": {"command": "create", "path": path, "file_text": code}},
        {"type": "text_editor_code_execution_tool_result", "tool_use_id": f"f{n}", "content": {"type": "text_editor_code_execution_create_result", "is_file_update": False}},
        {"type": "server_tool_use", "id": f"b{n}", "name": "bash_code_execution", "input": {"command": command}},
        {"type": "bash_code_execution_tool_result", "tool_use_id": f"b{n}", "content": {"type": "bash_code_execution_result", "stdout": stdout, "stderr": "", "return_code": rc, "content": []}},
    ]


def ev(**d):
    return "@@EVIDENCE " + json.dumps(d)


PLAN = {"question": "Are these three claims about integers true?", "known": [{"fact": "Goldbach is open", "source": "https://en.wikipedia.org/wiki/Goldbach%27s_conjecture"}],
        "hypotheses": [{"id": "H1", "statement": "Every even n from 4 to 10^5 is a sum of two primes", "test": "check all", "falsifier": "an even n with no pair"},
                       {"id": "H2", "statement": "n^2 + n + 41 is prime for every n >= 0", "test": "check n up to 100", "falsifier": "a composite value"},
                       {"id": "H3", "statement": "1 + 3 + ... + (2n-1) = n^2 for all n", "test": "SymPy summation", "falsifier": "the sum differs"}]}
EXPERIMENT_OUT = "\n".join([
    ev(h="H1", kind="exhaustive", result="supports", complete=True, detail="all 49,999 even n in [4, 10^5] have a pair"),
    ev(h="H2", kind="numerical", result="supports", detail="prime for n = 0..39"),
    ev(h="H3", kind="symbolic", result="supports", detail="summation(2k-1, (k,1,n)) - n**2 simplifies to 0"),
    "@@CHART " + json.dumps({"title": "Goldbach pairs", "type": "line", "x": [4, 6, 8], "series": [{"name": "pairs", "y": [1, 1, 1]}]}),
    "@@EVIDENCE not json", "noise line"])
CHALLENGE_OUT = ev(h="H2", kind="counterexample", result="refutes", detail="n = 40 gives 1681 = 41 * 41")


def script():
    return [
        response([tool_use("p", "submit_plan", PLAN)], "tool_use"), response([text("Plan ready.")], "end_turn"),
        response(code_blocks(1, "/tmp/exp.py", "print('experiments')", "python /tmp/exp.py", EXPERIMENT_OUT)
                 + [tool_use("s1", "submit_experiments", {"findings": [{"h": "H1", "method": "sieve", "finding": "no exception"}]})], "tool_use"),
        response([text("Done.")], "end_turn"),
        response(code_blocks(2, "/tmp/attack.py", "print('attack')", "python /tmp/attack.py", CHALLENGE_OUT)
                 + [tool_use("s2", "submit_report", {"attacks": [{"h": "H2", "attack": "n = 40", "outcome": "composite"}],
                                                     "summary": "H1 holds in range; H2 fails at 40; H3 is an identity.", "limitations": ["Finite range"]})], "tool_use"),
        response([text("Reported.")], "end_turn"),
    ]


def test_labels_come_only_from_code_output():
    client = FakeClient(script())
    events = []
    run = lab.investigate(client, "claude-sonnet-5-5", "today", "Test three integer claims", lambda k, d: events.append((k, d)))
    by = {v["id"]: v for v in run["verdicts"]}
    assert by["H1"]["label"] == "proven" and "every case in the claimed domain" in by["H1"]["why"].lower()
    assert by["H2"]["label"] == "disproven" and "1681" in by["H2"]["why"]
    assert by["H3"]["label"] == "proven" and "SymPy" in by["H3"]["why"]
    assert len(run["runs"]) == 4 and run["runs"][1]["stage"] == "experiment" and run["runs"][3]["stage"] == "challenge"
    assert len(run["evidence"]) == 4 and run["charts"][0]["title"] == "Goldbach pairs"
    kinds = [k for k, _ in events]
    assert kinds.count("stage") == 3 and "plan" in kinds and kinds.count("run") == 4 and "chart" in kinds
    exp_req = client.requests[2]
    assert any(t.get("type") == "code_execution_20250825" for t in exp_req["tools"]) and "@@EVIDENCE" in exp_req["system"]
    assert not any(t.get("type") == "code_execution_20250825" for t in client.requests[0]["tools"])   # Discover has no code tool
    md = lab.report_markdown({**run, "title": "Integer claims"})
    assert "### H2: Disproven" in md and "Finite range" in md
    assert "print('experiments')" in lab.reproducible_script(run["runs"])


def test_failed_runs_and_untested_claims_never_count():
    runs = lab.runs_from([{"role": "assistant", "content": code_blocks(1, "/tmp/a.py", "x", "python /tmp/a.py",
                                                                         ev(h="H1", kind="symbolic", result="supports"), rc=1)}], "experiment")
    assert runs[1]["status"] == "failed" and lab.markers(runs)[0] == []        # a crashed script's output is ignored
    assert lab.label("H1", [])[0] == "hypothesis"
    mixed = [{"h": "H1", "kind": "numerical", "result": "supports", "complete": False, "detail": "a", "run": 1, "stage": "experiment"},
             {"h": "H1", "kind": "numerical", "result": "refutes", "complete": False, "detail": "b", "run": 2, "stage": "challenge"}]
    assert lab.label("H1", mixed)[0] == "hypothesis"                            # conflicting evidence isn't a win
    partial = [{"h": "H1", "kind": "exhaustive", "result": "supports", "complete": False, "detail": "n<=1e6", "run": 1, "stage": "experiment"}]
    assert lab.label("H1", partial)[0] == "evidence"                            # a range isn't a proof


def test_a_vague_question_stops_early():
    client = FakeClient([response([text("Hmm.")], "end_turn")])
    with pytest.raises(lab.LabError):
        lab.investigate(client, "m", "today", "Is life good?", lambda k, d: None)


def test_lab_endpoint_needs_an_account_saves_history_and_limits_per_day(monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("LAB_DAILY", "1")
    from fastapi.testclient import TestClient
    from agent import accounts as acct, inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from server.app import Settings, create_app
    a = acct.Accounts(acct.Database("sqlite:///:memory:"))
    u = acct.User(acct.b64u(b"user-sam"), "Sam", acct.now_ms())
    a.db.run("INSERT INTO users (id, name, created) VALUES (%s, %s, %s)", (u.id, u.name, u.created))
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=FakeClient(script()), settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=a)
    anon = TestClient(app)
    assert anon.post("/api/lab/stream", json={"question": "Test three integer claims"}).status_code == 401
    c = TestClient(app)
    c.cookies.set("askh_session", a.start_session(u))
    assert c.get("/api/lab/status").json()["allowed"] is True
    body = c.post("/api/lab/stream", json={"question": "Test three integer claims"}).text
    assert "event: plan" in body and "event: run" in body and "event: final" in body
    final = json.loads(body.split("event: final\ndata: ")[1].split("\n\n")[0])
    assert final["saved"] is True and [v["label"] for v in final["verdicts"]] == ["proven", "disproven", "proven"]
    assert final["receipt"]["type"] == "lab" and final["signature"]
    runs = c.get("/api/lab/runs").json()["runs"]
    assert runs[0]["id"] == final["id"] and runs[0]["labels"] == ["proven", "disproven", "proven"]
    assert c.get(f"/api/lab/runs/{final['id']}").json()["plan"]["hypotheses"][1]["id"] == "H2"
    over = c.post("/api/lab/stream", json={"question": "Another one please"})
    assert over.status_code == 429 and "today's 1" in over.json()["message"]
    assert c.delete(f"/api/lab/runs/{final['id']}").json() == {"ok": True} and c.get("/api/lab/runs").json()["runs"] == []
