"""Solve mode: a tournament of approaches scored by code, a failure journal, sealed hidden tests and lessons that carry over."""
import json
from types import SimpleNamespace

import pytest

from agent import solve as sv
from fakes import FakeClient, response, text, tool_use
from test_lab import code_blocks

FRAME = {"measurable": True, "problem": "Merge overlapping intervals as fast as possible.", "entry": "def merge(xs: list[list[int]]) -> list[list[int]]",
         "metric": {"name": "runtime", "direction": "min", "unit": "ms", "how": "median of 5 runs on 100k random intervals"},
         "requirements": ["Output intervals are sorted and non-overlapping", "Every input point is covered"],
         "approaches": [{"name": "Sort and sweep", "idea": "sort by start, sweep"}, {"name": "Brute force", "idea": "pairwise merge until stable"},
                        {"name": "Interval tree", "idea": "insert into a tree"}]}
HIDDEN = "import json\nfrom solution import merge\n# cases...\nprint('@@HIDDEN ' + json.dumps({'passed': 20, 'total': 20, 'failures': []}))\n"


def score(name, passed, total, metric):
    return "@@SCORE " + json.dumps({"approach": name, "passed": passed, "total": total, "metric": metric})


def submit(entries):
    return response([tool_use("s", "submit_round", {"entries": entries})], "tool_use")


def bash(n, stdout, rc=0):
    return [{"type": "server_tool_use", "id": f"x{n}", "name": "bash_code_execution", "input": {"command": "python /work/bench.py"}},
            {"type": "bash_code_execution_tool_result", "tool_use_id": f"x{n}", "content": {"type": "bash_code_execution_result", "stdout": stdout, "stderr": "", "return_code": rc}}]


def boxed(resp, cid="cntr_1"):
    resp.container = SimpleNamespace(id=cid)   # the real API returns the sandbox's id with every code execution response
    return resp


def script(hidden_r2="20/20"):
    p, t = hidden_r2.split("/")
    r1 = (code_blocks(1, "/work/approaches/sort_and_sweep.py", "def merge(xs): ...", "python /work/bench.py sort_and_sweep", score("sort_and_sweep", 12, 12, 2.0))
          + bash(2, score("brute_force", 12, 12, 40.0)) + bash(3, "Traceback: interval_tree crashed", rc=1))
    r2 = bash(4, score("sort_and_sweep_r2", 12, 12, 1.2)) + bash(5, score("brute_force_r2", 12, 12, 35.0)) + bash(6, score("new_idea_r2", 9, 12, None))
    return [
        response([tool_use("f", "submit_frame", FRAME)], "tool_use"), response([text("Framed.")], "end_turn"),
        response([text("```python\n" + HIDDEN + "```")], "end_turn"),
        boxed(response(r1 + [tool_use("s1", "submit_round", {"entries": [
            {"approach": "sort_and_sweep", "what_happened": "fast", "lesson": "Sorting first makes one pass enough", "next_idea": "avoid copying lists"},
            {"approach": "brute_force", "what_happened": "slow", "lesson": "Quadratic merging is far too slow at 100k", "next_idea": ""},
            {"approach": "interval_tree", "what_happened": "crashed", "lesson": "Recursion depth blew up", "next_idea": "an iterative heap-based sweep"}]})], "tool_use")),
        response([text("Round 1 done.")], "end_turn"),
        response(r2 + [tool_use("s2", "submit_round", {"entries": [{"approach": "sort_and_sweep_r2", "what_happened": "faster", "lesson": "In-place merge saves allocations"}]})], "tool_use"),
        response([text("Round 2 done.")], "end_turn"),
        boxed(response(bash(7, "@@FILE sort_and_sweep_r2\ndef merge(xs):\n    return xs\n@@END\n@@FILE sort_and_sweep\ndef merge(xs):\n    return sorted(xs)\n@@END\n"), "end_turn")),
        response(bash(8, "@@HIDDEN " + json.dumps({"passed": int(p), "total": int(t), "failures": [] if p == t else ["[[1,3],[2,4]] -> wrong"]})) + [text("done")], "end_turn"),
        response(bash(9, "@@HIDDEN " + json.dumps({"passed": 20, "total": 20, "failures": []})) + [text("done")], "end_turn"),
    ]


def test_a_tournament_scored_by_code_and_verified_on_sealed_tests():
    client, events = FakeClient(script()), []
    rec = sv.solve(client, "main", "helper", "today", "Make interval merging fast", lambda k, d: events.append((k, d)), budget="quick")
    kinds = [k for k, _ in events]
    assert kinds.index("sealed") < kinds.index("round")                          # sealed before any solving
    assert rec["seal"]["sha256"] == sv.sha256(HIDDEN.strip()) and HIDDEN.strip() == rec["hidden_tests"]
    assert HIDDEN.strip() not in json.dumps([r["messages"] for r in client.requests[3:7]])   # the solver never saw them
    board = rec["leaderboard"]
    assert [b["approach"] for b in board[:2]] == ["sort_and_sweep_r2", "sort_and_sweep"] and board[0]["metric"] == 1.2
    crashed = next(e for e in rec["journal"] if e["approach"] == "interval_tree")
    assert crashed["result"].startswith("no score printed") and "Recursion" in crashed["lesson"]
    assert rec["status"] == "verified" and rec["winner"]["approach"] == "sort_and_sweep_r2" and rec["winner"]["hidden"]["passed"] == 20
    assert rec["winner"]["source"] == "def merge(xs):\n    return xs"
    round2 = client.requests[5]["system"]
    assert "FAILURE JOURNAL" in round2 and "Quadratic merging is far too slow" in round2 and "Refine sort_and_sweep" in round2
    assert "an iterative heap-based sweep" in round2                              # the crashed approach's idea came back as a fresh try
    verify_req = client.requests[-1]["messages"][-1]["content"]
    assert HIDDEN.strip() in verify_req and client.requests[-1].get("container") is None   # a fresh sandbox
    md = sv.report_markdown({**rec, "title": "Merge"})
    assert "| sort_and_sweep_r2 | 2 | 12/12 | 1.2 |" in md and rec["seal"]["sha256"] in md


def test_passing_its_own_tests_but_failing_the_sealed_ones_is_reported_honestly():
    rec = sv.solve(FakeClient(script(hidden_r2="17/20")), "main", "helper", "today", "Make interval merging fast", lambda k, d: None, budget="quick")
    assert rec["winner"]["approach"] == "sort_and_sweep"                         # the runner-up passed the sealed tests
    loser = next(f for f in rec["finalists"] if f["approach"] == "sort_and_sweep_r2")
    assert loser["hidden"]["passed"] == 17 and loser["hidden"]["failures"]


def test_problems_code_cant_measure_are_declined_plainly():
    frame = {"measurable": False, "why_not": "Team morale can't be scored by code.", "problem": "Fix our team's morale"}
    client = FakeClient([response([tool_use("f", "submit_frame", frame)], "tool_use"), response([text("ok")], "end_turn")])
    rec = sv.solve(client, "main", "helper", "today", "How do we fix our team's morale?", lambda k, d: None)
    assert rec["measurable"] is False and rec["frame"]["why_not"].startswith("Team morale") and len(client.requests) == 2


def test_scores_count_only_from_successful_runs():
    runs = [{"n": 1, "kind": "bash", "status": "ok", "stdout": score("a", 3, 3, 1.0) + "\n" + score("b", 2, 3, 9.0)},
            {"n": 2, "kind": "bash", "status": "failed", "stdout": score("c", 3, 3, 0.1)},
            {"n": 3, "kind": "file", "status": "ok", "stdout": score("d", 3, 3, 0.1)}]
    got = sv.scores(runs, 1)
    assert [(s["approach"], s["correct"], s["metric"]) for s in got] == [("a", True, 1.0), ("b", False, None)]


def test_lessons_carry_over_to_similar_problems():
    rec = {"frame": {"problem": "Merge overlapping intervals fast"}, "winner": {"approach": "sweep"},
           "journal": [{"approach": "sweep", "idea": "sort then sweep", "correct": True, "result": "correct", "lesson": "sorting first wins"},
                       {"approach": "brute", "idea": "pairwise merging", "correct": False, "result": "incorrect", "lesson": "quadratic is too slow"}]}
    ls = sv.lessons_from(rec)
    assert [l["outcome"] for l in ls] == ["won", "failed"]
    assert len(sv.relevant(ls, "Speed up merging overlapping intervals in Python")) == 2
    assert sv.relevant(ls, "Write a haiku about rain") == []
    assert "LESSONS FROM YOUR EARLIER EXPERIMENTS" in sv.lessons_text(ls)


def test_solve_through_the_api_saves_lessons_for_next_time():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from agent import accounts as acct, inbox as inboxmod, receipts as rcpt
    from agent.notes import MemoryNotesStore
    from server.app import Settings, create_app
    a = acct.Accounts(acct.Database("sqlite:///:memory:"))
    u = acct.User(acct.b64u(b"solver"), "Harsha", acct.now_ms())
    a.db.run("INSERT INTO users (id, name, created) VALUES (%s, %s, %s)", (u.id, u.name, u.created))
    client = FakeClient(script() + script())
    app = create_app(client=client, settings=Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame"),
                     notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(), receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), accounts=a)
    c = TestClient(app)
    c.cookies.set("askh_session", a.start_session(u))
    body = c.post("/api/lab/solve", json={"problem": "Make merging overlapping intervals fast", "budget": "quick"}).text
    final = json.loads(body.split("event: final\ndata: ")[1].split("\n\n")[0])
    assert final["status"] == "verified" and final["receipt"]["type"] == "solve" and final["lessons_saved"] == 6
    assert final["receipt"]["sealed_tests_sha256"] == final["seal"]["sha256"]
    assert c.post("/api/proof/verify", json={"receipt": final["receipt"], "signature": final["signature"]}).json()["valid"]
    assert c.get("/api/lab/runs").json()["runs"][0]["mode"] == "solve"
    body2 = c.post("/api/lab/solve", json={"problem": "Speed up merging overlapping intervals again", "budget": "quick"}).text
    assert "event: lessons" in body2
    frame_req = next(r for r in client.requests[10:] if "Frame stage" in r["system"])
    assert "LESSONS FROM YOUR EARLIER EXPERIMENTS" in frame_req["system"] and "Quadratic merging" in frame_req["system"]
