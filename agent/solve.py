"""Discovery Lab, Solve mode: several competing approaches, scored by code, improved over rounds, checked on sealed tests.

  1. Frame       Can success be measured by code? If not, say so plainly (and suggest Investigate instead). If yes:
                 the metric, the function to write, and 3-4 genuinely different approaches.
  2. Seal        A separate model call writes hidden tests the solver never sees. Their SHA-256 fingerprint is shown
                 (and signed) before solving starts, so anyone can check afterwards that the tests weren't changed to
                 fit the answer. The tests are revealed in the report.
  3. Rounds      In the sandbox the solver writes a harness (visible tests + a benchmark), implements each approach,
                 and runs it. Scores count only if a successful run printed them:
                     @@SCORE {"approach": "...", "passed": 12, "total": 12, "metric": 3.41}
                 Every approach and its outcome go into a failure journal; the next round reads the journal, drops
                 dead ends and refines the leaders. It stops when the best score stops improving or the budget ends.
  4. Verify      The finalists' code (read back from the sandbox with cat, so it's exactly what ran) is run against
                 the sealed tests in a fresh sandbox that never saw the solving.
  5. Learn       Lessons (what worked, what failed and why) are saved to the person's experiment journal and offered
                 to the next problem, so the process improves without pretending to retrain the model.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .lab import CODE_TOOL, LabError, _txt, runs_from

BUDGETS = {"quick": {"rounds": 2, "approaches": 3}, "standard": {"rounds": 3, "approaches": 3}, "thorough": {"rounds": 4, "approaches": 4}}
PLATEAU = 0.02          # stop when a round improves the best score by less than 2%
SCORE_RE = re.compile(r"^@@SCORE\s+(\{.*\})\s*$", re.M)
HIDDEN_RE = re.compile(r"^@@HIDDEN\s+(\{.*\})\s*$", re.M)

FRAME_PROMPT = """You are the Frame stage of Ask Harsha's Discovery Lab, Solve mode. Today is {today}.

Decide whether the problem's success can be measured by code in a sandbox (Python 3.11, numpy/scipy/sympy/pandas, 1 CPU, \
no internet, runs of about a minute). Code-measurable: making code faster or more correct, finding an algorithm that \
beats a baseline, fitting a model to given data, finding values that satisfy a condition. Not measurable: strategy, \
people, design taste, anything needing live systems or private data that wasn't given.

Call submit_frame once:
- measurable: true or false. If false, explain why_not in one or two plain sentences and what could make it measurable.
- problem: the problem restated precisely.
- entry: the exact Python function every approach must define, with types, e.g. "def solve(intervals: list[list[int]]) -> list[list[int]]".
- metric: what to optimize once an approach is correct: name, direction ("min" or "max"), unit, and how to measure it \
(e.g. "median runtime over 5 runs on n=100000 random inputs, seed 1").
- requirements: what counts as correct, as short testable statements.
- approaches: {n} genuinely different approaches (different ideas, not tweaks), each with a short name and the idea.
{lessons}"""

HIDDEN_PROMPT = """You write the sealed hidden tests for a coding problem. The solver will never see them; they decide \
whether the winning solution really works. Write ONE self-contained Python file that:
- imports the solution with: from solution import {func}
- checks correctness on edge cases (empty, one element, duplicates, negatives, extremes) and on larger random cases \
with a fixed seed, comparing against a simple, obviously-correct reference implementation you write in the file
- never crashes on a failing case: catch exceptions per case and count them as failures
- finally prints exactly one line: print("@@HIDDEN " + json.dumps({{"passed": p, "total": t, "failures": first_five_failure_descriptions}}))
Keep the whole run under 60 seconds. Reply with only the file content, no explanation, no code fences."""

ROUND_PROMPT = """You are the solver in Ask Harsha's Discovery Lab, Solve mode: a small engineering team that tries several \
approaches, measures them, and learns from what fails. You have a Python sandbox (code_execution): Python 3.11 with numpy, \
scipy, sympy, pandas; 1 CPU; no internet. Files persist between rounds in /work.

PROBLEM: {problem}
EVERY APPROACH DEFINES: {entry}
CORRECT MEANS: {requirements}
METRIC: {metric}
{inputs}
{harness}

THIS ROUND ({round} of {rounds}): implement and measure these approaches:
{todo}

{journal}

Rules:
- Write each approach to /work/approaches/<name>.py using the text editor tool's create command. The file must define the entry function.
- Score every approach by running the benchmark with bash: python /work/bench.py <name>. Scores only count if a successful \
run prints them, one line per approach: @@SCORE {{"approach": "<name>", "passed": <visible tests passed>, "total": <visible tests>, "metric": <number or null if incorrect>}}
- Fix crashes, but don't edit tests to make an approach pass.
Then call submit_round with one entry per approach: what happened and the lesson (why it won or failed), and one idea worth trying next."""

HARNESS_FIRST = """FIRST, build the harness (round 1 only):
- /work/visible_tests.py: a function check(fn) -> (passed, total) running at least 10 correctness cases (edge cases included) against a simple reference implementation.
- /work/bench.py: takes an approach name, imports /work/approaches/<name>.py, runs visible_tests.check, and if every test \
passes measures the METRIC as described; prints the @@SCORE line with json.dumps. Incorrect approaches get "metric": null."""

HARNESS_AGAIN = "The harness (/work/visible_tests.py, /work/bench.py) already exists. Don't change it."

SUBMIT_FRAME = {
    "name": "submit_frame", "description": "The framed problem, or why it can't be measured by code.",
    "input_schema": {"type": "object", "properties": {
        "measurable": {"type": "boolean"}, "why_not": {"type": "string"}, "problem": {"type": "string"}, "entry": {"type": "string"},
        "metric": {"type": "object", "properties": {"name": {"type": "string"}, "direction": {"type": "string", "enum": ["min", "max"]},
                                                     "unit": {"type": "string"}, "how": {"type": "string"}}, "required": ["name", "direction", "how"]},
        "requirements": {"type": "array", "items": {"type": "string"}},
        "approaches": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "idea": {"type": "string"}},
                                                  "required": ["name", "idea"]}}},
        "required": ["measurable"]},
}
SUBMIT_ROUND = {
    "name": "submit_round", "description": "What each approach did this round, the lesson, and what to try next.",
    "input_schema": {"type": "object", "properties": {"entries": {"type": "array", "items": {"type": "object", "properties": {
        "approach": {"type": "string"}, "what_happened": {"type": "string"}, "lesson": {"type": "string"}, "next_idea": {"type": "string"}},
        "required": ["approach", "what_happened", "lesson"]}}}, "required": ["entries"]},
}

VERIFY_PROMPT = """You run sealed tests in a fresh sandbox, exactly as given. Use the code_execution tool's text editor to \
create each file with exactly the content given (no edits), then run the command with bash. Don't fix or change anything. \
When done, reply with one word: done."""


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")[:32]
    return s if s and not s[0].isdigit() else ("a_" + s if s else "approach")


def func_name(entry: str) -> str:
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", entry or "")
    return m.group(1) if m else "solve"


def clean_frame(raw: dict, n: int) -> dict:
    metric = raw.get("metric") if isinstance(raw.get("metric"), dict) else {}
    seen, approaches = set(), []
    for a in (raw.get("approaches") or [])[:n]:
        if not isinstance(a, dict):
            continue
        s = slug(a.get("name"))
        while s in seen:
            s += "_2"
        seen.add(s)
        approaches.append({"name": s, "label": _txt(a.get("name"), 60) or s, "idea": _txt(a.get("idea"), 400)})
    return {"measurable": bool(raw.get("measurable")), "why_not": _txt(raw.get("why_not"), 600), "problem": _txt(raw.get("problem"), 1500),
            "entry": _txt(raw.get("entry"), 300) or "def solve(*args)",
            "metric": {"name": _txt(metric.get("name"), 80) or "score", "direction": "max" if metric.get("direction") == "max" else "min",
                       "unit": _txt(metric.get("unit"), 30), "how": _txt(metric.get("how"), 400)},
            "requirements": [_txt(r, 300) for r in (raw.get("requirements") or [])[:8] if _txt(r, 300)], "approaches": approaches}


def scores(runs: list[dict], round_no: int) -> list[dict]:
    """@@SCORE lines printed by successful bash runs. Anything else doesn't count."""
    out = []
    for r in runs:
        if r.get("kind") != "bash" or r.get("status") != "ok":
            continue
        for m in SCORE_RE.finditer(r.get("stdout", "")):
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            try:
                passed, total = int(d.get("passed", 0)), int(d.get("total", 0))
            except (TypeError, ValueError):
                continue
            metric = d.get("metric")
            metric = float(metric) if isinstance(metric, (int, float)) and metric == metric else None
            out.append({"approach": slug(str(d.get("approach", ""))), "passed": passed, "total": total, "correct": total > 0 and passed == total,
                        "metric": metric if total > 0 and passed == total else None, "run": r["n"], "round": round_no})
    return out


def better(a: float | None, b: float | None, direction: str) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return a < b if direction == "min" else a > b


def leaderboard(all_scores: list[dict], direction: str) -> list[dict]:
    """Each approach's best correct result (or its latest result if it was never correct), best first."""
    best: dict[str, dict] = {}
    for s in all_scores:
        cur = best.get(s["approach"])
        if cur is None or (s["correct"] and (not cur["correct"] or better(s["metric"], cur["metric"], direction))) or (not cur["correct"] and not s["correct"]):
            best[s["approach"]] = s
    rows = list(best.values())
    rows.sort(key=lambda s: (not s["correct"], (s["metric"] if direction == "min" else -s["metric"]) if s["metric"] is not None else float("inf"), -s["passed"]))
    return rows


def improvement(prev: float | None, cur: float | None, direction: str) -> float:
    if cur is None:
        return 0.0
    if prev is None:
        return 1.0
    if prev == 0:
        return 1.0 if better(cur, prev, direction) else 0.0
    gain = (prev - cur) / abs(prev) if direction == "min" else (cur - prev) / abs(prev)
    return max(0.0, gain)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```[a-zA-Z]*\n(.*?)\n```$", t, re.S)
    return m.group(1) if m else t


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    steps: int = 0

    def reply(self, r) -> None:
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.steps += r.steps

    def resp(self, resp) -> None:
        from .loop import _get
        u = _get(resp, "usage")
        if u is not None:
            self.input_tokens += _get(u, "input_tokens", 0) or 0
            self.output_tokens += _get(u, "output_tokens", 0) or 0


@dataclass
class State:
    frame: dict | None = None
    rounds: list[dict] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)


def lessons_text(lessons: list[dict]) -> str:
    if not lessons:
        return ""
    return ("\nLESSONS FROM YOUR EARLIER EXPERIMENTS (use them; don't repeat what already failed):\n"
            + "\n".join(f"- [{l.get('outcome', '')}] {l.get('approach', '')} on “{l.get('problem', '')[:80]}”: {l.get('lesson', '')}" for l in lessons[:6]))


def solve(client: Any, model: str, helper_model: str, today: str, problem: str, emit: Callable[[str, dict], None],
          budget: str = "standard", starting_code: str = "", tests: str = "", lessons: list[dict] | None = None,
          max_steps: int = 14) -> dict:
    from .loop import Agent
    from .tools import Tool
    plan = BUDGETS.get(budget, BUDGETS["standard"])
    usage = Usage()
    st = State()
    all_runs: list[dict] = []
    box = {"container": None, "stage": "frame"}
    lessons = lessons or []

    def tool_agent(system: str, submit: dict, take, code: bool, container=None, speed_kind="deep", steps=max_steps):
        a = Agent(client, model=model, max_steps=steps, max_tokens=8000, system=system, stream=True,
                  server_tools=[CODE_TOOL] if code else [], strict_grounding=False, record_tools=False, speed=speed_kind, container=container)
        a.cache_system = False
        if submit:
            a.tools.add(Tool(name=submit["name"], description=submit["description"], input_schema=submit["input_schema"], run=take))

        def on_event(kind, data):
            if kind == "server_tool_start":
                emit("status", {"label": {"bash_code_execution": "Running code", "text_editor_code_execution": "Writing code"}.get(data.get("name"), "Working")})
            elif kind == "response_content":
                for r in runs_from([{"role": "assistant", "content": data["content"]}], box["stage"], start=len(all_runs)):
                    all_runs.append(r)
                    emit("run", r)
        a.on_event = on_event
        return a

    # 1. Frame
    emit("stage", {"stage": "frame", "label": "Frame: can success be measured in code?"})

    def take_frame(args):
        st.frame = clean_frame(args, plan["approaches"])
        return {"ok": True, "note": "Received. Reply with one short sentence."}

    extra = ""
    if starting_code.strip():
        extra += f"\n\nSTARTING CODE (the baseline to beat; include it as one approach named 'baseline'):\n{starting_code[:20000]}"
    if tests.strip():
        extra += f"\n\nTHE PERSON'S OWN TESTS (must pass):\n{tests[:20000]}"
    a = tool_agent(FRAME_PROMPT.format(today=today, n=plan["approaches"], lessons=lessons_text(lessons)), SUBMIT_FRAME, take_frame, code=False)
    usage.reply(a.ask(problem + extra))
    if not st.frame:
        raise LabError("I couldn't frame that problem. Try describing what the code should do and what 'better' means.")
    frame = st.frame
    emit("frame", frame)
    if not frame["measurable"] or not frame["approaches"]:
        return {"mode": "solve", "problem": problem, "frame": frame, "measurable": False, "usage": usage.__dict__, "runs": all_runs,
                "rounds": [], "journal": [], "leaderboard": [], "lessons_used": lessons}

    # 2. Seal the hidden tests before any solving
    emit("stage", {"stage": "seal", "label": "Sealing hidden tests the solver will never see"})
    func = func_name(frame["entry"])
    resp = client.messages.create(model=model, max_tokens=6000, system=HIDDEN_PROMPT.format(func=func),
                                  messages=[{"role": "user", "content": f"Problem: {frame['problem']}\nFunction: {frame['entry']}\nCorrect means:\n"
                                             + "\n".join(f"- {r}" for r in frame["requirements"]) + (f"\nThe person's tests (also include these cases):\n{tests[:10000]}" if tests.strip() else "")}])
    usage.resp(resp)
    from .loop import _get
    hidden = strip_fences("".join(_get(b, "text", "") for b in (_get(resp, "content", []) or []) if _get(b, "type") == "text"))
    if "@@HIDDEN" not in hidden or f"import {func}" not in hidden:
        raise LabError("The hidden tests couldn't be written for this problem. Try stating the function's inputs and outputs more precisely.")
    seal = {"sha256": sha256(hidden), "bytes": len(hidden.encode())}
    emit("sealed", seal)

    # 3. Rounds
    all_scores: list[dict] = []
    best_prev: float | None = None
    todo = frame["approaches"]
    direction = frame["metric"]["direction"]
    stop_reason = f"Ran all {plan['rounds']} rounds."
    inputs = (f"STARTING CODE (implement as approach 'baseline'):\n{starting_code[:20000]}\n" if starting_code.strip() else "") + \
             (f"THE PERSON'S TESTS (put them in visible_tests.py too):\n{tests[:20000]}\n" if tests.strip() else "")
    for rnd in range(1, plan["rounds"] + 1):
        box["stage"] = f"round {rnd}"
        emit("stage", {"stage": "round", "round": rnd, "rounds": plan["rounds"], "label": f"Round {rnd}: {len(todo)} approach{'es' if len(todo) != 1 else ''}"})
        got: list[dict] = []

        def take_round(args, got=got):
            for e in (args.get("entries") or [])[:8]:
                if isinstance(e, dict):
                    got.append({"approach": slug(e.get("approach")), "what_happened": _txt(e.get("what_happened"), 400),
                                "lesson": _txt(e.get("lesson"), 400), "next_idea": _txt(e.get("next_idea"), 300)})
            return {"ok": True, "note": "Received. Reply with one short sentence."}

        journal = ("FAILURE JOURNAL SO FAR (scores were measured by code):\n" + "\n".join(
            f"- round {e['round']} {e['approach']}: {e['result']}. Lesson: {e['lesson']}" for e in st.entries[-16:])) if st.entries else ""
        prompt = ROUND_PROMPT.format(problem=frame["problem"], entry=frame["entry"], requirements="; ".join(frame["requirements"]) or "see problem",
                                     metric=f"{frame['metric']['name']} ({frame['metric']['direction']}, {frame['metric']['unit']}): {frame['metric']['how']}",
                                     inputs=inputs, harness=HARNESS_FIRST if rnd == 1 else HARNESS_AGAIN, round=rnd, rounds=plan["rounds"],
                                     todo="\n".join(f"- {t['name']}: {t['idea']}" for t in todo), journal=journal)
        before = len(all_runs)
        a = tool_agent(prompt, SUBMIT_ROUND, take_round, code=True, container=box["container"])
        usage.reply(a.ask(f"Do round {rnd} now."))
        if a.container_id:
            box["container"] = a.container_id
        round_scores = scores(all_runs[before:], rnd)
        all_scores += round_scores
        notes = {g["approach"]: g for g in got}
        for t in todo:
            sc = [s for s in round_scores if s["approach"] == t["name"]]
            s = sc[-1] if sc else None
            result = ("no score printed (it didn't run, or crashed)" if s is None else
                      f"incorrect ({s['passed']}/{s['total']} visible tests)" if not s["correct"] else
                      f"correct, {frame['metric']['name']} = {s['metric']:g}{(' ' + frame['metric']['unit']) if frame['metric']['unit'] else ''}" if s["metric"] is not None else "correct, no metric")
            n = notes.get(t["name"], {})
            st.entries.append({"round": rnd, "approach": t["name"], "idea": t["idea"], "result": result,
                               "correct": bool(s and s["correct"]), "metric": s["metric"] if s else None, "run": s["run"] if s else None,
                               "lesson": n.get("lesson") or "No lesson recorded.", "next_idea": n.get("next_idea", ""), "what_happened": n.get("what_happened", "")})
        board = leaderboard(all_scores, direction)
        best = next((b["metric"] for b in board if b["correct"] and b["metric"] is not None), None)
        gain = improvement(best_prev, best, direction)
        st.rounds.append({"round": rnd, "approaches": [t["name"] for t in todo], "best": best, "gain": round(gain, 4)})
        emit("round", {"round": rnd, "leaderboard": board, "journal": st.entries, "best": best, "gain": gain})
        if rnd > 1 and gain < PLATEAU:
            stop_reason = f"Stopped after round {rnd}: the best score improved by less than {int(PLATEAU * 100)}%."
            break
        best_prev = best
        if rnd == plan["rounds"]:
            break
        # next round: refine the two leaders with the journal's ideas, plus one fresh idea if a slot is left
        leaders = [b for b in board if b["correct"]][:2] or board[:1]
        todo = []
        for b in leaders:
            idea = next((e["next_idea"] for e in reversed(st.entries) if e["approach"] == b["approach"] and e["next_idea"]), "")
            todo.append({"name": slug(f"{b['approach']}_r{rnd + 1}"), "idea": f"Refine {b['approach']} (best so far: {b['metric']}). {idea or 'Use the journal to make it faster or more robust.'}"})
        fresh = [e["next_idea"] for e in st.entries if e["round"] == rnd and e["next_idea"] and not e["correct"]]
        if len(todo) < plan["approaches"]:
            todo.append({"name": slug(f"new_idea_r{rnd + 1}"), "idea": fresh[0] if fresh else "A genuinely different approach than any tried so far, informed by the journal."})

    board = leaderboard(all_scores, direction)
    finalists = [b for b in board if b["correct"]][:2]

    # 4. Read the finalists' exact code back from the sandbox, then run the sealed tests in a fresh sandbox
    sources: dict[str, str] = {}
    if finalists and box["container"]:
        box["stage"] = "collect"
        emit("stage", {"stage": "verify", "label": "Verifying finalists on the sealed tests in a fresh sandbox"})
        before = len(all_runs)
        a = tool_agent("Print files exactly as asked with bash. Nothing else.", None, None, code=True, container=box["container"], speed_kind="helper", steps=4)
        a.model = helper_model
        usage.reply(a.ask("Run exactly this one command: " + " ; ".join(f"echo '@@FILE {f['approach']}'; cat /work/approaches/{f['approach']}.py; echo '@@END'" for f in finalists)))
        for r in all_runs[before:]:
            for m in re.finditer(r"@@FILE (\S+)\n(.*?)\n?@@END", r.get("stdout", "") if r.get("status") == "ok" else "", re.S):
                sources[m.group(1)] = m.group(2)
    verified = []
    for f in finalists:
        src = sources.get(f["approach"])
        if not src:
            verified.append({**f, "hidden": None, "note": "Its code couldn't be read back from the sandbox."})
            continue
        box["stage"] = f"verify {f['approach']}"
        before = len(all_runs)
        a = tool_agent(VERIFY_PROMPT, None, None, code=True, container=None, speed_kind="helper", steps=6)
        a.model = helper_model
        usage.reply(a.ask(f"FILE /work/solution.py\n<<<\n{src}\n>>>\n\nFILE /work/hidden_tests.py\n<<<\n{hidden}\n>>>\n\nRUN\ncd /work && python hidden_tests.py"))
        res = None
        for r in all_runs[before:]:
            if r.get("kind") == "bash" and r.get("status") == "ok":
                for m in HIDDEN_RE.finditer(r.get("stdout", "")):
                    try:
                        d = json.loads(m.group(1))
                        res = {"passed": int(d.get("passed", 0)), "total": int(d.get("total", 0)),
                               "failures": [_txt(x, 200) for x in (d.get("failures") or [])[:5]], "run": r["n"]}
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        verified.append({**f, "hidden": res, "source": src})
    winner = next((v for v in verified if v.get("hidden") and v["hidden"]["total"] > 0 and v["hidden"]["passed"] == v["hidden"]["total"]), None)
    if winner:
        status = "verified"
    elif finalists:
        status = "failed_hidden"
    else:
        status = "no_solution"
    return {"mode": "solve", "problem": problem, "frame": frame, "measurable": True, "seal": seal, "hidden_tests": hidden,
            "rounds": st.rounds, "journal": st.entries, "leaderboard": board, "finalists": verified, "winner": winner,
            "status": status, "stop_reason": stop_reason, "runs": all_runs, "container": box["container"], "usage": usage.__dict__,
            "lessons_used": lessons, "budget": budget}


STATUS_WORDS = {"verified": "Verified solution", "failed_hidden": "Passed its own tests, failed the sealed ones",
                "no_solution": "No correct solution found", "unmeasurable": "Can't be measured in code"}


def lessons_from(rec: dict) -> list[dict]:
    """What to remember for next time: one line per approach, with the outcome measured by code."""
    out = []
    problem = rec.get("frame", {}).get("problem") or rec.get("problem", "")
    won = (rec.get("winner") or {}).get("approach")
    for e in rec.get("journal", []):
        outcome = "won" if e["approach"] == won else ("correct" if e["correct"] else "failed")
        out.append({"problem": problem[:200], "approach": e["approach"], "idea": e["idea"][:200], "outcome": outcome,
                    "result": e["result"][:160], "lesson": e["lesson"][:300]})
    return out[:20]


WORD = re.compile(r"[a-z]{3,}")
STOP = set("the and for with that this from into your you are how what when make find best code function list number numbers using use".split())


def relevant(lessons: list[dict], problem: str, k: int = 6) -> list[dict]:
    """The earlier lessons that share the most words with this problem (failures and winners first)."""
    want = set(WORD.findall(problem.lower())) - STOP
    scored = []
    for l in lessons:
        words = set(WORD.findall((l.get("problem", "") + " " + l.get("idea", "") + " " + l.get("lesson", "")).lower())) - STOP
        overlap = len(want & words)
        if overlap >= 2:
            scored.append((overlap + (1 if l.get("outcome") in ("won", "failed") else 0), l))
    scored.sort(key=lambda x: -x[0])
    return [l for _, l in scored[:k]]


def script(rec: dict) -> str:
    """The winning code plus the sealed tests, runnable anywhere: python solve_check.py"""
    w = rec.get("winner") or (rec.get("finalists") or [{}])[0]
    if not w or not w.get("source"):
        return "# No verified solution to export."
    return ("# Discovery Lab, Solve mode: the winning approach and the sealed hidden tests, exactly as they ran.\n"
            f"# Sealed tests SHA-256: {rec['seal']['sha256']}\n# Save the two parts as solution.py and hidden_tests.py, then run: python hidden_tests.py\n\n"
            f"# ===== solution.py ({w['approach']}) =====\n{w['source'].rstrip()}\n\n# ===== hidden_tests.py =====\n"
            + "\n".join("# " + line for line in rec["hidden_tests"].splitlines()) + "\n")


def report_markdown(rec: dict) -> str:
    f = rec["frame"]
    lines = [f"# {rec.get('title') or 'Solve report'}", "", f"**Problem.** {f.get('problem') or rec['problem']}", ""]
    if not rec.get("measurable"):
        return "\n".join(lines + ["## Can't be measured in code", "", f.get("why_not") or "", ""])
    m = f["metric"]
    lines += [f"**Goal.** Correct on every test, then {'lowest' if m['direction'] == 'min' else 'highest'} {m['name']}"
              + (f" ({m['unit']})" if m["unit"] else "") + f": {m['how']}", "",
              f"**Result.** {STATUS_WORDS.get(rec['status'], rec['status'])}. {rec.get('stop_reason', '')}", ""]
    w = rec.get("winner")
    if w:
        lines += [f"**Winner.** `{w['approach']}`, {m['name']} = {w['metric']:g} {m['unit']}, sealed tests {w['hidden']['passed']}/{w['hidden']['total']}.", ""]
    lines += ["## Leaderboard", "", "| Approach | Round | Visible tests | " + m["name"] + " |", "|---|---|---|---|"]
    for b in rec["leaderboard"]:
        value = "" if b["metric"] is None else f"{b['metric']:g}"
        lines.append(f"| {b['approach']} | {b['round']} | {b['passed']}/{b['total']} | {value} |")
    lines += ["", "## Experiment journal", ""]
    for e in rec["journal"]:
        lines.append(f"- Round {e['round']}, **{e['approach']}**: {e['result']}. {e['lesson']}")
    lines += ["", "## Sealed hidden tests", "", f"SHA-256 `{rec['seal']['sha256']}`, fixed before solving began. "
              "Hash the hidden_tests.py in the download to check it's the same file.", ""]
    for v in rec.get("finalists", []):
        h = v.get("hidden")
        lines.append(f"- {v['approach']}: " + (f"{h['passed']}/{h['total']} passed" + (f" (failures: {'; '.join(h['failures'])})" if h['failures'] else "") if h else v.get("note", "not run")))
    lines += ["", f"{len(rec['runs'])} sandbox actions. Scores count only when code that ran printed them."]
    return "\n".join(lines) + "\n"
