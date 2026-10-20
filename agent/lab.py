"""Discovery Lab: investigate a question with hypotheses, real experiments and a deliberate attempt to break them.

Three stages, each a short agent run:
  Discover   - look up what's already known (web search) and state 1-3 testable hypotheses.
  Experiment - write and run Python (NumPy, SciPy, SymPy) in Anthropic's code execution sandbox. The sandbox is
               isolated from this server: no network, no access to our keys or database.
  Challenge  - in the same sandbox, try hard to disprove each hypothesis: counterexamples, edge cases, other
               explanations.

The integrity rule: a result counts only if code that actually ran printed it. Experiment code reports evidence as
one-line JSON markers in its output:

    @@EVIDENCE {"h": "H1", "kind": "exhaustive", "result": "supports", "complete": false, "detail": "checked n <= 10^6"}
    @@CHART {"title": "...", "type": "line", "x": [...], "series": [{"name": "...", "y": [...]}]}

Labels are computed from those markers by code, never taken from the model's prose:
  Disproven              - a counterexample (or an exhaustive check) that ran successfully refutes it.
  Proven                 - a symbolic proof (SymPy reduced it to an identity) or an exhaustive check of the whole
                           claimed domain ("complete": true), and nothing refutes it.
  Experimental evidence  - tested and supported on the cases checked, but not proven for all cases.
  Hypothesis             - untested, inconclusive, or the evidence conflicts.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .loop import _get

CODE_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
EVIDENCE_KINDS = ("exhaustive", "numerical", "symbolic", "simulation", "counterexample", "benchmark", "statistical")
RESULTS = ("supports", "refutes", "inconclusive")
LABELS = {"proven": "Proven", "evidence": "Experimental evidence", "hypothesis": "Hypothesis", "disproven": "Disproven"}
MAX_HYPOTHESES = 3
MAX_CHART_POINTS = 2000

MARKERS = """Every result must come from code that runs. In your Python, print one line per finding:
@@EVIDENCE {"h": "H1", "kind": "<exhaustive|numerical|symbolic|simulation|counterexample|benchmark|statistical>", "result": "<supports|refutes|inconclusive>", "complete": <true only if you checked the ENTIRE claimed domain, e.g. a finite claim>, "detail": "<what was checked and what was found, with the key numbers>"}
For a counterexample, put the actual values in detail. For a SymPy proof, use kind "symbolic" and only when simplify/equals gave an exact identity.
To draw a chart, print: @@CHART {"title": "...", "type": "line|bar|scatter", "x": [...], "series": [{"name": "...", "y": [...]}], "xlabel": "...", "ylabel": "..."} (at most 2000 points; no matplotlib images).
Print markers with json.dumps so they're valid JSON. Keep other output short: no huge arrays."""

DISCOVER_PROMPT = """You are the Discover stage of Ask Harsha's Discovery Lab, a research partner that investigates questions \
with real experiments. Today is {today}.

Read the question. If it helps, use web_search (at most 2 searches) to learn what's already known: theorems, best \
algorithms, records, standard baselines. Then call submit_plan once with:
- question: the question restated precisely, with every term defined.
- known: up to 5 established facts that matter, each with its source URL if you found one (empty list if none).
- hypotheses: 1 to 3 hypotheses that code can test in a few minutes on 1 CPU with NumPy, SciPy and SymPy. Each needs a \
precise statement, how you'll test it, and what result would disprove it. Prefer one bold hypothesis plus a careful \
one. If the question is too vague, make it concrete and say how in the statement.
Don't claim anything is true yet. You have no code tool in this stage."""

EXPERIMENT_PROMPT = """You are the Experiment stage of Ask Harsha's Discovery Lab. You have a Python sandbox \
(code_execution): Python 3.11 with numpy, scipy, sympy, pandas; 1 CPU, 5 GB memory, no internet. Write a script to a file \
and run it with bash, fixing errors as you go. Keep each run under about a minute (use smaller sizes first).

The plan:
{plan}

Test every hypothesis. Use exact arithmetic or SymPy where you can; use exhaustive checks over a stated range; compare \
against a sensible baseline for algorithm questions; use a fixed random seed for anything random.

{markers}

When the experiments are done, call submit_experiments with one short entry per hypothesis: the method in plain words \
and what the output showed. Never report a number that your code didn't print."""

CHALLENGE_PROMPT = """You are the Challenge stage of Ask Harsha's Discovery Lab. Your job is to break the results below: \
assume they're wrong and try to prove it. The same Python sandbox is available (files from the experiments are still \
there).

The plan:
{plan}

What the experiments found:
{found}

For each hypothesis that's still standing, try at least two attacks with code: hunt for counterexamples (bigger ranges, \
special values, edge cases like 0, 1, negatives, duplicates, worst-case inputs), test the assumptions, and check a \
competing explanation. Report each attack with an @@EVIDENCE marker ("refutes" if it breaks the hypothesis, "supports" if \
the hypothesis survived it, "inconclusive" otherwise).

{markers}

Then call submit_report with: one entry per attack (what you tried and what happened), a summary that a smart \
non-expert can follow (3 to 6 sentences, no claims beyond the evidence), the limitations, and up to 3 next steps."""

SUBMIT_PLAN = {
    "name": "submit_plan",
    "description": "Hand the research plan to the Experiment stage.",
    "input_schema": {"type": "object", "properties": {
        "question": {"type": "string"},
        "known": {"type": "array", "items": {"type": "object", "properties": {"fact": {"type": "string"}, "source": {"type": "string"}}, "required": ["fact"]}},
        "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string", "description": "H1, H2 or H3"}, "statement": {"type": "string"},
            "test": {"type": "string"}, "falsifier": {"type": "string"}}, "required": ["id", "statement", "test", "falsifier"]}},
    }, "required": ["question", "hypotheses"]},
}
SUBMIT_EXPERIMENTS = {
    "name": "submit_experiments",
    "description": "Report what the experiments showed, one entry per hypothesis.",
    "input_schema": {"type": "object", "properties": {"findings": {"type": "array", "items": {"type": "object", "properties": {
        "h": {"type": "string"}, "method": {"type": "string"}, "finding": {"type": "string"}}, "required": ["h", "method", "finding"]}}},
        "required": ["findings"]},
}
SUBMIT_REPORT = {
    "name": "submit_report",
    "description": "The final report after trying to break the results.",
    "input_schema": {"type": "object", "properties": {
        "attacks": {"type": "array", "items": {"type": "object", "properties": {
            "h": {"type": "string"}, "attack": {"type": "string"}, "outcome": {"type": "string"}}, "required": ["h", "attack", "outcome"]}},
        "summary": {"type": "string"}, "limitations": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}}},
        "required": ["attacks", "summary"]},
}


def _txt(v, n: int) -> str:
    return " ".join(str(v or "").split())[:n]


def clean_plan(raw: dict) -> dict:
    hyps = []
    for i, h in enumerate((raw.get("hypotheses") or [])[:MAX_HYPOTHESES]):
        if not isinstance(h, dict) or not _txt(h.get("statement"), 600):
            continue
        hyps.append({"id": f"H{len(hyps) + 1}", "statement": _txt(h.get("statement"), 600), "test": _txt(h.get("test"), 600),
                     "falsifier": _txt(h.get("falsifier"), 400)})
    known = [{"fact": _txt(k.get("fact"), 400), "source": _txt(k.get("source"), 400)}
             for k in (raw.get("known") or [])[:5] if isinstance(k, dict) and _txt(k.get("fact"), 400)]
    return {"question": _txt(raw.get("question"), 800), "known": known, "hypotheses": hyps}


def plan_text(plan: dict) -> str:
    lines = [f"Question: {plan['question']}"]
    for h in plan["hypotheses"]:
        lines.append(f"{h['id']}: {h['statement']}\n  Test: {h['test']}\n  Disproved if: {h['falsifier']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- what actually ran in the sandbox
def runs_from(messages: list[dict], stage: str, start: int = 0) -> list[dict]:
    """Every code action in the conversation, paired with its sandbox result, in order."""
    uses: dict[str, dict] = {}
    order: list[str] = []
    results: dict[str, Any] = {}
    for m in messages:
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            t = _get(b, "type")
            if t == "server_tool_use" and _get(b, "name") in ("bash_code_execution", "text_editor_code_execution"):
                uses[_get(b, "id")] = {"name": _get(b, "name"), "input": dict(_get(b, "input") or {})}
                order.append(_get(b, "id"))
            elif t in ("bash_code_execution_tool_result", "text_editor_code_execution_tool_result"):
                results[_get(b, "tool_use_id")] = _get(b, "content")
    out = []
    for uid in order:
        use, res = uses[uid], results.get(uid)
        inp = use["input"]
        run = {"n": start + len(out) + 1, "stage": stage}
        if use["name"] == "bash_code_execution":
            run.update(kind="bash", command=str(inp.get("command", ""))[:4000])
        else:
            run.update(kind="file", action=str(inp.get("command", "")), path=str(inp.get("path", ""))[:300],
                       code=str(inp.get("file_text") or inp.get("new_str") or "")[:60000])
        if res is None:
            run.update(status="no result")
        elif str(_get(res, "type", "")).endswith("_error"):
            run.update(status="error", error=str(_get(res, "error_code", "error")))
        else:
            run.update(status="ok", stdout=str(_get(res, "stdout", "") or "")[:20000], stderr=str(_get(res, "stderr", "") or "")[:4000],
                       return_code=_get(res, "return_code"))
            if run["kind"] == "bash" and run["return_code"] not in (0, None):
                run["status"] = "failed"
        out.append(run)
    return out


MARKER = re.compile(r"^@@(EVIDENCE|CHART)\s+(\{.*\})\s*$", re.M)


def markers(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Evidence and charts printed by code that ran successfully. Anything else doesn't count."""
    ev, charts = [], []
    for r in runs:
        if r.get("kind") != "bash" or r.get("status") != "ok":
            continue
        for m in MARKER.finditer(r.get("stdout", "")):
            try:
                d = json.loads(m.group(2))
            except json.JSONDecodeError:
                continue
            if m.group(1) == "EVIDENCE":
                h = str(d.get("h", "")).upper().strip()
                if not re.fullmatch(r"H[1-9]", h):
                    continue
                kind = d.get("kind") if d.get("kind") in EVIDENCE_KINDS else "numerical"
                result = d.get("result") if d.get("result") in RESULTS else "inconclusive"
                ev.append({"h": h, "kind": kind, "result": result, "complete": bool(d.get("complete")),
                           "detail": _txt(d.get("detail"), 500), "run": r["n"], "stage": r["stage"]})
            else:
                c = clean_chart(d, r["n"])
                if c:
                    charts.append(c)
    return ev, charts[:8]


def clean_chart(d: dict, run: int) -> dict | None:
    x = d.get("x")
    series = d.get("series")
    if not isinstance(x, list) or not isinstance(series, list) or not x:
        return None
    x = x[:MAX_CHART_POINTS]
    clean = []
    for s in series[:6]:
        y = s.get("y") if isinstance(s, dict) else None
        if not isinstance(y, list):
            continue
        ys = [float(v) if isinstance(v, (int, float)) and v == v else None for v in y[:len(x)]]
        clean.append({"name": _txt(s.get("name"), 60) or f"series {len(clean) + 1}", "y": ys})
    if not clean:
        return None
    xs = [v if isinstance(v, (int, float)) else _txt(v, 30) for v in x]
    return {"title": _txt(d.get("title"), 120), "type": d.get("type") if d.get("type") in ("line", "bar", "scatter") else "line",
            "x": xs, "series": clean, "xlabel": _txt(d.get("xlabel"), 60), "ylabel": _txt(d.get("ylabel"), 60), "run": run}


def label(h: str, evidence: list[dict]) -> tuple[str, str]:
    """(label key, why) for one hypothesis, from code-printed evidence only."""
    mine = [e for e in evidence if e["h"] == h]
    refutes = [e for e in mine if e["result"] == "refutes"]
    supports = [e for e in mine if e["result"] == "supports"]
    hard_refute = [e for e in refutes if e["kind"] in ("counterexample", "exhaustive", "symbolic")]
    if hard_refute:
        e = hard_refute[0]
        return "disproven", f"Code found a {'counterexample' if e['kind'] == 'counterexample' else e['kind'] + ' refutation'} (run {e['run']}): {e['detail']}"
    if refutes and supports:
        return "hypothesis", f"The evidence conflicts: {len(supports)} result(s) support it and {len(refutes)} don't (e.g. run {refutes[0]['run']}: {refutes[0]['detail']})."
    if refutes:
        return "hypothesis", f"Tests didn't support it (run {refutes[0]['run']}): {refutes[0]['detail']}"
    proofs = [e for e in supports if e["kind"] == "symbolic" or (e["kind"] == "exhaustive" and e["complete"])]
    if proofs:
        e = proofs[0]
        how = "SymPy reduced it to an exact identity" if e["kind"] == "symbolic" else "every case in the claimed domain was checked"
        return "proven", f"{how[0].upper() + how[1:]} (run {e['run']}): {e['detail']}"
    if supports:
        kinds = sorted({e["kind"] for e in supports})
        return "evidence", (f"Supported by {len(supports)} {'/'.join(kinds)} result(s) that survived the challenges, but not proven for "
                            f"every case. Strongest: run {supports[-1]['run']}: {supports[-1]['detail']}")
    return "hypothesis", "No code result tested it yet."


def verdicts(plan: dict, evidence: list[dict]) -> list[dict]:
    out = []
    for h in plan["hypotheses"]:
        key, why = label(h["id"], evidence)
        out.append({**h, "label": key, "label_word": LABELS[key], "why": why,
                    "evidence": [e for e in evidence if e["h"] == h["id"]]})
    return out


def reproducible_script(runs: list[dict]) -> str:
    """The experiments as one Python file a person can run: each script the sandbox created, then the commands."""
    parts = ["# Discovery Lab experiments, exactly as they ran in the sandbox.", "# Run with Python 3.11+ and numpy, scipy, sympy installed.", ""]
    for r in runs:
        if r["kind"] == "file" and r.get("action") == "create" and r.get("code"):
            parts += [f"# ---- {r['path']} (run {r['n']}, {r['stage']})", r["code"].rstrip(), ""]
        elif r["kind"] == "bash":
            parts += [f"# $ {line}" for line in r["command"].splitlines()[:5]] + [""]
    return "\n".join(parts)


def evidence_fingerprint(evidence: list[dict]) -> str:
    keys = sorted(json.dumps({k: e[k] for k in ("h", "kind", "result", "complete")}, sort_keys=True) for e in evidence)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def report_markdown(run: dict) -> str:
    """The research report as Markdown (for the PDF and .md downloads)."""
    plan, rep = run["plan"], run.get("report") or {}
    lines = [f"# {run.get('title') or 'Discovery Lab report'}", "", f"**Question.** {plan['question']}", ""]
    if rep.get("summary"):
        lines += ["## Summary", "", rep["summary"], ""]
    lines += ["## Findings", ""]
    for v in run["verdicts"]:
        lines += [f"### {v['id']}: {v['label_word']}", "", v["statement"], "", f"**Why this label:** {v['why']}", ""]
        for e in v["evidence"]:
            lines.append(f"- Run {e['run']} ({e['stage']}, {e['kind']}): {e['result']}. {e['detail']}")
        lines.append("")
    if plan.get("known"):
        lines += ["## What was already known", ""] + [f"- {k['fact']}" + (f" ({k['source']})" if k.get("source") else "") for k in plan["known"]] + [""]
    if rep.get("attacks"):
        lines += ["## Challenges", ""] + [f"- **{a['h']}:** {a['attack']} → {a['outcome']}" for a in rep["attacks"]] + [""]
    if rep.get("limitations"):
        lines += ["## Limitations", ""] + [f"- {x}" for x in rep["limitations"]] + [""]
    if rep.get("next_steps"):
        lines += ["## Next steps", ""] + [f"- {x}" for x in rep["next_steps"]] + [""]
    lines += ["## How to read the labels", "",
              "- **Proven:** an exact symbolic identity, or every case in the claimed domain checked by code.",
              "- **Experimental evidence:** supported on the cases tested, not proven for all cases.",
              "- **Hypothesis:** untested, inconclusive, or the evidence conflicts.",
              "- **Disproven:** code found a counterexample.", "",
              f"{len(run['runs'])} sandbox actions. Every finding above was printed by code that ran; the full code is in the "
              "lab-experiments.py download."]
    return "\n".join(lines)


@dataclass
class Submission:
    """Collects what a stage hands back through its submit tool."""
    plan: dict | None = None
    findings: list[dict] = field(default_factory=list)
    report: dict | None = None


# ---------------------------------------------------------------- the pipeline
def investigate(client: Any, model: str, today: str, question: str, emit, web_tool: dict | None = None,
                prior: dict | None = None, container: str | None = None, max_steps: int = 10) -> dict:
    """Run Discover → Experiment → Challenge and return the full research record.

    `emit(kind, data)` streams progress: stage, status, plan, run, chart, verdicts. `prior` is an earlier run this one
    continues; `container` reuses its sandbox (its files are still there for up to 30 days).
    """
    from .loop import Agent
    from .tools import Tool
    sub = Submission()
    usage = {"input_tokens": 0, "output_tokens": 0, "web_searches": 0, "steps": 0}
    all_runs: list[dict] = []

    def agent_for(system: str, tools: list[dict], submit: dict, key: str, code: bool):
        a = Agent(client, model=model, max_steps=max_steps, max_tokens=8000, system=system, stream=True,
                  server_tools=([CODE_TOOL] if code else []) + tools, strict_grounding=False, record_tools=False,
                  speed="deep", container=container_box["id"])

        def take(args: dict) -> dict:
            if key == "plan":
                sub.plan = clean_plan(args)
            elif key == "experiments":
                sub.findings = [{"h": _txt(f.get("h"), 4).upper(), "method": _txt(f.get("method"), 500), "finding": _txt(f.get("finding"), 800)}
                                for f in (args.get("findings") or [])[:6] if isinstance(f, dict)]
            else:
                sub.report = {"attacks": [{"h": _txt(x.get("h"), 4).upper(), "attack": _txt(x.get("attack"), 500), "outcome": _txt(x.get("outcome"), 500)}
                                          for x in (args.get("attacks") or [])[:12] if isinstance(x, dict)],
                              "summary": _txt(args.get("summary"), 1500),
                              "limitations": [_txt(x, 300) for x in (args.get("limitations") or [])[:6] if _txt(x, 300)],
                              "next_steps": [_txt(x, 300) for x in (args.get("next_steps") or [])[:3] if _txt(x, 300)]}
            return {"ok": True, "note": "Received. Finish with one short sentence; don't repeat it."}
        a.tools.add(Tool(name=submit["name"], description=submit["description"], input_schema=submit["input_schema"], run=take))

        def on_event(kind, data):
            if kind == "server_tool_start":
                emit("status", {"label": {"bash_code_execution": "Running code", "text_editor_code_execution": "Writing code",
                                          "web_search": "Searching the web"}.get(data.get("name"), "Working")})
            elif kind == "web_search_start":
                emit("status", {"label": "Searching the web"})
            elif kind == "response_content":
                for r in runs_from([{"role": "assistant", "content": data["content"]}], stage_box["name"], start=len(all_runs)):
                    all_runs.append(r)
                    emit("run", r)
                    for c in markers([r])[1]:
                        emit("chart", c)
        a.on_event = on_event
        return a

    def run_stage(a, prompt_text: str):
        reply = a.ask(prompt_text)
        usage["input_tokens"] += reply.input_tokens
        usage["output_tokens"] += reply.output_tokens
        usage["web_searches"] += reply.web_searches
        usage["steps"] += reply.steps
        if a.container_id:
            container_box["id"] = a.container_id
        return reply

    container_box = {"id": container}
    stage_box = {"name": "discover"}

    # 1. Discover
    emit("stage", {"stage": "discover", "label": "Discover: what's known, and what to test"})
    ask = question
    if prior:
        ask = (f"This continues earlier research.\nEarlier question: {prior['plan']['question']}\nEarlier findings:\n"
               + "\n".join(f"- {v['id']} ({v['label_word']}): {v['statement']}. {v['why']}" for v in prior.get("verdicts", []))
               + f"\nEarlier summary: {(prior.get('report') or {}).get('summary', '')}\n\nNew request: {question}")
    a1 = agent_for(DISCOVER_PROMPT.format(today=today), [web_tool] if web_tool else [], SUBMIT_PLAN, "plan", code=False)
    run_stage(a1, ask)
    if not sub.plan or not sub.plan["hypotheses"]:
        raise LabError("I couldn't turn that into something testable. Try a more specific question, like a claim about numbers or an algorithm.")
    plan = sub.plan
    emit("plan", plan)

    # 2. Experiment
    stage_box["name"] = "experiment"
    emit("stage", {"stage": "experiment", "label": "Experiment: running code in the sandbox"})
    a2 = agent_for(EXPERIMENT_PROMPT.format(plan=plan_text(plan), markers=MARKERS), [], SUBMIT_EXPERIMENTS, "experiments", code=True)
    run_stage(a2, "Run the experiments now.")
    ev, _ = markers(all_runs)
    emit("verdicts", {"stage": "experiment", "verdicts": verdicts(plan, ev)})

    # 3. Challenge
    stage_box["name"] = "challenge"
    emit("stage", {"stage": "challenge", "label": "Challenge: trying to break the results"})
    found = "\n".join(f"- {v['id']} ({v['label_word']}): {v['why']}" for v in verdicts(plan, ev))
    found += "\n" + "\n".join(f"- {f['h']} method: {f['method']}. Finding: {f['finding']}" for f in sub.findings)
    a3 = agent_for(CHALLENGE_PROMPT.format(plan=plan_text(plan), found=found, markers=MARKERS), [], SUBMIT_REPORT, "report", code=True)
    run_stage(a3, "Try to break these results now.")

    ev, charts = markers(all_runs)
    final = verdicts(plan, ev)
    return {"question": question, "plan": plan, "findings": sub.findings, "report": sub.report or {"attacks": [], "summary": "", "limitations": [], "next_steps": []},
            "runs": all_runs, "evidence": ev, "charts": charts, "verdicts": final, "container": container_box["id"],
            "usage": usage, "evidence_fingerprint": evidence_fingerprint(ev)}


class LabError(Exception):
    """The investigation couldn't go on; the message is for the person."""


REPRODUCE_PROMPT = """You re-run experiments exactly as recorded, to check they're reproducible. Use the code_execution \
tool. Recreate each file with exactly the content given (no edits), then run each command in the same order. Don't fix, \
improve or add anything. When done, reply with one word: done."""


def reproduce(client: Any, model: str, runs: list[dict]) -> dict:
    """Replay the recorded file writes and commands in a fresh sandbox and compare the evidence they print."""
    from .loop import Agent
    steps = []
    for r in runs:
        if r["kind"] == "file" and r.get("action") == "create" and r.get("code"):
            steps.append(f"FILE {r['path']}\n<<<\n{r['code']}\n>>>")
        elif r["kind"] == "bash" and r.get("status") in ("ok", "failed"):
            steps.append(f"RUN\n{r['command']}")
    if not steps:
        raise LabError("There's no code to re-run in this investigation.")
    a = Agent(client, model=model, max_steps=12, max_tokens=8000, system=REPRODUCE_PROMPT, stream=False,
              server_tools=[CODE_TOOL], strict_grounding=False, record_tools=False, speed="helper")
    reply = a.ask("Recreate and run, in order:\n\n" + "\n\n".join(steps)[:150_000])
    new_runs = runs_from(a.messages, "reproduce")
    before, _ = markers(runs)
    after, _ = markers(new_runs)
    same = evidence_fingerprint(before) == evidence_fingerprint(after)
    return {"same": same, "before": len(before), "after": len(after), "runs": len(new_runs),
            "fingerprint_before": evidence_fingerprint(before), "fingerprint_after": evidence_fingerprint(after),
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}
