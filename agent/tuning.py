"""Tune: Ask Harsha measures its own settings and proposes faster ones, without spending more or answering worse.

  1. Read     the performance history (median and slowest answers, time to first words, cost, model calls).
  2. Propose  a small model suggests a few candidate settings, reading the journal of earlier tuning runs so it
              doesn't retry what already lost.
  3. Measure  the current settings and each candidate answer the same fixed benchmark questions through the real
              pipeline (model, tools, trust check), with real API calls. The current settings and the best candidate
              are then measured a second time, interleaved, so one lucky run can't win.
  4. Decide   a candidate wins only if it is at least 10% faster (median full answer), costs no more (5% slack for
              noise), and gets every benchmark question right that the current settings got right.
  5. Propose a change   the winner becomes agent/tuned.json in a GitHub pull request with the numbers. Nothing changes
              until you merge it; Render then deploys it.

What Tune may change is deliberately narrow: how hard the chat model thinks, whether simple questions go to the fast
model, and how many web searches an answer may run. It never touches the trust check, citations, grounding, Proof
Mode or any safety rule, and environment variables (CHAT_EFFORT, CHAT_THINKING) always override tuned values.
"""
from __future__ import annotations

import base64
import json
import os
import re
import statistics
import urllib.error
import urllib.request
from pathlib import Path

from . import speed
from .loop import _get

TUNED_PATH = Path(__file__).resolve().parent / "tuned.json"
KNOBS = {
    "chat_effort": {"values": ["low", "medium", "high"], "default": "medium",
                    "what": "How hard the chat model thinks before answering (Sonnet output effort)."},
    "chat_thinking": {"values": ["between_tools", "adaptive"], "default": "between_tools",
                      "what": "between_tools skips up-front thinking; adaptive lets it think first on hard questions."},
    "simple_to_fast_model": {"values": [False, True], "default": False,
                             "what": "Send short, simple questions (no Harsha, no math/money/health/law, nothing current) to the fast model."},
    "web_max_searches": {"values": [1, 2, 3], "default": 3, "what": "The most web searches one answer may run."},
}
FASTER_BY = 0.10       # a winner must be at least this much faster (median full answer)
COST_SLACK = 1.05      # and cost at most this much more (noise)

BENCH = [
    {"id": "fact-capital", "q": "What's the capital of Australia?", "expect": ["Canberra"]},
    {"id": "fact-water", "q": "At what temperature does water boil at sea level, in Celsius?", "expect": ["100"]},
    {"id": "math-percent", "q": "What is 15% of 240?", "expect": ["36"]},
    {"id": "math-days", "q": "How many hours are in a week?", "expect": ["168"]},
    {"id": "explain-dns", "q": "In two sentences, what does DNS do?", "expect": ["domain", "IP", "address"]},
    {"id": "howto-git", "q": "How do I undo my last git commit but keep the changes?", "expect": ["reset", "--soft"]},
    {"id": "harsha-role", "q": "What does Harsha do at IBM?", "expect": ["product"], "cite": True},
    {"id": "harsha-certs", "q": "Which certifications does Harsha have?", "expect": ["CSPO", "Salesforce", "SAFe"], "cite": True},
    {"id": "write-subject", "q": "Write a short, friendly email subject line for a team lunch invite.", "expect": [""]},
    {"id": "web-ceo", "q": "Who is the current CEO of Microsoft?", "expect": ["Nadella"], "web": True},
]

SIMPLE_BLOCK = re.compile(r"\b(harsha|his|he|him|resume|career|calculate|how much|how many|percent|%|price|cost|interest|tax|loan|"
                          r"invest|dose|symptom|medic|legal|law|contract|latest|today|current|news|now|this week|code|function|script|"
                          r"debug|error|file|spreadsheet|pdf|study|research|prove)\b", re.I)


def defaults() -> dict:
    return {k: v["default"] for k, v in KNOBS.items()}


def clean(cfg: dict | None) -> dict:
    out = defaults()
    for k, spec in KNOBS.items():
        if cfg and k in cfg and cfg[k] in spec["values"]:
            out[k] = cfg[k]
    return out


def load_tuned(path: Path = TUNED_PATH) -> dict:
    """The settings Tune last proposed and you merged (agent/tuned.json), or the defaults."""
    try:
        return clean(json.loads(path.read_text(encoding="utf-8")).get("settings"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return defaults()


def env_locked() -> list[str]:
    """Settings an environment variable already fixes (env always wins over tuned values)."""
    return [k for k, env in (("chat_effort", "CHAT_EFFORT"), ("chat_thinking", "CHAT_THINKING")) if os.environ.get(env)]


def speed_override(cfg: dict) -> dict:
    """What to pass to the agent so its chat calls use these settings (unless an env var fixes them)."""
    locked = env_locked()
    o = {}
    if "chat_effort" not in locked:
        o["effort"] = cfg["chat_effort"]
    if "chat_thinking" not in locked:
        o["thinking"] = cfg["chat_thinking"]
    return o


def is_simple(question: str) -> bool:
    q = question.strip()
    return 0 < len(q) <= 200 and not SIMPLE_BLOCK.search(q) and q.count("\n") < 2


# ---------------------------------------------------------------- measuring
def score(item: dict, text: str, cited: list[str], grounded: bool, trust_status: str | None) -> bool:
    low = (text or "").lower()
    if not low.strip():
        return False
    ok = any(e.lower() in low for e in item["expect"])   # "" means any non-empty answer counts
    if item.get("cite"):
        ok = ok and bool(cited) and grounded
    if trust_status == "contradicted":
        ok = False
    return ok


def summarize(rows: list[dict]) -> dict:
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    first = [r["first_text_s"] for r in rows if r.get("first_text_s") is not None]
    return {"answers": len(rows), "correct": sum(1 for r in rows if r["correct"]),
            "p50_seconds": round(statistics.median(secs), 2) if secs else None,
            "p90_seconds": round(sorted(secs)[max(0, round(0.9 * (len(secs) - 1)))], 2) if secs else None,
            "p50_first_words": round(statistics.median(first), 2) if first else None,
            "cost_usd": round(sum(r.get("cost_usd", 0) for r in rows), 5),
            "right": sorted(r["id"] for r in rows if r["correct"])}


def merge_runs(a: dict, b: dict) -> dict:
    """Two measurements of the same settings combined: medians over both, correct only if right both times."""
    rows = a["rows"] + b["rows"]
    s = summarize(rows)
    right_both = sorted(set(a["summary"]["right"]) & set(b["summary"]["right"]))
    s.update(right=right_both, correct=len(right_both), cost_usd=round((a["summary"]["cost_usd"] + b["summary"]["cost_usd"]) / 2, 5))
    return {"config": a["config"], "rows": rows, "summary": s}


def decide(baseline: dict, candidates: list[dict]) -> tuple[dict | None, list[dict]]:
    """The winner (or None) and a verdict for every candidate, from measured numbers only."""
    b = baseline["summary"]
    verdicts = []
    for c in candidates:
        s = c["summary"]
        lost = sorted(set(b["right"]) - set(s["right"]))
        faster = b["p50_seconds"] and s["p50_seconds"] is not None and s["p50_seconds"] <= b["p50_seconds"] * (1 - FASTER_BY)
        cheaper = s["cost_usd"] <= b["cost_usd"] * COST_SLACK + 1e-6
        if lost:
            why = "Answered worse: got " + ", ".join(lost) + " wrong that the current settings got right."
        elif not faster:
            why = f"Not at least {int(FASTER_BY * 100)}% faster ({s['p50_seconds']}s vs {b['p50_seconds']}s median)."
        elif not cheaper:
            why = f"Costs more (${s['cost_usd']:.4f} vs ${b['cost_usd']:.4f} for the benchmark)."
        else:
            why = ""
        verdicts.append({"config": c["config"], "summary": s, "ok": not why, "why": why or "Faster, no more expensive, and just as correct."})
    ok = [v for v in verdicts if v["ok"]]
    ok.sort(key=lambda v: v["summary"]["p50_seconds"])
    return (ok[0] if ok else None), verdicts


# ---------------------------------------------------------------- proposing
PROPOSE_TOOL = {
    "name": "propose", "description": "Candidate settings to measure.",
    "input_schema": {"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {
        "settings": {"type": "object", "description": "Values for the knobs; omit a knob to keep its current value."},
        "why": {"type": "string"}}, "required": ["settings", "why"]}}}, "required": ["candidates"]},
}
PROPOSE_PROMPT = """You tune an AI assistant's settings for speed without raising cost or lowering answer quality. You get \
the tunable knobs, the current settings, where the time goes in real traffic, and a journal of settings already tried. \
Propose {n} different candidate settings that are most likely to be faster at no extra cost. Don't repeat settings the \
journal shows already lost. Each candidate changes one or two knobs, with a one-sentence reason grounded in the numbers."""


def bottlenecks(perf: dict) -> str:
    if not perf or not perf.get("answers"):
        return "No real traffic recorded yet; reason from the knobs alone."
    parts = [f"{perf['answers']} answers: median {perf['p50_seconds']}s (slowest 10%: {perf['p90_seconds']}s), first words "
             f"{perf['p50_first_words']}s, trust check {perf.get('p50_trust_seconds')}s, {perf.get('avg_model_calls')} model calls per answer, "
             f"${perf.get('avg_cost_usd')} per answer, prompt cache used on {int((perf.get('cache_hit_share') or 0) * 100)}% of answers."]
    for label, v in (perf.get("by_label") or {}).items():
        parts.append(f"- {label}: {v['answers']} answers, median {v['p50_seconds']}s")
    return "\n".join(parts)


def propose(client, model: str, current: dict, perf: dict, journal: list[dict], n: int = 3) -> tuple[list[dict], int, int]:
    knobs = "\n".join(f"- {k}: one of {v['values']}. {v['what']}" for k, v in KNOBS.items())
    tried = "\n".join(f"- {json.dumps(j.get('config'))}: {j.get('why')}" for j in journal[-20:]) or "(nothing tried yet)"
    locked = env_locked()
    resp = client.messages.create(
        model=model, max_tokens=800, system=PROPOSE_PROMPT.format(n=n), tools=[PROPOSE_TOOL], tool_choice={"type": "tool", "name": "propose"},
        **speed.params(model, "helper"),
        messages=[{"role": "user", "content": f"KNOBS:\n{knobs}\n\nCURRENT: {json.dumps(current)}\n"
                   + (f"FIXED BY ENVIRONMENT (don't change): {', '.join(locked)}\n" if locked else "")
                   + f"\nWHERE THE TIME GOES:\n{bottlenecks(perf)}\n\nJOURNAL OF EARLIER TRIES:\n{tried}"}])
    u = _get(resp, "usage")
    i, o = ((_get(u, "input_tokens", 0) or 0), (_get(u, "output_tokens", 0) or 0)) if u is not None else (0, 0)
    tb = next((b for b in (_get(resp, "content", []) or []) if _get(b, "type") == "tool_use"), None)
    seen = {json.dumps(current, sort_keys=True)} | {json.dumps(clean(j.get("config")), sort_keys=True) for j in journal}
    out = []
    for c in ((_get(tb, "input", {}) or {}).get("candidates") or []) if tb is not None else []:
        cfg = clean({**current, **{k: v for k, v in (c.get("settings") or {}).items() if k not in locked}})
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append({"config": cfg, "why": " ".join(str(c.get("why") or "").split())[:300]})
    if not out:   # the model proposed nothing new: try one knob at a time in the obvious directions
        for k, v in (("chat_effort", "low"), ("simple_to_fast_model", True), ("web_max_searches", 2)):
            cfg = clean({**current, k: v})
            if json.dumps(cfg, sort_keys=True) not in seen and k not in locked:
                seen.add(json.dumps(cfg, sort_keys=True))
                out.append({"config": cfg, "why": f"Try {k} = {v}."})
    return out[:n], i, o


# ---------------------------------------------------------------- the pull request
class TuneError(Exception):
    pass


def tuned_file(cfg: dict, evidence: dict) -> str:
    return json.dumps({"settings": clean(cfg), "evidence": evidence,
                       "note": "Written by Ask Harsha's Tune. Environment variables (CHAT_EFFORT, CHAT_THINKING) override these."}, indent=2) + "\n"


def open_pull_request(repo: str, token: str, cfg: dict, evidence: dict, body: str, opener=None, stamp: str = "") -> dict:
    """Commit agent/tuned.json to a new branch and open a pull request. If the token can't open pull requests, return
    the compare link so the person can open it with one click."""
    if not repo or not token:
        raise TuneError("GitHub isn't connected (GITHUB_TOKEN and NOTES_REPO or TUNE_REPO), so the change can't be proposed as a pull request.")
    opener = opener or urllib.request.urlopen

    def call(method, path, payload=None):
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}{path}", method=method,
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                              "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "ask-harsha",
                                              **({"Content-Type": "application/json"} if payload is not None else {})})
        with opener(req, timeout=15) as r:
            return json.loads(r.read().decode() or "{}")

    try:
        base = call("GET", "").get("default_branch", "main")
        head_sha = call("GET", f"/git/ref/heads/{base}")["object"]["sha"]
        branch = f"tune/{stamp or 'settings'}"
        call("POST", "/git/refs", {"ref": f"refs/heads/{branch}", "sha": head_sha})
        path = "agent/tuned.json"
        existing = None
        try:
            existing = call("GET", f"/contents/{path}?ref={branch}").get("sha")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        put = {"message": "Tune: faster settings, same quality and cost", "branch": branch,
               "content": base64.b64encode(tuned_file(cfg, evidence).encode()).decode()}
        if existing:
            put["sha"] = existing
        call("PUT", f"/contents/{path}", put)
    except urllib.error.HTTPError as e:
        raise TuneError(f"GitHub refused the change ({e.code}). The token needs Contents: read and write on {repo}.") from e
    except (urllib.error.URLError, TimeoutError, OSError, KeyError) as e:
        raise TuneError(f"Couldn't reach GitHub: {e}") from e
    compare = f"https://github.com/{repo}/compare/{base}...{branch}?expand=1"
    try:
        pr = call("POST", "/pulls", {"title": "Tune: faster settings, same quality and cost", "head": branch, "base": base, "body": body})
        return {"branch": branch, "url": pr.get("html_url") or compare, "pull_request": True}
    except urllib.error.HTTPError:
        return {"branch": branch, "url": compare, "pull_request": False,
                "note": "The branch is ready, but the token can't open pull requests. Open this link to create it with one click."}


def pr_body(run: dict) -> str:
    b, w = run["baseline"]["summary"], run["winner"]["summary"]
    changed = {k: (run["baseline"]["config"][k], run["winner"]["config"][k]) for k in KNOBS if run["baseline"]["config"][k] != run["winner"]["config"][k]}
    lines = ["Proposed by Ask Harsha's **Tune**, measured with real API calls on the benchmark questions (measured twice, interleaved).", "",
             "| | Current | Proposed |", "|---|---|---|",
             f"| Median full answer | {b['p50_seconds']}s | {w['p50_seconds']}s |",
             f"| Slowest 10% | {b['p90_seconds']}s | {w['p90_seconds']}s |",
             f"| Median first words | {b['p50_first_words']}s | {w['p50_first_words']}s |",
             f"| Benchmark cost | ${b['cost_usd']:.4f} | ${w['cost_usd']:.4f} |",
             f"| Questions right (both runs) | {b['correct']}/{len(BENCH)} | {w['correct']}/{len(BENCH)} |", "",
             "**Changes:** " + ", ".join(f"`{k}`: {a} → {c}" for k, (a, c) in changed.items()), "",
             f"Signed report fingerprint: `{run.get('fingerprint', '')}`", "",
             "Nothing changes until this is merged. Environment variables still override these values."]
    return "\n".join(lines)
