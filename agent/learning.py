"""Job Pilot's Interview Conversion Learning Engine: which résumé strategy actually gets replies, measured, not assumed.

How it learns without fooling itself:
  - Each application is tailored with one of a few résumé strategies, assigned so that comparable applications (same match
    level, same sponsorship likelihood) get each strategy about equally often. That makes the comparison fair.
  - Outcomes are recorded per application: a recruiter reply, an interview, an offer, a rejection, or nothing after 21 days.
    They come from his own taps in the tracker, or from the laptop runner reading his inbox locally (only the outcome and
    date leave the laptop, never the email).
  - The engine reports response and interview rates with 95% confidence intervals and compares strategies with a
    two-proportion test, stratified so a strategy isn't credited for having drawn easier jobs.
  - It changes behavior only on real evidence: a strategy becomes the default when each arm has at least 30 decided
    outcomes and the difference is significant (p < 0.05) in the stratified test. Even then 20% of applications keep
    testing the alternative, so the evidence stays current. Until then it says plainly that there isn't enough data.
"""
from __future__ import annotations

import math
import random
import re
import time
from typing import Any

STRATEGIES = {
    "problem_first": {"label": "Lead with the problem",
                      "text": "Strategy for this version: open the summary with the problem this role is hired to solve and his closest "
                              "experience solving it; in each job, put first the bullet that's closest to that problem."},
    "requirements_first": {"label": "Lead with the requirements",
                           "text": "Strategy for this version: open the summary with the role's title and his strongest match to the "
                                   "listed must-haves; in each job, order bullets by the must-haves in the order the posting lists them."},
}
POSITIVE = ("recruiter_reply", "interview", "offer")
OUTCOMES = POSITIVE + ("rejected",)
NO_RESPONSE_DAYS = 21
MIN_PER_ARM = 30
EXPLORE = 0.2
DAY_MS = 86_400_000


def group_of(app: dict) -> str:
    """Comparable applications: same match level and sponsorship likelihood."""
    return f"{(app.get('qualify') or {}).get('match', 'unknown')}|{((app.get('screen') or {}).get('sponsor') or {}).get('level', 'unknown')}"


def outcome_of(app: dict, now_ms: int) -> str | None:
    """positive outcomes, rejected, no_response (21 days of silence), or None while it's too early to say."""
    o = app.get("outcome")
    if o in OUTCOMES:
        return o
    sub = app.get("submitted_at")
    if sub and now_ms - int(sub) >= NO_RESPONSE_DAYS * DAY_MS:
        return "no_response"
    return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cmh_test(strata: list[tuple[int, int, int, int]]) -> float | None:
    """Cochran–Mantel–Haenszel test across strata of (a_yes, a_n, b_yes, b_n). Returns a two-sided p-value."""
    num, var = 0.0, 0.0
    for ay, an, by, bn in strata:
        n = an + bn
        if an == 0 or bn == 0 or n < 2:
            continue
        m1 = ay + by
        num += ay - an * m1 / n
        var += an * bn * m1 * (n - m1) / (n * n * (n - 1))
    if var <= 0:
        return None
    chi = (abs(num) - 0.5) ** 2 / var if abs(num) > 0.5 else 0.0
    return 2 * (1 - _phi(math.sqrt(chi)))


def analyze(apps: list[dict], now_ms: int | None = None) -> dict:
    """Everything the learning report says, computed from submitted applications."""
    now_ms = now_ms or int(time.time() * 1000)
    sub = [a for a in apps if a.get("status") == "submitted" or a.get("submitted_at")]
    arms: dict[str, dict] = {}
    strata: dict[str, dict[str, list[int]]] = {}
    for a in sub:
        s = a.get("strategy") or "unassigned"
        o = outcome_of(a, now_ms)
        arm = arms.setdefault(s, {"submitted": 0, "decided": 0, "positive": 0, "interviews": 0, "rejected": 0, "no_response": 0, "pending": 0})
        arm["submitted"] += 1
        if o is None:
            arm["pending"] += 1
            continue
        arm["decided"] += 1
        arm["positive"] += o in POSITIVE
        arm["interviews"] += o in ("interview", "offer")
        arm["rejected"] += o == "rejected"
        arm["no_response"] += o == "no_response"
        st = strata.setdefault(group_of(a), {})
        cell = st.setdefault(s, [0, 0])
        cell[0] += o in POSITIVE
        cell[1] += 1
    for s, arm in arms.items():
        arm["label"] = STRATEGIES.get(s, {}).get("label", s)
        arm["response_rate"] = round(arm["positive"] / arm["decided"], 3) if arm["decided"] else None
        arm["response_ci"] = [round(x, 3) for x in wilson(arm["positive"], arm["decided"])]
        arm["interview_rate"] = round(arm["interviews"] / arm["decided"], 3) if arm["decided"] else None
    names = [s for s in STRATEGIES if s in arms]
    verdict = {"winner": None, "p_value": None, "message": ""}
    if len(names) >= 2:
        a, b = names[0], names[1]
        p = cmh_test([(c.get(a, [0, 0])[0], c.get(a, [0, 0])[1], c.get(b, [0, 0])[0], c.get(b, [0, 0])[1]) for c in strata.values()])
        verdict["p_value"] = round(p, 4) if p is not None else None
        need = max(0, MIN_PER_ARM - min(arms[a]["decided"], arms[b]["decided"]))
        if need:
            verdict["message"] = (f"Not enough evidence yet: each strategy needs {MIN_PER_ARM} applications with a known outcome "
                                  f"({need} more for the smaller one). Until then both are used equally.")
        elif p is not None and p < 0.05:
            ra, rb = arms[a]["response_rate"] or 0, arms[b]["response_rate"] or 0
            verdict["winner"] = a if ra > rb else b
            verdict["message"] = (f"“{arms[verdict['winner']]['label']}” gets more replies (p = {p:.3f}, compared within similar jobs). "
                                  f"It's now used for {int((1 - EXPLORE) * 100)}% of applications; the rest keep testing.")
        else:
            verdict["message"] = f"No real difference between strategies yet (p = {p:.2f}). Both stay in use equally." if p is not None else \
                "The strategies haven't been tried on comparable jobs yet."
    else:
        verdict["message"] = "No submitted applications with outcomes yet."
    return {"arms": arms, "verdict": verdict, "segments": segments(sub, now_ms), "submitted": len(sub),
            "decided": sum(a["decided"] for a in arms.values()), "pending": sum(a["pending"] for a in arms.values())}


def segments(sub: list[dict], now_ms: int) -> dict:
    """Observed response rates by match level, sponsorship likelihood, hiring system and location. Descriptive only."""
    keys = {"match": lambda a: (a.get("qualify") or {}).get("match", "unknown"),
            "sponsor": lambda a: ((a.get("screen") or {}).get("sponsor") or {}).get("level", "unknown"),
            "ats": lambda a: (a.get("job") or {}).get("ats", "unknown"),
            "location": lambda a: {0: "preferred city", 1: "remote", 2: "elsewhere in the US"}.get((a.get("screen") or {}).get("location_rank"), "unknown")}
    out = {}
    for name, fn in keys.items():
        rows: dict[str, list[int]] = {}
        for a in sub:
            o = outcome_of(a, now_ms)
            if o is None:
                continue
            r = rows.setdefault(fn(a), [0, 0])
            r[0] += o in POSITIVE
            r[1] += 1
        out[name] = {k: {"positive": v[0], "decided": v[1], "rate": round(v[0] / v[1], 3), "ci": [round(x, 3) for x in wilson(*v)]}
                     for k, v in rows.items()}
    return out


def assign(history: list[dict], app: dict, winner: str | None = None, rng: random.Random | None = None) -> str:
    """Balanced within comparable jobs; once a winner is proven, it gets 80% and the rest keep testing."""
    rng = rng or random.Random()
    names = list(STRATEGIES)
    if winner in STRATEGIES and rng.random() >= EXPLORE:
        return winner
    if winner in STRATEGIES:
        return rng.choice([n for n in names if n != winner])
    g = group_of(app)
    counts = {n: 0 for n in names}
    for h in history:
        if group_of(h) == g and h.get("strategy") in counts:
            counts[h["strategy"]] += 1
    low = min(counts.values())
    return rng.choice([n for n, c in counts.items() if c == low])
