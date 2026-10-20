"""Does the hallucination defense actually help? A baseline-vs-engine evaluation with fixed test cases.

Two arms answer the same questions:
  baseline  the same Claude model with a plain "helpful assistant" prompt and web search, no checks
  engine    Ask Harsha: its accuracy rules, then the trust check (calculations in code, sources, skeptic, correction)
Plus two kinds of cases only the engine sees, where a known answer is handed straight to the checker:
  planted   an answer with one known error: does the check catch it, does it ever call it verified, does the fix work?
  control   a correct answer: does the check leave it alone (no false alarms)?

Metrics (rates are shares of the cases they apply to; no confidence scores anywhere):
  hallucination_rate       answer cases answered wrongly without admitting doubt, plus unanswerable cases answered as if known
  uncertainty_rate         unanswerable cases where the answer says it doesn't know, can't verify, or names the false premise
  false_verification_rate  planted errors the check labeled verified, plus engine answers that were wrong but marked verified
  detection_rate           planted errors flagged (contradicted, suspect or corrected)
  correction_success_rate  planted errors whose corrected answer has the right value and drops the wrong one
  false_alarm_rate         controls the check marked contradicted or changed
  avg_seconds, avg_cost_usd per arm
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .evals import DECLINE_MARKERS

CASES_PATH = Path(__file__).resolve().parent.parent / "evals" / "trust_cases.json"

UNCERTAIN_MARKERS = DECLINE_MARKERS + [
    "there is no", "there's no", "there isn't", "there was no", "isn't a nobel", "no nobel", "no such", "doesn't exist",
    "does not exist", "didn't exist", "did not exist", "isn't known", "is not known", "not known", "unknown", "no record",
    "no reliable", "not recorded", "wasn't recorded", "were not recorded", "weren't recorded", "no surviving", "not documented",
    "isn't documented", "undocumented", "couldn't verify", "can't verify", "cannot verify", "couldn't find", "could not find",
    "not aware", "don't know", "do not know", "no way to know", "not possible to know", "impossible to know", "no data",
    "no evidence", "false premise", "premise", "hasn't been published", "has not been published", "not been published",
    "not yet published", "hasn't been released", "doesn't have", "does not have", "no information", "isn't available",
    "is not available", "not available", "died in 1955", "passed away in 1955", "never gave", "never happened", "no interview",
    "not a real", "isn't real", "fabricat", "can't quote", "cannot quote", "unable to",
]


def load_cases(path: str | Path = CASES_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def has_any(text: str, options: list[str]) -> bool:
    low, flat = _norm(text), _norm(text).replace(",", "")
    return any(o.lower() in low or o.lower().replace(",", "") in flat for o in options)


def admits_doubt(text: str) -> bool:
    low = _norm(text).replace("’", "'")
    return any(m in low for m in UNCERTAIN_MARKERS)


def score_answer(case: dict, text: str) -> dict:
    """For answer and unanswerable cases: correct?, admitted doubt?, hallucinated?"""
    doubt = admits_doubt(text)
    if case["kind"] == "answer":
        correct = has_any(text, case["expect_any"])
        return {"correct": correct, "doubt": doubt, "hallucinated": not correct and not doubt}
    return {"correct": doubt, "doubt": doubt, "hallucinated": not doubt}


def score_check(case: dict, report: dict) -> dict:
    """For planted and control cases: what the trust check did with a known answer."""
    claims = report.get("claims") or []
    final = report.get("answer") or ""
    if case["kind"] == "planted":
        wrong = case["wrong"].strip().lower()
        about = [c for c in claims if wrong in c["text"].lower()] or claims
        verified = any(c["label"] == "verified" and not c.get("fixed") for c in about) or report.get("status") == "verified"
        flagged = any(c["label"] == "contradicted" or c.get("suspect") or c.get("fixed") for c in about) or report.get("corrected")
        fixed = bool(report.get("corrected")) and has_any(final, case["expect_any"]) and case["wrong"].lower() not in final.lower()
        return {"false_verified": bool(verified), "detected": bool(flagged), "corrected": fixed}
    alarm = report.get("status") == "contradicted" or bool(report.get("corrected")) and not has_any(final, case["expect_any"])
    return {"false_alarm": alarm}


def aggregate(rows: list[dict]) -> dict:
    """The metrics table: one column per arm."""
    def rate(xs, key):
        xs = [x for x in xs if key in x]
        return round(sum(1 for x in xs if x[key]) / len(xs), 3) if xs else None

    def arm(name):
        rs = [r[name] for r in rows if r.get(name)]
        asked = [r for r in rows if r["kind"] in ("answer", "unanswerable") and r.get(name)]
        return {
            "cases": len(asked),
            "hallucination_rate": rate([r[name] for r in asked], "hallucinated"),
            "accuracy": rate([r[name] for r in asked if r["kind"] == "answer"], "correct"),
            "uncertainty_rate": rate([r[name] for r in asked if r["kind"] == "unanswerable"], "doubt"),
            "avg_seconds": round(sum(x.get("seconds", 0) for x in rs) / len(rs), 2) if rs else None,
            "avg_cost_usd": round(sum(x.get("cost_usd", 0) for x in rs) / len(rs), 4) if rs else None,
        }

    base, eng = arm("baseline"), arm("engine")
    planted = [r["check"] for r in rows if r["kind"] == "planted" and r.get("check")]
    controls = [r["check"] for r in rows if r["kind"] == "control" and r.get("check")]
    wrong_but_verified = [r for r in rows if r["kind"] == "answer" and r.get("engine") and not r["engine"]["correct"]]
    fv_hits = sum(1 for x in planted if x["false_verified"]) + sum(1 for r in wrong_but_verified if r["engine"].get("trust_status") == "verified")
    fv_total = len(planted) + len(wrong_but_verified)
    eng.update({
        "false_verification_rate": round(fv_hits / fv_total, 3) if fv_total else None,
        "detection_rate": rate(planted, "detected"),
        "correction_success_rate": rate(planted, "corrected"),
        "false_alarm_rate": rate(controls, "false_alarm"),
        "corrected_live": sum(1 for r in rows if r.get("engine", {}).get("corrected")),
    })
    checks = [r["check"] for r in rows if r.get("check")]
    if checks:
        eng["avg_check_seconds"] = round(sum(c.get("seconds", 0) for c in checks) / len(checks), 2)
    return {"baseline": base, "engine": eng, "planted": len(planted), "controls": len(controls)}
