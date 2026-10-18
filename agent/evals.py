"""Eval cases and scoring, shared by the command-line runner and the owner-only web button.

A case passes when the answer:
  - contains every `must_include` string and none of `must_not_include` (case-insensitive)
  - cites every ID in `must_cite`
  - declines, when `expect_decline` is true
  - gets the expected page label, when `expect_label` is set (record, memory, web, general, none)
  - changes no memory, when `expect_no_memory_change` is true
  - passes the grounding check

Optional case fields:
  only: "general" or "career"   run the case only against that agent
  web: true                     needs web search; skipped when it's off
`{year}` inside must_include is replaced with the current year.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

CASES_PATH = Path(__file__).resolve().parent.parent / "evals" / "cases.json"

DECLINE_MARKERS = [
    "isn't in", "is not in", "not in harsha's record", "not in his record", "doesn't cover", "does not cover",
    "ask him directly", "contact him", "contact harsha", "reach out", "discuss with harsha", "discussed with harsha",
    "best discussed", "don't have", "do not have", "can't share", "cannot share", "not able to", "can't confirm",
    "cannot confirm", "only harsha can", "can't save", "cannot save", "no record", "not something i",
    "can't say", "cannot say", "doesn't say", "does not say", "doesn't mention", "does not mention", "not stated",
]


def load_cases(path: str | Path = CASES_PATH, mode: str = "general", web: bool = True) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for c in cases:
        if c.get("only") and c["only"] != mode:
            continue
        if c.get("web") and not web:
            continue
        out.append(c)
    return out


def score(case: dict, text: str, report=None, label: str | None = None, memory_changes: list | None = None) -> list[str]:
    year = str(datetime.now().year)
    low = text.lower()
    failures = []
    for s in case.get("must_include", []):
        s = s.replace("{year}", year)
        if s.lower() not in low:
            failures.append(f"missing '{s}'")
    for s in case.get("must_not_include", []):
        if s.lower() in low:
            failures.append(f"contains '{s}'")
    for cid in case.get("must_cite", []):
        if f"[{cid}]" not in text:
            failures.append(f"no citation [{cid}]")
    if case.get("expect_decline") and not any(m in low for m in DECLINE_MARKERS):
        failures.append("did not decline")
    if case.get("expect_label") and label is not None and label != case["expect_label"]:
        failures.append(f"label '{label}', expected '{case['expect_label']}'")
    if case.get("expect_no_memory_change") and memory_changes:
        failures.append("changed memory")
    if report is not None and not report.ok:
        failures.append(report.summary())
    return failures
