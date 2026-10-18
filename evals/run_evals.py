#!/usr/bin/env python3
"""Live evals: run each case against the real model and score it.

    python evals/run_evals.py                 # all cases
    python evals/run_evals.py --model claude-haiku-5-5
    python evals/run_evals.py --only ai-launch,contact

A case passes when the answer contains every must_include string, cites every must_cite ID,
contains none of must_not_include, passes the grounding check, and (for expect_decline) declines.
Results are also written to evals/results-<model>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.kb import KnowledgeBase  # noqa: E402
from agent.loop import DEFAULT_MODEL, Agent  # noqa: E402
from cli import load_dotenv, make_client  # noqa: E402

DECLINE_MARKERS = ["isn't in", "is not in", "not in harsha's record", "not in his record", "doesn't cover", "ask him directly", "discuss with harsha", "discussed with harsha", "best discussed", "don't have"]


def score(case: dict, reply) -> list[str]:
    text = reply.text
    low = text.lower()
    failures = []
    for s in case.get("must_include", []):
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
    if reply.report and not reply.report.ok:
        failures.append(reply.report.summary())
    return failures


def main() -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--only", help="Comma-separated case IDs.")
    args = p.parse_args()

    cases = json.loads((ROOT / "evals" / "cases.json").read_text())
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    kb, client = KnowledgeBase.load(), make_client()

    rows, passed = [], 0
    for case in cases:
        agent = Agent(client, kb=kb, model=args.model)  # fresh conversation per case
        reply = agent.ask(case["question"])
        failures = score(case, reply)
        passed += not failures
        rows.append({"id": case["id"], "pass": not failures, "failures": failures, "answer": reply.text,
                     "steps": reply.steps, "tokens": reply.input_tokens + reply.output_tokens})
        mark = "PASS" if not failures else "FAIL"
        print(f"{mark}  {case['id']:<18} {'; '.join(failures)}")

    total_tokens = sum(r["tokens"] for r in rows)
    print(f"\n{passed}/{len(cases)} passed · {total_tokens} tokens · model {args.model}")
    out = ROOT / "evals" / f"results-{args.model}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Details: {out}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
