#!/usr/bin/env python3
"""Live evals: run each case against the real model and score it.

    python evals/run_evals.py                 # all cases
    python evals/run_evals.py --model claude-haiku-5-5
    python evals/run_evals.py --only ai-launch,contact

Scoring rules live in agent/evals.py and are shared with the owner-only "Run evals" button in the
web app, which tests the full general assistant (including general-knowledge and web cases).
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

from agent.evals import load_cases, score as score_case  # noqa: E402


def score(case: dict, reply) -> list[str]:
    return score_case(case, reply.text, reply.report)


def main() -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--only", help="Comma-separated case IDs.")
    args = p.parse_args()

    cases = load_cases(mode="career", web=False)  # this runner tests the career-only CLI agent
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
