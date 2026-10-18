#!/usr/bin/env python3
"""Ask Harsha, command-line chat.

    python cli.py                      # interactive chat
    python cli.py --trace              # show every tool call and the grounding check
    python cli.py -q "What did he do at Cybage?"   # one question, then exit

Commands inside the chat: /trace  /reset  /facts  /help  /quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent.kb import KnowledgeBase
from agent.loop import DEFAULT_MODEL, Agent

ROOT = Path(__file__).resolve().parent
DIM, BOLD, RED, GREEN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    DIM = BOLD = RED = GREEN = YELLOW = RESET = ""


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value lines, no extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")  # allow trailing comments
        os.environ.setdefault(key.strip(), value)


def make_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("The 'anthropic' package is missing. Run: pip install -r requirements.txt")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return anthropic.Anthropic()


class Tracer:
    def __init__(self, on: bool):
        self.on = on

    def __call__(self, kind: str, data: dict) -> None:
        if not self.on:
            return
        if kind == "tool_call":
            print(f"{DIM}  → {data['name']}({json.dumps(data['input'], ensure_ascii=False)}){RESET}")
        elif kind == "tool_result":
            out = data["output"]
            try:
                ids = [r["id"] for r in json.loads(out).get("results", [])]
                summary = f"facts {', '.join(ids)}" if ids else "no facts"
            except (json.JSONDecodeError, AttributeError):
                summary = out[:120]
            color = RED if data["is_error"] else DIM
            print(f"{color}  ← {summary}{RESET}")


def print_reply(reply, trace: bool) -> None:
    print(f"\n{BOLD}Ask Harsha:{RESET} {reply.text}\n")
    status = reply.report.summary() if reply.report else ""
    color = GREEN if reply.report and reply.report.ok else YELLOW
    if trace or (reply.report and not reply.report.ok):
        print(f"{color}  [{status} · {reply.steps} steps · {reply.input_tokens}+{reply.output_tokens} tokens · {reply.seconds}s]{RESET}\n")


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description="Ask questions about Harsha Reddy's experience.")
    p.add_argument("-q", "--question", help="Ask one question and exit.")
    p.add_argument("--model", default=os.environ.get("ASK_HARSHA_MODEL", DEFAULT_MODEL))
    p.add_argument("--trace", action="store_true", help="Show tool calls and the grounding check.")
    p.add_argument("--kb", default=str(ROOT / "knowledge" / "profile.json"), help="Path to the facts file.")
    args = p.parse_args(argv)

    kb = KnowledgeBase.load(args.kb)
    tracer = Tracer(args.trace)
    agent = Agent(make_client(), kb=kb, model=args.model, on_event=tracer)

    if args.question:
        print_reply(agent.ask(args.question), args.trace)
        return 0

    print(f"{BOLD}Ask Harsha{RESET} · {len(kb.facts)} verified facts · model {args.model}")
    print(f"{DIM}Ask about his roles, impact, skills or certifications. /help for commands.{RESET}\n")
    while True:
        try:
            q = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in ("/quit", "/exit"):
            return 0
        if q == "/help":
            print("/trace toggle tool tracing · /reset start over · /facts list topics · /quit exit\n")
            continue
        if q == "/trace":
            tracer.on = not tracer.on
            print(f"Tracing {'on' if tracer.on else 'off'}.\n")
            continue
        if q == "/reset":
            agent.reset()
            print("Conversation cleared.\n")
            continue
        if q == "/facts":
            for topic, n in sorted(kb.topics().items()):
                print(f"  {topic:<15} {n} fact{'s' if n != 1 else ''}")
            print()
            continue
        try:
            print_reply(agent.ask(q), tracer.on)
        except Exception as e:  # network or API errors: the agent rolled back its history, so the session continues
            print(f"{RED}Request failed: {type(e).__name__}: {e}{RESET}\n")


if __name__ == "__main__":
    raise SystemExit(main())
