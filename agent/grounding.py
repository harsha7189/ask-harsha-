"""Post-answer grounding check, run on every reply.

It doesn't trust the model to follow the citation rules; it verifies them:
  1. Every cited ID must exist in the knowledge base.
  2. Every cited ID must have been retrieved by a tool during this turn.
  3. Every number in the answer must appear in at least one retrieved fact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION = re.compile(r"\[(F\d{2,3})\]")
# Numbers worth checking: 35%, $8M+, 12,000+, 11, 2024, 3+ ... (ignores single digits used as list counts like "3 points").
NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?\s?(?:%|[kKmMbB](?![a-z]))?\+?")


MULT = {"k": 1e3, "m": 1e6, "b": 1e9}


def _key(s: str) -> tuple[str, float] | None:
    """Turn '35%' into ('pct', 35), '$8M+' and '8,000,000' into ('num', 8e6), '12K' into ('num', 12000).

    Percentages only match percentages, so '60%' can't be 'supported' by '60 days'.
    """
    t = s.strip().replace(",", "").replace(" ", "").replace("+", "").lower().lstrip("$")
    if not t:
        return None
    if t.endswith("%"):
        return ("pct", float(t[:-1]))
    if t[-1] in MULT:
        return ("num", float(t[:-1]) * MULT[t[-1]])
    try:
        return ("num", float(t))
    except ValueError:
        return None


@dataclass
class GroundingReport:
    cited: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)      # cited but not in the KB
    unretrieved_ids: list[str] = field(default_factory=list)  # in the KB but never fetched this turn
    unsupported_numbers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unknown_ids or self.unretrieved_ids or self.unsupported_numbers)

    def summary(self) -> str:
        if self.ok:
            n = len(self.cited)
            return f"grounded ({n} citation{'s' if n != 1 else ''})"
        parts = []
        if self.unknown_ids:
            parts.append(f"unknown IDs {', '.join(self.unknown_ids)}")
        if self.unretrieved_ids:
            parts.append(f"cited without lookup {', '.join(self.unretrieved_ids)}")
        if self.unsupported_numbers:
            parts.append(f"numbers not in retrieved facts {', '.join(self.unsupported_numbers)}")
        return "check: " + "; ".join(parts)


def check(answer: str, retrieved_ids: set[str], kb) -> GroundingReport:
    report = GroundingReport()
    seen = []
    for cid in CITATION.findall(answer):
        if cid not in seen:
            seen.append(cid)
    report.cited = seen
    report.unknown_ids = [c for c in seen if c not in kb.by_id]
    report.unretrieved_ids = [c for c in seen if c in kb.by_id and c not in retrieved_ids]

    evidence = " ".join(kb.by_id[i].text for i in retrieved_ids if i in kb.by_id)
    supported = {k for k in (_key(m) for m in NUMBER.findall(evidence)) if k}
    for m in NUMBER.findall(CITATION.sub("", answer)):
        k = _key(m)
        if k is None:
            continue
        if k[0] == "num" and k[1] < 10:
            continue  # small counts ("3 releases", list numbering) aren't worth flagging
        if k not in supported:
            report.unsupported_numbers.append(m.strip())
    return report
