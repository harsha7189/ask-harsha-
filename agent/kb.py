"""Knowledge base: loads Harsha's facts and retrieves them with a small BM25 ranker.

Written from scratch (no vector database) so every ranking decision is explainable:
a fact scores higher when it shares rare words with the question.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "profile.json"

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did",
    "what", "which", "who", "whom", "how", "when", "where", "why", "his", "he", "him",
    "harsha", "reddy", "tell", "me", "about", "any", "can", "you", "your", "i", "it", "that",
    "this", "as", "into", "while", "s",
}

# A few domain synonyms so recruiter phrasing still finds the right facts.
SYNONYMS = {
    "job": ["role"], "position": ["role"], "title": ["role"], "work": ["role"], "worked": ["role"],
    "company": ["role"], "employer": ["role"], "school": ["degree"], "university": ["degree"],
    "college": ["degree"], "study": ["degree"], "studied": ["degree"], "certified": ["certification"],
    "certs": ["certification"], "certificates": ["certification"], "llm": ["ai"], "genai": ["ai"],
    "agents": ["agent"], "email": ["contact"], "reach": ["contact"], "hire": ["contact"],
    "impact": ["metrics"], "results": ["metrics"], "numbers": ["metrics"], "achievements": ["metrics"],
    "streaming": ["video"], "drm": ["entitlements"], "integration": ["api"], "integrations": ["api"],
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords. '$8M+' becomes '8m'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def expand(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for t in tokens:
        out.extend(SYNONYMS.get(t, []))
    return out


@dataclass
class Fact:
    id: str
    topic: str
    text: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "topic": self.topic, "text": self.text}


class KnowledgeBase:
    def __init__(self, facts: list[Fact], subject: str = "Harsha Reddy"):
        if not facts:
            raise ValueError("The knowledge base has no facts.")
        ids = [f.id for f in facts]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate fact IDs in the knowledge base.")
        self.subject = subject
        self.facts = facts
        self.by_id = {f.id: f for f in facts}
        # Index: each fact is a bag of words from its text, topic and tags.
        self._docs = [tokenize(f"{f.text} {f.topic} {' '.join(f.tags)} {' '.join(f.tags)}") for f in facts]
        self._tf = [Counter(d) for d in self._docs]
        self._avg_len = sum(len(d) for d in self._docs) / len(self._docs)
        df = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "KnowledgeBase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        facts = [Fact(id=f["id"], topic=f["topic"], text=f["text"], tags=f.get("tags", [])) for f in data["facts"]]
        return cls(facts, subject=data.get("subject", "Harsha Reddy"))

    def search(self, query: str, k: int = 5, k1: float = 1.4, b: float = 0.75) -> list[tuple[Fact, float]]:
        """Okapi BM25. Returns up to k facts with a positive score, best first."""
        q = expand(tokenize(query))
        if not q:
            return []
        scored = []
        for fact, tf, doc in zip(self.facts, self._tf, self._docs):
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * len(doc) / self._avg_len)
                score += self._idf.get(term, 0.0) * freq * (k1 + 1) / denom
            if score > 0:
                scored.append((fact, round(score, 3)))
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]

    def topics(self) -> dict[str, int]:
        return dict(Counter(f.topic for f in self.facts))

    def by_topic(self, topic: str) -> list[Fact]:
        return [f for f in self.facts if f.topic == topic]
