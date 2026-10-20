"""Durable operating data: performance history, stats, evaluation results and small bits of server state.

Everything here used to live in the server's memory, so a restart or deploy wiped it and two servers couldn't share
it. With a database it's stored there; without one (local runs, tests) it falls back to memory.

What's stored is outcomes and timings, never questions, answers or anything personal:
  metrics   (kind, ts, body)  e.g. kind "answer": seconds, first words, cost, cache, trust level; "proof"; "trust";
                              "health"; "trust_eval"; "eval"; "tune"
  app_state (key, body)       e.g. the last evaluation's full rows, the last health check, the date it last ran
Old metrics are pruned after RETENTION_DAYS.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from collections import defaultdict

RETENTION_DAYS = 120
MEMORY_CAP = 5000


def now_ms() -> int:
    return int(time.time() * 1000)


class Telemetry:
    def __init__(self, db=None):
        self.db = db
        self._mem: dict[str, list[dict]] = defaultdict(list)
        self._state: dict[str, object] = {}
        self._lock = threading.Lock()
        self._pruned = 0.0

    @property
    def durable(self) -> bool:
        return self.db is not None

    def add(self, kind: str, row: dict) -> None:
        row = {**row, "ts": row.get("ts") or now_ms()}
        if self.db is None:
            with self._lock:
                rows = self._mem[kind]
                rows.append(row)
                del rows[:-MEMORY_CAP]
            return
        try:
            self.db.run("INSERT INTO metrics (id, kind, ts, body) VALUES (%s, %s, %s, %s)",
                        (secrets.token_hex(8), kind[:40], row["ts"], json.dumps(row, ensure_ascii=False)[:20000]))
            if time.monotonic() - self._pruned > 6 * 3600:
                self._pruned = time.monotonic()
                self.db.run("DELETE FROM metrics WHERE ts < %s", (now_ms() - RETENTION_DAYS * 86_400_000,))
        except Exception:
            pass   # operating data must never break an answer

    def recent(self, kind: str, days: float = 30, limit: int = 2000) -> list[dict]:
        """Rows of one kind from the last `days` days, oldest first."""
        since = now_ms() - int(days * 86_400_000)
        if self.db is None:
            with self._lock:
                return [r for r in self._mem.get(kind, []) if r["ts"] >= since][-limit:]
        try:
            rows = self.db.all("SELECT body FROM metrics WHERE kind = %s AND ts >= %s ORDER BY ts DESC LIMIT %s", (kind, since, limit))
        except Exception:
            return []
        out = []
        for (body,) in reversed(rows):
            try:
                out.append(json.loads(body))
            except json.JSONDecodeError:
                continue
        return out

    def put(self, key: str, value) -> None:
        if self.db is None:
            with self._lock:
                self._state[key] = json.loads(json.dumps(value))
            return
        body = json.dumps(value, ensure_ascii=False)
        try:
            if self.db.one("SELECT 1 FROM app_state WHERE key = %s", (key,)):
                self.db.run("UPDATE app_state SET body = %s, updated = %s WHERE key = %s", (body, now_ms(), key))
            else:
                self.db.run("INSERT INTO app_state (key, body, updated) VALUES (%s, %s, %s)", (key, body, now_ms()))
        except Exception:
            pass

    def get(self, key: str, default=None):
        if self.db is None:
            with self._lock:
                v = self._state.get(key, default)
                return json.loads(json.dumps(v)) if v is not None else default
        try:
            row = self.db.one("SELECT body FROM app_state WHERE key = %s", (key,))
            return json.loads(row[0]) if row else default
        except Exception:
            return default

    def claim(self, key: str, value: str) -> bool:
        """Set key to value only if it isn't already that value. True if this call set it (a cheap once-a-day lock)."""
        if self.get(key) == value:
            return False
        self.put(key, value)
        return True


class Scoreboard:
    """A stats object (Proof Mode's or the trust check's) whose runs are stored durably instead of in memory."""

    def __init__(self, telemetry: Telemetry, kind: str, summarize, days: float = 30):
        self.telemetry, self.kind, self.summarize, self.days = telemetry, kind, summarize, days

    def add(self, row: dict) -> None:
        self.telemetry.add(self.kind, row)

    def summary(self) -> dict:
        return {**self.summarize(self.telemetry.recent(self.kind, self.days)), "days": self.days, "durable": self.telemetry.durable}


def performance(rows: list[dict]) -> dict:
    """Answer speed and cost over a set of 'answer' rows: medians, the slow tail, and where the time went."""
    def pct(xs, p):
        xs = sorted(x for x in xs if isinstance(x, (int, float)))
        if not xs:
            return None
        k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
        return round(xs[k], 2)

    total = [r.get("seconds") for r in rows]
    first = [r.get("first_text_s") for r in rows]
    trust_s = [r.get("trust_s") for r in rows]
    by_label: dict[str, list] = defaultdict(list)
    for r in rows:
        by_label[r.get("label") or "other"].append(r.get("seconds"))
    calls = [len(r.get("model_calls_s") or []) for r in rows]
    return {
        "answers": len(rows),
        "p50_seconds": pct(total, 50), "p90_seconds": pct(total, 90),
        "p50_first_words": pct(first, 50), "p90_first_words": pct(first, 90),
        "p50_trust_seconds": pct(trust_s, 50),
        "avg_model_calls": round(sum(calls) / len(calls), 2) if calls else None,
        "avg_cost_usd": round(sum(r.get("cost_usd", 0) or 0 for r in rows) / len(rows), 5) if rows else None,
        "cache_hit_share": round(sum(1 for r in rows if (r.get("cache_read") or 0) > 0) / len(rows), 3) if rows else None,
        "fallbacks": sum(1 for r in rows if r.get("fallback")),
        "by_label": {k: {"answers": len(v), "p50_seconds": pct(v, 50)} for k, v in sorted(by_label.items())},
    }
