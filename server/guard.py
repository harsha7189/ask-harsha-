"""Abuse and cost protection for the public endpoint.

Both guards keep state in memory, which fits a single small server. If you ever run several
instances, move these counters to a shared store such as Redis.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone


class RateLimiter:
    """Sliding window: at most `limit` requests per `window` seconds per key (visitor IP)."""

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._hits) > 10_000:  # drop idle visitors so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True

    def retry_after(self, key: str) -> int:
        with self._lock:
            q = self._hits.get(key)
            if not q:
                return 0
            return max(1, int(self.window - (self.clock() - q[0])) + 1)


class DailyBudget:
    """Stops answering once today's estimated spend (UTC day) reaches `limit_usd`."""

    def __init__(self, limit_usd: float, price_in_per_mtok: float, price_out_per_mtok: float, today=None):
        self.limit_usd = limit_usd
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._day = self._today()
        self._spent = 0.0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._spent = day, 0.0

    def exhausted(self) -> bool:
        with self._lock:
            self._roll()
            return self._spent >= self.limit_usd

    def add(self, input_tokens: int, output_tokens: int) -> float:
        cost = input_tokens / 1e6 * self.price_in + output_tokens / 1e6 * self.price_out
        with self._lock:
            self._roll()
            self._spent += cost
            return cost

    def add_flat(self, usd: float) -> float:
        """Add a fixed cost, such as web search fees."""
        with self._lock:
            self._roll()
            self._spent += usd
            return usd

    @property
    def spent(self) -> float:
        with self._lock:
            self._roll()
            return self._spent
