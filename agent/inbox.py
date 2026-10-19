"""Ask the real Harsha: questions the agent hands to Harsha when a human would answer better.

The agent offers this when a question about Harsha isn't covered by his record, or when a general question is
genuinely complex or uncertain. A visitor who chooses to send it gets a private link; Harsha answers from his
owner Inbox; the visitor sees the answer at that link (and the page reminds them). Harsha can also save his
answer to memory so the agent knows it next time.

Storage: inbox.json on the `notes` branch, next to the agent's memory. The repository is public, so the whole
file is encrypted (Fernet: AES with an integrity check). The key comes from INBOX_KEY if set, otherwise it's
derived from the owner passcode with scrypt. Changing the passcode without setting INBOX_KEY makes the old
inbox unreadable, so the store refuses to write rather than overwrite it.

Each item's id is a random, unguessable token; it doubles as the visitor's link. Nothing else about the visitor
is kept: no IP address, name or email.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

MAX_OPEN = 300
MAX_QUESTION = 1000
MAX_CONTEXT = 1200
MAX_ANSWER = 2000
KINDS = ("about_harsha", "complex", "uncertain", "judgment")


class InboxUnavailable(Exception):
    """The inbox can't be read or written right now; the message is safe to show."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clip(s, n: int) -> str:
    return " ".join(str(s or "").split())[:n] if n < 1500 else str(s or "").strip()[:n]


@dataclass
class Item:
    id: str
    question: str
    reason: str = ""
    kind: str = "complex"
    context: str = ""          # what the agent answered, so Harsha sees what the visitor already got
    note: str = ""             # optional note from the visitor
    status: str = "open"       # open, answered, dismissed
    created: str = field(default_factory=_now)
    answer: str = ""
    answered_at: str = ""
    note_id: str = ""          # memory note saved from the answer, if any

    def public(self) -> dict:
        """What the visitor's link shows."""
        return {"id": self.id, "status": "answered" if self.status == "answered" else "waiting",
                "question": self.question, "answer": self.answer if self.status == "answered" else "",
                "answered_at": self.answered_at if self.status == "answered" else ""}


def make_key(inbox_key: str = "", owner_passcode: str = "") -> bytes | None:
    if inbox_key:
        raw = hashlib.sha256(inbox_key.encode()).digest()
    elif owner_passcode:
        raw = hashlib.scrypt(owner_passcode.encode(), salt=b"ask-harsha-inbox-v1", n=2 ** 14, r=8, p=1, dklen=32)
    else:
        return None
    return base64.urlsafe_b64encode(raw)


class InboxStore:
    """In-memory inbox; subclasses persist the encrypted file."""

    def __init__(self, key: bytes | None):
        self.fernet = Fernet(key) if key else None
        self._items: list[Item] = []
        self._loaded = False
        self._lock = threading.Lock()

    # persistence hooks: read() returns the stored text or None if there's no file yet
    def read(self) -> str | None:
        return None

    def write(self, text: str) -> None:
        pass

    @property
    def enabled(self) -> bool:
        return self.fernet is not None

    def _load(self) -> None:
        if not self.enabled:
            raise InboxUnavailable("Ask the real Harsha isn't set up on this server.")
        text = self.read()
        if not text:
            self._items, self._loaded = [], True
            return
        try:
            data = json.loads(self.fernet.decrypt(json.loads(text)["enc"].encode()))
        except (InvalidToken, KeyError, ValueError) as e:
            raise InboxUnavailable("The inbox can't be unlocked with the current key, so it wasn't changed.") from e
        self._items = [Item(**i) for i in data.get("items", [])]
        self._loaded = True

    def _ready(self) -> None:
        if not self._loaded:
            self._load()

    def _save(self) -> None:
        payload = json.dumps({"items": [asdict(i) for i in self._items]}, ensure_ascii=False).encode()
        self.write(json.dumps({"v": 1, "enc": self.fernet.encrypt(payload).decode()}))

    def reload(self) -> None:
        with self._lock:
            self._load()

    def submit(self, question: str, reason: str = "", kind: str = "complex", context: str = "", note: str = "") -> Item:
        question = _clip(question, MAX_QUESTION)
        if not question:
            raise ValueError("There's no question to send.")
        with self._lock:
            self._ready()
            if sum(1 for i in self._items if i.status == "open") >= MAX_OPEN:
                raise InboxUnavailable("Harsha's inbox is full right now. Please try again in a few days.")
            item = Item(id=secrets.token_urlsafe(9), question=question, reason=_clip(reason, 200),
                        kind=kind if kind in KINDS else "complex", context=_clip(context, MAX_CONTEXT), note=_clip(note, 500))
            self._items.append(item)
            try:
                self._save()
            except Exception:
                self._items.pop()
                raise
            return item

    def get(self, item_id: str) -> Item | None:
        with self._lock:
            self._ready()
            return next((i for i in self._items if i.id == item_id), None)

    def all(self) -> list[Item]:
        with self._lock:
            self._ready()
            return list(self._items)

    def _update(self, item_id: str, **changes) -> Item:
        with self._lock:
            self._ready()
            item = next((i for i in self._items if i.id == item_id), None)
            if item is None:
                raise KeyError(item_id)
            before = asdict(item)
            for k, v in changes.items():
                setattr(item, k, v)
            try:
                self._save()
            except Exception:
                for k, v in before.items():
                    setattr(item, k, v)
                raise
            return item

    def answer(self, item_id: str, answer: str, note_id: str = "") -> Item:
        answer = str(answer or "").strip()[:MAX_ANSWER]
        if not answer:
            raise ValueError("Write an answer first.")
        return self._update(item_id, status="answered", answer=answer, answered_at=_now(), note_id=note_id)

    def dismiss(self, item_id: str) -> Item:
        return self._update(item_id, status="dismissed")


class MemoryInboxStore(InboxStore):
    def __init__(self, key: bytes | None = None):
        super().__init__(key if key is not None else Fernet.generate_key())
        self.text: str | None = None

    def read(self):
        return self.text

    def write(self, text):
        self.text = text


class FileInboxStore(InboxStore):
    def __init__(self, path: str | Path, key: bytes | None):
        super().__init__(key)
        self.path = Path(path)

    def read(self):
        return self.path.read_text(encoding="utf-8") if self.path.exists() else None

    def write(self, text):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text, encoding="utf-8")


class GitHubInboxStore(InboxStore):
    """inbox.json on a branch of the GitHub repo (the same branch as the notes), through the contents API."""

    API = "https://api.github.com"

    def __init__(self, repo: str, token: str, key: bytes | None, branch: str = "notes", path: str = "inbox.json", opener=None):
        super().__init__(key)
        self.repo, self.token, self.branch, self.path = repo, token, branch, path
        self._sha: str | None = None
        self._open = opener or urllib.request.urlopen

    def _request(self, method: str, body: dict | None = None):
        url = f"{self.API}/repos/{self.repo}/contents/{self.path}" + (f"?ref={self.branch}" if method == "GET" else "")
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None, headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "ask-harsha",
            **({"Content-Type": "application/json"} if body else {}),
        })
        with self._open(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def read(self):
        try:
            data = self._request("GET")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self._sha = None
                return None
            raise InboxUnavailable(f"GitHub returned {e.code} reading the inbox.") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise InboxUnavailable("Couldn't reach GitHub to read the inbox.") from e
        self._sha = data.get("sha")
        return base64.b64decode(data.get("content", "")).decode("utf-8")

    def write(self, text):
        body = {"message": "Inbox update", "branch": self.branch, "content": base64.b64encode(text.encode()).decode()}
        if self._sha:
            body["sha"] = self._sha
        try:
            data = self._request("PUT", body)
            self._sha = data["content"]["sha"]
        except urllib.error.HTTPError as e:
            if e.code in (409, 422):  # changed elsewhere: reload on next use instead of overwriting
                self._loaded = False
                raise InboxUnavailable("The inbox changed elsewhere. Please try again.") from e
            raise InboxUnavailable(f"GitHub returned {e.code} saving to the inbox.") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise InboxUnavailable("Couldn't reach GitHub to save to the inbox.") from e
