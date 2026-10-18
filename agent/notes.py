"""Memory: notes Harsha teaches the assistant ("remember that ..."), searchable and citable as N001, N002 ...

Storage backends:
  - GitHubNotesStore: notes.json on a separate branch of the project's GitHub repo. Survives restarts and
    redeploys (Render's free disk does not), and every change is a commit, so there's a full edit history.
    Writing to a branch other than the deployed one means saving a note doesn't trigger a redeploy.
  - FileNotesStore: a local JSON file, for laptop runs.
  - MemoryNotesStore: in memory, for tests.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .kb import Fact, KnowledgeBase

MAX_NOTE_CHARS = 500
MAX_NOTES = 2000


class NotesUnavailable(Exception):
    """Storage can't be read or written right now. Callers must not overwrite anything."""


@dataclass
class Note:
    id: str
    text: str
    created: str

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "created": self.created}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _next_id(notes: list[Note]) -> str:
    n = max((int(x.id[1:]) for x in notes if x.id[1:].isdigit()), default=0) + 1
    return f"N{n:03d}"


class NotesStore:
    """Base class: keeps notes in memory and calls _save() after each change."""

    def __init__(self):
        self._lock = threading.RLock()
        self._notes: list[Note] = []
        self.version = 0  # bumps on every change so search indexes know to rebuild

    # backends override these
    def _load(self) -> list[Note]:
        return []

    def _save(self, notes: list[Note], message: str) -> None:
        pass

    def _ensure_ready(self) -> None:
        """Called (under the lock) before any change. Backends that can fail to load reload here."""

    def reload(self) -> None:
        with self._lock:
            self._notes = self._load()
            self.version += 1

    def all(self) -> list[Note]:
        with self._lock:
            return list(self._notes)

    def get(self, note_id: str) -> Note | None:
        with self._lock:
            return next((n for n in self._notes if n.id == note_id), None)

    def add(self, text: str) -> Note:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("The note is empty.")
        if len(text) > MAX_NOTE_CHARS:
            raise ValueError(f"Notes must be under {MAX_NOTE_CHARS} characters. Split it into smaller notes.")
        with self._lock:
            self._ensure_ready()
            if len(self._notes) >= MAX_NOTES:
                raise ValueError("Memory is full. Remove some notes first.")
            note = Note(_next_id(self._notes), text, _now())
            new = self._notes + [note]
            self._save(new, f"Remember {note.id}: {text[:60]}")
            self._notes = new
            self.version += 1
            return note

    def delete(self, note_id: str) -> Note:
        with self._lock:
            self._ensure_ready()
            note = self.get(note_id)
            if note is None:
                raise KeyError(note_id)
            new = [n for n in self._notes if n.id != note_id]
            self._save(new, f"Forget {note_id}")
            self._notes = new
            self.version += 1
            return note


class MemoryNotesStore(NotesStore):
    def __init__(self, notes: list[str] | None = None):
        super().__init__()
        for t in notes or []:
            self.add(t)


class FileNotesStore(NotesStore):
    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.reload()

    def _load(self) -> list[Note]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [Note(**n) for n in data.get("notes", [])]

    def _save(self, notes: list[Note], message: str) -> None:
        self.path.write_text(json.dumps({"notes": [n.to_dict() for n in notes]}, indent=2, ensure_ascii=False), encoding="utf-8")


class GitHubNotesStore(NotesStore):
    """notes.json on a branch of a GitHub repo, read and written through the contents API."""

    API = "https://api.github.com"

    def __init__(self, repo: str, token: str, branch: str = "notes", path: str = "notes.json", opener=None):
        super().__init__()
        self.repo, self.token, self.branch, self.path = repo, token, branch, path
        self._sha: str | None = None
        self._loaded = False
        self._open = opener or urllib.request.urlopen
        try:
            self.reload()
        except NotesUnavailable:
            pass  # start empty and read-only; add() retries the load before writing

    def _request(self, method: str, body: dict | None = None):
        url = f"{self.API}/repos/{self.repo}/contents/{self.path}"
        if method == "GET":
            url += f"?ref={self.branch}"
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None, headers={
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ask-harsha",
            **({"Content-Type": "application/json"} if body else {}),
        })
        with self._open(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def _load(self) -> list[Note]:
        try:
            data = self._request("GET")
        except urllib.error.HTTPError as e:
            if e.code == 404:  # no file yet: start empty, first save creates it
                self._sha, self._loaded = None, True
                return []
            raise NotesUnavailable(f"GitHub returned {e.code} reading notes") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise NotesUnavailable(f"Could not reach GitHub: {e}") from e
        self._sha = data.get("sha")
        content = base64.b64decode(data.get("content", "")).decode("utf-8") or '{"notes": []}'
        self._loaded = True
        return [Note(**n) for n in json.loads(content).get("notes", [])]

    def _ensure_ready(self) -> None:
        if not self._loaded:  # startup load failed: never write over a file we couldn't read
            self._notes = self._load()
            self.version += 1

    def _save(self, notes: list[Note], message: str) -> None:
        body = {
            "message": message,
            "branch": self.branch,
            "content": base64.b64encode(json.dumps({"notes": [n.to_dict() for n in notes]}, indent=2, ensure_ascii=False).encode()).decode(),
        }
        if self._sha:
            body["sha"] = self._sha
        try:
            data = self._request("PUT", body)
            self._sha = data["content"]["sha"]
        except urllib.error.HTTPError as e:
            if e.code in (409, 422):  # the file changed elsewhere (e.g. edited on GitHub): reload, don't clobber
                self._notes = self._load()
                self.version += 1
                raise NotesUnavailable("Memory changed elsewhere, so it was reloaded. Please try again.") from e
            raise NotesUnavailable(f"GitHub returned {e.code} saving notes") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise NotesUnavailable(f"Could not reach GitHub: {e}") from e


class NotesIndex:
    """BM25 search over notes, rebuilt automatically when the store changes."""

    def __init__(self, store: NotesStore):
        self.store = store
        self._version = -1
        self._kb: KnowledgeBase | None = None

    def _refresh(self) -> None:
        if self._version == self.store.version:
            return
        notes = self.store.all()
        facts = [Fact(id=n.id, topic="note", text=n.text) for n in notes]
        self._kb = KnowledgeBase(facts) if facts else None
        self._version = self.store.version

    def search(self, query: str, k: int = 5) -> list[tuple[Note, float]]:
        self._refresh()
        if self._kb is None:
            return []
        return [(self.store.get(f.id), s) for f, s in self._kb.search(query, k=k) if self.store.get(f.id)]
