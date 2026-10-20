"""Accounts with passkeys, and saved chats for signed-in people.

Sign-in uses passkeys (WebAuthn): the person's device creates a key pair for this site only, unlocked with Face ID,
a fingerprint or the device PIN. The server keeps only the public key. No passwords, emails or names are needed;
an optional nickname labels the passkey in their password manager.

Sessions are random tokens in an HttpOnly cookie; the database stores only a SHA-256 hash of each token.

Storage is Postgres when DATABASE_URL is set (Supabase in production), otherwise SQLite (local runs and tests).
People who aren't signed in can chat, but nothing of theirs is stored.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

MAX_CHATS = 300
MAX_CHAT_BYTES = 1_500_000
SESSION_DAYS = 90
CHALLENGE_SECONDS = 300

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT NOT NULL, created BIGINT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS credentials (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
       public_key TEXT NOT NULL, sign_count BIGINT NOT NULL DEFAULT 0, created BIGINT NOT NULL, last_used BIGINT)""",
    """CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
       created BIGINT NOT NULL, expires BIGINT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS chats (user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, id TEXT NOT NULL,
       title TEXT NOT NULL, created BIGINT NOT NULL, updated BIGINT NOT NULL, body TEXT NOT NULL, PRIMARY KEY (user_id, id))""",
    """CREATE INDEX IF NOT EXISTS chats_user_updated ON chats (user_id, updated)""",
    """CREATE INDEX IF NOT EXISTS credentials_user ON credentials (user_id)""",
]


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def unb64u(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def now_ms() -> int:
    return int(time.time() * 1000)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Database:
    """A tiny wrapper so the same SQL runs on Postgres (psycopg, %s) and SQLite (?)."""

    def __init__(self, url: str = ""):
        self.lock = threading.Lock()
        if url.startswith(("postgres://", "postgresql://")):
            import psycopg  # imported only when Postgres is used
            self.kind = "postgres"
            # prepare_threshold=None: works through Supabase's connection pooler
            self.conn = psycopg.connect(url, autocommit=True, prepare_threshold=None)
        else:
            self.kind = "sqlite"
            self.conn = sqlite3.connect(url.removeprefix("sqlite:///") or ":memory:", check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")
        for stmt in SCHEMA:
            self.run(stmt)

    def _sql(self, sql: str) -> str:
        return sql if self.kind == "postgres" else sql.replace("%s", "?")

    def run(self, sql: str, args: tuple = ()) -> None:
        with self.lock:
            self._reconnect_if_closed()
            cur = self.conn.cursor()
            cur.execute(self._sql(sql), args)
            if self.kind == "sqlite":
                self.conn.commit()

    def all(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self.lock:
            self._reconnect_if_closed()
            cur = self.conn.cursor()
            cur.execute(self._sql(sql), args)
            return list(cur.fetchall())

    def one(self, sql: str, args: tuple = ()) -> tuple | None:
        rows = self.all(sql, args)
        return rows[0] if rows else None

    def _reconnect_if_closed(self) -> None:
        if self.kind == "postgres" and self.conn.closed:
            import psycopg
            self.conn = psycopg.connect(self.conn.info.dsn, autocommit=True, prepare_threshold=None)


@dataclass
class User:
    id: str
    name: str
    created: int


class Accounts:
    def __init__(self, db: Database, rp_name: str = "Ask Harsha"):
        self.db = db
        self.rp_name = rp_name
        self._challenges: dict[str, tuple[float, bytes, dict]] = {}
        self._lock = threading.Lock()

    # ---- one-time challenges for passkey ceremonies (kept in memory for 5 minutes) ----
    def _remember(self, challenge: bytes, data: dict) -> str:
        cid = secrets.token_urlsafe(16)
        now = time.monotonic()
        with self._lock:
            for k in [k for k, (t, _, _) in self._challenges.items() if now - t > CHALLENGE_SECONDS]:
                del self._challenges[k]
            self._challenges[cid] = (now, challenge, data)
        return cid

    def _take(self, cid: str) -> tuple[bytes, dict]:
        with self._lock:
            item = self._challenges.pop(cid, None)
        if not item or time.monotonic() - item[0] > CHALLENGE_SECONDS:
            raise ValueError("That sign-in took too long or was already used. Please try again.")
        return item[1], item[2]

    # ---- passkey registration (create an account) ----
    def registration_options(self, rp_id: str, nickname: str) -> tuple[str, str]:
        from webauthn import generate_registration_options, options_to_json
        from webauthn.helpers.structs import (AuthenticatorSelectionCriteria, ResidentKeyRequirement,
                                              UserVerificationRequirement)
        name = " ".join(nickname.split())[:40] or "Ask Harsha user"
        user_id = secrets.token_bytes(16)
        opts = generate_registration_options(
            rp_id=rp_id, rp_name=self.rp_name, user_id=user_id, user_name=name, user_display_name=name,
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.REQUIRED, user_verification=UserVerificationRequirement.PREFERRED))
        cid = self._remember(opts.challenge, {"user_id": b64u(user_id), "name": name, "rp_id": rp_id})
        return cid, options_to_json(opts)

    def register(self, cid: str, credential: dict, origin: str) -> User:
        from webauthn import verify_registration_response
        challenge, data = self._take(cid)
        v = verify_registration_response(credential=credential, expected_challenge=challenge,
                                         expected_rp_id=data["rp_id"], expected_origin=origin)
        user = User(data["user_id"], data["name"], now_ms())
        self.db.run("INSERT INTO users (id, name, created) VALUES (%s, %s, %s)", (user.id, user.name, user.created))
        self.db.run("INSERT INTO credentials (id, user_id, public_key, sign_count, created) VALUES (%s, %s, %s, %s, %s)",
                    (b64u(v.credential_id), user.id, b64u(v.credential_public_key), v.sign_count, now_ms()))
        return user

    # ---- passkey sign-in ----
    def authentication_options(self, rp_id: str) -> tuple[str, str]:
        from webauthn import generate_authentication_options, options_to_json
        from webauthn.helpers.structs import UserVerificationRequirement
        opts = generate_authentication_options(rp_id=rp_id, user_verification=UserVerificationRequirement.PREFERRED)
        return self._remember(opts.challenge, {"rp_id": rp_id}), options_to_json(opts)

    def authenticate(self, cid: str, credential: dict, origin: str) -> User:
        from webauthn import verify_authentication_response
        challenge, data = self._take(cid)
        cred_id = str(credential.get("id", ""))
        row = self.db.one("SELECT user_id, public_key, sign_count FROM credentials WHERE id = %s", (cred_id,))
        if not row:
            raise ValueError("This passkey isn't linked to an account here. It may have been deleted.")
        v = verify_authentication_response(credential=credential, expected_challenge=challenge, expected_rp_id=data["rp_id"],
                                           expected_origin=origin, credential_public_key=unb64u(row[1]),
                                           credential_current_sign_count=int(row[2]))
        self.db.run("UPDATE credentials SET sign_count = %s, last_used = %s WHERE id = %s", (v.new_sign_count, now_ms(), cred_id))
        return self.user(row[0])

    # ---- sessions ----
    def start_session(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        self.db.run("INSERT INTO sessions (token_hash, user_id, created, expires) VALUES (%s, %s, %s, %s)",
                    (token_hash(token), user.id, now_ms(), now_ms() + SESSION_DAYS * 86_400_000))
        return token

    def session_user(self, token: str) -> User | None:
        if not token:
            return None
        row = self.db.one("SELECT u.id, u.name, u.created, s.expires FROM sessions s JOIN users u ON u.id = s.user_id "
                          "WHERE s.token_hash = %s", (token_hash(token),))
        if not row or row[3] < now_ms():
            return None
        return User(row[0], row[1], row[2])

    def end_session(self, token: str) -> None:
        if token:
            self.db.run("DELETE FROM sessions WHERE token_hash = %s", (token_hash(token),))

    def user(self, user_id: str) -> User:
        row = self.db.one("SELECT id, name, created FROM users WHERE id = %s", (user_id,))
        if not row:
            raise ValueError("Account not found.")
        return User(*row)

    def delete_user(self, user_id: str) -> None:
        for table in ("chats", "sessions", "credentials"):
            self.db.run(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
        self.db.run("DELETE FROM users WHERE id = %s", (user_id,))

    # ---- chats ----
    @staticmethod
    def _summary(chat_id: str, title: str, created: int, updated: int, body: dict) -> dict:
        msgs = body.get("messages") or []
        answers = [m.get("data") or {} for m in msgs if m.get("role") == "assistant"]
        return {"id": chat_id, "title": title, "created": created, "updated": updated, "wx": body.get("wx"),
                "questions": sum(1 for m in msgs if m.get("role") == "user"),
                "last_answer": (answers[-1].get("answer", "") if answers else "")[:400],
                "labels": [[a.get("label"), bool(a.get("grounded"))] for a in answers][-20:]}

    def list_chats(self, user_id: str, search: str = "") -> list[dict]:
        rows = self.db.all("SELECT id, title, created, updated, body FROM chats WHERE user_id = %s ORDER BY updated DESC LIMIT %s",
                           (user_id, MAX_CHATS))
        out = []
        term = search.strip().lower()
        for cid, title, created, updated, body in rows:
            if term and term not in title.lower() and term not in body.lower():
                continue
            out.append(self._summary(cid, title, created, updated, json.loads(body)))
        return out

    def get_chat(self, user_id: str, chat_id: str) -> dict | None:
        row = self.db.one("SELECT id, title, created, updated, body FROM chats WHERE user_id = %s AND id = %s", (user_id, chat_id))
        if not row:
            return None
        body = json.loads(row[4])
        return {"id": row[0], "title": row[1], "created": row[2], "updated": row[3], "wx": body.get("wx"), "messages": body.get("messages", [])}

    def save_chat(self, user_id: str, chat: dict) -> dict:
        chat_id = str(chat.get("id", ""))[:40]
        if not chat_id:
            raise ValueError("Missing chat id.")
        title = " ".join(str(chat.get("title", "")).split())[:120] or "Untitled chat"
        body = json.dumps({"wx": chat.get("wx"), "messages": (chat.get("messages") or [])[-200:]}, ensure_ascii=False)
        if len(body.encode()) > MAX_CHAT_BYTES:
            raise ValueError("This chat is too long to save. Start a new chat to keep going.")
        created = int(chat.get("created") or now_ms())
        updated = int(chat.get("updated") or now_ms())
        exists = self.db.one("SELECT 1 FROM chats WHERE user_id = %s AND id = %s", (user_id, chat_id))
        if exists:
            self.db.run("UPDATE chats SET title = %s, updated = %s, body = %s WHERE user_id = %s AND id = %s",
                        (title, updated, body, user_id, chat_id))
        else:
            self.db.run("INSERT INTO chats (user_id, id, title, created, updated, body) VALUES (%s, %s, %s, %s, %s, %s)",
                        (user_id, chat_id, title, created, updated, body))
            count = self.db.one("SELECT COUNT(*) FROM chats WHERE user_id = %s", (user_id,))[0]
            if count > MAX_CHATS:  # keep the newest MAX_CHATS
                self.db.run("DELETE FROM chats WHERE user_id = %s AND id IN (SELECT id FROM chats WHERE user_id = %s "
                            "ORDER BY updated ASC LIMIT %s)", (user_id, user_id, count - MAX_CHATS))
        return {"id": chat_id, "title": title, "updated": updated}

    def rename_chat(self, user_id: str, chat_id: str, title: str) -> None:
        self.db.run("UPDATE chats SET title = %s WHERE user_id = %s AND id = %s",
                    (" ".join(title.split())[:120] or "Untitled chat", user_id, chat_id))

    def delete_chat(self, user_id: str, chat_id: str) -> None:
        self.db.run("DELETE FROM chats WHERE user_id = %s AND id = %s", (user_id, chat_id))

    def delete_all_chats(self, user_id: str) -> None:
        self.db.run("DELETE FROM chats WHERE user_id = %s", (user_id,))

    def export(self, user_id: str) -> list[dict]:
        rows = self.db.all("SELECT id FROM chats WHERE user_id = %s ORDER BY updated DESC", (user_id,))
        return [c for c in (self.get_chat(user_id, r[0]) for r in rows) if c]


def connect(url: str) -> Accounts | None:
    """Accounts on the given database, or None when no database is configured."""
    if not url:
        return None
    return Accounts(Database(url))


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x))
