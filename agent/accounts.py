"""Accounts with passkeys, and saved chats for signed-in people.

Sign-in uses passkeys (WebAuthn): the person's device creates a key pair for this site only, unlocked with Face ID,
a fingerprint or the device PIN. The server keeps only the public key. No passwords, emails or names are needed;
an optional nickname labels the passkey in their password manager.

Sessions are random tokens in an HttpOnly cookie; the database stores only a SHA-256 hash of each token.

Recovery, without email: every account gets a one-time recovery code (20 characters, about 98 bits) shown once when
the passkey is made. Only its hash is stored. Using it signs the person in on a new device, signs out every other
session, and is used up; adding a passkey afterwards issues a fresh code. People can also add more than one passkey
(say, iPhone and Windows Hello) so losing one device doesn't lock them out.

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
    """CREATE TABLE IF NOT EXISTS recovery (user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
       code_hash TEXT NOT NULL UNIQUE, created BIGINT NOT NULL)""",
]
# Columns added after the first release. Each runs once; "already exists" errors are ignored.
MIGRATIONS = [
    "ALTER TABLE credentials ADD COLUMN label TEXT",
]
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"   # no 0/O, 1/I/L or U, so codes are easy to read and type
CODE_LENGTH = 20


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def unb64u(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def now_ms() -> int:
    return int(time.time() * 1000)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_recovery_code() -> str:
    raw = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
    return "-".join(raw[i:i + 4] for i in range(0, CODE_LENGTH, 4))


def normalize_code(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch.isalnum())


def code_hash(code: str) -> str:
    return hashlib.sha256(("askh-recovery:" + normalize_code(code)).encode()).hexdigest()


def device_label(user_agent: str) -> str:
    """A friendly name for where a passkey was made, from the browser's user agent ("iPhone", "Windows")."""
    ua = user_agent or ""
    for needle, name in (("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"), ("CrOS", "Chromebook"),
                         ("Windows", "Windows"), ("Macintosh", "Mac"), ("Linux", "Linux")):
        if needle in ua:
            return name
    return "Device"


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
        for stmt in MIGRATIONS:
            try:
                self.run(stmt)
            except Exception:  # the column is already there
                pass

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

    def register(self, cid: str, credential: dict, origin: str, label: str = "") -> User:
        from webauthn import verify_registration_response
        challenge, data = self._take(cid)
        if data.get("add"):
            raise ValueError("That request was for adding a passkey. Please try again.")
        v = verify_registration_response(credential=credential, expected_challenge=challenge,
                                         expected_rp_id=data["rp_id"], expected_origin=origin)
        user = User(data["user_id"], data["name"], now_ms())
        self.db.run("INSERT INTO users (id, name, created) VALUES (%s, %s, %s)", (user.id, user.name, user.created))
        self._save_credential(user.id, v, label)
        return user

    def _save_credential(self, user_id: str, v, label: str) -> None:
        self.db.run("INSERT INTO credentials (id, user_id, public_key, sign_count, created, label) VALUES (%s, %s, %s, %s, %s, %s)",
                    (b64u(v.credential_id), user_id, b64u(v.credential_public_key), v.sign_count, now_ms(), label[:40] or None))

    # ---- more passkeys on the same account (a second device, so losing one isn't the end) ----
    def add_passkey_options(self, user: User, rp_id: str) -> tuple[str, str]:
        from webauthn import generate_registration_options, options_to_json
        from webauthn.helpers.structs import (AuthenticatorSelectionCriteria, PublicKeyCredentialDescriptor,
                                              ResidentKeyRequirement, UserVerificationRequirement)
        existing = [PublicKeyCredentialDescriptor(id=unb64u(r[0]))
                    for r in self.db.all("SELECT id FROM credentials WHERE user_id = %s", (user.id,))]
        opts = generate_registration_options(
            rp_id=rp_id, rp_name=self.rp_name, user_id=unb64u(user.id), user_name=user.name, user_display_name=user.name,
            exclude_credentials=existing,
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.REQUIRED, user_verification=UserVerificationRequirement.PREFERRED))
        cid = self._remember(opts.challenge, {"user_id": user.id, "rp_id": rp_id, "add": True})
        return cid, options_to_json(opts)

    def add_passkey(self, user: User, cid: str, credential: dict, origin: str, label: str = "") -> None:
        from webauthn import verify_registration_response
        challenge, data = self._take(cid)
        if not data.get("add") or data.get("user_id") != user.id:
            raise ValueError("That request doesn't match your account. Please try again.")
        v = verify_registration_response(credential=credential, expected_challenge=challenge,
                                         expected_rp_id=data["rp_id"], expected_origin=origin)
        if self.db.one("SELECT 1 FROM credentials WHERE id = %s", (b64u(v.credential_id),)):
            raise ValueError("This passkey is already on your account.")
        self._save_credential(user.id, v, label)

    def passkeys(self, user_id: str) -> list[dict]:
        rows = self.db.all("SELECT id, label, created, last_used FROM credentials WHERE user_id = %s ORDER BY created", (user_id,))
        return [{"id": r[0], "label": r[1] or "Passkey", "created": r[2], "last_used": r[3]} for r in rows]

    def remove_passkey(self, user_id: str, cred_id: str) -> None:
        if not self.db.one("SELECT 1 FROM credentials WHERE user_id = %s AND id = %s", (user_id, cred_id)):
            raise ValueError("That passkey isn't on your account.")
        if self.db.one("SELECT COUNT(*) FROM credentials WHERE user_id = %s", (user_id,))[0] <= 1:
            raise ValueError("This is your only passkey. Add another one before removing it.")
        self.db.run("DELETE FROM credentials WHERE user_id = %s AND id = %s", (user_id, cred_id))

    # ---- recovery code ----
    def make_recovery_code(self, user_id: str) -> str:
        """A new one-time recovery code for this account. Any older code stops working."""
        code = new_recovery_code()
        self.db.run("DELETE FROM recovery WHERE user_id = %s", (user_id,))
        self.db.run("INSERT INTO recovery (user_id, code_hash, created) VALUES (%s, %s, %s)", (user_id, code_hash(code), now_ms()))
        return code

    def recovery_status(self, user_id: str) -> int | None:
        """When the current recovery code was made, or None if the account has none."""
        row = self.db.one("SELECT created FROM recovery WHERE user_id = %s", (user_id,))
        return row[0] if row else None

    def recover(self, code: str) -> User:
        """Sign in with a recovery code. The code is used up and every other session is signed out."""
        if len(normalize_code(code)) != CODE_LENGTH:
            raise ValueError("Recovery codes are 20 letters and numbers, like ABCD-EFGH-JKMN-PQRS-TVWX.")
        row = self.db.one("SELECT user_id FROM recovery WHERE code_hash = %s", (code_hash(code),))
        if not row:
            raise ValueError("That recovery code didn't work. Check it, or it may already have been used.")
        self.db.run("DELETE FROM recovery WHERE user_id = %s", (row[0],))
        self.db.run("DELETE FROM sessions WHERE user_id = %s", (row[0],))
        return self.user(row[0])

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
        for table in ("chats", "sessions", "credentials", "recovery"):
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


def fix_password(url: str) -> str:
    """Percent-encode the password in a postgres URL, so passwords with @ # / ? % : still work when pasted as is."""
    from urllib.parse import quote, unquote
    url = url.strip()
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    userinfo, host = rest.rsplit("@", 1)
    if ":" not in userinfo:
        return url
    user, password = userinfo.split(":", 1)
    password = password.strip("[]") if password.startswith("[") and password.endswith("]") and password != "[YOUR-PASSWORD]" else password
    return f"{scheme}://{user}:{quote(unquote(password), safe='')}@{host}"


def connect(url: str) -> Accounts | None:
    """Accounts on the given database, or None when no database is configured."""
    if not url:
        return None
    if "[YOUR-PASSWORD]" in url:
        raise ValueError("DATABASE_URL still contains [YOUR-PASSWORD]. Replace it with your Supabase database password.")
    return Accounts(Database(fix_password(url)))


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x))
