"""Verified answers: every answer is signed, so a shared copy can be proven real and an edited one caught.

How it works:
  1. When the agent answers, the server builds a receipt (question, answer, model, time, sources, grounding result)
     and signs it with an Ed25519 private key. The page gets the receipt and the signature with the answer.
  2. If the person taps "Share verified", the page sends that pair back. The server checks the signature, which
     proves the server itself produced exactly that answer, and stores it under a random id: /v/<id>.
  3. Anyone opening /v/<id> sees the original, the signature status, and can paste what they were shown to check
     it word for word. The public key is published at /.well-known/ask-harsha-key.json, so anyone can check a
     downloaded receipt without trusting this server.

Keys: SIGNING_KEY (base64 of 32 random bytes) if set; otherwise derived from INBOX_KEY or the owner passcode with
scrypt. Changing the secret changes the key, so set SIGNING_KEY if you plan to change the passcode.

Storage: shared receipts are files on the `notes` branch (receipts/<id>.json), encrypted with the inbox key,
because the repository is public. Only people with the link can read one. Nothing is stored unless someone shares.
"""
from __future__ import annotations

import base64
import functools
import hashlib
import json
import secrets
import urllib.error
import urllib.request
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

VERSION = 1
MAX_RECEIPT_BYTES = 40_000


class ReceiptError(Exception):
    """Shown to the visitor as is."""


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64u(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def canonical(receipt: dict) -> bytes:
    """One exact byte form of a receipt, so the same content always gives the same signature."""
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class Signer:
    def __init__(self, seed: bytes):
        self.key = Ed25519PrivateKey.from_private_bytes(seed)
        self.public = self.key.public_key()
        raw = self.public.public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.public_b64 = _b64u(raw)
        self.fingerprint = hashlib.sha256(raw).hexdigest()[:16]

    @classmethod
    @functools.lru_cache(maxsize=8)
    def from_secrets(cls, signing_key: str = "", inbox_key: str = "", owner_passcode: str = "") -> "Signer | None":
        if signing_key:
            try:
                seed = base64.b64decode(signing_key)
            except ValueError:
                seed = b""
            if len(seed) != 32:
                seed = hashlib.sha256(signing_key.encode()).digest()
        elif inbox_key or owner_passcode:
            seed = hashlib.scrypt((inbox_key or owner_passcode).encode(), salt=b"ask-harsha-signing-v1", n=2 ** 14, r=8, p=1, dklen=32)
        else:
            return None
        return cls(seed)

    def sign(self, receipt: dict) -> str:
        return _b64u(self.key.sign(canonical(receipt)))

    def verify(self, receipt: dict, signature: str) -> bool:
        try:
            self.public.verify(_unb64u(signature), canonical(receipt))
            return True
        except (InvalidSignature, ValueError):
            return False


def verify_with_public_key(public_b64: str, receipt: dict, signature: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(_unb64u(public_b64)).verify(_unb64u(signature), canonical(receipt))
        return True
    except (InvalidSignature, ValueError):
        return False


def build(question: str, answer: str, model: str, label: str, grounded: bool, citations: list[dict],
          web_sources: list[dict], attachments: list[str], key_fingerprint: str) -> dict:
    return {
        "v": VERSION,
        "type": "ask-harsha-answer",
        "nonce": secrets.token_urlsafe(8),
        "issued_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": model,
        "question": question[:2000],
        "answer": answer,
        "label": label,
        "grounded": bool(grounded),
        "citations": [{"id": c["id"], "text": c["text"]} for c in citations][:20],
        "web_sources": [{"url": w.get("url", ""), "title": w.get("title", "")} for w in web_sources][:8],
        "attachments": list(attachments)[:4],
        "key": key_fingerprint,
    }


class ReceiptStore:
    """Encrypted receipts by id. Base class keeps them in memory (tests)."""

    def __init__(self, fernet_key: bytes | None):
        self.fernet = Fernet(fernet_key) if fernet_key else None
        self._files: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self.fernet is not None

    def _read(self, rid: str) -> str | None:
        return self._files.get(rid)

    def _write(self, rid: str, text: str) -> None:
        self._files[rid] = text

    def put(self, receipt: dict, signature: str) -> str:
        rid = secrets.token_urlsafe(9)
        body = json.dumps({"receipt": receipt, "sig": signature}, ensure_ascii=False).encode()
        self._write(rid, json.dumps({"v": 1, "enc": self.fernet.encrypt(body).decode()}))
        return rid

    def get(self, rid: str) -> dict | None:
        text = self._read(rid)
        if not text:
            return None
        try:
            return json.loads(self.fernet.decrypt(json.loads(text)["enc"].encode()))
        except (InvalidToken, KeyError, ValueError):
            raise ReceiptError("This receipt can't be opened with the current key.")


class GitHubReceiptStore(ReceiptStore):
    API = "https://api.github.com"

    def __init__(self, repo: str, token: str, fernet_key: bytes | None, branch: str = "notes", opener=None):
        super().__init__(fernet_key)
        self.repo, self.token, self.branch = repo, token, branch
        self._open = opener or urllib.request.urlopen

    def _req(self, method: str, path: str, body: dict | None = None):
        url = f"{self.API}/repos/{self.repo}/contents/{path}" + (f"?ref={self.branch}" if method == "GET" else "")
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None, headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "ask-harsha",
            **({"Content-Type": "application/json"} if body else {}),
        })
        with self._open(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def _read(self, rid):
        try:
            data = self._req("GET", f"receipts/{rid}.json")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise ReceiptError(f"GitHub returned {e.code} reading the receipt.") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ReceiptError("Couldn't reach storage to read the receipt.") from e
        return base64.b64decode(data.get("content", "")).decode()

    def _write(self, rid, text):
        try:
            self._req("PUT", f"receipts/{rid}.json", {"message": "Shared receipt", "branch": self.branch,
                                                       "content": base64.b64encode(text.encode()).decode()})
        except urllib.error.HTTPError as e:
            raise ReceiptError(f"GitHub returned {e.code} saving the receipt.") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ReceiptError("Couldn't reach storage to save the receipt.") from e
