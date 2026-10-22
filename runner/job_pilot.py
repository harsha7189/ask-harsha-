#!/usr/bin/env python3
"""Job Pilot's laptop runner: fills in and (once dry runs look right) submits the applications Ask Harsha prepared.

Why a laptop: job sites block form-filling from cloud servers, and your laptop has a normal home connection.

What it does for each ready application, at night:
  1. Opens the posting in a real Chromium browser (its own profile, separate from your everyday browser).
  2. Gets to the form: Greenhouse, Lever and Ashby directly; Workday and iCIMS through their Apply flow, creating an
     account if needed (a unique strong password per site, saved in your system keychain) and clicking the email
     verification link (read over IMAP with a Gmail app password, only messages from those hiring systems).
  3. Fills every page: it reads each field's label, options and whether it's required, asks Ask Harsha for the
     answers (which come from your answer sheet or that job's tailored résumé, quoted), uploads the tailored résumé,
     and stops the application if a required question has no supported answer.
  4. Clicks Submit, waits for the confirmation, and reports "submitted" with a screenshot as proof. (If dry run is
     turned on in Job Pilot's settings, it stops before Submit and reports "filled" instead.)
  5. After the night's applications, reads your inbox here on the laptop for replies to earlier applications
     (rejections, recruiter replies, interview requests) and sends only "which application, what outcome, when".
It never solves CAPTCHAs or gets around bot checks: an application that shows one waits for you.

Setup:
  pip install playwright keyring && python -m playwright install chromium
  python job_pilot.py setup        # server, runner token, email for accounts, Gmail app password (optional)
  python job_pilot.py run          # process tonight's ready applications (respects the nightly cap)
  python job_pilot.py schedule     # run it every night at 1:00
"""
from __future__ import annotations

import argparse
import base64
import email
import email.utils
import imaplib
import json
import os
import random
import re
import secrets
import string
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta
from email.header import decode_header
from pathlib import Path

HOME = Path(os.environ.get("JOB_PILOT_HOME", Path.home() / ".job_pilot"))
CONFIG = HOME / "config.json"
SERVICE = "ask-harsha-job-pilot"

CONFIRMED = re.compile(r"thanks? (you )?for (applying|your application|submitting)|application (has been |was )?(successfully )?(submitted|received)|"
                       r"we('ve| have) received your application|successfully (submitted|applied)", re.I)
SUBMIT_TEXT = re.compile(r"^\s*(submit( application)?|send application|apply|finish|complete application)\s*$", re.I)
NEXT_TEXT = re.compile(r"^\s*(save and continue|continue|next|save & continue)\s*$", re.I)


# ---------------------------------------------------------------- talking to Ask Harsha
class Api:
    def __init__(self, server: str, token: str, opener=None):
        self.server, self.token = server.rstrip("/"), token
        self.open = opener or urllib.request.urlopen

    def _req(self, method: str, path: str, body=None, raw=False):
        req = urllib.request.Request(self.server + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"X-Runner-Token": self.token, "Content-Type": "application/json", "User-Agent": "job-pilot-runner"})
        with self.open(req, timeout=120) as r:
            data = r.read()
        return data if raw else json.loads(data.decode() or "{}")

    def next(self):
        return self._req("GET", "/api/jobs/runner/next")

    def file(self, app_id: str, kind: str) -> bytes:
        return self._req("GET", f"/api/jobs/runner/file/{app_id}/{kind}", raw=True)

    def answers(self, app_id: str, fields: list[dict]) -> list[dict]:
        return self._req("POST", "/api/jobs/runner/answers", {"app_id": app_id, "fields": fields})["answers"]

    def step(self, app_id: str, snapshot: dict) -> dict:
        return self._req("POST", "/api/jobs/runner/step", {"app_id": app_id, **snapshot})

    def watch(self) -> list[dict]:
        return self._req("GET", "/api/jobs/runner/watch").get("apps", [])

    def outcomes(self, items: list[dict]) -> dict:
        return self._req("POST", "/api/jobs/runner/outcomes", {"items": items})

    def report(self, app_id: str, status: str, step: str = "", detail: str = "", screenshots=None, confirmation: str = ""):
        return self._req("POST", "/api/jobs/runner/report", {"app_id": app_id, "status": status, "step": step[:200], "detail": detail[:2000],
                                                             "screenshots": screenshots or [], "confirmation": confirmation[:4000]})


# ---------------------------------------------------------------- passwords and email verification
class Keys:
    """Per-site passwords in the system keychain (macOS Keychain, Windows Credential Manager)."""

    def __init__(self, backend=None):
        if backend is None:
            import keyring as backend
        self.k = backend

    def password_for(self, site: str, user: str) -> tuple[str, bool]:
        """(password, is_new)."""
        existing = self.k.get_password(SERVICE, f"{site}:{user}")
        if existing:
            return existing, False
        alphabet = string.ascii_letters + string.digits
        pw = "".join(secrets.choice(alphabet) for _ in range(14)) + secrets.choice("!@#$%") + secrets.choice(string.digits) + "Aa"
        self.k.set_password(SERVICE, f"{site}:{user}", pw)
        return pw, True

    def get(self, name: str) -> str | None:
        return self.k.get_password(SERVICE, name)

    def set(self, name: str, value: str) -> None:
        self.k.set_password(SERVICE, name, value)


class Mailbox:
    """Reads only verification emails from hiring systems, over IMAP (Gmail: imap.gmail.com with an app password)."""
    SENDERS = ("myworkday", "workday", "icims", "greenhouse", "lever", "ashby")

    def __init__(self, user: str, app_password: str, host: str = "imap.gmail.com"):
        self.user, self.pw, self.host = user, app_password, host

    def verification_link(self, wait_s: int = 180, since: datetime | None = None) -> str | None:
        since = since or datetime.now() - timedelta(minutes=10)
        end = time.time() + wait_s
        while time.time() < end:
            try:
                m = imaplib.IMAP4_SSL(self.host)
                m.login(self.user, self.pw)
                m.select("INBOX")
                _, ids = m.search(None, f'(SINCE "{since.strftime("%d-%b-%Y")}")')
                for i in reversed(ids[0].split()[-30:]):
                    _, data = m.fetch(i, "(RFC822)")
                    msg = email.message_from_bytes(data[0][1])
                    sender = str(msg.get("From", "")).lower()
                    if not any(s in sender for s in self.SENDERS):
                        continue
                    link = find_verify_link(_body(msg))
                    if link:
                        m.logout()
                        return link
                m.logout()
            except Exception:
                pass
            time.sleep(10)
        return None


    def recent(self, days: int = 14, limit: int = 300) -> list[dict]:
        """Recent inbox messages, read here on the laptop. Only matched outcomes (app, outcome, date) are ever sent anywhere."""
        out = []
        m = imaplib.IMAP4_SSL(self.host)
        m.login(self.user, self.pw)
        m.select("INBOX", readonly=True)
        _, ids = m.search(None, f'(SINCE "{(datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")}")')
        for i in reversed(ids[0].split()[-limit:]):
            _, data = m.fetch(i, "(BODY.PEEK[])")     # PEEK: doesn't mark anything as read
            msg = email.message_from_bytes(data[0][1])
            try:
                at = int(email.utils.parsedate_to_datetime(msg.get("Date")).timestamp() * 1000)
            except Exception:
                at = int(time.time() * 1000)
            out.append({"from": _hdr(msg.get("From", "")), "subject": _hdr(msg.get("Subject", "")), "body": _plain(_body(msg))[:5000], "at": at})
        m.logout()
        return out


def _hdr(v) -> str:
    try:
        return "".join(t.decode(c or "utf-8", "replace") if isinstance(t, bytes) else t for t, c in decode_header(str(v)))
    except Exception:
        return str(v)


def _plain(html_text: str) -> str:
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", html_text or "")
    return " ".join(re.sub(r"<[^>]+>", " ", t).replace("&nbsp;", " ").replace("&amp;", "&").split())


# what an email from an employer means (decided here, on the laptop)
REJECT = re.compile(r"(unfortunately|regret to inform|not (be )?(moving|move) forward|decided to (move|proceed) (forward )?with other|"
                    r"pursue other candidates|position has been filled|no longer (under )?consider|will not be (moving|proceeding)|"
                    r"not selected|other candidates whose)", re.I)
INTERVIEW = re.compile(r"(schedule (a|an|your) (call|interview|chat|conversation|time)|interview (invitation|request)|invite you to (an? )?interview|"
                       r"next steps? in (the|our) (hiring|interview) process|availability for (a|an) (call|interview|chat)|phone screen|"
                       r"book (a )?time|calendly\.com|goodtime\.io|would love to (chat|connect|speak))", re.I)
OFFER = re.compile(r"(pleased to (extend|offer)|offer letter|formal offer)", re.I)
AUTOMATED = re.compile(r"no-?reply|do-?not-?reply|notifications?@|myworkday|icims|greenhouse|lever\.co|ashbyhq|talent@|careers@|jobs@|recruiting@|news(letter)?@|marketing|info@|hello@|updates?@|events?@|team@|support@", re.I)


def _norm(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    n = re.sub(r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|plc|lp|llp|holdings|group|technologies|technology|the)\b", " ", n)
    return " ".join(n.split())


def classify_email(subject: str, body: str, sender: str) -> str | None:
    t = f"{subject}\n{body}"
    if OFFER.search(t):
        return "offer"
    if REJECT.search(t):
        return "rejected"
    if INTERVIEW.search(t):
        return "interview"
    if re.search(r"thank you for (applying|your application|your interest)|received your application|application (was )?received", t, re.I):
        return None                      # a receipt is not a response
    if not AUTOMATED.search(sender) and re.search(r"\b(application|position|role|r[eé]sum[eé]|candidacy|opportunity|your background)\b", t, re.I):
        return "recruiter_reply"         # a real person at the company wrote back
    return None


def match_outcomes(messages: list[dict], watch: list[dict]) -> list[dict]:
    """Ties inbox messages to submitted applications by company name, and decides what each one means."""
    out, best = [], {}
    rank = {"rejected": 1, "recruiter_reply": 2, "interview": 3, "offer": 4}
    for w in watch:
        name = _norm(w.get("company", ""))
        if len(name) < 3:
            continue
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])")
        compact = name.replace(" ", "")
        for msg in messages:
            if msg["at"] < (w.get("submitted_at") or 0):
                continue
            sender = msg["from"].lower()
            head = _norm(msg["from"] + " " + msg["subject"])
            domain = (re.search(r"@([\w.-]+)", sender) or [None, ""])[1]
            if not (pat.search(head) or (compact and compact in domain.replace("-", ""))):
                continue
            o = classify_email(msg["subject"], msg["body"], sender)
            if o and rank[o] > rank.get(best.get(w["id"], {}).get("outcome"), 0) and rank[o] > rank.get(w.get("outcome"), 0):
                best[w["id"]] = {"app_id": w["id"], "outcome": o, "at": msg["at"]}
    out = list(best.values())
    return out


def _body(msg) -> str:
    parts = []
    for p in msg.walk() if msg.is_multipart() else [msg]:
        if p.get_content_type() in ("text/html", "text/plain"):
            try:
                parts.append(p.get_payload(decode=True).decode(p.get_content_charset() or "utf-8", "replace"))
            except Exception:
                continue
    return "\n".join(parts)


def find_verify_link(text: str) -> str | None:
    links = re.findall(r'https?://[^\s"\'<>]+', text or "")
    for l in links:
        if re.search(r"verif|activat|confirm|validate", l, re.I):
            return l.replace("&amp;", "&")
    return None


# ---------------------------------------------------------------- reading a page
SNAP_JS = r"""
() => {
  const vis = e => { if (!e) return false; const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0'; };
  const txt = e => (e ? (e.innerText || e.textContent || '') : '').replace(/\s+/g, ' ').trim();
  const short = (t, n) => (t || '').slice(0, n);
  const byIds = ids => (ids || '').split(/\s+/).map(id => txt(document.getElementById(id))).join(' ').trim();
  const QBOX = 'fieldset, [role=group], [role=radiogroup], [data-automation-id^="formField"], [class*="question" i], [class*="field" i], [class*="Field"], .form-group, li';
  const questionOf = e => {
    const box = e.closest(QBOX); if (!box) return '';
    const l = box.querySelector('legend, label, [class*="label" i], [class*="title" i], h3, h4, p');
    let t = l ? txt(l) : '';
    if (!t) t = txt(box).slice(0, 160);
    return short(t, 200);
  };
  const labelOf = e => {
    if (e.id) { const l = document.querySelector(`label[for="${CSS.escape(e.id)}"]`); if (l && txt(l)) return txt(l); }
    if (e.getAttribute('aria-labelledby')) { const t = byIds(e.getAttribute('aria-labelledby')); if (t) return t; }
    if (e.getAttribute('aria-label')) return e.getAttribute('aria-label');
    const wrap = e.closest('label'); if (wrap && txt(wrap)) return txt(wrap);
    const q = questionOf(e); if (q) return q;
    return e.getAttribute('placeholder') || e.name || '';
  };
  const required = (e, label) => !!(e.required || e.getAttribute('aria-required') === 'true' || /\*\s*$/.test(label || ''));
  document.querySelectorAll('[data-jp]').forEach(x => x.removeAttribute('data-jp'));
  const out = []; let n = 0;
  const mark = (e, item) => { e.setAttribute('data-jp', String(n)); item.i = n++; out.push(item); };
  const seen = new Set();
  // native radio groups as one question
  for (const r of document.querySelectorAll('input[type=radio]')) {
    if (seen.has('r:' + r.name)) continue; seen.add('r:' + r.name);
    const group = [...document.querySelectorAll(`input[type=radio][name="${CSS.escape(r.name)}"]`)];
    if (!group.some(x => vis(x) || vis(x.closest('label')))) continue;
    const q = (r.closest('fieldset') && r.closest('fieldset').querySelector('legend')) ? txt(r.closest('fieldset').querySelector('legend')) : questionOf(r);
    mark(r, {kind: 'radio', name: r.name, label: short(q || r.name, 300), options: group.map(x => short(labelOf(x), 120)),
             value: (group.find(x => x.checked) ? labelOf(group.find(x => x.checked)) : ''), required: group.some(x => x.required)});
  }
  // aria radio groups (custom widgets)
  for (const g of document.querySelectorAll('[role=radiogroup]')) {
    if (!vis(g) || g.querySelector('input[type=radio]')) continue;
    const opts = [...g.querySelectorAll('[role=radio]')];
    const sel = opts.find(o => o.getAttribute('aria-checked') === 'true');
    mark(g, {kind: 'radio', aria: true, label: short(g.getAttribute('aria-label') || byIds(g.getAttribute('aria-labelledby')) || questionOf(g), 300),
             options: opts.map(o => short(txt(o) || o.getAttribute('aria-label'), 120)), value: sel ? txt(sel) : '', required: g.getAttribute('aria-required') === 'true'});
  }
  for (const e of document.querySelectorAll('input, select, textarea, [role=combobox], [aria-haspopup=listbox], [contenteditable=true]')) {
    if (e.getAttribute('data-jp') || e.type === 'radio') continue;
    const tag = e.tagName;
    let kind = tag === 'SELECT' ? 'select' : tag === 'TEXTAREA' ? 'textarea' : (e.getAttribute('role') === 'combobox' || e.getAttribute('aria-haspopup') === 'listbox' || e.getAttribute('aria-autocomplete')) ? 'combobox' : (e.type || 'text').toLowerCase();
    if (['hidden', 'submit', 'button', 'image', 'reset'].includes(kind)) continue;
    if (kind !== 'file' && !vis(e)) continue;
    if (kind === 'combobox' && tag !== 'INPUT' && e.querySelector('input[role=combobox]')) continue;
    const label = short(labelOf(e), 300);
    let options = [], value = '';
    if (kind === 'select') { options = [...e.options].map(o => o.text.trim()).filter(t => t && !/^(select|choose|--|please)/i.test(t)).slice(0, 60); value = e.selectedIndex > 0 ? e.options[e.selectedIndex].text.trim() : ''; }
    else if (kind === 'checkbox') value = e.checked ? 'checked' : '';
    else if (kind === 'file') value = (e.files && e.files.length) ? e.files[0].name : '';
    else if (kind === 'combobox' && tag !== 'INPUT') value = /^(select|choose)/i.test(txt(e)) ? '' : txt(e);
    else value = kind === 'password' ? (e.value ? '••••' : '') : (e.value || '');
    mark(e, {kind, label, options, value: short(value, 200), required: required(e, label), name: e.name || '', ac: e.getAttribute('autocomplete') || ''});
  }
  // visible options of an open dropdown
  for (const o of document.querySelectorAll('[role=option]')) {
    if (!vis(o) || out.length > 260) continue;
    mark(o, {kind: 'option', text: short(txt(o), 120)});
  }
  // buttons and button-like links, with the question they belong to (for Yes/No style buttons)
  for (const b of document.querySelectorAll('button, [role=button], input[type=submit], input[type=button], a')) {
    if (b.getAttribute('data-jp') || !vis(b) || out.length > 260) continue;
    const t = short(txt(b) || b.value || b.getAttribute('aria-label') || '', 80);
    if (!t) continue;
    if (b.tagName === 'A' && !/apply|next|continue|submit|review|sign in|log ?in|create account|start|begin|save/i.test(t)) continue;
    const q = questionOf(b);
    mark(b, {kind: b.tagName === 'A' ? 'link' : 'button', text: t, context: q && q !== t ? q : '',
             pressed: b.getAttribute('aria-pressed') === 'true' || b.getAttribute('aria-checked') === 'true' || /selected|active/i.test(b.className || ''),
             disabled: !!b.disabled || b.getAttribute('aria-disabled') === 'true'});
  }
  const errors = [...document.querySelectorAll('[role=alert], [aria-live=assertive], [class*="error" i], [class*="invalid" i], [data-automation-id*="error" i]')]
    .filter(vis).map(txt).filter(t => t && t.length < 200).slice(0, 12);
  // a CAPTCHA only counts when there's an actual challenge to solve (not the invisible spam-check badge)
  let captcha = false;
  for (const f of document.querySelectorAll('iframe')) {
    const src = f.src || ''; const r = f.getBoundingClientRect();
    if (!vis(f) || r.width < 60 || r.height < 60) continue;
    if (/recaptcha\/(api2|enterprise)\/(anchor|bframe)/.test(src) && !/size=invisible/.test(src)) captcha = true;
    if (/hcaptcha\.com/.test(src) || /challenges\.cloudflare\.com/.test(src)) captcha = true;
  }
  if (/verify you are human|are you a robot|complete the security check/i.test(txt(document.body).slice(0, 4000))) captcha = true;
  return {url: location.href, title: document.title, text: txt(document.body).slice(0, 3500), elements: out, errors, captcha};
}
"""


def human_pause(a: float = 0.4, b: float = 1.4) -> None:
    time.sleep(random.uniform(a, b))


BLOCKED_CLICKS = re.compile(r"(with|via)\s+(linkedin|indeed|google|facebook|apple|seek|glassdoor)|withdraw|delete|sign\s*out|log\s*out|unsubscribe", re.I)


class Runner:
    """Each page is read into a short list of fields and buttons; Ask Harsha decides what to fill and click (values come
    only from his answer sheet and the job's résumé, checked on the server); the runner does it here."""
    MAX_STEPS = 32
    SETTLE_MS = 7000

    def __init__(self, api: Api, keys: Keys, mailbox: Mailbox | None, email_addr: str, page_factory, pause=human_pause, log=print):
        self.api, self.keys, self.mail, self.email = api, keys, mailbox, email_addr
        self.page_factory, self.pause, self.log = page_factory, pause, log

    # ---- one application ----
    def apply(self, job: dict, dry_run: bool, profile: dict) -> str:
        app_id = job["id"]
        page = self.page_factory()
        tmp = Path(tempfile.mkdtemp(prefix="jp_"))
        name = (f"{profile.get('first_name', '')}_{profile.get('last_name', '')}_Resume".strip("_") or "Resume").replace(" ", "_")
        pdf, docx = tmp / f"{name}.pdf", tmp / f"{name}.docx"
        pdf.write_bytes(self.api.file(app_id, "pdf"))
        docx.write_bytes(self.api.file(app_id, "docx"))
        resume = docx if job.get("ats") in ("workday", "icims") else pdf     # Word parses best in Workday and iCIMS
        history: list[str] = []
        last_sig, stuck, step = "", 0, 0
        try:
            # sites that upload through their own "Attach" / "Select files" button open a file picker: it gets the résumé
            page.on("filechooser", lambda fc: fc.set_files(str(resume)))
            page.goto(self.start_url(job), wait_until="domcontentloaded")
            self.settle(page)
            for step in range(1, self.MAX_STEPS + 1):
                body = self.body(page)
                m = CONFIRMED.search(body)
                if m and step > 1:
                    return self.done(app_id, page, body, m)
                snap = page.evaluate(SNAP_JS)
                if snap["captcha"]:
                    return self.stop(app_id, page, "needs_you", f"page {step}", "This site showed a CAPTCHA puzzle. Finish this one yourself; the posting link is in the tracker.")
                sig = snap["url"] + "|" + "|".join(f"{e.get('label', e.get('text', ''))}={e.get('value', '')}" for e in snap["elements"])
                stuck = stuck + 1 if sig == last_sig else 0
                last_sig = sig
                if stuck >= 3:
                    return self.stop(app_id, page, "needs_you", f"page {step}", "Stuck on this page" + (": " + "; ".join(snap["errors"][:3]) if snap["errors"] else "."))
                plan = self.api.step(app_id, {**snap, "step": step, "stuck": stuck, "history": history[-12:], "dry_run": dry_run})
                status, reason = plan.get("status", "continue"), plan.get("reason", "")
                if status == "submitted":
                    m = CONFIRMED.search(body) or re.search(r"thank|received|submitted|success", body, re.I)
                    if m:
                        return self.done(app_id, page, body, m)
                if status == "needs_you":
                    return self.stop(app_id, page, "needs_you", f"page {step}", reason or "It couldn't finish this page.")
                if status == "filled":
                    return self.stop(app_id, page, "filled", "final page", "Dry run: every page is filled; stopped before Submit.")
                if status == "verify_email":
                    self.verify_email(page, job)
                    history.append("opened the email verification link")
                    continue
                for act in plan.get("actions", [])[:30]:
                    history.append(self.do(page, act, snap, resume))
                    self.pause(0.2, 0.6)
                self.settle(page)
            return self.stop(app_id, page, "needs_you", f"page {step}", "The application took more steps than expected.")
        except NeedsYou as e:
            return self.stop(app_id, page, "needs_you", f"page {step}", str(e))
        except Exception as e:
            return self.stop(app_id, page, "failed", f"page {step}", f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            try:
                page.close()
            except Exception:
                pass

    def start_url(self, job: dict) -> str:
        u = job["url"]
        if job.get("ats") == "lever" and "/apply" not in u:
            return u.rstrip("/") + "/apply"
        if job.get("ats") == "ashby" and "/application" not in u:
            return u.rstrip("/") + "/application"
        return u

    def settle(self, page) -> None:
        try:
            page.wait_for_load_state("networkidle", timeout=self.SETTLE_MS)
        except Exception:
            pass
        self.pause(0.6, 1.2)

    def body(self, page) -> str:
        try:
            return page.inner_text("body")
        except Exception:
            return ""

    # ---- doing one action ----
    def do(self, page, act: dict, snap: dict, resume: Path) -> str:
        op, i = act.get("op"), act.get("i")
        el_info = next((e for e in snap["elements"] if e["i"] == i), {}) if i is not None else {}
        what = el_info.get("label") or el_info.get("text") or ""
        el = page.query_selector(f'[data-jp="{i}"]') if i is not None else None
        try:
            if op == "wait":
                time.sleep(2)
                return "waited"
            if el is None:
                return f"couldn't find [{i}]"
            if op == "click":
                if BLOCKED_CLICKS.search(el_info.get("text", "")):
                    return f"skipped clicking “{el_info.get('text')}”"
                self.click(el)
                return f"clicked “{what[:60]}”"
            if op == "upload":
                el.set_input_files(str(resume))
                return f"uploaded the résumé to “{what[:60]}”"
            if op == "credential":
                field = act.get("field", "email")
                site = re.sub(r"^https?://", "", page.url).split("/")[0]
                value = self.email if field == "email" else self.keys.password_for(site, self.email)[0]
                el.fill(value)
                return f"filled the {field}"
            v = str(act.get("value", ""))
            if op == "fill":
                self.fill(page, el, el_info, v)
                return f"filled “{what[:60]}”"
            if op == "choose":
                self.choose(page, el, el_info, v)
                return f"chose “{v[:40]}” for “{what[:60]}”"
            if op == "check":
                try:
                    el.check()
                except Exception:
                    self.click(el)
                return f"ticked “{what[:60]}”"
            return f"unknown action {op}"
        except Exception as e:
            return f"{op} on “{what[:40]}” failed ({type(e).__name__})"

    def click(self, el) -> None:
        try:
            el.scroll_into_view_if_needed(timeout=3000)
        except Exception:
            pass
        try:
            el.click(timeout=5000)
        except Exception:
            try:
                el.click(timeout=3000, force=True)     # e.g. Workday puts a transparent layer over its buttons
            except Exception:
                el.evaluate("e => e.click()")

    def fill(self, page, el, info: dict, v: str) -> None:
        if info.get("kind") == "combobox":
            return self.choose(page, el, info, v)
        try:
            el.fill(v, timeout=5000)
        except Exception:
            self.click(el)
            page.keyboard.press("Control+A")
            page.keyboard.type(v, delay=15)

    def choose(self, page, el, info: dict, v: str) -> None:
        kind = info.get("kind")
        if kind == "select":
            opts = info.get("options") or []
            label = next((o for o in opts if o.lower() == v.lower()), None) or next((o for o in opts if v.lower() in o.lower()), v)
            el.select_option(label=label)
            return
        if kind == "radio" and not info.get("aria"):
            for r in page.query_selector_all(f'input[type=radio][name="{info.get("name")}"]'):
                lab = r.evaluate("e => { const l = e.closest('label') || (e.id && document.querySelector(`label[for='${e.id}']`)); return (l ? l.innerText : e.value || '').trim(); }")
                if lab.lower() == v.lower() or v.lower() in lab.lower():
                    try:
                        r.check()
                    except Exception:
                        r.evaluate("e => e.click()")
                    return
            raise NeedsYou(f"Couldn't select “{v}” for “{info.get('label', '')[:80]}”.")
        if kind == "radio":
            for o in el.query_selector_all("[role=radio]"):
                if (o.inner_text() or "").strip().lower() == v.lower():
                    self.click(o)
                    return
        # a dropdown or type-ahead: open it, type, pick the matching option
        self.click(el)
        time.sleep(0.4)
        try:
            if el.evaluate("e => e.tagName") == "INPUT":
                el.fill("")
                page.keyboard.type(v[:40], delay=20)
                time.sleep(0.9)
        except Exception:
            pass
        options = [o for o in page.query_selector_all("[role=option]") if o.is_visible()]
        best = next((o for o in options if (o.inner_text() or "").strip().lower() == v.lower()), None) \
            or next((o for o in options if v.lower() in (o.inner_text() or "").lower()), None)
        if best:
            self.click(best)
        else:
            page.keyboard.press("Enter")

    def verify_email(self, page, job: dict) -> None:
        if not self.mail:
            raise NeedsYou("The site sent a verification email, and the runner has no mailbox access set up.")
        link = self.mail.verification_link(since=datetime.now() - timedelta(minutes=15))
        if not link:
            raise NeedsYou("The verification email didn't arrive in time.")
        page.goto(link, wait_until="domcontentloaded")
        self.settle(page)
        page.goto(self.start_url(job), wait_until="domcontentloaded")
        self.settle(page)

    # ---- finishing ----
    def done(self, app_id, page, body: str, m) -> str:
        self.api.report(app_id, "submitted", "confirmation", "Submitted, and the site confirmed it.", [self.shot(page)],
                        confirmation=body[max(0, m.start() - 200):m.end() + 300])
        self.log("  submitted")
        return "submitted"

    def stop(self, app_id, page, status, step, detail):
        self.api.report(app_id, status, step, detail, [self.shot(page)])
        self.log(f"  {status}: {detail}")
        return status

    def shot(self, page) -> str:
        try:
            return base64.b64encode(page.screenshot(full_page=True, type="jpeg", quality=55)).decode()
        except Exception:
            return ""

    # ---- replies from employers ----
    def check_outcomes(self) -> dict:
        if not self.mail:
            return {"recorded": 0, "note": "no mailbox set up"}
        found = match_outcomes(self.mail.recent(), self.api.watch())
        return self.api.outcomes(found) if found else {"recorded": 0}

    # ---- the night ----
    def run(self, max_apps: int = 15, between=(90, 240)) -> dict:
        done = {"submitted": 0, "filled": 0, "needs_you": 0, "failed": 0}
        for _ in range(max_apps):
            nxt = self.api.next()
            job = nxt.get("app")
            if not job:
                self.log(f"Done: {nxt.get('reason', 'nothing left')}")
                break
            self.log(f"{job['title']} at {job['company']} ({job['ats']})")
            status = self.apply(job, nxt.get("dry_run", False), nxt.get("profile") or {})
            done[status] = done.get(status, 0) + 1
            time.sleep(random.uniform(*between))    # pacing, so it reads like a person, not a burst
        try:
            done["outcomes"] = self.check_outcomes().get("recorded", 0)
        except Exception as e:
            self.log(f"Couldn't check the inbox for replies ({type(e).__name__})")
        return done


class NeedsYou(Exception):
    pass


# ---------------------------------------------------------------- the command line
def load_config() -> dict:
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {}


def cmd_setup(args) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["server"] = input(f"Ask Harsha address [{cfg.get('server', 'https://ask-harsha.onrender.com')}]: ").strip() or cfg.get("server", "https://ask-harsha.onrender.com")
    cfg["email"] = input(f"Email to use for job-site accounts [{cfg.get('email', '')}]: ").strip() or cfg.get("email", "")
    keys = Keys()
    tok = re.sub(r"[^A-Za-z0-9_-]", "", input("Runner token from Job Pilot's settings (leave empty to keep): "))
    if tok:
        keys.set("runner_token", tok)
    app_pw = input("Gmail app password for reading verification emails (optional, leave empty to skip): ").strip()
    app_pw = re.sub(r"[^A-Za-z0-9]", "", app_pw)     # Google shows it in groups of 4; spaces or stray symbols from copying are dropped
    if app_pw and len(app_pw) != 16:
        print(f"That app password has {len(app_pw)} letters; Google's have 16. Run setup again and paste it carefully.")
    if app_pw:
        keys.set("imap_password", app_pw)
    CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"Saved. Settings in {CONFIG}; secrets in your system keychain.")


def cmd_run(args) -> None:
    from playwright.sync_api import sync_playwright
    cfg = load_config()
    keys = Keys()
    token = keys.get("runner_token")
    if not cfg.get("server") or not token:
        sys.exit("Run: python job_pilot.py setup")
    imap = keys.get("imap_password")
    api = Api(cfg["server"], token)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(str(HOME / "browser"), headless=not args.show, viewport={"width": 1280, "height": 900},
                                                   locale="en-US", timezone_id="America/Chicago")
        runner = Runner(api, keys, Mailbox(cfg["email"], imap) if imap else None, cfg["email"], page_factory=lambda: ctx.new_page(), log=log)
        log(f"=== Job Pilot run {datetime.now():%Y-%m-%d %H:%M} ===")
        try:
            log(str(runner.run(max_apps=args.max, between=(5, 15) if args.fast else (90, 240))))
        except Exception as e:
            log(f"Stopped: {type(e).__name__}: {e}")
            raise
        finally:
            ctx.close()


def log(line: str) -> None:
    """Prints, and keeps a copy in ~/.job_pilot/runner.log (scheduled runs have no window to print to)."""
    print(line, flush=True)
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        with open(HOME / "runner.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def cmd_outcomes(args) -> None:
    cfg, keys = load_config(), Keys()
    imap = keys.get("imap_password")
    if not imap:
        sys.exit("Add a Gmail app password with: python job_pilot.py setup")
    r = Runner(Api(cfg["server"], keys.get("runner_token")), keys, Mailbox(cfg["email"], imap), cfg["email"], page_factory=lambda: None)
    print(r.check_outcomes())


def cmd_schedule(args) -> None:
    py, me = sys.executable, str(Path(__file__).resolve())
    if sys.platform == "darwin":
        plist = Path.home() / "Library/LaunchAgents/com.askharsha.jobpilot.plist"
        plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.askharsha.jobpilot</string>
<key>ProgramArguments</key><array><string>{py}</string><string>{me}</string><string>run</string></array>
<key>StartCalendarInterval</key><dict><key>Hour</key><integer>1</integer><key>Minute</key><integer>0</integer></dict>
<key>StandardOutPath</key><string>{HOME}/runner.log</string><key>StandardErrorPath</key><string>{HOME}/runner.log</string>
</dict></plist>""")
        os.system(f"launchctl unload {plist} 2>/dev/null; launchctl load {plist}")
        print(f"Scheduled nightly at 1:00. Keep the Mac plugged in; to wake it: sudo pmset repeat wake MTWRFSU 00:55:00. Log: {HOME}/runner.log")
    elif sys.platform.startswith("win"):
        os.system(f'schtasks /Create /F /SC DAILY /ST 01:00 /TN "AskHarshaJobPilot" /TR "\\"{py}\\" \\"{me}\\" run"')
        print("Scheduled nightly at 1:00 (Task Scheduler). Next: in Task Scheduler, open AskHarshaJobPilot > Properties and tick "
              "'Wake the computer to run this task' (Conditions) and 'Run task as soon as possible after a scheduled start is missed' (Settings). "
              f"Log: {HOME / 'runner.log'}")
    else:
        print(f"Add this line with `crontab -e`:\n0 1 * * * {py} {me} run >> {HOME}/runner.log 2>&1")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Ask Harsha Job Pilot runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("run")
    r.add_argument("--show", action="store_true", help="show the browser window")
    r.add_argument("--max", type=int, default=15)
    r.add_argument("--fast", action="store_true", help="short pauses (for testing)")
    sub.add_parser("schedule")
    sub.add_parser("outcomes", help="match employer replies in your inbox to applications (runs after every night too)")
    args = ap.parse_args(argv)
    {"setup": cmd_setup, "run": cmd_run, "schedule": cmd_schedule, "outcomes": cmd_outcomes}[args.cmd](args)


if __name__ == "__main__":
    main()
