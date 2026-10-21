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

CAPTCHA = re.compile(r"recaptcha|hcaptcha|turnstile|captcha|are you a robot|i'm not a robot|verify you are human", re.I)
CONFIRMED = re.compile(r"thank you for (applying|your application|your interest)|application (has been |was )?(submitted|received)|"
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


# ---------------------------------------------------------------- reading a form
SCAN_JS = r"""
() => {
  const vis = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const txt = e => (e ? e.innerText || e.textContent || '' : '').replace(/\s+/g, ' ').trim();
  const labelOf = e => {
    if (e.id) { const l = document.querySelector(`label[for="${CSS.escape(e.id)}"]`); if (l && txt(l)) return txt(l); }
    if (e.getAttribute('aria-label')) return e.getAttribute('aria-label');
    const lb = e.getAttribute('aria-labelledby'); if (lb) { const t = lb.split(' ').map(id => txt(document.getElementById(id))).join(' ').trim(); if (t) return t; }
    const wrap = e.closest('label'); if (wrap && txt(wrap)) return txt(wrap);
    const fs = e.closest('fieldset'); if (fs && fs.querySelector('legend')) return txt(fs.querySelector('legend'));
    const box = e.closest('[data-automation-id^="formField"], .field, .application-question, .form-group, li, div');
    if (box) { const l = box.querySelector('label, legend, [class*="label"]'); if (l && txt(l)) return txt(l); }
    return e.getAttribute('placeholder') || e.name || '';
  };
  const out = []; let n = 0;
  const seenRadio = new Set();
  for (const e of document.querySelectorAll('input, select, textarea, button[aria-haspopup="listbox"]')) {
    const type = (e.tagName === 'SELECT') ? 'select' : (e.tagName === 'TEXTAREA') ? 'textarea' : (e.tagName === 'BUTTON') ? 'listbox' : (e.type || 'text').toLowerCase();
    if (['hidden', 'submit', 'button', 'image', 'reset', 'search', 'password'].includes(type)) continue;
    if (type !== 'file' && !vis(e)) continue;
    let options = [];
    if (type === 'select') options = [...e.options].map(o => o.text.trim()).filter(t => t && !/^select|^choose|^--/i.test(t));
    if (type === 'radio') {
      if (seenRadio.has(e.name)) continue; seenRadio.add(e.name);
      const group = [...document.querySelectorAll(`input[type=radio][name="${CSS.escape(e.name)}"]`)];
      options = group.map(r => labelOf(r)).filter(Boolean);
      const fs = e.closest('fieldset'); const q = fs && fs.querySelector('legend') ? txt(fs.querySelector('legend')) : (e.closest('[role=radiogroup]') ? (e.closest('[role=radiogroup]').getAttribute('aria-label') || '') : '');
      e.setAttribute('data-jp', String(n)); out.push({n: n++, type, label: q || e.name, options, required: e.required, name: e.name}); continue;
    }
    if (type === 'listbox') options = [];
    e.setAttribute('data-jp', String(n));
    const filled = type === 'checkbox' ? e.checked : type === 'select' ? (e.selectedIndex > 0) : type === 'listbox' ? !/select one/i.test(txt(e)) : !!e.value;
    out.push({n: n++, type, label: labelOf(e), options, required: e.required || e.getAttribute('aria-required') === 'true', filled, name: e.name || ''});
  }
  return out;
}
"""


def human_pause(a: float = 0.4, b: float = 1.4) -> None:
    time.sleep(random.uniform(a, b))


class Runner:
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
        step = "opening the posting"
        try:
            page.goto(job["url"], wait_until="domcontentloaded")
            self.pause()
            step = "getting to the form"
            self.to_form(page, job, profile)
            for n in range(1, 12):
                step = f"page {n} of the form"
                if self.captcha(page):
                    return self.stop(app_id, page, "needs_you", step, "This site showed a CAPTCHA. Finish this one yourself; the posting link is in the tracker.")
                unresolved = self.fill_page(page, app_id, resume)
                if unresolved:
                    return self.stop(app_id, page, "needs_you", step, "No supported answer for: " + "; ".join(unresolved[:4]))
                nxt, submit = self.button(page, NEXT_TEXT), self.button(page, SUBMIT_TEXT)
                if submit and not nxt:
                    if dry_run:
                        return self.stop(app_id, page, "filled", "final page", "Dry run: every page is filled; stopped before Submit.")
                    submit.click()
                    return self.confirm(app_id, page)
                if not nxt:
                    return self.stop(app_id, page, "needs_you", step, "Couldn't find a Next or Submit button on this page.")
                nxt.click()
                page.wait_for_load_state("domcontentloaded")
                self.pause(0.8, 2.0)
            return self.stop(app_id, page, "needs_you", step, "The form had more pages than expected.")
        except CaptchaShown:
            return self.stop(app_id, page, "needs_you", step, "This site showed a CAPTCHA. Finish this one yourself; the posting link is in the tracker.")
        except NeedsYou as e:
            return self.stop(app_id, page, "needs_you", step, str(e))
        except Exception as e:
            return self.stop(app_id, page, "failed", step, f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            try:
                page.close()
            except Exception:
                pass

    def stop(self, app_id, page, status, step, detail):
        self.api.report(app_id, status, step, detail, [self.shot(page)])
        self.log(f"  {status}: {detail}")
        return status

    def shot(self, page) -> str:
        try:
            return base64.b64encode(page.screenshot(full_page=True, type="jpeg", quality=55)).decode()
        except Exception:
            return ""

    def confirm(self, app_id, page) -> str:
        for _ in range(20):
            time.sleep(1)
            body = page.inner_text("body") if page else ""
            if CONFIRMED.search(body):
                m = CONFIRMED.search(body)
                self.api.report(app_id, "submitted", "confirmation", "Submitted, and the site confirmed it.", [self.shot(page)],
                                confirmation=body[max(0, m.start() - 200):m.end() + 300])
                return "submitted"
        return self.stop(app_id, page, "needs_you", "after Submit", "Clicked Submit but didn't see a confirmation. Check your email or the site.")

    # ---- getting to the form ----
    def to_form(self, page, job: dict, profile: dict) -> None:
        ats = job.get("ats")
        if ats == "lever" and "/apply" not in page.url:
            page.goto(page.url.rstrip("/") + "/apply", wait_until="domcontentloaded")
        elif ats == "ashby" and "/application" not in page.url:
            page.goto(page.url.rstrip("/") + "/application", wait_until="domcontentloaded")
        elif ats == "greenhouse":
            b = self.button(page, re.compile(r"^\s*apply( for this job| now)?\s*$", re.I))
            if b:
                b.click()
                self.pause()
        elif ats in ("workday", "icims"):
            self.account_flow(page, job, profile)

    def account_flow(self, page, job: dict, profile: dict) -> None:
        """Workday/iCIMS: Apply → (apply manually) → sign in, or create an account and verify the email."""
        for pattern in (r"^\s*apply\s*$", r"apply manually|apply for this job online|^\s*apply now\s*$"):
            b = self.button(page, re.compile(pattern, re.I))
            if b:
                b.click()
                page.wait_for_load_state("domcontentloaded")
                self.pause()
        site = re.sub(r"^https?://", "", page.url).split("/")[0]
        pw, new = self.keys.password_for(site, self.email)
        started = datetime.now()
        if new:
            create = self.button(page, re.compile(r"create (an )?account|sign up|new user", re.I))
            if create:
                create.click()
                self.pause()
        self.fill_credentials(page, pw, confirm=new)
        if self.captcha(page):
            raise CaptchaShown()
        b = self.button(page, re.compile(r"^\s*(create account|sign up|register|sign in|log in|submit|continue)\s*$", re.I))
        if b:
            b.click()
            page.wait_for_load_state("domcontentloaded")
            self.pause(1, 2)
        body = page.inner_text("body")
        if re.search(r"verify your (email|account)|verification (email|link)|check your email", body, re.I):
            if not self.mail:
                raise NeedsYou("The site sent a verification email, and the runner has no mailbox access set up.")
            link = self.mail.verification_link(since=started)
            if not link:
                raise NeedsYou("The verification email didn't arrive in time.")
            page.goto(link, wait_until="domcontentloaded")
            self.pause(1, 2)
            page.goto(job["url"], wait_until="domcontentloaded")
            self.account_flow(page, job, profile)   # now it signs in

    def fill_credentials(self, page, pw: str, confirm: bool) -> None:
        for sel in ('input[type=email]', 'input[autocomplete=username]', 'input[name*=email i]', 'input[data-automation-id=email]'):
            e = page.query_selector(sel)
            if e and e.is_visible():
                e.fill(self.email)
                break
        pws = [e for e in page.query_selector_all("input[type=password]") if e.is_visible()]
        for i, e in enumerate(pws if confirm else pws[:1]):
            e.fill(pw)
        for c in page.query_selector_all("input[type=checkbox]"):
            lab = (c.get_attribute("aria-label") or "") + " " + (page.evaluate("(e)=>{const l=e.closest('label')||document.querySelector(`label[for='${e.id}']`);return l?l.innerText:''}", c) or "")
            if re.search(r"agree|terms|privacy|consent", lab, re.I) and c.is_visible() and not c.is_checked():
                c.check()

    # ---- filling one page ----
    def fill_page(self, page, app_id: str, resume: Path) -> list[str]:
        fields = page.evaluate(SCAN_JS)
        for f in fields:
            if f["type"] == "file" and re.search(r"cover\s*letter|transcript|portfolio|writing\s+sample", f.get("label", ""), re.I):
                continue                    # only the résumé is ever uploaded
            if f["type"] == "file":
                el = page.query_selector(f'[data-jp="{f["n"]}"]')
                if el and not el.evaluate("e => e.files && e.files.length"):
                    el.set_input_files(str(resume))
                    self.pause()
        todo = [f for f in fields if f["type"] != "file" and not f.get("filled") and f["label"]]
        if not todo:
            return []
        answers = self.api.answers(app_id, [{"label": f["label"], "type": f["type"], "options": f["options"], "required": f["required"]} for f in todo])
        by_label = {a["label"]: a for a in answers}
        unresolved = []
        for f in todo:
            a = by_label.get(f["label"][:400]) or {}
            v = a.get("value")
            if v in (None, ""):
                if f["required"]:
                    unresolved.append(f["label"][:120])
                continue
            self.set_value(page, f, str(v))
            self.pause(0.2, 0.7)
        return unresolved

    def set_value(self, page, f: dict, v: str) -> None:
        sel = f'[data-jp="{f["n"]}"]'
        t = f["type"]
        if t == "select":
            page.select_option(sel, label=v)
        elif t == "radio":
            radios = page.query_selector_all(f'input[type=radio][name="{f["name"]}"]')
            for r in radios:
                lab = page.evaluate("(e)=>{const l=e.closest('label')||document.querySelector(`label[for='${e.id}']`);return l?l.innerText.trim():(e.value||'')}", r)
                if lab.strip().lower() == v.strip().lower():
                    r.check()
                    return
        elif t == "checkbox":
            if v.strip().lower() in ("yes", "true", "1", "i agree", "agree"):
                page.check(sel)
        elif t == "listbox":   # Workday-style dropdown: open it, pick the option by its text
            page.click(sel)
            self.pause(0.3, 0.8)
            opt = page.query_selector(f'[role=option]:has-text("{v}")')
            if opt:
                opt.click()
        else:
            page.fill(sel, v)

    # ---- helpers ----
    def button(self, page, pattern: re.Pattern):
        for b in page.query_selector_all("button, a[role=button], input[type=submit], a"):
            try:
                if not b.is_visible():
                    continue
                label = (b.inner_text() or b.get_attribute("value") or b.get_attribute("aria-label") or "").strip()
            except Exception:
                continue
            if label and pattern.search(label):
                return b
        return None

    def captcha(self, page) -> bool:
        try:
            frames = " ".join(f.url for f in page.frames)
            return bool(CAPTCHA.search(frames)) or bool(page.query_selector("[class*=captcha i], [id*=captcha i], iframe[title*=challenge i]"))
        except Exception:
            return False

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
            status = self.apply(job, nxt.get("dry_run", True), nxt.get("profile") or {})
            done[status] = done.get(status, 0) + 1
            time.sleep(random.uniform(*between))    # pacing, so it reads like a person, not a burst
        try:
            done["outcomes"] = self.check_outcomes().get("recorded", 0)
        except Exception as e:
            self.log(f"Couldn't check the inbox for replies ({type(e).__name__})")
        return done


class CaptchaShown(Exception):
    pass


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
