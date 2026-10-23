"""The laptop runner, driven through a real headless browser against local mock job sites:
a Greenhouse-style one-page form, a Workday-style flow with account creation, email verification and several pages,
and a page with a CAPTCHA (which must stop, never be solved)."""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
import build_sponsor_index as bsi  # noqa: E402
import job_pilot as jp  # noqa: E402
import tempfile as _tf  # noqa: E402
jp.DEBUG = __import__("pathlib").Path(_tf.mkdtemp())

pw_sync = pytest.importorskip("playwright.sync_api")

PAGE = "<!doctype html><html><body>{}</body></html>"
GH_FORM = """
<h1>Product Manager</h1>
<form action="/gh/{slug}/done" method="get">
  <label for="fn">First Name</label><input id="fn" name="fn" required>
  <label for="ln">Last Name</label><input id="ln" name="ln" required>
  <label for="em">Email</label><input id="em" name="em" type="email" required>
  <label for="cv">Resume/CV</label><input id="cv" name="cv" type="file">
  <label for="auth">Are you legally authorized to work in the United States?</label>
  <select id="auth" name="auth" required><option>Select...</option><option>Yes</option><option>No</option></select>
  <fieldset><legend>Do you have Salesforce experience?</legend>
    <label><input type="radio" name="sf" value="y"> Yes</label><label><input type="radio" name="sf" value="n"> No</label></fieldset>
  {extra}
  <button type="submit">Submit Application</button>
</form>"""

WD_STEP1 = """<form action="/wd/step2" method="get">
  <label for="a">First Name</label><input id="a" name="a" required>
  <label for="p">Phone Number</label><input id="p" name="p" type="tel">
  <label for="r">Upload Resume</label><input id="r" name="r" type="file">
  <button type="submit">Save and Continue</button></form>"""
WD_STEP2 = """<form action="/wd/review" method="get">
  <label for="s">Will you now or in the future require sponsorship for employment visa status?</label>
  <select id="s" name="s" required><option>Select One</option><option>Yes</option><option>No</option></select>
  <div><label id="hl">How did you hear about us?</label>
    <button type="button" id="hear" aria-haspopup="listbox" aria-labelledby="hl" onclick="document.getElementById('opts').style.display='block'">Select One</button>
    <ul id="opts" style="display:none"><li role="option" onclick="document.getElementById('hear').innerText=this.innerText;this.parentNode.style.display='none'">LinkedIn</li>
    <li role="option" onclick="document.getElementById('hear').innerText=this.innerText">Referral</li></ul></div>
  <button type="submit">Save and Continue</button></form>"""


ASHBY = """<div id="app">Loading…</div><script>
setTimeout(() => {
  document.getElementById('app').innerHTML = `
  <h1>Product Manager</h1>
  <form id="f" onsubmit="return false">
    <div class="field"><label for="nm">Name</label><input id="nm" required></div>
    <div class="field"><label for="em">Email</label><input id="em" type="email" required></div>
    <div class="field"><label for="cv">Resume</label><input id="cv" type="file" style="display:none"><button type="button" onclick="document.getElementById('cv').click()">Upload File</button></div>
    <div class="question"><label>Are you legally authorized to work in the United States?</label>
      <button type="button" class="yn" data-q="auth" onclick="pick(this)">Yes</button><button type="button" class="yn" data-q="auth" onclick="pick(this)">No</button></div>
    <div class="question"><label>Will you now or in the future require visa sponsorship?</label>
      <button type="button" class="yn" data-q="spon" onclick="pick(this)">Yes</button><button type="button" class="yn" data-q="spon" onclick="pick(this)">No</button></div>
    <div class="field"><label id="locl">Location</label><input id="loc" role="combobox" aria-labelledby="locl" aria-autocomplete="list" oninput="opts(this.value)">
      <ul id="lb" role="listbox"></ul></div>
    <div id="err" role="alert"></div>
    <button type="button" onclick="send()">Submit Application</button>
  </form>`;
}, 400);
const picked = {};
function pick(b) { picked[b.dataset.q] = b.innerText; document.querySelectorAll(`[data-q=${b.dataset.q}]`).forEach(x => x.setAttribute('aria-pressed', x === b)); }
function opts(v) { const lb = document.getElementById('lb'); lb.innerHTML = '';
  ['Austin, Texas, United States', 'Austin, Minnesota, United States'].filter(o => o.toLowerCase().includes(v.toLowerCase().split(',')[0])).forEach(o => {
    const li = document.createElement('li'); li.setAttribute('role', 'option'); li.innerText = o; li.onclick = () => { document.getElementById('loc').value = o; lb.innerHTML = ''; }; lb.appendChild(li); }); }
function send() {
  const ok = document.getElementById('nm').value && document.getElementById('em').value && picked.auth && picked.spon && document.getElementById('loc').value.startsWith('Austin, Texas');
  if (!ok) { document.getElementById('err').innerText = 'Please fill in all required fields.'; return; }
  fetch('/ashby/submit?auth=' + picked.auth + '&spon=' + picked.spon + '&loc=' + encodeURIComponent(document.getElementById('loc').value) + '&file=' + (document.getElementById('cv').files.length))
    .then(() => { document.getElementById('app').innerHTML = '<h2>Your application was successfully submitted.</h2>'; });
}
</script>"""
INVISIBLE = '<iframe src="https://www.google.com/recaptcha/api2/anchor?k=x&size=invisible" style="width:256px;height:60px;position:fixed;bottom:0;right:-200px"></iframe>'
VISIBLE = '<iframe src="https://www.google.com/recaptcha/api2/anchor?k=x&size=normal" style="width:304px;height:78px"></iframe>'


class Site:
    """Mock job sites with just enough state for accounts: registered, verified."""

    def __init__(self):
        self.users, self.verified, self.submitted, self.got = {}, set(), [], []
        site = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, body, code=200, loc=None):
                self.send_response(code)
                if loc:
                    self.send_header("Location", loc)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE.format(body).encode())

            def do_GET(self):
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                p = u.path
                site.got.append(p)
                if p.startswith("/gh/") and p.endswith("/done"):
                    site.submitted.append(q)
                    return self.send("<h2>Thank you for applying!</h2><p>We have received your application.</p>")
                if p == "/gh/basic":
                    return self.send(GH_FORM.format(slug="basic", extra=""))
                if p == "/gh/why":
                    return self.send(GH_FORM.format(slug="why", extra='<label for="w">Why do you want to work here?</label><textarea id="w" name="w" required></textarea>'))
                if p == "/gh/captcha":
                    return self.send(GH_FORM.format(slug="captcha", extra=VISIBLE))
                if p == "/gh/frame":
                    inner = GH_FORM.format(slug="frame", extra="").replace('"', "&quot;")
                    return self.send(f'<h1>Careers</h1><iframe src="/gh/frameinner" style="width:900px;height:900px"></iframe>')
                if p == "/gh/frameinner":
                    return self.send(GH_FORM.format(slug="frame", extra=""))
                if p == "/gh/terms":
                    return self.send(GH_FORM.format(slug="terms", extra='<label class="cb"><input type="checkbox" name="agree" value="1" required '
                                                    'style="position:absolute;opacity:0;width:1px;height:1px"><span>I agree to the terms and conditions</span></label>'))
                if p == "/gh/solve":
                    return self.send(GH_FORM.format(slug="solve", extra=VISIBLE + '<textarea name="g-recaptcha-response" style="display:none"></textarea>'
                                                    '<script>setTimeout(()=>{document.querySelector("[name=g-recaptcha-response]").value="x".repeat(40)},2500)</script>'))
                if p == "/gh/invisible":
                    return self.send(GH_FORM.format(slug="invisible", extra=INVISIBLE))
                if p == "/ashby/job/application":
                    return self.send(ASHBY)
                if p == "/ashby/submit":
                    site.submitted.append({"ashby": True, **q})
                    return self.send("ok")
                if p == "/wd/job":
                    return self.send('<div data-automation-id="legalNotice" style="position:fixed;inset:0;background:#fff;z-index:9">'
                                     '<p>We use cookies.</p><button data-automation-id="legalNoticeAcceptButton" '
                                     'onclick="this.parentNode.remove()">Accept Cookies</button></div>'
                                     '<h1>Senior Product Manager</h1><a href="/wd/choice">Apply</a>')
                if p == "/wd/choice":
                    return self.send('<a href="/wd/signin">Apply Manually</a><a href="/wd/signin">Use My Last Application</a>')
                if p == "/wd/signin":
                    return self.send('<form action="/wd/login" method="post"><label for="e">Email Address</label><input id="e" name="e" type="email">'
                                     '<label for="pw">Password</label><input id="pw" name="pw" type="password"><button type="submit">Sign In</button></form>'
                                     '<a href="/wd/create">Create Account</a>')
                if p == "/wd/create":
                    return self.send('<form action="/wd/register" method="post"><label for="e">Email Address</label><input id="e" name="e" type="email">'
                                     '<label for="p1">Password</label><input id="p1" name="p1" type="password">'
                                     '<label for="p2">Verify New Password</label><input id="p2" name="p2" type="password">'
                                     '<label><input type="checkbox" name="t" value="1" required> I agree to the terms and privacy policy</label>'
                                     '<button type="submit">Create Account</button></form>')
                if p == "/wd/verify":
                    site.verified.add(q.get("u"))
                    return self.send("<p>Your email is verified.</p>")
                if p == "/wd/step1":
                    return self.send(WD_STEP1)
                if p == "/wd/step2":
                    return self.send(WD_STEP2)
                if p == "/wd/review":
                    site.reviewed = q
                    return self.send('<h2>Review</h2><form action="/wd/done" method="get"><button type="submit">Submit</button></form>')
                if p == "/wd/done":
                    site.submitted.append({"wd": True})
                    return self.send("<h2>Application submitted</h2><p>Thank you for applying.</p>")
                return self.send("not found", 404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                q = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode()).items()}
                if self.path == "/wd/register":
                    if q.get("p1") != q.get("p2") or q.get("t") != "1":
                        return self.send("<p>Passwords must match and you must agree to the terms.</p>")
                    site.users[q["e"]] = q["p1"]
                    return self.send("<p>Please verify your email. Check your email for a verification link.</p>")
                if self.path == "/wd/login":
                    if site.users.get(q.get("e")) == q.get("pw") and q.get("e") in site.verified:
                        return self.send("", 303, "/wd/step1")
                    return self.send("<p>Wrong email or password.</p>")
                return self.send("not found", 404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


class FakeApi:
    ANSWERS = {r"first name": "Harsha", r"last name": "Reddy", r"email": "harsha@example.com", r"phone": "5125550100",
               r"legally authorized": "Yes", r"require (visa )?sponsorship": "Yes", r"salesforce": "Yes", r"how did you hear": "LinkedIn",
               r"^name$": "Harsha Reddy", r"^location$": "Austin, Texas, United States", r"agree|terms": "Yes"}

    def __init__(self, jobs, dry_run=True):
        self.jobs, self.dry, self.reports, self.asked = list(jobs), dry_run, [], []

    def next(self):
        return {"app": self.jobs.pop(0), "dry_run": self.dry, "profile": {"first_name": "Harsha", "last_name": "Reddy"}} if self.jobs else {"app": None, "reason": "Nothing ready."}

    def file(self, app_id, kind):
        return b"%PDF-1.4 fake" if kind == "pdf" else b"PK fake docx"

    def answers(self, app_id, fields):
        self.asked += [f["label"] for f in fields]
        out = []
        for f in fields:
            v = next((v for pat, v in self.ANSWERS.items() if re.search(pat, f["label"], re.I)), None)
            out.append({"label": f["label"][:400], "value": v, "needs_you": v is None})
        return out

    def report(self, app_id, status, step="", detail="", screenshots=None, confirmation=""):
        self.reports.append({"id": app_id, "status": status, "step": step, "detail": detail, "shots": len([s for s in screenshots or [] if s]),
                             "confirmation": confirmation})

    ADVANCE = re.compile(r"^\s*(apply manually|apply|save and continue|continue|next|submit( application)?)\s*$", re.I)

    def step(self, app_id, snap):
        """The real server-side checks (navigator.validate) around a scripted planner standing in for Claude."""
        from agent import navigator as nav
        self.steps = getattr(self, "steps", 0) + 1
        answers = {a["label"]: a for a in self.answers(app_id, nav.fields_needing_answers(snap))}
        _validate = nav.validate
        nav = type("N", (), {"FILLABLE": nav.FILLABLE, "validate": staticmethod(lambda plan, sn, an: _validate(plan, sn, an, allow_submit=not snap.get("dry_run")))})
        missing = [e["label"] for e in snap["elements"] if e.get("required") and e["kind"] in nav.FILLABLE and not e.get("value")
                   and (answers.get(e["label"][:400]) or {}).get("value") is None]
        if missing:
            return {"status": "needs_you", "actions": [], "reason": "No supported answer for: " + "; ".join(missing)}
        els, text, acts = snap["elements"], snap["text"], []
        if re.search(r"successfully submitted|thank you for applying", text, re.I):
            return nav.validate({"status": "submitted", "actions": []}, snap, answers)
        if re.search(r"check your email for a verification link", text, re.I):
            return {"status": "verify_email", "actions": []}
        pw = [e for e in els if e["kind"] == "password"]
        if pw:
            if not any("filled the password" in h for h in snap["history"]):
                link = next((e for e in els if e["kind"] in ("link", "button") and re.search(r"create account", e["text"], re.I)), None)
                if link and not any(re.search(r"verify", e.get("label", ""), re.I) for e in pw):
                    return nav.validate({"status": "continue", "actions": [{"op": "click", "i": link["i"]}]}, snap, answers)
            acts += [{"op": "credential", "i": e["i"], "field": "email"} for e in els if e["kind"] == "email"]
            acts += [{"op": "credential", "i": e["i"], "field": "password"} for e in pw]
            acts += [{"op": "check", "i": e["i"]} for e in els if e["kind"] == "checkbox"]
            btn = next(e for e in els if e["kind"] == "button" and re.search(r"create account|sign in", e["text"], re.I))
            return nav.validate({"status": "continue", "actions": acts + [{"op": "click", "i": btn["i"]}]}, snap, answers)
        for e in els:
            if e["kind"] == "checkbox" and not e.get("value"):
                acts.append({"op": "check", "i": e["i"]})
            elif e["kind"] in nav.FILLABLE and not e.get("value") and e.get("label"):
                acts.append({"op": "choose" if e["kind"] in ("select", "radio", "combobox") else "fill", "i": e["i"]})
            if e["kind"] == "file" and re.search(r"resume|cv", e.get("label", ""), re.I) and not e.get("value"):
                acts.append({"op": "upload", "i": e["i"]})
            if e["kind"] == "button" and e.get("context") and not e.get("pressed") and re.match(r"^(yes|no)$", e["text"], re.I):
                acts.append({"op": "click", "i": e["i"]})            # tries both Yes and No; validation keeps the right one
        nxt = next((e for e in els if e["kind"] in ("button", "link") and self.ADVANCE.match(e["text"])), None)
        if nxt:
            acts.append({"op": "click", "i": nxt["i"]})
        return nav.validate({"status": "continue", "actions": acts}, snap, answers)


class DictKeys:
    def __init__(self):
        self.d = {}

    def get_password(self, s, k):
        return self.d.get((s, k))

    def set_password(self, s, k, v):
        self.d[(s, k)] = v


class FakeMail:
    def __init__(self, site, user):
        self.site, self.user, self.asked = site, user, 0

    def verification_link(self, wait_s=180, since=None):
        self.asked += 1
        return f"{self.site.url}/wd/verify?u={self.user}" if self.user in self.site.users else None


@pytest.fixture(scope="module")
def browser():
    with pw_sync.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture(scope="module")
def site():
    s = Site()
    yield s
    s.httpd.shutdown()


def make(browser, api, site=None, user="harsha@example.com"):
    ctx = browser.new_context()
    r = jp.Runner(api, jp.Keys(DictKeys()), FakeMail(site, user) if site else None, user, page_factory=ctx.new_page,
                     pause=lambda *a: None, log=lambda *a: None)
    r.SETTLE_MS = 1500
    return r


def job(site, path, ats, i="a1"):
    return {"id": i, "company": "Acme", "title": "Product Manager", "url": site.url + path, "ats": ats, "answers": []}


def test_greenhouse_dry_run_fills_and_stops_before_submit(browser, site):
    api = FakeApi([job(site, "/gh/basic", "greenhouse")])
    out = make(browser, api).run(between=(0, 0))
    assert out["filled"] == 1 and api.reports[-1]["status"] == "filled" and api.reports[-1]["shots"] == 1
    assert not [s for s in site.submitted if "fn" in s and s.get("fn") == "Harsha" and "basic" in str(s)]
    assert "Do you have Salesforce experience?" in api.asked and "Are you legally authorized to work in the United States?" in api.asked


def test_greenhouse_submits_with_confirmation(browser, site):
    before = len(site.submitted)
    api = FakeApi([job(site, "/gh/basic", "greenhouse")], dry_run=False)
    out = make(browser, api).run(between=(0, 0))
    assert out["submitted"] == 1
    sent = site.submitted[before]
    assert sent["fn"] == "Harsha" and sent["em"] == "harsha@example.com" and sent["auth"] == "Yes" and sent["sf"] == "y"
    assert "Thank you for applying" in api.reports[-1]["confirmation"]


def test_unsupported_required_question_stops(browser, site):
    before = len(site.submitted)
    api = FakeApi([job(site, "/gh/why", "greenhouse")], dry_run=False)
    make(browser, api).run(between=(0, 0))
    r = api.reports[-1]
    assert r["status"] == "needs_you" and "Why do you want to work here?" in r["detail"]
    assert len(site.submitted) == before            # nothing guessed, nothing sent


def test_captcha_is_never_solved(browser, site):
    before = len(site.submitted)
    api = FakeApi([job(site, "/gh/captcha", "greenhouse")], dry_run=False)
    make(browser, api).run(between=(0, 0))
    assert api.reports[-1]["status"] == "needs_you" and "CAPTCHA" in api.reports[-1]["detail"]
    assert len(site.submitted) == before


def test_workday_creates_account_verifies_email_and_fills_every_page(browser, site):
    api = FakeApi([job(site, "/wd/job", "workday", "w1")], dry_run=False)
    keys = DictKeys()
    ctx = browser.new_context()
    mail = FakeMail(site, "harsha@example.com")
    r = jp.Runner(api, jp.Keys(keys), mail, "harsha@example.com", page_factory=ctx.new_page, pause=lambda *a: None, log=lambda *a: None)
    r.SETTLE_MS = 1500
    out = r.run(between=(0, 0))
    assert out["submitted"] == 1, api.reports
    assert "harsha@example.com" in site.users and "harsha@example.com" in site.verified and mail.asked == 1
    stored = [v for (s, k), v in keys.d.items() if k.endswith(":harsha@example.com")]
    assert stored == [site.users["harsha@example.com"]] and len(stored[0]) >= 16       # one strong password, kept in the keychain
    assert site.reviewed["s"] == "Yes"                                                   # sponsorship answer came from the sheet
    assert "How did you hear about us?" in api.asked
    assert "/wd/step1" in site.got and "/wd/step2" in site.got


def test_verify_link_and_password_helpers():
    assert jp.find_verify_link('Click <a href="https://x.myworkdayjobs.com/verify?token=a&amp;b=1">here</a>') == "https://x.myworkdayjobs.com/verify?token=a&b=1"
    assert jp.find_verify_link("https://example.com/unsubscribe") is None
    k = jp.Keys(DictKeys())
    a, new = k.password_for("site.com", "me@x.com")
    b, again = k.password_for("site.com", "me@x.com")
    assert new and not again and a == b and re.search(r"[A-Z]", a) and re.search(r"\d", a) and re.search(r"[!@#$%]", a)
    assert k.password_for("other.com", "me@x.com")[0] != a


def test_sponsor_index_from_dol_csv(tmp_path):
    f = tmp_path / "LCA_Disclosure_Data_FY2025_Q4.csv"
    f.write_text("CASE_STATUS,VISA_CLASS,EMPLOYER_NAME,JOB_TITLE\n"
                 "Certified,H-1B,\"Acme, Inc.\",Senior Product Manager\n"
                 "Certified - Withdrawn,H-1B,ACME INC,Product Owner\n"
                 "Denied,H-1B,Acme Inc,Product Manager\n"
                 "Certified,H-1B,Acme Inc,Software Engineer\n"
                 "Certified,E-3 Australian,Acme Inc,Product Manager\n"
                 "Certified,H-1B,Widget LLC,Production Manager\n"
                 "Certified,H-1B,Widget LLC,Product Marketing Manager\n")
    idx = bsi.build([str(f)])
    assert idx["acme"] == {"pm": 2, "total": 3, "years": ["2025"]}
    assert idx["widget"]["pm"] == 0 and idx["widget"]["total"] == 2


def test_ashby_style_page_with_yes_no_buttons_and_type_ahead(browser, site):
    before = len(site.submitted)
    api = FakeApi([job(site, "/ashby/job", "ashby", "a9")], dry_run=False)
    out = make(browser, api).run(between=(0, 0))
    assert out["submitted"] == 1, api.reports
    sent = site.submitted[before]
    assert sent["auth"] == "Yes" and sent["spon"] == "Yes" and sent["loc"] == "Austin, Texas, United States" and sent["file"] == "1"


def test_invisible_spam_check_is_not_a_captcha(browser, site):
    api = FakeApi([job(site, "/gh/invisible", "greenhouse")], dry_run=False)
    assert make(browser, api).run(between=(0, 0))["submitted"] == 1, api.reports


def test_server_never_lets_the_model_choose_values():
    """Open by design, with hard lines: his sheet data and protected answers win, no made-up numbers, no social sign-in."""
    from agent import navigator as nav
    snap = {"text": "Apply", "elements": [
        {"i": 0, "kind": "text", "label": "First Name", "value": "", "required": True},
        {"i": 1, "kind": "select", "label": "Will you require sponsorship?", "options": ["Yes", "No"], "value": "", "required": True},
        {"i": 2, "kind": "button", "text": "No", "context": "Are you authorized to work in the US?"},
        {"i": 3, "kind": "button", "text": "Yes", "context": "Are you authorized to work in the US?"},
        {"i": 4, "kind": "button", "text": "Apply with LinkedIn"},
        {"i": 5, "kind": "file", "label": "Cover Letter", "value": ""},
        {"i": 6, "kind": "text", "label": "How did you hear about us?", "value": ""},
        {"i": 7, "kind": "textarea", "label": "Tell us about a product you shipped", "value": ""},
        {"i": 8, "kind": "button", "text": "Submit Application"}]}
    answers = {"First Name": {"value": "Harsha", "source": "sheet"}, "Will you require sponsorship?": {"value": "Yes", "source": "sheet"},
               "Are you authorized to work in the US?": {"value": "Yes", "source": "sheet"}}
    plan = {"status": "continue", "actions": [{"op": "fill", "i": 0, "value": "Bob"}, {"op": "choose", "i": 1, "value": "No"},
                                             {"op": "click", "i": 2}, {"op": "click", "i": 3}, {"op": "click", "i": 4},
                                             {"op": "upload", "i": 5}, {"op": "fill", "i": 6, "value": "Company careers site"},
                                             {"op": "fill", "i": 7, "value": "I grew revenue 900% at IBM."}, {"op": "click", "i": 8}]}
    out = nav.validate(plan, snap, answers, allowed_text="Harsha Reddy IBM Product Manager Cut resolution time 11 to 4 minutes")
    assert out["actions"] == [{"op": "fill", "i": 0, "value": "Harsha"}, {"op": "choose", "i": 1, "value": "Yes"},
                              {"op": "click", "i": 3}, {"op": "fill", "i": 6, "value": "Company careers site"}, {"op": "click", "i": 8}]
    assert len(out["dropped"]) == 4                     # wrong Yes/No, social sign-in, cover letter upload, made-up number
    assert nav.validate({"status": "submitted", "actions": []}, snap, answers)["status"] == "continue"     # no confirmation on the page
    assert nav.validate(plan, snap, answers, allow_submit=False)["status"] == "filled"

def test_upload_through_the_sites_own_button(browser, site):
    """No visible file box: the runner clicks "Upload File" and the file picker gets the résumé."""
    from agent import navigator as nav
    before = len(site.submitted)
    api = FakeApi([job(site, "/ashby/job", "ashby", "a10")], dry_run=False)
    real = api.step

    def step(app_id, snap):
        out = real(app_id, snap)
        btn = next((e for e in snap["elements"] if e["kind"] == "button" and e["text"] == "Upload File"), None)
        if btn and out["status"] == "continue":
            out["actions"] = [a for a in out["actions"] if a["op"] != "upload"]
            out["actions"].insert(0, {"op": "click", "i": btn["i"]})
        return out
    api.step = step
    make(browser, api).run(between=(0, 0))
    assert site.submitted[before]["file"] == "1"


def test_each_page_is_recorded_for_troubleshooting(browser, site):
    api = FakeApi([job(site, "/gh/basic", "greenhouse", "rec1")], dry_run=False)
    make(browser, api).run(between=(0, 0))
    folder = next(p for p in jp.DEBUG.rglob("*rec1*") if p.is_dir())
    files = sorted(p.name for p in folder.iterdir())
    assert "step01.json" in files and "step01.jpg" in files
    rec = (folder / "step01.json").read_text()
    assert "First Name" in rec and "plan" in rec


def test_signed_in_on_the_account_page_goes_back_to_the_job(browser, site):
    """Workday sometimes lands on Candidate Home after sign-in; the runner returns to the posting instead of re-creating the account."""
    api = FakeApi([job(site, "/wd/job", "workday", "w2")], dry_run=False)
    real, seen = api.step, {"home": 0}

    def step(app_id, snap):
        if "/wd/step1" in snap["url"] and not seen["home"]:
            seen["home"] = 1
            return {"status": "reopen_job", "actions": []}       # as if it had landed on Candidate Home
        return real(app_id, snap)
    api.step = step
    keys = DictKeys()
    ctx = browser.new_context()
    r = jp.Runner(api, jp.Keys(keys), FakeMail(site, "harsha@example.com"), "harsha@example.com", page_factory=ctx.new_page,
                  pause=lambda *a: None, log=lambda *a: None)
    r.SETTLE_MS = 1500
    assert r.run(between=(0, 0))["submitted"] == 1, api.reports


def test_numbers_only_into_number_boxes():
    from agent import navigator as nav
    snap = {"text": "", "elements": [{"i": 0, "kind": "number", "label": "Salary expectations", "value": "", "required": True}]}
    out = nav.validate({"status": "continue", "actions": [{"op": "fill", "i": 0}]}, snap, {"Salary expectations": {"value": "$130,000 - $145,000"}})
    assert out["actions"] == [{"op": "fill", "i": 0, "value": "130000"}]


def test_form_inside_a_frame(browser, site):
    before = len(site.submitted)
    api = FakeApi([job(site, "/gh/frame", "icims", "f1")], dry_run=False)
    assert make(browser, api).run(between=(0, 0))["submitted"] == 1, api.reports
    assert site.submitted[before]["fn"] == "Harsha"


def test_styled_terms_checkbox_gets_ticked(browser, site):
    before = len(site.submitted)
    api = FakeApi([job(site, "/gh/terms", "greenhouse", "t1")], dry_run=False)
    assert make(browser, api).run(between=(0, 0))["submitted"] == 1, api.reports
    assert site.submitted[before]["agree"] == "1"


def test_assist_mode_waits_while_you_solve_the_captcha(browser, site):
    api = FakeApi([job(site, "/gh/solve", "greenhouse", "s1")], dry_run=False)
    r = make(browser, api)
    r.assist = True
    assert r.run(between=(0, 0))["submitted"] == 1, api.reports
    api2 = FakeApi([job(site, "/gh/captcha", "greenhouse", "s2")], dry_run=False)
    assert make(browser, api2).run(between=(0, 0))["needs_you"] == 1          # unattended: it waits for you, never solves it
