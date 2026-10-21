"""Job Pilot, part 1: find product roles, screen them, and judge how likely the employer is to sponsor a visa.

Finding: companies' own hiring systems publish their open jobs. We read those public feeds (no scraping of sites
that forbid it):
  Greenhouse  boards-api.greenhouse.io/v1/boards/{board}/jobs           (also lists each job's application questions)
  Lever       api.lever.co/v0/postings/{company}
  Ashby       api.ashbyhq.com/posting-api/job-board/{org}
  Workday     {tenant}.wd{n}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs   (the career site's own search)
  iCIMS       the posting page itself (no public feed)
New companies are discovered with web search (only URLs the search really returned are used) and added to a
registry, so the search widens every night.

Screening is code, so it's predictable: product titles only, no director/VP/principal, at most the required years of
experience, US locations with Austin and Dallas first, excluded companies skipped.

Sponsorship likelihood, from two signals:
  - the posting itself: "unable to sponsor", "without sponsorship now or in the future", "US citizens only",
    "security clearance" exclude a job; "visa sponsorship available" raises it
  - the employer's history: certified H-1B labor condition applications (US Department of Labor public data),
    especially for product roles. A likelihood, never a promise.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

ATS_PATTERNS = [
    ("greenhouse", re.compile(r"https?://(?:boards|job-boards)(?:\.eu)?\.greenhouse\.io/(?:embed/job_app\?for=)?([\w-]+)(?:/jobs/(\d+))?", re.I)),
    ("lever", re.compile(r"https?://jobs\.(?:eu\.)?lever\.co/([\w.-]+)(?:/([0-9a-f-]{36}))?", re.I)),
    ("ashby", re.compile(r"https?://jobs\.ashbyhq\.com/([\w.%-]+)(?:/([0-9a-f-]{36}))?", re.I)),
    ("workday", re.compile(r"https?://([\w-]+)\.(wd\d+)\.myworkdayjobs\.com/(?:[a-z]{2}-[A-Z]{2}/)?([\w-]+)(/job/[^?#\s]+)?", re.I)),
    ("icims", re.compile(r"https?://([\w-]+)\.icims\.com/jobs/(\d+)", re.I)),
]

TITLE_IN = re.compile(r"\b(product\s+manager|product\s+owner|product\s+management|technical\s+product|product\s+lead)\b", re.I)
TITLE_OUT = re.compile(r"\b(director|vice\s+president|vp|head\s+of|chief|principal|staff|group\s+product|distinguished|"
                       r"product\s+marketing|project\s+manager|program\s+manager|intern(ship)?|apprentice|sales|account\s+manager|"
                       r"manager,\s+product\s+management|manager\s+of\s+product)\b", re.I)
US_STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD "
             "TN TX UT VT VA WA WV WI WY DC").split()
STATE_NAMES = {"texas", "california", "washington", "new york", "massachusetts", "illinois", "georgia", "colorado", "florida",
               "north carolina", "virginia", "arizona", "utah", "oregon", "pennsylvania", "new jersey", "minnesota", "ohio", "michigan"}
NON_US = re.compile(r"\b(india|canada|united kingdom|uk|london|germany|berlin|france|paris|ireland|dublin|netherlands|amsterdam|spain|"
                    r"poland|israel|singapore|australia|sydney|japan|tokyo|brazil|mexico|philippines|toronto|vancouver|bangalore|bengaluru|"
                    r"hyderabad|pune|emea|apac|latam)\b", re.I)

NO_SPONSOR = re.compile(
    r"(not|unable to|cannot|can't|won't|will not|do not|does not|don't|no)\s+(currently\s+)?(be\s+able\s+to\s+)?(provide|offer|sponsor|support)\w*"
    r"[^.]{0,40}\b(visa|sponsorship|h-?1b|immigration)"
    r"|without\s+(the\s+)?(need\s+for\s+)?(current\s+or\s+future\s+|future\s+)?(visa\s+)?sponsorship"
    r"|sponsorship\s+(is\s+)?not\s+(available|offered|provided)"
    r"|must\s+be\s+(a\s+)?u\.?s\.?\s+citizen|u\.?s\.?\s+citizens?\s+only|(active|current)\s+(security\s+)?clearance|green\s+card\s+holders?\s+only",
    re.I)
YES_SPONSOR = re.compile(r"(visa\s+sponsorship\s+(is\s+)?(available|offered|provided)|will\s+(consider\s+)?sponsor|h-?1b\s+(sponsorship|transfer)s?\s+(available|welcome|considered)|open\s+to\s+sponsor)", re.I)
YEARS = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*(\d{1,2})?\s*\+?\s*(?:years?|yrs?)", re.I)


@dataclass
class Job:
    ats: str
    board: str
    job_id: str
    company: str
    title: str
    location: str
    url: str
    description: str = ""
    posted: str = ""
    remote: bool = False
    questions: list = field(default_factory=list)   # Greenhouse lists its application questions
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.ats}:{self.board}:{self.job_id}"

    def to_dict(self) -> dict:
        return {"key": self.key, "ats": self.ats, "board": self.board, "job_id": self.job_id, "company": self.company, "title": self.title,
                "location": self.location, "url": self.url, "description": self.description[:20000], "posted": self.posted,
                "remote": self.remote, "questions": self.questions[:60], "extra": self.extra}


def text_of(s: str) -> str:
    """HTML (often double-escaped, as in Greenhouse) to plain text."""
    t = html.unescape(html.unescape(s or ""))
    t = re.sub(r"<(br|/p|/li|/h\d|/div)\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"<li[^>]*>", "\n- ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def parse_url(url: str) -> dict | None:
    """Which hiring system a posting URL belongs to, and its board/company and job id."""
    for ats, pat in ATS_PATTERNS:
        m = pat.search(url or "")
        if not m:
            continue
        if ats == "workday":
            tenant, dc, site, path = m.group(1), m.group(2), m.group(3), m.group(4)
            return {"ats": "workday", "board": f"{tenant}|{dc}|{site}", "job_id": path or "", "url": url}
        if ats == "icims":
            return {"ats": "icims", "board": m.group(1), "job_id": m.group(2), "url": url}
        return {"ats": ats, "board": urllib.parse.unquote(m.group(1)), "job_id": m.group(2) or "", "url": url}
    return None


# ---------------------------------------------------------------- reading the feeds
class Fetcher:
    """HTTP for the job feeds. `opener` is injectable so tests run offline."""

    def __init__(self, opener: Callable | None = None, timeout: int = 15):
        self.open = opener or urllib.request.urlopen
        self.timeout = timeout

    def get(self, url: str, method: str = "GET", body: dict | None = None, raw: bool = False):
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"User-Agent": "Mozilla/5.0 (Ask Harsha Job Pilot)", "Accept": "application/json, text/html",
                                              **({"Content-Type": "application/json"} if body is not None else {})})
        with self.open(req, timeout=self.timeout) as r:
            data = r.read().decode("utf-8", "replace")
        return data if raw else json.loads(data or "{}")


def greenhouse(f: Fetcher, board: str) -> list[Job]:
    data = f.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true")
    name = board
    try:
        name = f.get(f"https://boards-api.greenhouse.io/v1/boards/{board}").get("name") or board
    except Exception:
        pass
    out = []
    for j in data.get("jobs", []):
        loc = (j.get("location") or {}).get("name", "")
        out.append(Job("greenhouse", board, str(j.get("id")), name, j.get("title", ""), loc, j.get("absolute_url", ""),
                       text_of(j.get("content", "")), j.get("updated_at", ""), "remote" in loc.lower()))
    return out


def greenhouse_questions(f: Fetcher, board: str, job_id: str) -> list[dict]:
    d = f.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}?questions=true")
    qs = []
    for q in d.get("questions", []):
        for fld in q.get("fields", []):
            qs.append({"label": q.get("label", ""), "required": bool(q.get("required")), "type": fld.get("type", ""),
                       "name": fld.get("name", ""), "options": [v.get("label") for v in fld.get("values", [])][:40]})
    return qs


def lever(f: Fetcher, company: str) -> list[Job]:
    out = []
    for j in f.get(f"https://api.lever.co/v0/postings/{company}?mode=json") or []:
        cats = j.get("categories") or {}
        loc = cats.get("location", "") or ", ".join(cats.get("allLocations") or [])
        desc = text_of((j.get("description") or "") + "\n" + "\n".join(f"{l.get('text', '')}\n{l.get('content', '')}" for l in j.get("lists", []))
                       + "\n" + (j.get("additional") or ""))
        out.append(Job("lever", company, j.get("id", ""), company.replace("-", " ").title(), j.get("text", ""), loc, j.get("hostedUrl", ""),
                       desc, str(j.get("createdAt", "")), (j.get("workplaceType") or "") == "remote"))
    return out


def ashby(f: Fetcher, org: str) -> list[Job]:
    data = f.get(f"https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=true")
    out = []
    for j in data.get("jobs", []):
        if j.get("isListed") is False:
            continue
        out.append(Job("ashby", org, j.get("id", ""), org.replace("-", " ").title(), j.get("title", ""), j.get("location", ""),
                       j.get("jobUrl", ""), text_of(j.get("descriptionHtml") or j.get("descriptionPlain") or ""), j.get("publishedAt", ""),
                       bool(j.get("isRemote"))))
    return out


def workday(f: Fetcher, board: str, search: str = "product manager", limit: int = 40) -> list[Job]:
    tenant, dc, site = board.split("|")
    host = f"https://{tenant}.{dc}.myworkdayjobs.com"
    data = f.get(f"{host}/wday/cxs/{tenant}/{site}/jobs", method="POST",
                 body={"appliedFacets": {}, "limit": min(limit, 20), "offset": 0, "searchText": search})
    out = []
    for p in data.get("jobPostings", [])[:limit]:
        path = p.get("externalPath", "")
        if not TITLE_IN.search(p.get("title", "")):
            continue
        desc, company = "", tenant
        try:
            d = f.get(f"{host}/wday/cxs/{tenant}/{site}{path}")
            info = d.get("jobPostingInfo") or {}
            desc = text_of(info.get("jobDescription", ""))
            company = (d.get("hiringOrganization") or {}).get("name") or tenant
        except Exception:
            pass
        out.append(Job("workday", board, path, company, p.get("title", ""), p.get("locationsText", ""), f"{host}/{site}{path}",
                       desc, p.get("postedOn", ""), "remote" in (p.get("locationsText", "") or "").lower()))
    return out


def icims(f: Fetcher, board: str, job_id: str, url: str) -> list[Job]:
    page = f.get(f"https://{board}.icims.com/jobs/{job_id}/job?in_iframe=1", raw=True)
    title = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S | re.I)
    loc = re.search(r"(?:Job Locations?|Location)\s*</?[^>]*>\s*([^<]{2,80})", page, re.I)
    company = re.sub(r"^careers-|-?careers$", "", board).replace("-", " ").title()
    return [Job("icims", board, job_id, company, text_of(title.group(1)) if title else "", (loc.group(1).strip() if loc else ""),
                url or f"https://{board}.icims.com/jobs/{job_id}/job", text_of(page)[:20000])]


def board_jobs(f: Fetcher, ats: str, board: str, job_id: str = "", url: str = "") -> list[Job]:
    if ats == "greenhouse":
        return greenhouse(f, board)
    if ats == "lever":
        return lever(f, board)
    if ats == "ashby":
        return ashby(f, board)
    if ats == "workday":
        return workday(f, board)
    if ats == "icims" and job_id:
        return icims(f, board, job_id, url)
    return []


# ---------------------------------------------------------------- discovering new companies with web search
DISCOVER_SITES = ["boards.greenhouse.io", "job-boards.greenhouse.io", "jobs.lever.co", "jobs.ashbyhq.com", "myworkdayjobs.com", "icims.com"]


def discovery_queries(rules: dict) -> list[str]:
    places = (rules.get("preferred_locations") or ["Austin", "Dallas"])[:3] + ["Remote US"]
    titles = ["Product Manager", "Product Owner"]
    qs = []
    for site in ("greenhouse.io", "lever.co", "ashbyhq.com", "myworkdayjobs.com", "icims.com"):
        for t in titles:
            qs.append(f'"{t}" {places[len(qs) % len(places)]} site:{site}')
    return qs


def urls_from_search(messages: list[dict]) -> list[str]:
    """URLs that web search actually returned (never ones a model typed)."""
    from .factcheck import _search_urls
    return sorted(_search_urls(messages))


def discover(client, model: str, rules: dict, usage: dict, max_searches: int = 8) -> list[dict]:
    """Run web searches for product roles on hiring-system sites; return the boards/postings found."""
    from .loop import Agent
    qs = discovery_queries(rules)[:max_searches]
    a = Agent(client, model=model, max_steps=4, max_tokens=1500, record_tools=False, strict_grounding=False, speed="helper",
              system="Run each web search you're given, exactly as written, one after another. Then reply: done.",
              server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": max_searches}])
    a.cache_system = False
    r = a.ask("Searches:\n" + "\n".join(qs))
    usage["input_tokens"] = usage.get("input_tokens", 0) + r.input_tokens
    usage["output_tokens"] = usage.get("output_tokens", 0) + r.output_tokens
    usage["web_searches"] = usage.get("web_searches", 0) + r.web_searches
    found, seen = [], set()
    for u in urls_from_search(a.messages):
        p = parse_url(u)
        if p and (p["ats"], p["board"]) not in seen:
            seen.add((p["ats"], p["board"]))
            found.append(p)
    return found


# ---------------------------------------------------------------- screening
def years_required(text: str) -> int | None:
    """The smallest 'N years' that's about experience, e.g. '3+ years of product management experience' -> 3."""
    best = None
    for m in YEARS.finditer(text or ""):
        tail = (text[m.end():m.end() + 80] + " " + text[max(0, m.start() - 60):m.start()]).lower()
        if not re.search(r"experience|product|management|pm\b|work", tail):
            continue
        n = int(m.group(1))
        if 1 <= n <= 20:
            best = n if best is None else min(best, n)
    return best


def location_rank(job: Job, preferred: list[str]) -> int | None:
    """0 = a preferred city, 1 = remote in the US, 2 = elsewhere in the US, None = outside the US."""
    loc = f"{job.location} ".lower()
    if NON_US.search(loc) and not re.search(r"\b(us|usa|united states)\b", loc):
        return None
    for i, city in enumerate(preferred):
        if city.lower() in loc:
            return 0
    if job.remote or "remote" in loc:
        return 1
    if any(re.search(rf"\b{s}\b", job.location or "") for s in US_STATES) or any(n in loc for n in STATE_NAMES) or re.search(r"\b(us|usa|united states)\b", loc):
        return 2
    return 2 if not loc.strip() else 2


def sponsorship_signal(text: str) -> dict:
    no, yes = NO_SPONSOR.search(text or ""), YES_SPONSOR.search(text or "")
    if no:
        return {"signal": "no", "quote": (text[max(0, no.start() - 60):no.end() + 40]).strip()}
    if yes:
        return {"signal": "yes", "quote": (text[max(0, yes.start() - 40):yes.end() + 40]).strip()}
    return {"signal": "none", "quote": ""}


def norm_company(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    n = re.sub(r"\b(inc|incorporated|llc|l l c|ltd|limited|corp|corporation|co|company|plc|lp|llp|holdings|group|technologies|technology|the)\b", " ", n)
    return " ".join(n.split())


def sponsor_history(index: dict, company: str) -> dict:
    """From the Department of Labor index: {"pm": certified product-role filings, "total": all, "years": [...]}."""
    if not index:
        return {"known": False}
    key = norm_company(company)
    hit = index.get(key)
    if hit is None and key:
        first = key.split()[0]
        cands = [k for k in index if k.split()[:1] == [first]]
        hit = index.get(min(cands, key=len)) if len(cands) == 1 else None
    if hit is None:
        return {"known": False}
    return {"known": True, "pm": int(hit.get("pm", 0)), "total": int(hit.get("total", 0)), "years": hit.get("years", [])}


def sponsor_likelihood(job: Job, index: dict) -> dict:
    sig = sponsorship_signal(job.description)
    hist = sponsor_history(index, job.company)
    if sig["signal"] == "no":
        level = "excluded"
    elif sig["signal"] == "yes" or (hist.get("known") and hist["pm"] >= 3):
        level = "high"
    elif hist.get("known") and hist["total"] >= 5:
        level = "medium"
    elif hist.get("known"):
        level = "low"
    else:
        level = "unknown"
    why = {"excluded": "The posting says it won't sponsor.",
           "high": "The posting offers sponsorship." if sig["signal"] == "yes" else f"{hist.get('pm')} certified H-1B filings for product roles.",
           "medium": f"{hist.get('total')} certified H-1B filings overall, few or none for product roles.",
           "low": f"Only {hist.get('total')} certified H-1B filing(s) on record.",
           "unknown": "No H-1B history found for this employer name, and the posting doesn't say."}[level]
    return {"level": level, "why": why, "posting": sig, "history": hist}


def screen(job: Job, rules: dict, index: dict, seen: set[str]) -> dict:
    """Code-only screening. Returns {"ok": bool, "reasons": [...], ...} with everything the ranking needs."""
    reasons = []
    if job.key in seen:
        reasons.append("Already seen")
    if not TITLE_IN.search(job.title) or TITLE_OUT.search(job.title):
        reasons.append(f"Title isn't a fit: {job.title}")
    company = norm_company(job.company)
    if any(norm_company(x) and norm_company(x) in company for x in rules.get("exclude_companies", [])):
        reasons.append("Company excluded")
    loc = location_rank(job, rules.get("preferred_locations") or ["Austin", "Dallas"])
    if loc is None:
        reasons.append(f"Outside the US: {job.location}")
    elif loc == 2 and not rules.get("open_to_relocate", True):
        reasons.append("Not a preferred location")
    yrs = years_required(job.description)
    if yrs is not None and yrs > int(rules.get("max_years_required", 5)):
        reasons.append(f"Asks for {yrs}+ years")
    if yrs is not None and yrs < int(rules.get("min_years_required", 2)):
        reasons.append(f"Too junior ({yrs} years)")
    sp = sponsor_likelihood(job, index)
    if rules.get("needs_sponsorship") and sp["level"] == "excluded":
        reasons.append("Says it won't sponsor")
    if rules.get("needs_sponsorship") and rules.get("sponsor_filter") == "likely" and sp["level"] in ("low",):
        reasons.append("Unlikely to sponsor")
    if len(job.description) < 300:
        reasons.append("No job description to tailor to")
    return {"ok": not reasons, "reasons": reasons, "location_rank": loc, "years": yrs, "sponsor": sp}


def priority(job: Job, s: dict) -> float:
    """Order for the night: preferred cities first, then remote, likely sponsors, fresher posts."""
    score = {0: 30, 1: 20, 2: 10}.get(s["location_rank"], 0)
    score += {"high": 25, "medium": 15, "unknown": 8, "low": 0, "excluded": -100}[s["sponsor"]["level"]]
    if s["years"] is not None and 3 <= s["years"] <= 5:
        score += 10
    return score
