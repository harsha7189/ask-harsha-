# Ask Harsha

A command-line AI agent that answers recruiter and hiring-manager questions about Harsha Reddy's experience, built from scratch on the Claude Messages API. There's no agent framework: the tool-calling loop, the retriever and the grounding check are all plain Python you can read in one sitting.

Example session with `--trace` (the answer wording varies run to run; token counts and timing depend on the model):

```
You: Has he shipped anything with AI?
  → search_facts({"query": "AI agent launch adoption"})
  ← facts F14, F25, F62, F52, F24
Ask Harsha: Yes. He took a net-new AI-agent capability from concept to production and drove
35% adoption across 200+ users in 60 days [F14]. He also built a 500+ case gold evaluation set,
which saw accuracy rise from 82% to 91% [F24].
  [grounded (2 citations) · 2 steps · ...]
```

## Why it's built this way

The product problem: a portfolio chatbot is only useful if a recruiter can trust it. One invented number is worse than no chatbot. So the design goal is **every claim traceable to a fact Harsha can defend**.

| Decision | What it does | Trade-off |
|---|---|---|
| Facts file, not a free-text résumé | Each fact has an ID (`F14`) so answers can cite it | Facts must be curated by hand |
| BM25 retrieval written from scratch | Keyword ranking with explainable scores, no vector DB | Weaker on paraphrase than embeddings; offset with a small synonym map |
| Hand-written agent loop | Full control of tool results, errors, step limits and history | More code than a framework |
| Post-answer grounding check | Verifies citations exist, were actually retrieved, and that every number appears in a retrieved fact | A heuristic: it catches invented numbers and fake citations, not every subtle paraphrase |
| Evals with decline cases | Measures refusal of unknowns, salary questions and prompt injection, not just right answers | Live evals cost a few cents per run |

## How the loop works

`agent/loop.py`, one user turn:

1. Send the conversation, system prompt and tool specs to `client.messages.create`.
2. If `stop_reason == "tool_use"`, run each `tool_use` block and reply with one user message of `tool_result` blocks (with `is_error: true` on failures). Repeat.
3. Any other stop reason ends the turn; the text blocks are the answer.
4. Run `grounding.check` on the answer.

Guards: a step limit (default 8) against runaway loops, history rollback if an API call fails mid-turn, and assistant turns stored exactly as returned so multi-turn conversations stay valid.

## Project layout

```
agent/
  kb.py         facts loader + BM25 retriever
  tools.py      tool schemas the model sees + the Python that runs them
  prompts.py    system prompt with the grounding rules
  loop.py       the agent loop
  grounding.py  citation and number verification
  notes.py      memory: notes store (GitHub branch / local file) + search
knowledge/profile.json   the facts (edit this to change what the agent knows)
cli.py          interactive chat
server/         web app: FastAPI API, rate limit + budget guards, phone chat page
evals/          live eval cases and runner
tests/          offline tests with a scripted fake model (no API key needed)
```

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your API key from platform.claude.com
python cli.py                 # chat
python cli.py --trace         # chat and show tool calls + grounding check
python cli.py -q "What did he do at Cybage?"
```

Inside the chat: `/trace`, `/reset`, `/facts`, `/quit`.

The default model is `claude-sonnet-5-5`. Set `ASK_HARSHA_MODEL=claude-haiku-5-5` in `.env` for the cheapest option.

## General assistant with memory

The web app is a general AI assistant: it answers anything from the model's own knowledge, uses Claude's web search tool for current information (showing its sources), and still answers questions about Harsha only from his record, with citations.

Harsha can teach it over time. After unlocking owner mode with his passcode, he says "remember that ..." or "forget ...", and the assistant saves or removes a note (IDs like `N003`). Notes are searchable and citable like career facts, and visitors can read them but never change them.

Notes live in `notes.json` on a separate `notes` branch of this repository, written through the GitHub API. That means they survive restarts (Render's free disk doesn't), every change is a commit you can review or revert, and saving a note doesn't redeploy the app. If GitHub can't be read, the store refuses to write rather than risk overwriting memory.

| Who | Can do |
|---|---|
| Visitor | Ask anything; read answers that cite Harsha's facts and notes |
| Harsha (passcode) | Everything above, plus remember/forget, the Memory screen, no per-visitor rate limit |

## Streaming and live evals

Answers stream word by word over Server-Sent Events (`POST /api/chat/stream`), with live status lines such as "Searching the web" or "Looking up Harsha's career facts" while the agent loop works. If streaming isn't available, the page falls back to `POST /api/chat`.

In owner mode, the lock menu has **Run evals**: the server asks every case in `evals/cases.json` as a visitor would, scores each answer with the shared rules in `agent/evals.py`, and streams PASS/FAIL results to the page with the reasons, the answer, time and cost. The cases cover career facts with required citations, ownership boundaries, honest declines (unknown claims, salary, prompt injection, people management), general knowledge, the current date, web search, and a visitor trying to plant a false memory. Each run is logged as one `eval_run` line, so scores can be compared over time.

## Chat pop-up for any website

One line adds a floating **Ask my AI** button to any site; tapping it opens the agent in a panel (full screen on phones):

```html
<script src="https://ask-harsha.onrender.com/widget.js" defer></script>
```

Optional attributes: `data-label`, `data-greeting` (`""` turns off the greeting bubble) and `data-side="left"`. The widget lives in a shadow root so the host page's CSS can't break it, wakes the server when the page loads so the first answer isn't slowed by a cold start, loads the chat only when opened, and exposes `window.AskHarsha.open("question")`. Inside the pop-up the chat runs in embed mode (`/?embed=1`): no app header, and starter questions about Harsha's work instead of the weather greeting. `/?q=…` opens the chat and asks a question straight away.

`/portfolio` serves Harsha's streaming-style portfolio with the pop-up built in, so one link gives visitors both.

## Chat history: "Previously on…"

Every conversation is saved in the visitor's own browser (never on the server). The play-screen icon opens **Previously on…**, a drawer styled like a streaming show:

- **The recap.** It opens with a short reel, "Previously on Ask Harsha", flying in one line from each of your latest chats with its day and weather. Skip or replay it; tap a line to jump into that chat. With reduced motion turned on, it shows the latest line without animating.
- **Seasons and episodes.** Chats are grouped into seasons, one per week counted from your first chat ("Season 3 · This week"), and numbered as episodes. Each shows the weather of the day it started, its first answer line, and how it was sourced: ✓ verified (cited and checked against Harsha's record), web, or file.
- **+ New episode**, search across titles and answers, and ⋯ on each episode to rename, download as Markdown or delete. Download all, delete all, or turn saving off at the bottom.

Up to 100 chats are kept, oldest dropped first when storage fills; attached files aren't kept, only their names. The Memory screen lives in the owner menu to keep the header to icons.

## Second opinions (model jury)

Under a general or web answer, **⚖️ Second opinions** asks ChatGPT and Gemini the same question in parallel (`agent/jury.py`, plain HTTPS, no extra SDKs), shows each answer as it arrives, then has Claude compare all three: an agreement level (strong, partly, disagree), what they agree on, where they differ with each model's position, what only one said, and what to double-check. It doesn't pick a winner; it shows where to look.

Set `OPENAI_API_KEY` and/or `GEMINI_API_KEY` to turn it on (the button appears only when at least one is set). Model names are settings, `OPENAI_MODEL` (default `gpt-6.1-sol`) and `GEMINI_MODEL` (default `gemini-3.8-flash`); if a configured model has been retired, the juror lists the provider's current models, picks the best match and remembers it. Visitors get 5 second opinions an hour; each juror call is charged to the daily budget at `JURY_PRICE_PER_CALL` (default $0.01). The question is sent to OpenAI and Google only when someone taps the button, and the page says so.

## Home screen greeting

The home screen opens with today's weather where the visitor is ("Hot & sunny Monday, 96°F in Austin, feels like 101°") and suggestions that fit it: ways to keep cool in a heat wave, cozy recipes on a rainy day, driving tips in snow, a wind-down routine on a clear night. In the morning it also links the morning brief. `agent/weather.py` finds the approximate location from the visitor's IP address (ipwho.is) and the weather from Open-Meteo, both free with no key, cached in memory and never logged. US visitors see °F, everyone else °C, and it falls back to a plain greeting if the weather can't be reached.

## PM Toolkit

`/tools` holds six focused agents for product work (`agent/toolkit.py`). Each is one streamed model call with its own instructions and no access to Harsha's record; input is wrapped as data, and nothing is stored or logged.

| Tool | Input | Output |
|---|---|---|
| Morning brief | Your topics (saved on your device) | News from the last two days (web search, with sources), a 5-minute lesson, one thing to try today, and a 3-question quiz with a daily streak |
| Meeting | Transcript, notes or a file, or **record live** on the phone | Summary, decisions, action items with owners (tick them off), open questions, follow-up email |
| Customer feedback | Survey answers, reviews, tickets, CSV | Themes with counts and verbatim quotes, urgent issues, feature requests, next steps |
| Customer interview | Prepare: what you want to learn. Analyze: a transcript | A non-leading interview guide, or insights with quotes, jobs, pains and gains |
| Release notes | Ticket titles, PR titles, commits; audience and version | Benefit-first notes grouped into New, Improved and Fixed |
| Spec review | A PRD or spec, pasted or as PDF/Word | Verdict, scores across eight areas, top issues quoted from the spec, missing sections, stakeholder questions |

Every tool has **Try an example**, and results can be copied, downloaded as Markdown or shared. The morning brief is cached per topic set per day, so everyone with the same topics shares one generation. Inputs are limited to 60,000 characters for visitors (150,000 for Harsha). Live recording uses the phone's speech recognition, restarting automatically after pauses on iPhone and keeping the screen awake; it can't tell speakers apart.

## Verified answers (tamper-proof sharing)

Every answer is signed by the server with an Ed25519 key (`agent/receipts.py`): the exact question, answer, time, model, sources and grounding result. **✓ Share verified** under an answer sends the signed receipt back; the server checks the signature (so only answers it really gave can be shared), stores it, and returns a link and QR code. A share card image can be saved or shared from the phone.

Anyone opening `/v/<id>` sees **Verified** or **Not verified**, the original question and answer with sources, and a box to paste what they were shown: an edited screenshot is caught word by word. The public key is at `/.well-known/ask-harsha-key.json`, and the signed receipt can be downloaded, so it can be checked without trusting this server. Receipts are stored encrypted on the `notes` branch, only when someone shares. Set `SIGNING_KEY` (base64 of 32 random bytes) to keep the same key if the passcode changes.

## Other AIs can use it (MCP)

`/mcp` is a Model Context Protocol server (Streamable HTTP, stateless, no sign-in), so Claude or any MCP client can use Ask Harsha as a tool:

| Tool | What it does | Cost |
|---|---|---|
| `ask_harsha_agent` | The full agent's answer, with sources and a verification link | one model call |
| `search_harsha_facts` | Search Harsha's verified facts (IDs like F14) | free |
| `get_harsha_topic` | Every fact on one topic | free |
| `verify_answer` | Check an Ask Harsha verification link | free |

To try it in Claude, add a custom connector with the URL `https://<your-app>.onrender.com/mcp`. Agent questions over MCP share the daily budget and are limited to 30 per 10 minutes per client.

## Ask the real Harsha

When a question needs a human, the agent says so. It calls `suggest_asking_harsha` when a question about Harsha isn't covered by his record, or when a general question is genuinely complex or uncertain and a real person's judgment would help (never for emergencies or medical, legal or financial decisions). It still answers first; then a card pops up under the answer: **Ask the real Harsha?**

- The visitor chooses whether to send it, can edit the question and add a note, and gets a private link. Their browser also remembers it and shows **The real Harsha answered** when he does.
- Harsha sees an Inbox in owner mode ("● Owner mode · 2 questions for you"), with the question, the visitor's note and what the agent told them. He answers once, and can tick **Also teach the agent** to save the answer to memory, so the agent knows it next time.
- `agent/inbox.py` stores the inbox as `inbox.json` on the `notes` branch, **encrypted** (Fernet), because the repository is public. The key comes from `INBOX_KEY` if set, otherwise from the owner passcode. Nothing about the visitor is kept: the random link token is the only identifier. Visitors can send 5 questions a day.

## Photos and files

The **+** next to the message box offers **Take a photo** (opens the camera), **Choose photos** and **Choose a file**. The agent reads photos (signs, labels, menus, screenshots, homework), PDFs up to 10 MB (30 pages for visitors, 100 for Harsha), Word documents (.docx) and text files (CSV, JSON, Markdown, code and so on). You can also paste or drag files in on a computer.

- Photos are shrunk on the phone to at most 1568 px on the long side (what the model actually uses), so a 12 MP iPhone photo uploads as about 50 KB.
- `agent/attachments.py` re-checks every file on the server (real image or PDF bytes, size and page limits), pulls the text out of Word files, and turns each into a content block.
- Files stay attached for the rest of the chat. They're sent as the first exchange in the conversation with a prompt-cache marker, so follow-up questions about the same file are read from cache at about a tenth of the price. The budget counts cache writes at 1.25x and cache reads at 0.1x.
- Answers from files are labelled "Read from your file". Nothing about the files is stored or logged except their types.

## Fact-check any answer

Every answer has a **Fact-check this answer** button. Tapping it runs a separate check (`agent/factcheck.py`, `POST /api/factcheck/stream`) that:

1. Splits the answer into up to 8 short claims and labels each one: about Harsha, about the world, or opinion.
2. Checks Harsha claims only against his record and notes, and world claims with fresh web searches.
3. Shows a verdict per claim (confirmed, contradicted, unverified, not checkable) with a one-line reason and the sources.

The model's verdicts aren't taken on trust. Code throws out any web source the search didn't actually return in this check and any record ID that wasn't provided or looked up, and a "confirmed" or "contradicted" verdict left with no real source is downgraded to "unverified". A check costs a few cents and counts toward the same rate limit and daily budget as a question.

## Assist mode: a voice for people who can't speak clearly, or at all

`/assist` is its own screen (and can be its own home-screen icon). It's for people whose speech is hard to understand (after a stroke, or with ALS, cerebral palsy or Parkinson's) and people who can't speak at all.

- **Speak:** the phone's own speech recognition listens, and `agent/assist.py` turns what it heard, which is often wrong for atypical speech, into up to three full sentences the person might mean. They tap one, and the phone says it out loud.
- **Type a few words:** "water cold" becomes "Can I have some cold water, please?" and a couple of other readings, plus "Say exactly …" and "None of these. Let me fix it".
- **Hear them:** it listens to the other person, then suggests replies.
- **Phrase boards** (Mine, Basics, Feelings, Body, Doctor, Food, People) speak instantly with one tap, with no model call, so they keep working offline or when the daily budget runs out. "Mine" holds saved phrases and ones the person adds.
- **Finish my sentence:** while typing, it suggests whole sentences this person has said or saved (their own first, most used first) and completes the word being typed. "wat" offers "Can I have some water, please?" in one tap. This runs on the phone, so it's instant and free.
- **Show big** puts the sentence on the whole screen for noisy rooms.
- **It learns this person:** every pick or fix is stored on their phone as "what was heard → what they meant", and recent pairs go along with each request, so it learns how their speech gets misheard. That history and the saved phrases never leave the device except inside a request, and the server logs only counts, never what was said.

It uses a fast model (`ASSIST_MODEL`, default `claude-haiku-5-5`) and its own per-visitor limit (`ASSIST_RATE_LIMIT_PER_WINDOW`, default 60 per 10 minutes), because for someone using it as their voice, waiting is the worst failure. Sign language from the camera is deliberately not included: phone-camera recognition isn't reliable enough yet for something people rely on.

## Web app (works on any phone)

```bash
uvicorn server.app:app --reload        # then open http://localhost:8000
```

`server/` wraps the same agent in a small FastAPI app with a phone-first chat page. Visitors don't need a Claude account and never see your API key. It adds per-visitor rate limiting, a daily spending cap and input limits, and the page can be added to an iPhone home screen. See **DEPLOY.md** to put it online for free on Render.

## Test it

```bash
python -m pytest -q                 # offline tests, no API key, about a second
python evals/run_evals.py           # career cases against the CLI agent (the web app's Run evals covers all 17)
```

The eval cases cover factual answers that must cite the right fact, ownership boundaries ("did he build the Redis architecture himself?" must credit engineering), and three cases that must decline: an invented job, a salary question, and a prompt-injection attempt.

## Updating the facts

Add or edit entries in `knowledge/profile.json`. Keep IDs unique and keep each fact to one defensible claim. The tests check that IDs are unique and that key searches still find the right fact.

## Ideas for v2

- Embedding retrieval alongside BM25 (hybrid search) and a reranking step
- A weekly report of declined questions, to show which facts to add next
