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
- **Quick phrases and saved phrases** speak instantly, with no model call, so they keep working offline or when the daily budget runs out.
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
