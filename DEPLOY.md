# Put Ask Harsha online (and on your iPhone)

About 20 minutes. You need a GitHub account, a Render account (free) and your Claude API key.

## 1. Try it on your laptop first

From the `ask-harsha` folder, with your virtual environment active and `.env` filled in:

```bash
pip install -r requirements.txt
python -m pytest -q                    # 41 passed
uvicorn server.app:app --reload
```

Open http://localhost:8000 and ask a question. Stop the server with Ctrl+C.

## 2. Put the code on GitHub

1. On github.com, create a new **empty** repository named `ask-harsha` (no README, no .gitignore).
2. In the project folder:

```bash
git init
git add .
git status                             # make sure .env is NOT in the list
git commit -m "Ask Harsha: grounded Q&A agent with web app"
git branch -M main
git remote add origin https://github.com/<your-username>/ask-harsha.git
git push -u origin main
```

## 3. Deploy on Render

1. Sign in at render.com with your GitHub account.
2. Click **New**, then **Blueprint**, and pick your `ask-harsha` repository. Render reads `render.yaml` and sets everything up.
3. When it asks for `ANTHROPIC_API_KEY`, paste your key. This is the only secret, and it lives only in Render.
4. Click **Apply**. The first build takes a few minutes.
5. When it shows **Live**, open the URL it gives you (something like `https://ask-harsha.onrender.com`).

The free plan sleeps after 15 minutes without visitors, and the first visit after that takes about a minute to wake up. A paid instance stays awake if that matters for interviews.

## 4. Add it to your iPhone home screen

1. Open your Render URL in **Safari**.
2. Tap **Share**, then **Add to Home Screen**, then **Add**.

You get a red "H" icon that opens Ask Harsha full screen, like an app.

## 5. Protect your spending

The server already limits each visitor to 12 questions per 10 minutes and stops answering for the day after an estimated $2 of usage (`DAILY_BUDGET_USD`). As a second safety net, set a monthly spend limit for your organization in the Claude Console at platform.claude.com.

To change a limit, edit the environment variable in Render (your service, then **Environment**) and save. Render restarts the service.

## 6. Watch how it's used

In Render, open your service, then **Logs**. Every answer writes one line like:

```json
{"event": "answer", "grounded": true, "steps": 2, "tokens_in": 3400, "tokens_out": 120, "cost_usd": 0.0080, "declined": false, "spent_today_usd": 0.041}
```

`declined: true` means the agent said something wasn't in your record. If you set `LOG_QUESTIONS=true`, the line also includes the question, which shows you which facts to add next. Recruiters' questions are then stored in your logs, so leave it off unless you want that.

## 7. Updating your facts

Edit `knowledge/profile.json`, run `python -m pytest -q`, then commit and push. Render redeploys automatically.

## Linking it from your portfolio

Put the Render URL behind an "Ask Harsha" button. If another website will call the API directly, add that site's address to `ALLOWED_ORIGINS` in Render.

## Turn on memory and owner mode

These two secrets let you teach the assistant from your phone. Add them in Render: open your service, then **Environment**, then **Add Environment Variable**, and save. Render restarts the service.

1. **OWNER_PASSCODE**: a passcode only you know, at least 12 characters. You'll type it once on your phone to unlock owner mode.
2. **GITHUB_TOKEN**: lets the app save notes to the `notes` branch of your repository.
   1. On github.com go to **Settings**, **Developer settings**, **Personal access tokens**, **Fine-grained tokens**, then **Generate new token** (direct link: https://github.com/settings/personal-access-tokens/new).
   2. Name it `ask-harsha-memory` and pick an expiration (a year is fine; set a reminder to renew it).
   3. Under **Repository access**, choose **Only select repositories** and pick `ask-harsha-`.
   4. Under **Permissions**, set **Contents** to **Read and write**. Leave everything else as is.
   5. Generate it, copy it, and paste it as `GITHUB_TOKEN` in Render.

`NOTES_REPO` and `NOTES_BRANCH` are already set by the blueprint. Without a token the app still works, but notes are kept only until the next restart.

Then, on your phone: open the app, tap the lock icon, and enter your passcode. You'll see **Owner** next to the name and a **Memory** button.

Everything you teach the assistant can be shown to any visitor, so keep passwords, account numbers and other private details out.
