# Job Pilot laptop runner

Ask Harsha does the thinking on the server: it finds roles, screens them, researches each one, tailors the résumé, checks it twice and prepares answers. This runner does the clicking on your laptop at night, because job sites block form-filling from cloud servers.

## Setup (once, about 15 minutes)

1. Install Python 3.11 or newer, then:
   ```
   pip install playwright keyring openpyxl
   python -m playwright install chromium
   ```
2. Copy `job_pilot.py` and `build_sponsor_index.py` from this folder to your laptop.
3. In Ask Harsha → Job Pilot → Laptop runner, make a runner token. Then run:
   ```
   python job_pilot.py setup
   ```
   It asks for the server address, the token, the email to use for job-site accounts and, optionally, a Gmail app password (Google Account → Security → App passwords). The app password lets it click Workday and iCIMS verification links; it only reads emails from hiring systems. The token and passwords are kept in your system keychain, not in files.
4. H-1B history: download the last two fiscal years of "LCA Programs (H-1B, H-1B1, E-3)" disclosure files from https://www.dol.gov/agencies/eta/foreign-labor/performance, then:
   ```
   python build_sponsor_index.py LCA_Disclosure_Data_FY2025_Q4.xlsx LCA_Disclosure_Data_FY2026_Q3.xlsx --upload
   ```
   Only three numbers per employer are uploaded: certified product-role filings, all certified filings, and the years.
5. Watch it once: `python job_pilot.py run --show --max 2`
6. Schedule it nightly at 1:00: `python job_pilot.py schedule`. Keep the laptop plugged in. On a Mac, also run `sudo pmset repeat wake MTWRFSU 00:55:00` so it wakes up.

## What it will and won't do

- Dry run is on until you turn it off in Job Pilot's settings: it fills every page, stops before Submit and sends a screenshot to the tracker.
- Each job-site account gets its own strong password, saved in your keychain under "ask-harsha-job-pilot".
- If a required question has no answer supported by your answer sheet or that job's tailored résumé, it stops and the application waits for you.
- It never solves CAPTCHAs or gets around bot checks. Those applications wait for you in the morning.
- It stops at the nightly cap (10 by default) and paces itself between applications.

Logs are in `~/.job_pilot/runner.log`.
