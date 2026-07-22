# Live Recommender Demo

A small live demo for your talk: attendees scan a QR code, rate 5-10 topics
on their phone, and instantly see topics the model predicts they'll love
(and one it predicts they'll skip) — based on item-item collaborative
filtering trained on your class survey history.

## How it works

1. **Historical data** (your class CSVs, wide format: one row per student,
   one column per topic, values 1-5) is used to compute an **item-item
   similarity matrix** — how strongly each pair of topics' ratings correlate
   across students. This is the classic Amazon-style "people who rated X
   this way also rated Y that way" approach, just item-based instead of
   user-based (works better here since you have far more topics than a
   typical retail catalog, and no user history to build on for a brand-new
   audience member).
2. From that matrix, the build script auto-selects a **quiz subset** (5-10
   topics) — the ones most predictive of the rest of the catalog and best
   supported by historical data.
3. At the talk, each attendee rates only those quiz items. The app predicts
   their rating for every *other* topic using a mean-centered weighted-sum
   of similarities, and returns the top 3 predicted topics (and the lowest
   one, for fun).
4. Nothing is logged or stored per-user — it's a pure stateless calculation
   per request. No login, no student ID needed.

## Project layout

```
recsys-demo/
  build_model.py     # offline: CSV -> model.json (run this once with your real data)
  recommender.py      # runtime prediction logic, shared by app.py
  app.py               # FastAPI web app (quiz page + /recommend endpoint)
  templates/index.html # mobile-friendly quiz UI
  static/style.css
  data/sample_survey.csv  # synthetic example data, for testing before you plug in real data
  generate_qr.py       # makes a QR code image pointing at your deployed URL
  requirements.txt
  Procfile             # tells Railway how to run the app
```

## 1. Build the model from your real data

Export your class survey history to a single wide CSV: one row per student,
one column per topic (whatever you've been calling them, e.g. `rap_music`,
`broccoli`, `winter_weather`), values 1-5, blanks OK for topics a student
didn't rate. If your CSVs are split across years, concatenate them first
(rows can just be stacked — students don't need to match across files).

Then:

```bash
pip install -r requirements.txt
python build_model.py data/your_real_survey.csv --id-col student_id --quiz-size 8
```

This writes `model.json`. Check the printed quiz item list — if you'd
rather hand-pick which topics get asked live (maybe for variety/comedy
value), force it:

```bash
python build_model.py data/your_real_survey.csv --quiz-items rap_music,broccoli,winter_weather,cilantro,horror_movies,camping,cats,true_crime_podcasts
```

Try it first against the included `data/sample_survey.csv` (synthetic, just
for testing the pipeline end to end) before using your real data.

## 2. Run it locally

```bash
uvicorn app:app --reload
```

Open http://localhost:8000 and try the quiz.

## 3. Deploy to Railway

1. Push this folder to a new GitHub repo.
2. In Railway: **New Project → Deploy from GitHub repo** → select the repo.
3. Railway auto-detects Python via `requirements.txt` + `Procfile` and
   deploys. No config needed beyond that.
4. Under the service's **Settings → Networking**, click **Generate Domain**
   to get a public URL (something like `your-app.up.railway.app`).
5. Visit the URL to confirm the quiz loads.

Railway keeps the container running (no cold-start sleep like some free
tiers), which matters for a live demo — the app should be ready to go the
instant someone scans the code.

**Before the talk:** load the URL yourself a few minutes ahead of time,
run through the quiz once, and keep the tab open — this both double-checks
everything works and keeps the container warm.

## 4. Make the QR code

```bash
python generate_qr.py https://your-app.up.railway.app
```

Drop the resulting `qr_code.png` into your slides.

## Updating the model later

If you collect more survey data later, just re-run `build_model.py` against
the updated CSV and re-deploy (push to GitHub — Railway auto-redeploys).

## Optional enhancements (not built, but easy follow-ons)

- **Live results wall**: a second page showing a running tally/heatmap of
  what the room has rated so far, for a fun visual during the talk. Would
  need a tiny in-memory counter in `app.py` (still no database needed, since
  it can just reset when the app restarts).
- **Explain the recommendation**: show attendees *why* — e.g. "you rated
  camping high, and camping correlates with X" — nice for the "how recsys
  actually works" narrative of a talk.
- **CSV export of live session ratings** (anonymous) if you want to reuse
  the room's answers as future training data.
