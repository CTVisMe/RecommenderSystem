"""
Live recommender demo web app.

- GET  /            quiz page (mobile-friendly), 5-10 questions from model.json
- POST /recommend   {ratings: {item: 1-5, ...}} -> top recommendations
- GET  /health       for Railway healthcheck

No login, no student identification, no persistence of individual answers.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recommender import load_model, top_recommendations, bottom_recommendations

app = FastAPI(title="Class Taste Recommender")
app.mount("/static", StaticFiles(directory="static"), name="static")

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


class RatingsIn(BaseModel):
    ratings: dict


@app.get("/health")
def health():
    return {"status": "ok"}


def render_quiz_page(quiz_items: list, n_students: int) -> str:
    html = TEMPLATE_PATH.read_text()
    questions_html = "\n".join(
        f'''    <div class="question">
      <label>{item.replace("_", " ")}</label>
      <div class="scale" data-item="{item}">
        {"".join(f'<button type="button" class="scale-btn" data-value="{n}">{n}</button>' for n in range(1, 6))}
      </div>
    </div>'''
        for item in quiz_items
    )
    html = html.replace("__QUESTIONS_HTML__", questions_html)
    html = html.replace("__N_STUDENTS__", str(n_students))
    html = html.replace("__TOTAL__", str(len(quiz_items)))
    return html


@app.get("/", response_class=HTMLResponse)
def quiz():
    model = load_model()
    return render_quiz_page(model["quiz_items"], model["n_students"])


@app.post("/recommend")
def recommend(payload: RatingsIn):
    ratings = {k: float(v) for k, v in payload.ratings.items() if 1 <= float(v) <= 5}
    top = top_recommendations(ratings, k=3)
    bottom = bottom_recommendations(ratings, k=1)
    return {"top": top, "avoid": bottom}
