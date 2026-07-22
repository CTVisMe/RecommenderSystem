"""
Runtime prediction logic: item-based collaborative filtering using a
precomputed item-item similarity matrix (see build_model.py).

Given a new (anonymous, live-audience) user's ratings on the quiz items,
predict their rating for every other catalog item and return the top ones.
"""
import json
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.json"

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH) as f:
            _model_cache = json.load(f)
    return _model_cache


def predict_all(ratings: dict) -> dict:
    """ratings: {item_name: 1-5} for the items the live user answered.
    Returns {item_name: predicted_score} for every OTHER catalog item,
    using the standard mean-centered weighted-sum item-based CF formula:

        pred(j) = mean_j + sum_i sim(i,j) * (r_i - mean_i)  /  sum_i |sim(i,j)|

    where i ranges over the quiz items the user actually rated.
    """
    model = load_model()
    sim = model["similarity"]
    stats = model["item_stats"]

    rated_items = [i for i in ratings if i in sim]
    preds = {}
    for j in model["items"]:
        if j in ratings:
            continue  # don't recommend something they already told us about
        mean_j = stats[j]["mean"]
        num, den = 0.0, 0.0
        for i in rated_items:
            s = sim[i].get(j, 0.0)
            num += s * (ratings[i] - stats[i]["mean"])
            den += abs(s)
        pred = mean_j + (num / den if den > 1e-9 else 0.0)
        pred = max(1.0, min(5.0, pred))
        preds[j] = pred
    return preds


def top_recommendations(ratings: dict, k: int = 3) -> list:
    preds = predict_all(ratings)
    ranked = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
    return [{"item": item, "predicted_rating": round(score, 2)} for item, score in ranked[:k]]


def bottom_recommendations(ratings: dict, k: int = 1) -> list:
    """Lowest-predicted items — fun to show as 'what you'll probably hate'."""
    preds = predict_all(ratings)
    ranked = sorted(preds.items(), key=lambda kv: kv[1])
    return [{"item": item, "predicted_rating": round(score, 2)} for item, score in ranked[:k]]
