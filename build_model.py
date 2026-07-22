"""
Builds the recommender model from historical survey data.

Input: a wide-format CSV where each row is one student and each column
(after the id column) is a topic rated 1-5. Missing values are allowed
(not every student rated every topic) and should be left BLANK in the CSV
(an empty cell), not coded as 0 or any other number — 0 is a valid-looking
float and would otherwise get treated as a real (very negative) rating,
dragging down that topic's average and corrupting its correlations with
everything else.

Output: model.json containing
  - items: full list of topics
  - item_stats: mean/std/n per topic (for mean-centering predictions)
  - similarity: item-item Pearson correlation matrix (pairwise-complete)
  - quiz_items: the subset of topics chosen to ask live at the talk

Run:
    python build_model.py data/combined_survey.csv --id-col StudentID --quiz-size 8
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def load_ratings(csv_path: str, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
    # Blank cells already come through as NaN (real missing) via read_csv.
    # Coerce anything non-numeric (typos, stray text) to NaN too.
    df = df.apply(pd.to_numeric, errors="coerce")
    # Safety net, not a missing-value convention: a rating outside 1-5 is a
    # data-entry error, not a signal, so drop it rather than let it skew the
    # mean/correlations. If this ever fires on real data, it's worth checking
    # the source CSV — it means something other than a blank slipped through.
    out_of_range = ((df < 1) | (df > 5)) & df.notna()
    if out_of_range.any().any():
        bad_cols = df.columns[out_of_range.any()].tolist()
        print(f"Warning: found out-of-range (non 1-5) values in columns {bad_cols}; treating as missing.")
    df = df.where((df >= 1) & (df <= 5))
    return df


def compute_similarity(df: pd.DataFrame, min_overlap: int = 8) -> pd.DataFrame:
    """Pairwise Pearson correlation between items, using only students who
    rated both items. Pairs with too little overlap are set to 0 (no signal)
    rather than left as a noisy correlation from a handful of students."""
    items = df.columns.tolist()
    sim = pd.DataFrame(np.eye(len(items)), index=items, columns=items)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            pair = df[[a, b]].dropna()
            if len(pair) >= min_overlap and pair[a].std() > 0 and pair[b].std() > 0:
                corr = pair[a].corr(pair[b])
            else:
                corr = 0.0
            if pd.isna(corr):
                corr = 0.0
            sim.loc[a, b] = corr
            sim.loc[b, a] = corr
    return sim


def choose_quiz_items(sim: pd.DataFrame, item_stats: dict, quiz_size: int) -> list:
    """Pick the subset of items to actually ask at the talk. We want items
    that are: (a) strong predictors of the rest of the catalog (high average
    |correlation| with other items), and (b) reasonably well-rated historically
    (enough n to trust the correlations). This is a simple greedy pick, not
    globally optimal, but works well for 5-10 slots."""
    items = sim.index.tolist()
    predictive_power = (sim.abs().sum(axis=1) - 1) / (len(items) - 1)
    trust = pd.Series({k: min(item_stats[k]["n"] / 30.0, 1.0) for k in items})
    score = predictive_power * trust
    ranked = score.sort_values(ascending=False)
    return ranked.index[:quiz_size].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--id-col", default="StudentID")
    ap.add_argument("--quiz-size", type=int, default=8)
    ap.add_argument("--min-overlap", type=int, default=8)
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--quiz-items", default=None,
                     help="Comma-separated list to force specific quiz items instead of auto-selecting")
    args = ap.parse_args()

    df = load_ratings(args.csv_path, args.id_col)
    if df.shape[1] < 2:
        sys.exit("Need at least 2 topic columns in the CSV.")

    item_stats = {
        col: {
            "mean": float(df[col].mean()) if df[col].notna().any() else 3.0,
            "std": float(df[col].std()) if df[col].notna().sum() > 1 else 1.0,
            "n": int(df[col].notna().sum()),
        }
        for col in df.columns
    }

    sim = compute_similarity(df, min_overlap=args.min_overlap)

    if args.quiz_items:
        quiz_items = [q.strip() for q in args.quiz_items.split(",")]
        missing = [q for q in quiz_items if q not in df.columns]
        if missing:
            sys.exit(f"--quiz-items not found in CSV columns: {missing}")
    else:
        quiz_items = choose_quiz_items(sim, item_stats, args.quiz_size)

    model = {
        "items": df.columns.tolist(),
        "item_stats": item_stats,
        "similarity": sim.round(4).to_dict(),
        "quiz_items": quiz_items,
        "n_students": int(len(df)),
    }

    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)

    print(f"Wrote {args.out}")
    print(f"  {len(df.columns)} topics, {len(df)} historical students")
    print(f"  Quiz items ({len(quiz_items)}): {quiz_items}")


if __name__ == "__main__":
    main()
