"""
Hybrid: ALS matrix factorization when a playlist has seed tracks (so ALS can
fold it in), falls back to the popularity ranking when it has zero seed
tracks (title-only, nothing for ALS to fold in). Evaluated over the FULL
eval set so it's directly comparable to the popularity and KNN baselines
(both of which cover all tasks including the zero-seed ones).

Usage:
  python src/hybrid_model.py --name dev50k --factors 64 --iterations 15 --k 500
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from metrics import evaluate_all

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="dev50k")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--regularization", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--k", type=int, default=500)
    args = ap.parse_args()

    pdir = os.path.join(BASE, "data", "processed", args.name)
    train = pd.read_parquet(os.path.join(pdir, "train_interactions.parquet"))
    eval_tasks = pd.read_parquet(os.path.join(pdir, "eval_tasks.parquet"))

    pid_cat = train["pid"].astype("category")
    track_cat = train["track_uri"].astype("category")
    pid_codes = pid_cat.cat.codes.values
    track_codes = track_cat.cat.codes.values
    pid_index = {pid: i for i, pid in enumerate(pid_cat.cat.categories)}
    track_categories = track_cat.cat.categories
    n_users = len(pid_cat.cat.categories)
    n_items = len(track_categories)

    mat = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32) * args.alpha, (pid_codes, track_codes)),
        shape=(n_users, n_items),
    )

    t0 = time.time()
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        random_state=42,
    )
    model.fit(mat)
    print(f"Trained ALS in {time.time()-t0:.1f}s")

    track_pop = np.asarray(mat.sum(axis=0)).flatten()
    pop_rank_uris = [track_categories[i] for i in np.argsort(-track_pop)]

    has_seed_mask = eval_tasks["pid"].isin(pid_index) & (eval_tasks["n_seed"] > 0)
    als_tasks = eval_tasks[has_seed_mask].reset_index(drop=True)
    fallback_tasks = eval_tasks[~has_seed_mask].reset_index(drop=True)
    print(f"ALS-served: {len(als_tasks)}, popularity-fallback: {len(fallback_tasks)}")

    results = []

    if len(als_tasks):
        eval_user_ids = np.array([pid_index[p] for p in als_tasks["pid"]])
        ids, scores = model.recommend(
            eval_user_ids, mat[eval_user_ids], N=args.k, filter_already_liked_items=True,
        )
        for row_idx, task_row in enumerate(als_tasks.itertuples()):
            hidden = set(task_row.hidden_track_uris)
            rec_uris = [track_categories[i] for i in ids[row_idx]]
            m = evaluate_all(rec_uris, hidden, k=args.k)
            m["n_seed"] = task_row.n_seed
            m["served_by"] = "als"
            results.append(m)

    for task_row in fallback_tasks.itertuples():
        hidden = set(task_row.hidden_track_uris)
        seed = set(task_row.seed_track_uris)
        rec = [t for t in pop_rank_uris if t not in seed][: args.k]
        m = evaluate_all(rec, hidden, k=args.k)
        m["n_seed"] = task_row.n_seed
        m["served_by"] = "popularity_fallback"
        results.append(m)

    df = pd.DataFrame(results)
    overall = df[["r_precision", "ndcg", "clicks"]].mean().to_dict()
    by_seed = df.groupby("n_seed")[["r_precision", "ndcg", "clicks"]].mean()
    by_source = df.groupby("served_by")[["r_precision", "ndcg", "clicks"]].mean()

    print("Overall:", overall)
    print("\nBy seed length:")
    print(by_seed)
    print("\nBy source:")
    print(by_source)

    out = {
        "params": vars(args),
        "overall": overall,
        "by_seed_length": by_seed.reset_index().to_dict(orient="records"),
        "by_source": by_source.reset_index().to_dict(orient="records"),
    }
    out_path = os.path.join(BASE, "results", "metrics", f"hybrid_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
