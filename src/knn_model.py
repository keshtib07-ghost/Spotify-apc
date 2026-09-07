"""
Item-based collaborative filtering via playlist co-occurrence.

For each eval playlist, finds all training playlists that share at least
one seed track, pools their tracks, and scores each candidate track by
co-occurrence count / sqrt(popularity) -- an inverse-item-frequency style
adjustment so it doesn't just re-rank by raw popularity. This is a
lightweight alternative to ALS: no training step, purely lookup-based, but
each eval task costs a sparse row-slice instead of a fixed matrix multiply.

Falls back to a popularity ranking whenever a playlist has no seed tracks
(nothing to look up) or its seed tracks are unseen in training data.

Usage:
  python src/knn_model.py --name dev50k --k 500
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from metrics import evaluate_all

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="dev50k")
    ap.add_argument("--k", type=int, default=500)
    args = ap.parse_args()

    pdir = os.path.join(BASE, "data", "processed", args.name)
    train = pd.read_parquet(os.path.join(pdir, "train_interactions.parquet"))
    eval_tasks = pd.read_parquet(os.path.join(pdir, "eval_tasks.parquet"))

    pid_cat = train["pid"].astype("category")
    track_cat = train["track_uri"].astype("category")
    pid_codes = pid_cat.cat.codes.values
    track_codes = track_cat.cat.codes.values
    track_index = {u: i for i, u in enumerate(track_cat.cat.categories)}
    track_categories = track_cat.cat.categories
    n_users = len(pid_cat.cat.categories)
    n_items = len(track_categories)

    X = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (pid_codes, track_codes)),
        shape=(n_users, n_items),
    )
    Xcsc = X.tocsc()
    track_pop = np.asarray(X.sum(axis=0)).flatten()
    pop_rank = np.argsort(-track_pop)
    inv_sqrt_pop = 1.0 / np.sqrt(track_pop + 1.0)

    t0 = time.time()
    results = []
    fallback_count = 0
    for row in eval_tasks.itertuples():
        seed = row.seed_track_uris
        hidden = set(row.hidden_track_uris)
        seed_idx = [track_index[u] for u in seed if u in track_index]

        if not seed_idx:
            fallback_count += 1
            rec = [track_categories[i] for i in pop_rank[: args.k]]
        else:
            playlists = set()
            for si in seed_idx:
                col = Xcsc.getcol(si)
                playlists.update(col.indices.tolist())
            playlists = np.array(list(playlists))
            pooled = np.asarray(X[playlists, :].sum(axis=0)).flatten()
            score = pooled * inv_sqrt_pop
            for si in seed_idx:
                score[si] = -1  # never recommend a seed track back
            top_idx = np.argpartition(-score, args.k)[: args.k]
            top_idx = top_idx[np.argsort(-score[top_idx])]
            rec = [track_categories[i] for i in top_idx]

        m = evaluate_all(rec, hidden, k=args.k)
        m["n_seed"] = row.n_seed
        results.append(m)

    print(f"Evaluated {len(results)} tasks in {time.time()-t0:.1f}s "
          f"({fallback_count} fell back to popularity, 0 seed tracks)")

    df = pd.DataFrame(results)
    overall = df[["r_precision", "ndcg", "clicks"]].mean().to_dict()
    by_seed = df.groupby("n_seed")[["r_precision", "ndcg", "clicks"]].mean()

    print("Overall:", overall)
    print("\nBy seed length:")
    print(by_seed)

    out = {"overall": overall, "by_seed_length": by_seed.reset_index().to_dict(orient="records")}
    out_path = os.path.join(BASE, "results", "metrics", f"knn_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
