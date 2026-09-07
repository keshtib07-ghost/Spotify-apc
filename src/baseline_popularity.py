"""
Popularity baseline: recommend the globally most popular tracks (that aren't
already in the seed), regardless of playlist content. This is the floor any
real model must beat.

Usage:
  python src/baseline_popularity.py --name dev50k --k 500
"""
import argparse
import json
import os

import pandas as pd

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

    popularity = train["track_uri"].value_counts()
    top_tracks = popularity.index.tolist()

    results = []
    for _, row in eval_tasks.iterrows():
        seed = set(row["seed_track_uris"])
        hidden = set(row["hidden_track_uris"])
        rec = [t for t in top_tracks if t not in seed][: args.k]
        m = evaluate_all(rec, hidden, k=args.k)
        m["n_seed"] = row["n_seed"]
        results.append(m)

    df = pd.DataFrame(results)
    overall = df[["r_precision", "ndcg", "clicks"]].mean().to_dict()
    by_seed = df.groupby("n_seed")[["r_precision", "ndcg", "clicks"]].mean()

    print("Overall:", overall)
    print("\nBy seed length:")
    print(by_seed)

    out = {"overall": overall, "by_seed_length": by_seed.reset_index().to_dict(orient="records")}
    out_path = os.path.join(BASE, "results", "metrics", f"baseline_popularity_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
