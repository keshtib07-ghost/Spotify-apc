"""
Build a train/eval split that mimics the Automatic Playlist Continuation task.

- Randomly holds out `n_eval` playlists as evaluation playlists.
- For each eval playlist, reveals only the first `k` tracks (a "seed"),
  chosen per-playlist from a mix of scenario lengths like the official
  challenge (0, 1, 5, 10, 25 seeds), and hides the rest as ground truth.
- Writes:
    data/processed/<name>/train_interactions.parquet
        (train playlists in full + eval playlists' seed tracks only -
         this is what any model is allowed to see)
    data/processed/<name>/eval_tasks.parquet
        (pid, seed_track_uris list, hidden_track_uris list)

Usage:
  python src/make_eval_split.py --name dev50k --n-eval 2000 --seed 42
"""
import argparse
import os
import random

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")

SEED_SCENARIOS = [0, 1, 5, 10, 25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="dev50k")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pdir = os.path.join(BASE, "data", "processed", args.name)
    interactions = pd.read_parquet(os.path.join(pdir, "interactions.parquet"))

    all_pids = interactions["pid"].unique().tolist()
    rng.shuffle(all_pids)
    eval_pids = set(all_pids[: args.n_eval])

    interactions = interactions.sort_values(["pid", "pos"])
    grouped = interactions.groupby("pid")["track_uri"].apply(list)

    train_rows = []
    eval_rows = []

    for pid, track_list in grouped.items():
        if pid not in eval_pids:
            for pos, uri in enumerate(track_list):
                train_rows.append({"pid": pid, "pos": pos, "track_uri": uri})
            continue

        n = len(track_list)
        possible = [k for k in SEED_SCENARIOS if k < n]
        k = rng.choice(possible) if possible else 0
        seed_tracks = track_list[:k]
        hidden_tracks = track_list[k:]
        if not hidden_tracks:
            continue

        for pos, uri in enumerate(seed_tracks):
            train_rows.append({"pid": pid, "pos": pos, "track_uri": uri})

        eval_rows.append({
            "pid": pid,
            "n_seed": k,
            "seed_track_uris": seed_tracks,
            "hidden_track_uris": hidden_tracks,
        })

    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)

    train_df.to_parquet(os.path.join(pdir, "train_interactions.parquet"), index=False)
    eval_df.to_parquet(os.path.join(pdir, "eval_tasks.parquet"), index=False)

    print(f"train playlists: {len(all_pids) - len(eval_pids)}")
    print(f"eval playlists: {len(eval_df)}")
    print(f"train interactions (visible): {len(train_df)}")
    print(f"seed length distribution:\n{eval_df['n_seed'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
