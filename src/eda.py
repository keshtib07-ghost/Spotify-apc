"""
Exploratory data analysis on a processed MPD subset.

Reads data/processed/<name>/{playlists,tracks,interactions}.parquet,
prints summary stats, and saves figures to results/figures/ and a
metrics summary to results/metrics/.

Usage:
  python src/eda.py --name dev50k
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(BASE, "results", "figures")
METRIC_DIR = os.path.join(BASE, "results", "metrics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="dev50k")
    args = ap.parse_args()

    pdir = os.path.join(BASE, "data", "processed", args.name)
    playlists = pd.read_parquet(os.path.join(pdir, "playlists.parquet"))
    tracks = pd.read_parquet(os.path.join(pdir, "tracks.parquet"))
    interactions = pd.read_parquet(os.path.join(pdir, "interactions.parquet"))

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    n_playlists = len(playlists)
    n_tracks = tracks["track_uri"].nunique()
    n_artists = tracks["artist_uri"].nunique()
    n_albums = tracks["album_uri"].nunique()
    n_interactions = len(interactions)

    track_counts = interactions["track_uri"].value_counts()
    artist_counts = interactions.merge(
        tracks[["track_uri", "artist_uri"]], on="track_uri", how="left"
    )["artist_uri"].value_counts()

    summary = {
        "dataset": args.name,
        "n_playlists": int(n_playlists),
        "n_unique_tracks": int(n_tracks),
        "n_unique_artists": int(n_artists),
        "n_unique_albums": int(n_albums),
        "n_interactions": int(n_interactions),
        "avg_tracks_per_playlist": float(playlists["num_tracks"].mean()),
        "median_tracks_per_playlist": float(playlists["num_tracks"].median()),
        "max_tracks_per_playlist": int(playlists["num_tracks"].max()),
        "min_tracks_per_playlist": int(playlists["num_tracks"].min()),
        "avg_followers": float(playlists["num_followers"].mean()),
        "pct_collaborative": float(playlists["collaborative"].mean() * 100),
        "top10_track_share_pct": float(track_counts.head(10).sum() / n_interactions * 100),
        "tracks_appearing_once_pct": float((track_counts == 1).sum() / n_tracks * 100),
    }

    with open(os.path.join(METRIC_DIR, f"eda_{args.name}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for k, v in summary.items():
        print(f"{k}: {v}")

    # Figure 1: playlist length distribution
    plt.figure(figsize=(6, 4))
    playlists["num_tracks"].clip(upper=250).hist(bins=50)
    plt.xlabel("Tracks per playlist (clipped at 250)")
    plt.ylabel("Number of playlists")
    plt.title("Playlist length distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"playlist_length_{args.name}.png"), dpi=120)
    plt.close()

    # Figure 2: track popularity long tail (log-log rank vs frequency)
    plt.figure(figsize=(6, 4))
    freqs = track_counts.values
    plt.loglog(range(1, len(freqs) + 1), freqs)
    plt.xlabel("Track rank (log)")
    plt.ylabel("Number of playlists containing track (log)")
    plt.title("Track popularity long tail")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"track_longtail_{args.name}.png"), dpi=120)
    plt.close()

    # Figure 3: artist popularity long tail
    plt.figure(figsize=(6, 4))
    afreqs = artist_counts.values
    plt.loglog(range(1, len(afreqs) + 1), afreqs)
    plt.xlabel("Artist rank (log)")
    plt.ylabel("Number of playlist appearances (log)")
    plt.title("Artist popularity long tail")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"artist_longtail_{args.name}.png"), dpi=120)
    plt.close()

    print(f"\nSaved summary to results/metrics/eda_{args.name}.json")
    print(f"Saved 3 figures to results/figures/")


if __name__ == "__main__":
    main()
