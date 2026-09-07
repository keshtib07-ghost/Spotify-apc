"""
Convert raw MPD slice JSON files into compact Parquet tables.

Reads data/raw/mpd.slice.<start>-<end>.json files one at a time (so memory use
stays bounded regardless of how many slices we process) and produces three
tables under data/processed/:

  playlists.parquet   one row per playlist (pid, name, num_followers, ...)
  tracks.parquet      one row per unique track (track_uri, name, artist, ...)
  interactions.parquet  one row per (pid, pos, track_uri) - the playlist contents

Usage:
  python src/build_dataset.py --n-slices 50 --out-name dev50k
  python src/build_dataset.py --n-slices 1000 --out-name full
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def slice_filenames(n_slices):
    names = sorted(
        os.listdir(RAW_DIR),
        key=lambda f: int(f.split(".")[2].split("-")[0]),
    )
    names = [f for f in names if f.startswith("mpd.slice.")]
    return names[:n_slices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-slices", type=int, default=50, help="number of 1000-playlist slice files to process")
    ap.add_argument("--out-name", type=str, default="dev50k", help="subfolder name under data/processed")
    args = ap.parse_args()

    files = slice_filenames(args.n_slices)
    print(f"Processing {len(files)} slice files -> data/processed/{args.out_name}")

    playlist_rows = []
    interaction_rows = []
    track_seen = {}  # track_uri -> row dict
    t0 = time.time()

    for i, fname in enumerate(files):
        with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as fh:
            data = json.load(fh)

        for pl in data["playlists"]:
            playlist_rows.append({
                "pid": pl["pid"],
                "name": pl.get("name", ""),
                "collaborative": pl.get("collaborative", "false") == "true",
                "num_tracks": pl["num_tracks"],
                "num_albums": pl["num_albums"],
                "num_artists": pl["num_artists"],
                "num_followers": pl["num_followers"],
                "num_edits": pl.get("num_edits", 0),
                "duration_ms": pl.get("duration_ms", 0),
                "modified_at": pl.get("modified_at", 0),
            })
            for t in pl["tracks"]:
                uri = t["track_uri"]
                interaction_rows.append({
                    "pid": pl["pid"],
                    "pos": t["pos"],
                    "track_uri": uri,
                })
                if uri not in track_seen:
                    track_seen[uri] = {
                        "track_uri": uri,
                        "track_name": t.get("track_name", ""),
                        "artist_uri": t.get("artist_uri", ""),
                        "artist_name": t.get("artist_name", ""),
                        "album_uri": t.get("album_uri", ""),
                        "album_name": t.get("album_name", ""),
                        "duration_ms": t.get("duration_ms", 0),
                    }

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(files)}] playlists={len(playlist_rows)} "
                  f"unique_tracks={len(track_seen)} elapsed={elapsed:.1f}s", flush=True)

    out_dir = os.path.join(PROCESSED_DIR, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(playlist_rows).to_parquet(os.path.join(out_dir, "playlists.parquet"), index=False)
    pd.DataFrame(list(track_seen.values())).to_parquet(os.path.join(out_dir, "tracks.parquet"), index=False)
    pd.DataFrame(interaction_rows).to_parquet(os.path.join(out_dir, "interactions.parquet"), index=False)

    print(f"Done in {time.time()-t0:.1f}s. Wrote to {out_dir}")
    print(f"  playlists: {len(playlist_rows)}")
    print(f"  unique tracks: {len(track_seen)}")
    print(f"  interactions: {len(interaction_rows)}")


if __name__ == "__main__":
    sys.exit(main())
