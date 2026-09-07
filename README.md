# Spotify Automatic Playlist Continuation

End-to-end ML study on the Spotify Million Playlist Dataset (MPD): given a
playlist with only some tracks known, predict the rest.
ES 335 Course project, IIT Gandhinagar.

**[Full report / write-up](https://keshtib07-ghost.github.io/spotify-apc/)**

## Results

Hybrid model (ALS matrix factorization + popularity fallback for
title-only playlists) vs. a popularity-only baseline, evaluated on our own
held-out split of the full 1M-playlist dataset (10,000 eval playlists),
using the RecSys Challenge 2018 metrics:

| Model | R-precision | NDCG | Clicks |
|---|---|---|---|
| Popularity | 0.024 | 0.075 | 21.2 |
| **Hybrid (ALS + popularity fallback)** | **0.095** | **0.219** | **9.1** |

See the [full report](https://keshtib07-ghost.github.io/spotify-apc/) for
methodology, EDA, and discussion, and [LOG.md](LOG.md) for the dated
project log.

## Project layout

```
src/                  pipeline scripts (see docs/index.md "Reproducing this")
data/raw/             raw MPD JSON slices (gitignored, not tracked)
data/processed/       built Parquet tables + eval splits (gitignored)
results/metrics/      saved metrics JSON per model
results/figures/      saved plots
docs/                 GitHub Pages report source
```