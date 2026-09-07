# Project log

## 2026-09-07
- Set up folders, environment, git

## 2026-09-08
- Recreated venv with Python 3.13 (3.14 lacked wheels for `implicit`).
- `src/build_dataset.py`: streams raw MPD slice JSON -> compact Parquet
  (playlists / tracks / interactions), one slice at a time so memory stays
  bounded. Built a 50k-playlist dev subset (`data/processed/dev50k`).
- `src/eda.py`: EDA on dev50k. Headline finding: 57.7% of tracks appear in
  only one playlist (heavy long tail) -> pure collaborative filtering will
  struggle on rare tracks, motivating a hybrid/fallback approach later.
- `src/make_eval_split.py`: builds an Automatic-Playlist-Continuation-style
  eval task -- holds out 2000 playlists, reveals only the first k tracks
  (k in {0,1,5,10,25}, mimicking the official challenge's seed scenarios),
  hides the rest as ground truth. Also writes `train_interactions.parquet`,
  the only data any model is allowed to see (no leakage of hidden tracks).
- `src/metrics.py`: R-precision, NDCG, and "clicks" metrics matching the
  RecSys Challenge 2018 definitions.
- `src/baseline_popularity.py`: popularity-only baseline.
  Overall: R-precision 0.024, NDCG 0.075, clicks 20.4.
- `src/als_model.py`: implicit ALS matrix factorization (factors=64,
  iterations=15). Eval playlists are included as rows in the training
  matrix (seed tracks only) so ALS folds them in naturally.
  Overall: R-precision 0.117, NDCG 0.264, clicks 6.4 -- a ~5x improvement
  in R-precision over popularity.
  Limitation: playlists with 0 seed tracks (title-only) have no
  interactions to fold in, so ALS can't recommend for them at all -- next
  step is a hybrid that falls back to popularity for that case.
- Next: hybrid cold-start fallback, scale pipeline to the full ~1M
  playlists, write up report.