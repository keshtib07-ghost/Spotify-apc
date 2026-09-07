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
- `src/knn_model.py`: item-based CF via playlist co-occurrence (no training
  step, pure lookup). On the full 2000-task eval set (same tasks as above,
  including the 441 zero-seed ones, which fall back to popularity):
  R-precision 0.092, NDCG 0.207, clicks 10.6.
- `src/hybrid_model.py`: ALS when a playlist has seed tracks, popularity
  fallback when it doesn't. Evaluated on the same full 2000-task set as KNN
  (unlike the earlier `als_model.py` number, which only covered the 1559
  tasks ALS could fold in) so it's a fair comparison:
  R-precision 0.097, NDCG 0.223, clicks 9.3 -- best of the three.
- Decision: KNN's per-task cost scales with how many playlists share a
  seed track, which gets expensive for popular tracks at 1M-playlist scale.
  ALS's cost is a fixed matrix multiply regardless of data size. So the
  hybrid (ALS + popularity fallback) is the model we scale to the full
  dataset; KNN stays as a documented dev-scale comparison point in the
  report.
- Built the full ~1M-playlist processed dataset (2.26M unique tracks, 66.3M
  interactions; 47.4% of tracks are still singletons even at full scale).
  Eval split: 10,000 held-out playlists, same seed-length scenarios as dev.
- Found and fixed a real perf bug: `baseline_popularity.py` and the
  fallback path in `hybrid_model.py` were filtering seed tracks against the
  *entire* ranked track list per eval task (2.26M-item Python list
  comprehension x 10,000 tasks). Fixed by pre-slicing to just the top
  `k + max_seed_len` candidates before filtering -- popularity baseline
  went from a projected ~19 minutes to 22 seconds.
- Full-scale results (10,000 eval playlists), confirming the dev50k
  findings hold at real scale:
    popularity: R-precision 0.024, NDCG 0.075, clicks 21.2
    hybrid (ALS+popularity fallback): R-precision 0.095, NDCG 0.219,
      clicks 9.1  (~4x R-precision over popularity; ALS training took 375s
      for 1M playlists x 2.26M tracks, 15 iterations, factors=64)
- Next: write up the report (methodology, results, discussion of the
  long-tail/cold-start limitation), then GitHub Pages.