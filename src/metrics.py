"""
Metrics matching the RecSys Challenge 2018 (Spotify MPD) definitions:
  - R-precision
  - NDCG (binary relevance, official discounting)
  - Recommended Songs "clicks" (how many groups of 10 you'd have to scroll
    through before finding a relevant track; capped at 51 if never found)

recommended: ordered list of track_uris (our top-K predictions)
ground_truth: set of track_uris that were actually hidden from the playlist
"""
import numpy as np


def r_precision(recommended, ground_truth):
    if not ground_truth:
        return 0.0
    top = recommended[: len(ground_truth)]
    hits = sum(1 for t in top if t in ground_truth)
    return hits / len(ground_truth)


def ndcg(recommended, ground_truth, k=500):
    if not ground_truth:
        return 0.0
    rec = recommended[:k]
    dcg = 0.0
    for i, t in enumerate(rec):
        if t in ground_truth:
            rel = 1.0
            if i == 0:
                dcg += rel
            else:
                dcg += rel / np.log2(i + 2)
    n_rel = min(len(ground_truth), k)
    idcg = 1.0 + sum(1.0 / np.log2(i + 2) for i in range(1, n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def clicks(recommended, ground_truth, k=500):
    rec = recommended[:k]
    for i, t in enumerate(rec):
        if t in ground_truth:
            return i // 10
    return 51


def evaluate_all(recommended, ground_truth, k=500):
    return {
        "r_precision": r_precision(recommended, ground_truth),
        "ndcg": ndcg(recommended, ground_truth, k=k),
        "clicks": clicks(recommended, ground_truth, k=k),
    }
