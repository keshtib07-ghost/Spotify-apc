"""
Combine saved metrics JSONs into one comparison table + bar chart.

Usage:
  python src/compare_models.py --name dev50k --models popularity knn hybrid
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
METRIC_DIR = os.path.join(BASE, "results", "metrics")
FIG_DIR = os.path.join(BASE, "results", "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="dev50k")
    ap.add_argument("--models", nargs="+", default=["baseline_popularity", "knn", "hybrid"])
    args = ap.parse_args()

    rows = []
    for m in args.models:
        path = os.path.join(METRIC_DIR, f"{m}_{args.name}.json")
        with open(path) as f:
            data = json.load(f)
        row = data["overall"]
        row["model"] = m
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")[["r_precision", "ndcg", "clicks"]]
    print(df)

    md_path = os.path.join(METRIC_DIR, f"comparison_{args.name}.md")
    with open(md_path, "w") as f:
        f.write(f"# Model comparison on `{args.name}`\n\n")
        f.write(df.to_markdown(floatfmt=".4f"))
        f.write("\n")
    print(f"Saved table to {md_path}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(
        axes, ["r_precision", "ndcg", "clicks"],
        ["R-precision (higher better)", "NDCG (higher better)", "Clicks (lower better)"],
    ):
        df[metric].plot.bar(ax=ax, color=["#888", "#4c72b0", "#55a868"][: len(df)])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"comparison_{args.name}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
