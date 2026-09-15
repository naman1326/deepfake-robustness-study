#!/usr/bin/env python3
"""
plot_blur_packetloss.py

Generates the blur/packet-loss tradeoff plot from the already-computed
condition_metrics_summary.csv (from analyze_results.py), matching the same
style as accuracy_vs_bitrate.png and accuracy_vs_resolution.png.

Usage:
    python plot_blur_packetloss.py \
        --summary_csv "C:\\path\\to\\analysis\\condition_metrics_summary.csv" \
        --output_dir  "C:\\path\\to\\analysis"
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    baseline_row = df[df["condition_type"] == "baseline"]

    subset = df[df["condition_type"].isin(["blur", "packet_loss_sim"])].copy()
    # Order: blur sigma1.5, blur sigma3.0, then packet loss 1pct
    order = ["sigma1.5", "sigma3.0", "1pct"]
    subset["order"] = subset["condition_value"].apply(
        lambda v: order.index(v) if v in order else -1)
    subset = subset.sort_values("order")

    x_labels = [
        f"{row['condition_type']}\n{row['condition_value']}"
        for _, row in subset.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_labels, subset["auc"], marker="o", label="AUC")
    ax.plot(x_labels, subset["accuracy"], marker="s", label="Accuracy")
    if not baseline_row.empty:
        ax.axhline(baseline_row["auc"].values[0], linestyle="--", color="gray",
                   label=f"Baseline AUC ({baseline_row['auc'].values[0]:.3f})")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Score")
    ax.set_title("Detector performance vs. blur and packet-loss degradation")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    out_path = os.path.join(args.output_dir, "accuracy_vs_blur_packetloss.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[DONE] Saved {out_path}")


if __name__ == "__main__":
    main()
