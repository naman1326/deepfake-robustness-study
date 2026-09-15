#!/usr/bin/env python3
"""
analyze_results.py

Computes AUC, EER, and threshold-accuracy per degradation condition from
inference_results_full.csv (produced by batch_infer.py), and produces the
accuracy/AUC-vs-degradation tradeoff plots that form the core results
figures for the paper.

Usage:
    python analyze_results.py \
        --results_csv "C:\\path\\to\\inference_results_full.csv" \
        --output_dir  "C:\\path\\to\\analysis"

Requires: pip install scikit-learn matplotlib pandas
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNGs
import matplotlib.pyplot as plt


def compute_eer(y_true, y_score):
    """Equal Error Rate: the point where false positive rate == false
    negative rate (1 - true positive rate)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # index where fpr and fnr are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold for accuracy calculation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    print(f"[INFO] Loaded {len(df)} rows")

    # Drop rows with no face detected (empty avg_fake_prob)
    before = len(df)
    df = df.dropna(subset=["avg_fake_prob"])
    df = df[df["avg_fake_prob"] != ""]
    df["avg_fake_prob"] = df["avg_fake_prob"].astype(float)
    print(f"[INFO] Dropped {before - len(df)} rows with no detectable face")

    df["y_true"] = (df["label"] == "fake").astype(int)

    summary_rows = []
    for (ctype, cval), group in df.groupby(["condition_type", "condition_value"]):
        y_true = group["y_true"].values
        y_score = group["avg_fake_prob"].values

        # Need both classes present to compute AUC/EER meaningfully
        if len(np.unique(y_true)) < 2:
            print(f"[WARN] Skipping {ctype}/{cval}: only one class present")
            continue

        auc = roc_auc_score(y_true, y_score)
        eer = compute_eer(y_true, y_score)
        y_pred = (y_score >= args.threshold).astype(int)
        accuracy = (y_pred == y_true).mean()

        summary_rows.append({
            "condition_type": ctype,
            "condition_value": cval,
            "n_videos": len(group),
            "auc": auc,
            "eer": eer,
            "accuracy": accuracy,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "condition_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[DONE] Wrote per-condition metrics to {summary_csv}")
    print(summary_df.to_string(index=False))

    # --- Plot 1: Accuracy/AUC vs bitrate ---
    bitrate_order = ["500kbps", "1000kbps", "2000kbps", "5000kbps"]
    baseline_row = summary_df[summary_df["condition_type"] == "baseline"]
    bitrate_df = summary_df[summary_df["condition_type"] == "bitrate"].copy()
    bitrate_df["order"] = bitrate_df["condition_value"].apply(
        lambda v: bitrate_order.index(v) if v in bitrate_order else -1)
    bitrate_df = bitrate_df.sort_values("order")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    x_labels = list(bitrate_df["condition_value"])
    ax1.plot(x_labels, bitrate_df["auc"], marker="o", label="AUC")
    ax1.plot(x_labels, bitrate_df["accuracy"], marker="s", label="Accuracy")
    if not baseline_row.empty:
        ax1.axhline(baseline_row["auc"].values[0], linestyle="--", color="gray",
                    label=f"Baseline AUC ({baseline_row['auc'].values[0]:.3f})")
    ax1.set_xlabel("Bitrate condition")
    ax1.set_ylabel("Score")
    ax1.set_title("Detector performance vs. bitrate degradation")
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "accuracy_vs_bitrate.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Accuracy/AUC vs resolution ---
    resolution_order = ["240p", "360p", "480p", "720p"]
    resolution_df = summary_df[summary_df["condition_type"] == "resolution"].copy()
    resolution_df["order"] = resolution_df["condition_value"].apply(
        lambda v: resolution_order.index(v) if v in resolution_order else -1)
    resolution_df = resolution_df.sort_values("order")

    fig, ax2 = plt.subplots(figsize=(7, 5))
    x_labels = list(resolution_df["condition_value"])
    ax2.plot(x_labels, resolution_df["auc"], marker="o", label="AUC")
    ax2.plot(x_labels, resolution_df["accuracy"], marker="s", label="Accuracy")
    if not baseline_row.empty:
        ax2.axhline(baseline_row["auc"].values[0], linestyle="--", color="gray",
                    label=f"Baseline AUC ({baseline_row['auc'].values[0]:.3f})")
    ax2.set_xlabel("Resolution condition")
    ax2.set_ylabel("Score")
    ax2.set_title("Detector performance vs. resolution degradation")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "accuracy_vs_resolution.png"), dpi=150)
    plt.close(fig)

    print(f"\n[DONE] Saved plots to {args.output_dir}\\accuracy_vs_bitrate.png "
          f"and {args.output_dir}\\accuracy_vs_resolution.png")


if __name__ == "__main__":
    main()
