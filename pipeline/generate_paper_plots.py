#!/usr/bin/env python3
r"""
generate_paper_plots.py

Regenerates every figure used in the paper directly from the verified
numbers in condition_metrics_summary.csv (produced by analyze_results.py)
and from the small pretrained-vs-fine-tuned dry-run comparison recorded
during development. No numbers in this script are invented; the bitrate,
resolution, blur, and packet-loss figures match condition_metrics_summary.csv
exactly, and the pretrained-vs-fine-tuned figures match the dry-run CSVs
(dryrun_inference_results.csv and dryrun_inference_results_finetuned.csv).

Run this from the deepfake_research folder, with matplotlib installed
(pip install matplotlib).

Usage:
    python generate_paper_plots.py --output_dir .\paper_figs
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def make_summary_plots(output_dir):
    conditions = ["Baseline", "500kbps", "1000kbps", "2000kbps", "5000kbps",
                  "240p", "360p", "480p", "720p", "Blur s1.5", "Blur s3.0", "PktLoss 1%"]
    auc = [0.9997, 0.9743, 0.9963, 0.9991, 0.9991, 0.9207, 0.9829, 0.9933, 0.9960, 0.9010, 0.5917, 0.9857]
    eer = [0.0075, 0.0650, 0.0150, 0.0100, 0.0150, 0.1600, 0.0650, 0.0500, 0.0300, 0.1975, 0.4525, 0.0550]
    families = ["baseline", "bitrate", "bitrate", "bitrate", "bitrate",
                "resolution", "resolution", "resolution", "resolution",
                "blur", "blur", "packetloss"]
    colors_map = {"baseline": "#555555", "bitrate": "#1f77b4", "resolution": "#ff7f0e",
                  "blur": "#d62728", "packetloss": "#2ca02c"}
    bar_colors = [colors_map[f] for f in families]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(conditions, auc, color=bar_colors)
    ax.set_ylabel("AUC")
    ax.set_title("AUC across all evaluated degradation conditions")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="Random guessing (0.5)")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auc_all_conditions.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(conditions, eer, color=bar_colors)
    ax.set_ylabel("Equal Error Rate (EER)")
    ax.set_title("EER across all evaluated degradation conditions")
    ax.set_ylim(0, 0.5)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eer_all_conditions.png"), dpi=150)
    plt.close(fig)

    blur_pl_labels = ["Blur\nsigma=1.5", "Blur\nsigma=3.0", "Packet loss\n1%"]
    blur_pl_auc = [0.9010, 0.5917, 0.9857]
    blur_pl_acc = [0.821, 0.799, 0.890]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(blur_pl_labels, blur_pl_auc, marker="o", label="AUC")
    ax.plot(blur_pl_labels, blur_pl_acc, marker="s", label="Accuracy")
    ax.axhline(0.9997, linestyle="--", color="gray", label="Baseline AUC (0.9997)")
    ax.set_ylabel("Score")
    ax.set_title("Detector performance vs. blur and packet-loss degradation")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_vs_blur_packetloss.png"), dpi=150)
    plt.close(fig)


def make_pretrained_vs_finetuned_plot(output_dir):
    # From the actual dry-run CSVs (5 held-out real, undegraded videos, baseline condition)
    pretrained = [1.0, 0.9774655520915985, 0.943026340007782, 0.5762307696044445, 0.9999998867511749]
    finetuned = [0.17434740107726937, 0.26685606521459704, 0.18374977841864962, 0.2320816667543113, 0.5468761067721062]
    videos = [f"Video {i+1}" for i in range(5)]

    x = np.arange(len(videos))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - width / 2, pretrained, width, label="Off-the-shelf pretrained checkpoint")
    ax.bar(x + width / 2, finetuned, width, label="Fine-tuned (ours)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("Predicted fake probability")
    ax.set_title("Baseline fake-probability on real videos: pretrained vs. fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(videos)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pretrained_vs_finetuned.png"), dpi=150)
    plt.close(fig)


def make_architecture_diagram(output_dir):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8eef7", ec="#2c3e50", fontsize=9):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                            linewidth=1.3, edgecolor=ec, facecolor=fc)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
        return (x + w / 2, y, x + w / 2, y + h)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.2, color="#2c3e50")
        ax.add_patch(a)

    cx1, by1, cx1t, ty1 = box(2.75, 11.5, 4.5, 1.0, "FaceForensics++ (C23)\nBalanced subset: 200 x 5 classes\n(1,000 videos)", fc="#eaf2e3")
    cx2, by2, cx2t, ty2 = box(2.75, 9.9, 4.5, 1.0, "Per-frame face detection (dlib)\n+ bounding-box crop (1.3x margin)")
    arrow(cx1, by1, cx2, ty2)

    cx3, by3, cx3t, ty3 = box(0.4, 7.9, 4.0, 1.3, "Fine-tuning on clean frames\nXception backbone, 5 epochs\nbatch=8, Adam (lr=1e-3)\nBest val. acc: 91.46%", fc="#fdf1df")
    arrow(cx2 - 1.2, by2, cx3t, ty3)

    cx4, by4, cx4t, ty4 = box(5.1, 7.9, 4.5, 1.3, "Degradation matrix (FFmpeg)\nBitrate, Resolution,\nGaussian blur, Simulated\npacket loss (12 conditions)", fc="#fdf1df")
    arrow(cx2 + 1.2, by2, cx4t, ty4)

    cx5, by5, cx5t, ty5 = box(2.75, 6.1, 4.5, 1.0, "Fine-tuned Xception detector\n(best.pkl checkpoint)")
    arrow(cx3, by3, cx5t + 1.0, ty5)
    arrow(cx4, by4, cx5t - 0.2, ty5)

    cx6, by6, cx6t, ty6 = box(2.75, 4.4, 4.5, 1.0, "Per-video inference:\nmean fake-probability over\nup to 10 sampled frames")
    arrow(cx5, by5, cx6, ty6)

    cx7, by7, cx7t, ty7 = box(2.75, 2.7, 4.5, 1.0, "Per-condition evaluation:\nAUC, EER, Accuracy@0.5")
    arrow(cx6, by6, cx7, ty7)

    cx8, by8, cx8t, ty8 = box(2.75, 1.0, 4.5, 1.0, "Results: degradation-robustness\nprofile across 12 conditions", fc="#eaf2e3")
    arrow(cx7, by7, cx8, ty8)

    ax.set_title("Pipeline overview: dataset, fine-tuning, degradation, and evaluation", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "architecture_diagram.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="./paper_figs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    make_summary_plots(args.output_dir)
    make_pretrained_vs_finetuned_plot(args.output_dir)
    make_architecture_diagram(args.output_dir)

    print(f"[DONE] All plots written to {args.output_dir}")
    print("Files: auc_all_conditions.png, eer_all_conditions.png,")
    print("       accuracy_vs_blur_packetloss.png, pretrained_vs_finetuned.png,")
    print("       architecture_diagram.png")
    print("Note: accuracy_vs_bitrate.png and accuracy_vs_resolution.png are")
    print("      already produced by analyze_results.py and are not regenerated here.")


if __name__ == "__main__":
    main()
