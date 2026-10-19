#!/usr/bin/env python3
r"""
generate_all_figures.py

Regenerates every figure used in the CallGuard-X paper.

DATA PROVENANCE (nothing here is invented):
  * OURS_*            : condition_metrics_summary.csv produced by analyze_results.py
  * TRAIN_LOG         : the console log of train_CNN.py (5 epochs, batch size 8)
  * DRYRUN_*          : dryrun_inference_results*.csv (5 real videos, baseline condition)
  * HF_*              : model card of Akram and Dodo (2026), FF++ C23, Hugging Face
                        repo MUmairAB/deepfake-detection-ff-cn-transformer
  * FRONTIERS_*       : Table 5 of Li, Azzali and Farid, Front. Artif. Intell. 9:1835651 (2026)
  * XCEPTION_LCF      : Xception AUC values reported in arXiv:2605.29092

Usage (from any folder, needs matplotlib and numpy):
    python generate_all_figures.py --output_dir ./figs
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OURS_COLOR = "#e8710a"
LIT_COLOR = "#4a6fa5"
GREY = "#8a8a8a"
BASE_AUC = 0.999675

# ---------------------------------------------------------------- our results
COND = ["Baseline", "500kbps", "1000kbps", "2000kbps", "5000kbps",
        "240p", "360p", "480p", "720p", "Blur s=1.5", "Blur s=3.0", "PktLoss 1%"]
AUC = [0.999675, 0.97425, 0.9963, 0.999125, 0.999075, 0.920725, 0.982925, 0.99325, 0.996025, 0.901025, 0.591725, 0.985725]
EER = [0.0075, 0.065, 0.015, 0.01, 0.015, 0.16, 0.065, 0.05, 0.03, 0.1975, 0.4525, 0.055]
ACC = [0.995, 0.945, 0.979, 0.992, 0.996, 0.875, 0.946, 0.966, 0.981, 0.821, 0.799, 0.890]
FAMILY = ["baseline"] + ["bitrate"] * 4 + ["resolution"] * 4 + ["blur"] * 2 + ["packet"]
FCOL = {"baseline": "#555555", "bitrate": "#1f77b4", "resolution": "#ff7f0e", "blur": "#d62728", "packet": "#2ca02c"}
P_MAJ = 0.8  # 800 fake of 1000 videos


def kappa(acc, p_maj):
    """chance-corrected accuracy: improvement over a majority-class classifier"""
    return (acc - p_maj) / (1.0 - p_maj)


# ------------------------------------------------------------ literature data
HF_MODELS = ["Xception", "EfficientNet-B3", "CNN+BiLSTM", "ViT-Base/16", "Hybrid CNN-Transf."]
HF_AUC = [0.9944, 0.9976, 0.9805, 0.7066, 0.9458]
HF_ACC = [0.9744, 0.9829, 0.9744, 0.6695, 0.8462]
HF_FS = [0.7015, 0.7343, 0.7591, 0.6880, 0.6273]
HF_PARAMS = [20.8, 10.7, 13.7, 85.8, 74.9]

FR_MODELS = ["XceptionNet", "FSD-GAN", "FakeTracer", "MesoNet", "Hybrid+Landmark"]
FR = {  # clean, JPEG, noise, low-light, blur  (accuracy %), balanced test sets
    "XceptionNet": (96.8, 89.5, 88.7, 87.0, 85.3),
    "FSD-GAN": (96.2, 92.0, 91.8, 90.5, 89.6),
    "FakeTracer": (94.5, 91.4, 89.7, 91.1, 90.2),
    "MesoNet": (88.1, 68.5, 65.9, 66.2, 65.1),
    "Hybrid+Landmark": (90.3, 86.0, 85.2, 84.9, 83.5),
}
XCEPTION_LCF = (0.9805, 0.6248)  # in-domain C23 AUC, FaceShifter AUC


def fr_kappa(v):
    return (v / 100.0 - 0.5) / 0.5


# ----------------------------------------------------------------- helpers
from matplotlib.transforms import Bbox


def save_panels(fig, groups, outdir, names, pad=0.06):
    """Save each group of axes in `groups` as its own PNG (union of tight boxes)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for axes, name in zip(groups, names):
        boxes = [a.get_tightbbox(rend).transformed(fig.dpi_scale_trans.inverted()) for a in axes]
        b = Bbox.union(boxes)
        b = Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)
        path = os.path.join(outdir, name + ".png")
        fig.savefig(path, dpi=220, bbox_inches=b)
        print("saved", path)
    plt.close(fig)


def savefig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def label_bars(ax, bars, fmt="{:.3f}", fs=7.5, dy=0.01):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy if h >= 0 else dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs)


# =============================================================== DIAGRAMS
def rbox(ax, x, y, w, h, text, fc="#e8eef7", ec="#2c3e50", fs=12, lw=1.5, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)


def arr(ax, x1, y1, x2, y2, color="#2c3e50", style="-|>", lw=1.6, ls="-", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15, linewidth=lw,
                        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def fig_system_model(outdir):
    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 11.6)
    ax.axis("off")
    w = 3.9
    gap = 0.6
    xs = [0.2 + i * (w + gap) for i in range(7)]
    y, h = 6.4, 3.1
    stages = [
        ("1  Sender\ncamera or virtual\ncamera (real or\ndeepfake feed)", "#eaf2e3"),
        ("2  Video encoder\nH.264 rate control\n500 to 5000 kbps\n240p to 720p", "#fdf1df"),
        ("3  Network channel\nbandwidth drift,\npacket loss, jitter", "#fdf1df"),
        ("4  Receiver client\ndecoder and client\nfilters (blur\nsigma 1.5 and 3.0)", "#fdf1df"),
        ("5  Face localisation\ndlib HOG, 1.3x crop\nresize 299x299\nnormalise [-1, 1]", "#e8eef7"),
        ("6  CallGuard-X\nfine-tuned Xception\n2-way softmax\n(20.8 M params)", "#e8eef7"),
        ("7  Decision\nmean p(fake) over\nK <= 10 frames,\nthreshold, warning", "#eaf2e3"),
    ]
    for (t, c), x in zip(stages, xs):
        rbox(ax, x, y, w, h, t, fc=c, fs=10.8)
    for i in range(6):
        arr(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    ops = [(xs[1] + w / 2, r"$D_b,\ D_r$"), (xs[2] + w / 2, r"$D_p$"), (xs[3] + w / 2, r"$D_\sigma$")]
    for cx, t in ops:
        ax.text(cx, y + h + 0.45, t, ha="center", fontsize=13, color="#b03a2e", weight="bold")
    ax.plot([xs[1], xs[3] + w], [y + h + 1.05] * 2, color="#b03a2e", lw=1.4)
    ax.text((xs[1] + xs[3] + w) / 2, y + h + 1.3, "degradation channel modelled in this study (12 conditions)",
            ha="center", fontsize=12, color="#b03a2e")
    # placement lane
    bx0, bx1 = xs[4], xs[5] + w
    ax.plot([bx0, bx1], [5.7, 5.7], color=GREY, lw=1.4)
    ax.plot([bx0, bx0], [5.7, y - 0.1], color=GREY, lw=1.4)
    ax.plot([bx1, bx1], [5.7, y - 0.1], color=GREY, lw=1.4)
    ax.text((bx0 + bx1) / 2, 5.05, "stages 5 and 6 can be deployed at one of three places (Section XI)", ha="center", fontsize=11.5, weight="bold")
    pw = 5.2
    pxs = [11.4, 17.1, 22.8]
    ptxt = ["Client side\non-device inference,\nno video leaves call",
            "Edge server\nmedia-server (SFU) tap,\nshared GPU",
            "Cloud service\nelastic GPUs,\nhighest latency"]
    for px, t in zip(pxs, ptxt):
        rbox(ax, px, 1.9, pw, 2.2, t, fc="#f4f4f4", fs=10.8)
        ax.plot([px + pw / 2] * 2, [4.1, 4.75], color=GREY, lw=1.1)
    ax.plot([pxs[0] + pw / 2, pxs[2] + pw / 2], [4.75, 4.75], color=GREY, lw=1.1)
    ax.plot([(bx0 + bx1) / 2] * 2, [4.75, 5.7], color=GREY, lw=1.1)
    ax.text(5.6, 3.0, "Evaluation harness:\nstages 1 to 4 are replaced\nby offline FFmpeg operators\napplied to FaceForensics++\nclips", ha="center", va="center", fontsize=11, style="italic")
    savefig(fig, outdir, "system_model.png")


def fig_methodology(outdir):
    fig, ax = plt.subplots(figsize=(16, 8.4))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 17)
    ax.axis("off")

    def lane(y, title):
        ax.text(0.2, y, title, fontsize=13, weight="bold", color="#2c3e50")
        ax.plot([0.2, 31.8], [y - 0.35] * 2, color="#c8c8c8", lw=1)

    # ---- lane A
    lane(16.4, "A  Data preparation")
    A = [("FaceForensics++ C23\n5 classes x 200\n= 1,000 videos", "#eaf2e3"),
         ("Video-level split\n850 train / 150 val\n(seed 42)", "#e8eef7"),
         ("10 frames per video\nevenly spaced", "#e8eef7"),
         ("dlib face crop, 1.3x\nresize 299x299\nnormalise [-1, 1]", "#e8eef7"),
         ("~8,500 train and\n~1,500 val frames\n0 = real, 1 = fake", "#eaf2e3")]
    wA, gA = 5.6, 0.9
    ya, hA = 13.3, 2.2
    for i, (t, c) in enumerate(A):
        x = 0.3 + i * (wA + gA)
        rbox(ax, x, ya, wA, hA, t, fc=c, fs=11.5)
        if i < 4:
            arr(ax, x + wA, ya + hA / 2, x + wA + gA, ya + hA / 2)
    xA_last = 0.3 + 4 * (wA + gA) + wA / 2
    # connector A -> B : down the right margin, then left above lane B
    ax.plot([xA_last, xA_last], [ya, 12.2], color="#555555", ls="--", lw=1.4)
    ax.plot([xA_last, 1.6], [12.2, 12.2], color="#555555", ls="--", lw=1.4)
    arr(ax, 1.6, 12.2, 1.6, 10.35, color="#555555", ls="--", lw=1.4)
    ax.text(15.5, 12.45, "frames used for fine-tuning", fontsize=10.5, style="italic", color="#555555", ha="center")

    # ---- lane B
    lane(11.7, "B  Detector (Xception) and fine-tuning")
    B = [("Input\n299x299x3", "#f4f4f4", 2.9),
         ("Entry flow\n2 conv, 3 sep-conv\nblocks (128,256,728)", "#e8eef7", 4.5),
         ("Middle flow\n8 x [3 sep-conv, 728]\nresidual", "#e8eef7", 4.2),
         ("Exit flow\nsep-conv 1024,\n1536, 2048", "#e8eef7", 3.5),
         ("Global avg\npool (2048)", "#e8eef7", 2.9),
         ("Dropout p=0.5\nFC 2048 to 2\nsoftmax", "#fdf1df", 3.3),
         ("Cross-entropy, Adam\nlr 1e-3, batch 8, 5\nepochs; final weights\nexported (best.pkl)", "#eaf2e3", 4.7)]
    x = 0.3
    yb, hb = 8.2, 2.15
    gapb = 0.5
    prev = None
    endx = None
    for t, c, wi in B:
        rbox(ax, x, yb, wi, hb, t, fc=c, fs=10.4)
        if prev is not None:
            arr(ax, prev, yb + hb / 2, x, yb + hb / 2)
        prev = x + wi
        endx = x + wi / 2
        x += wi + gapb
    # checkpoint connector B -> C (right margin, then left above inference box)
    ax.plot([endx, endx], [yb, 6.6], color="#555555", ls="--", lw=1.4)
    ax.text(endx + 0.3, 7.1, "checkpoint\nbest.pkl", fontsize=10.5, style="italic", color="#555555", va="center")

    # ---- lane C
    lane(6.0, "C  Degradation, inference and evaluation")
    C = [("1,000 clean\nvideos", "#eaf2e3", 3.4),
         ("FFmpeg operators\nDb, Dr, Dsigma, Dp\n12 conditions", "#fdf1df", 4.6),
         ("12,000\ndegraded videos", "#fdf1df", 3.6),
         ("Per-video inference\nmean p(fake)\nK <= 10 frames", "#e8eef7", 4.6),
         ("Per-condition metrics\nAUC, EER, Acc@0.5\nn = 1,000 each", "#e8eef7", 4.8),
         ("Plots, tables and\ncomparison with\nliterature", "#eaf2e3", 4.2)]
    x = 0.3
    yc, hc = 2.2, 2.2
    prev = None
    pos = []
    for t, c, wi in C:
        rbox(ax, x, yc, wi, hc, t, fc=c, fs=10.8)
        if prev is not None:
            arr(ax, prev, yc + hc / 2, x, yc + hc / 2)
        pos.append((x, wi))
        prev = x + wi
        x += wi + 1.0
    inf_cx = pos[3][0] + pos[3][1] / 2
    ax.plot([endx, endx], [6.6, 5.2], color="#555555", ls="--", lw=1.4)
    ax.plot([endx, inf_cx], [5.2, 5.2], color="#555555", ls="--", lw=1.4)
    arr(ax, inf_cx, 5.2, inf_cx, yc + hc + 0.05, color="#555555", ls="--", lw=1.4)
    savefig(fig, outdir, "methodology_combined.png")


# =============================================================== OWN RESULTS
def fig_training(outdir):
    ep = [1, 2, 3, 4, 5]
    tr_acc = [0.8699, 0.9238, 0.9560, 0.9695, 0.9769]
    va_acc = [0.9146, 0.9086, 0.9126, 0.8939, 0.8819]
    tr_loss = [0.0657, 0.0246, 0.0154, 0.0103, 0.0080]
    va_loss = [0.0377, 0.0456, 0.0394, 0.0559, 0.0468]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].plot(ep, tr_acc, "o-", label="Training", color=LIT_COLOR)
    axs[0].plot(ep, va_acc, "s-", label="Validation", color=OURS_COLOR)
    axs[0].annotate("highest val. acc 0.9146\n(epoch 1)", xy=(1, 0.9146), xytext=(1.5, 0.995), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=GREY))
    axs[0].annotate("weights exported as\nbest.pkl and evaluated\n(epoch 5, val. acc 0.8819)", xy=(5, 0.8819), xytext=(2.7, 0.835), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=GREY))
    axs[0].set_ylim(0.82, 1.02)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Frame-level accuracy")
    axs[0].set_title("Accuracy during fine-tuning")
    axs[0].set_xticks(ep)
    axs[0].legend()
    axs[1].plot(ep, tr_loss, "o-", label="Training", color=LIT_COLOR)
    axs[1].plot(ep, va_loss, "s-", label="Validation", color=OURS_COLOR)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Epoch loss (as logged by train\\_CNN.py)")
    axs[1].set_title("Loss during fine-tuning")
    axs[1].set_xticks(ep)
    axs[1].legend()
    fig.tight_layout()
    savefig(fig, outdir, "training_curves.png")


def fig_accuracy_kappa(outdir):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = [FCOL[f] for f in FAMILY]
    b = axs[0].bar(COND, ACC, color=cols)
    axs[0].axhline(P_MAJ, color="black", ls="--", lw=1.1, label="Majority-class accuracy (0.800)")
    axs[0].set_ylim(0.6, 1.02)
    axs[0].set_ylabel("Accuracy at threshold 0.5")
    axs[0].set_title("Accuracy per condition")
    axs[0].tick_params(axis="x", rotation=55)
    axs[0].legend(loc="lower left", fontsize=9.5)
    label_bars(axs[0], b, "{:.3f}", 9.5, 0.004)
    k = [kappa(a, P_MAJ) for a in ACC]
    b2 = axs[1].bar(COND, k, color=cols)
    axs[1].axhline(0, color="black", lw=1)
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_ylabel(r"Chance-corrected accuracy $\kappa_{maj}$")
    axs[1].set_title("Improvement over a majority-class classifier")
    axs[1].tick_params(axis="x", rotation=55)
    label_bars(axs[1], b2, "{:.2f}", 9.5, 0.01)
    fig.tight_layout()
    save_panels(fig, [[axs[0]], [axs[1]]], outdir, ["acc_a_majority", "acc_b_kappa"])


def fig_heatmap(outdir):
    data = np.array([AUC, EER, ACC]).T
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    # invert EER colour semantics by overlaying value text only
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["AUC", "EER", "Accuracy"])
    ax.set_yticks(range(len(COND)))
    ax.set_yticklabels(COND)
    for i in range(len(COND)):
        for j in range(3):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9.5)
    ax.set_title("Metric summary across conditions")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, outdir, "fig_heatmap.png")


# ============================================================ COMPARISON
def fig_cmp1(outdir):
    names = ["CallGuard-X\n(ours)"] + [m.replace(" ", "\n", 1) if len(m) > 11 else m for m in HF_MODELS]
    auc = [0.999675] + HF_AUC
    acc = [0.995] + HF_ACC
    cols = [OURS_COLOR] + [LIT_COLOR] * 5
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    b = axs[0].bar(names, auc, color=cols)
    axs[0].set_ylim(0.5, 1.03)
    axs[0].set_ylabel("In-domain AUC")
    axs[0].set_title("AUC on FaceForensics++ C23")
    label_bars(axs[0], b, "{:.4f}", 9.5, 0.004)
    b = axs[1].bar(names, acc, color=cols)
    axs[1].axhline(0.8, color="black", ls="--", lw=1, label="Majority-class accuracy (~0.80, 4:1 split)")
    axs[1].set_ylim(0.5, 1.05)
    axs[1].set_ylabel("In-domain accuracy")
    axs[1].set_title("Accuracy on FaceForensics++ C23")
    axs[1].legend(loc="lower left", fontsize=9)
    label_bars(axs[1], b, "{:.4f}", 9.5, 0.004)
    for a in axs:
        plt.setp(a.get_xticklabels(), rotation=30, ha="right", fontsize=9.5)
    fig.tight_layout()
    save_panels(fig, [[axs[0]], [axs[1]]], outdir, ["cmp1a_auc", "cmp1b_acc"])


def fig_cmp2(outdir):
    # blur: chance-corrected accuracy
    labels = FR_MODELS + ["Ours\nsigma=1.5", "Ours\nsigma=3.0"]
    clean_k = [fr_kappa(FR[m][0]) for m in FR_MODELS] + [kappa(0.995, P_MAJ)] * 2
    blur_k = [fr_kappa(FR[m][4]) for m in FR_MODELS] + [kappa(0.821, P_MAJ), kappa(0.799, P_MAJ)]
    ret = [b / c for b, c in zip(blur_k, clean_k)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels))
    w = 0.38
    axs[0].bar(x - w / 2, clean_k, w, label="Clean", color=GREY)
    bb = axs[0].bar(x + w / 2, blur_k, w, label="Blurred", color=[LIT_COLOR] * 5 + [OURS_COLOR] * 2)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([l.replace("\n"," ") for l in labels], fontsize=9.5, rotation=35, ha="right")
    axs[0].set_ylabel(r"Chance-corrected accuracy $\kappa$")
    axs[0].set_title("Clean versus blurred input")
    axs[0].axhline(0, color="black", lw=0.8)
    axs[0].legend(fontsize=9.5, loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2)
    label_bars(axs[0], bb, "{:.2f}", 9.5, 0.01)
    bars = axs[1].bar(labels, [max(r, 0) for r in ret], color=[LIT_COLOR] * 5 + [OURS_COLOR] * 2)
    axs[1].set_ylabel(r"Retained $\kappa$ ($\kappa_{blur}/\kappa_{clean}$)")
    axs[1].set_title("Fraction of clean performance retained")
    axs[1].set_ylim(0, 1.05)
    plt.setp(axs[1].get_xticklabels(), rotation=35, ha="right", fontsize=9.5)
    for bar, r in zip(bars, ret):
        axs[1].text(bar.get_x() + bar.get_width() / 2, max(r, 0) + 0.015, f"{r:.2f}", ha="center", fontsize=9.5)
    fig.tight_layout()
    save_panels(fig, [[axs[0]], [axs[1]]], outdir, ["cmp2a_blur_kappa", "cmp2b_blur_retained"])


def fig_cmp3(outdir):
    # (a) compression retention
    lit_ret = [fr_kappa(FR[m][1]) / fr_kappa(FR[m][0]) for m in FR_MODELS]
    br = [("500 kbps", 0.945), ("1000 kbps", 0.979), ("2000 kbps", 0.992), ("5000 kbps", 0.996)]
    ours_ret = [kappa(a, P_MAJ) / kappa(0.995, P_MAJ) for _, a in br]
    labels = [m + "\n(JPEG)" for m in FR_MODELS] + ["Ours\n" + n for n, _ in br]
    vals = lit_ret + ours_ret
    cols = [LIT_COLOR] * 5 + [OURS_COLOR] * 4
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    b = axs[0].bar(labels, vals, color=cols)
    axs[0].set_ylim(0, 1.12)
    axs[0].set_ylabel(r"Retained $\kappa$ under compression")
    axs[0].set_title("Compression robustness")
    axs[0].tick_params(axis="x", labelsize=9.5, rotation=45)
    label_bars(axs[0], b, "{:.2f}", 9.5, 0.01)
    # (b) heat map: rows = models, cols = clean / compression / blur (kappa)
    rows = FR_MODELS + ["Ours (milder: 1000 kbps, s=1.5)", "Ours (harsher: 500 kbps, s=3.0)"]
    M = []
    for m in FR_MODELS:
        M.append([fr_kappa(FR[m][0]), fr_kappa(FR[m][1]), fr_kappa(FR[m][4])])
    M.append([kappa(0.995, P_MAJ), kappa(0.979, P_MAJ), kappa(0.821, P_MAJ)])
    M.append([kappa(0.995, P_MAJ), kappa(0.945, P_MAJ), kappa(0.799, P_MAJ)])
    M = np.array(M)
    im = axs[1].imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_xticklabels(["Clean", "Compression", "Blur"], fontsize=9.5)
    axs[1].set_yticks(range(len(rows)))
    axs[1].set_yticklabels(rows, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(3):
            axs[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9.5)
    axs[1].set_title(r"$\kappa$ heat map across models")
    cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_panels(fig, [[axs[0]], [axs[1], cb.ax]], outdir, ["cmp3a_compression", "cmp3b_heatmap"])


def fig_cmp4(outdir):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) params vs AUC
    offs = {"Xception": (10, -14), "EfficientNet-B3": (-40, 12), "CNN+BiLSTM": (8, -16), "ViT-Base/16": (-70, 8), "Hybrid CNN-Transf.": (-60, 10)}
    for n, p, a in zip(HF_MODELS, HF_PARAMS, HF_AUC):
        axs[0].scatter(p, a, s=70, color=LIT_COLOR)
        axs[0].annotate(n, (p, a), textcoords="offset points", xytext=offs[n], fontsize=9.5)
    axs[0].scatter(20.8, 0.999675, s=110, color=OURS_COLOR, marker="*")
    axs[0].annotate("CallGuard-X (ours)", (20.8, 0.999675), textcoords="offset points", xytext=(22, 12), fontsize=10, color=OURS_COLOR)
    axs[0].set_xlabel("Parameters (millions)")
    axs[0].set_ylabel("In-domain AUC")
    axs[0].set_title("Model size versus in-domain AUC")
    axs[0].set_ylim(0.68, 1.02)
    axs[0].grid(alpha=0.3)
    # (b) in-domain vs FaceShifter
    names = HF_MODELS + ["Xception\n(arXiv 2605.29092)"]
    ind = HF_AUC + [XCEPTION_LCF[0]]
    fs = HF_FS + [XCEPTION_LCF[1]]
    x = np.arange(len(names))
    w = 0.38
    axs[1].bar(x - w / 2, ind, w, label="In-domain (C23)", color=LIT_COLOR)
    b = axs[1].bar(x + w / 2, fs, w, label="Unseen FaceShifter", color="#c0504d")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([n.replace(" ", "\n", 1) if len(n) > 12 and "\n" not in n else n for n in names], fontsize=8.5)
    axs[1].set_ylim(0.5, 1.05)
    axs[1].set_ylabel("AUC")
    axs[1].set_title("Cross-manipulation generalisation (literature)")
    axs[1].legend(fontsize=9.5, loc="upper right")
    label_bars(axs[1], b, "{:.2f}", 9.5, 0.005)
    fig.tight_layout()
    save_panels(fig, [[axs[0]], [axs[1]]], outdir, ["cmp4a_size_auc", "cmp4b_shift_pairs"])


def fig_cmp5(outdir, drydir):
    pre = [1.0, 0.9774655520915985, 0.943026340007782, 0.5762307696044445, 0.9999998867511749]
    fin = [0.17434740107726937, 0.26685606521459704, 0.18374977841864962, 0.2320816667543113, 0.5468761067721062]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(5)
    w = 0.38
    axs[0].bar(x - w / 2, pre, w, label="Off-the-shelf checkpoint", color=GREY)
    axs[0].bar(x + w / 2, fin, w, label="Fine-tuned (ours)", color=OURS_COLOR)
    axs[0].axhline(0.5, color="black", ls=":", lw=1)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"Video {i + 1}" for i in x])
    axs[0].set_ylabel("Predicted fake probability")
    axs[0].set_title("Five real videos, baseline condition")
    axs[0].set_ylim(0, 1.28)
    axs[0].legend(fontsize=9.5, loc="upper center", ncol=2)
    # (b) AUC loss under distribution shift
    lit = [("Xception\n(HF)", 0.9944 - 0.7015), ("EfficientNet-B3\n(HF)", 0.9976 - 0.7343), ("CNN+BiLSTM\n(HF)", 0.9805 - 0.7591),
           ("ViT-Base\n(HF)", 0.7066 - 0.6880), ("Hybrid\n(HF)", 0.9458 - 0.6273), ("Xception\n(2605.29092)", 0.9805 - 0.6248)]
    ours = [("Ours\n500 kbps", BASE_AUC - 0.97425), ("Ours\n240p", BASE_AUC - 0.920725), ("Ours\nblur s=1.5", BASE_AUC - 0.901025),
            ("Ours\nblur s=3.0", BASE_AUC - 0.591725)]
    lab = [n for n, _ in lit] + [n for n, _ in ours]
    val = [v for _, v in lit] + [v for _, v in ours]
    col = ["#c0504d"] * len(lit) + [OURS_COLOR] * len(ours)
    b = axs[1].bar(lab, val, color=col)
    axs[1].set_ylabel("AUC lost relative to in-domain / clean")
    axs[1].set_title("AUC loss: unseen manipulation vs. degradation")
    axs[1].tick_params(axis="x", labelsize=9.5, rotation=60)
    label_bars(axs[1], b, "{:.2f}", 9.5, 0.005)
    fig.tight_layout()
    save_panels(fig, [[axs[0]], [axs[1]]], outdir, ["pretrained_vs_finetuned", "cmp5b_auc_loss"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="./figs")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_system_model(args.output_dir)
    fig_methodology(args.output_dir)
    fig_training(args.output_dir)
    fig_accuracy_kappa(args.output_dir)
    fig_heatmap(args.output_dir)
    fig_cmp1(args.output_dir)
    fig_cmp2(args.output_dir)
    fig_cmp3(args.output_dir)
    fig_cmp4(args.output_dir)
    fig_cmp5(args.output_dir, None)


if __name__ == "__main__":
    main()
