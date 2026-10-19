#!/usr/bin/env python3
r"""
heldout_analysis.py

Adds three analyses that tighten the claims of the CallGuard-X paper.

 1. HELD-OUT metrics. 850 of the 1,000 benchmark clips were used (in clean form) to fine-tune
    the detector. The 150 clips in data_list/my_val.txt were never used for training. This
    script recovers those clips from the frame names in my_val.txt and recomputes AUC, EER,
    accuracy and chance-corrected accuracy per condition on them only.
 2. PER-METHOD metrics. AUC per manipulation method (Deepfakes, Face2Face, FaceSwap,
    NeuralTextures) against the real clips, per condition, on all 1,000 clips.
 3. THRESHOLD diagnostics. TPR (fake recall), TNR (real recall) and balanced accuracy at
    tau = 0.5 per condition. These show whether strong blur makes the detector label
    (almost) everything as fake.

Usage (run from deepfake_research; paths are examples):
    python heldout_analysis.py ^
        --results_csv .\inference_results_combined.csv ^
        --val_list    C:\Users\ghoda\Downloads\Deepfake-Detection\data_list\my_val.txt ^
        --output_dir  .\analysis_heldout

Requires: pandas, numpy, scikit-learn.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[k] + fnr[k]) / 2)


def key_from_frame_path(p):
    """frame files are named <method>_<videoname>_<NNN>.jpg (see extract_frames.py)"""
    name = os.path.basename(p.replace("\\", "/"))
    stem = os.path.splitext(name)[0]
    m = re.match(r"^([A-Za-z0-9]+?)_(.+)_(\d{3})$", stem)
    if not m:
        return None
    return m.group(1), m.group(2)


def metrics(g, tau=0.5):
    y = g["y"].values
    s = g["avg_fake_prob"].values
    out = {"n": len(g), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    if len(np.unique(y)) < 2:
        return out
    pred = (s >= tau).astype(int)
    acc = float((pred == y).mean())
    pmaj = max(y.mean(), 1 - y.mean())
    tpr = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    tnr = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    out.update({"auc": float(roc_auc_score(y, s)), "eer": eer(y, s), "acc": acc,
                "majority_rate": float(pmaj), "kappa_maj": float((acc - pmaj) / (1 - pmaj)),
                "tpr_fake_recall": tpr, "tnr_real_recall": tnr, "balanced_acc": (tpr + tnr) / 2})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--output_dir", default="./analysis_heldout")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    df = pd.read_csv(a.results_csv)
    df = df[df["avg_fake_prob"].notna() & (df["avg_fake_prob"].astype(str) != "")].copy()
    df["avg_fake_prob"] = df["avg_fake_prob"].astype(float)
    df["y"] = (df["label"] == "fake").astype(int)
    df["video"] = df["source_path"].apply(lambda p: os.path.splitext(os.path.basename(p.replace("\\", "/")))[0])
    df["key"] = list(zip(df["method"], df["video"]))
    df["cond"] = df["condition_type"] + "_" + df["condition_value"].astype(str)

    held = set()
    with open(a.val_list) as f:
        for line in f:
            if line.strip():
                k = key_from_frame_path(line.split()[0])
                if k:
                    held.add(k)
    print(f"held-out clips recovered from {os.path.basename(a.val_list)}: {len(held)}")
    present = {k for k in df["key"].unique()}
    print(f"of which present in the results file: {len(held & present)}")

    # 1. held-out metrics
    rows = []
    for cond, g in df[df["key"].isin(held)].groupby("cond"):
        rows.append({"condition": cond, **metrics(g)})
    ho = pd.DataFrame(rows)
    ho.to_csv(os.path.join(a.output_dir, "heldout_metrics.csv"), index=False)
    print("\nHELD-OUT clips only (never used for fine-tuning)")
    print(ho.round(4).to_string(index=False))

    # 2. per-method AUC (fake method vs the real clips)
    rows = []
    real = df[df["method"] == "original"]
    for cond in sorted(df["cond"].unique()):
        for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            g = pd.concat([real[real["cond"] == cond], df[(df["method"] == m) & (df["cond"] == cond)]])
            r = metrics(g)
            rows.append({"condition": cond, "method": m, "auc": r.get("auc"), "eer": r.get("eer"), "tpr_fake_recall": r.get("tpr_fake_recall")})
    pm = pd.DataFrame(rows)
    pm.to_csv(os.path.join(a.output_dir, "per_method_metrics.csv"), index=False)
    print("\nPER-METHOD AUC (all 1,000 clips)")
    print(pm.pivot(index="condition", columns="method", values="auc").round(4).to_string())

    # 3. threshold diagnostics on all clips
    rows = []
    for cond, g in df.groupby("cond"):
        rows.append({"condition": cond, **metrics(g)})
    td = pd.DataFrame(rows)
    td.to_csv(os.path.join(a.output_dir, "threshold_diagnostics_all.csv"), index=False)
    print("\nTHRESHOLD DIAGNOSTICS (all 1,000 clips)")
    print(td[["condition", "acc", "majority_rate", "tpr_fake_recall", "tnr_real_recall", "balanced_acc"]].round(4).to_string(index=False))
    print(f"\nCSV files written to {a.output_dir}")


if __name__ == "__main__":
    main()
