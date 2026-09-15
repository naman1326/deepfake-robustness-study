#!/usr/bin/env python3
"""
select_subset.py

Samples a balanced subset of videos from a FaceForensics++ (C23) download
and writes a manifest CSV listing each selected video with its class/method.

Expected input folder structure (standard FF++ layout):

    <dataset_root>/
        original/c23/videos/*.mp4
        Deepfakes/c23/videos/*.mp4
        Face2Face/c23/videos/*.mp4
        FaceSwap/c23/videos/*.mp4
        NeuralTextures/c23/videos/*.mp4

If your download doesn't nest videos under c23/videos (some mirrors flatten
this), pass --flat and the script will look directly inside each method
folder instead.

Usage:
    python select_subset.py \
        --dataset_root /path/to/FaceForensics++_C23 \
        --output_dir   /path/to/subset \
        --per_class    200 \
        --seed         42

Output:
    <output_dir>/subset_manifest.csv   (source_path, label, method)
    <output_dir>/<method>/<filename>.mp4   (copied/symlinked video files)
"""

import argparse
import csv
import os
import random
import shutil
import sys

REAL_METHOD = "original"
FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def find_videos(dataset_root: str, method: str, flat: bool) -> list[str]:
    if flat:
        method_dir = os.path.join(dataset_root, method)
    else:
        method_dir = os.path.join(dataset_root, method, "c23", "videos")

    if not os.path.isdir(method_dir):
        # try the flat layout as a fallback so the script is forgiving
        alt_dir = os.path.join(dataset_root, method)
        if os.path.isdir(alt_dir):
            method_dir = alt_dir
        else:
            print(f"[WARN] Could not find video folder for '{method}' "
                  f"(looked in: {method_dir})", file=sys.stderr)
            return []

    videos = [
        os.path.join(method_dir, f)
        for f in sorted(os.listdir(method_dir))
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]
    return videos


def sample_and_copy(videos: list[str], n: int, out_dir: str, seed: int) -> list[str]:
    random.seed(seed)
    if len(videos) < n:
        print(f"[WARN] Requested {n} but only found {len(videos)} videos "
              f"in {out_dir}; using all available.", file=sys.stderr)
        chosen = videos
    else:
        chosen = random.sample(videos, n)

    os.makedirs(out_dir, exist_ok=True)
    copied_paths = []
    for src in chosen:
        dst = os.path.join(out_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied_paths.append(dst)
    return copied_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True,
                         help="Path to the FaceForensics++_C23 folder")
    parser.add_argument("--output_dir", required=True,
                         help="Where to write the sampled subset + manifest")
    parser.add_argument("--per_class", type=int, default=200,
                         help="Videos to sample PER METHOD (real gets the "
                              "same count as each individual fake method, "
                              "so total fakes = per_class * 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flat", action="store_true",
                         help="Use if videos sit directly inside each "
                              "method folder rather than under c23/videos")
    args = parser.parse_args()

    manifest_rows = []

    # Real videos
    real_videos = find_videos(args.dataset_root, REAL_METHOD, args.flat)
    real_out_dir = os.path.join(args.output_dir, REAL_METHOD)
    real_chosen = sample_and_copy(real_videos, args.per_class, real_out_dir, args.seed)
    for path in real_chosen:
        manifest_rows.append({"source_path": path, "label": "real", "method": "original"})
    print(f"[OK] {REAL_METHOD}: sampled {len(real_chosen)} videos")

    # Fake videos, one method at a time
    for method in FAKE_METHODS:
        videos = find_videos(args.dataset_root, method, args.flat)
        out_dir = os.path.join(args.output_dir, method)
        chosen = sample_and_copy(videos, args.per_class, out_dir, args.seed)
        for path in chosen:
            manifest_rows.append({"source_path": path, "label": "fake", "method": method})
        print(f"[OK] {method}: sampled {len(chosen)} videos")

    manifest_path = os.path.join(args.output_dir, "subset_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "label", "method"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[DONE] Wrote manifest with {len(manifest_rows)} videos to {manifest_path}")
    print(f"       Real: {len(real_chosen)} | Fake: {len(manifest_rows) - len(real_chosen)}")


if __name__ == "__main__":
    main()
