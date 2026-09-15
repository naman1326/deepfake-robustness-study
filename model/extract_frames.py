#!/usr/bin/env python3
"""
extract_frames.py

Extracts face-cropped frames from the CLEAN (undegraded) subset videos
(subset_manifest.csv, produced by select_subset.py) and writes:
    1. JPEG frame crops, organized as <output_dir>/<label>/<filename>.jpg
    2. train_list.txt and val_list.txt in the exact "<path> <label>" format
       MyDataset expects, with the video-level split done BEFORE frame
       extraction so no video's frames leak across train/val.

Must be run from inside the cloned HongguLiu/Deepfake-Detection repo folder
(imports get_boundingbox from detect_from_video.py).

Label convention (matches detect_from_video.py / model_selection):
    0 = real, 1 = fake

Usage:
    python extract_frames.py \
        --subset_manifest "C:\\path\\to\\subset\\subset_manifest.csv" \
        --output_dir "C:\\path\\to\\frames" \
        --frames_per_video 10 \
        --val_fraction 0.15
"""

import argparse
import csv
import os
import random
import sys

import cv2
import dlib
from tqdm import tqdm

from detect_from_video import get_boundingbox


def extract_frames_from_video(video_path, face_detector, n_frames):
    """Returns a list of cropped-face BGR images (numpy arrays)."""
    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        return []

    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, num_frames // n_frames) if num_frames > 0 else 1

    crops = []
    frame_idx = 0
    while True:
        ret, image = reader.read()
        if not ret or image is None:
            break

        if frame_idx % stride == 0:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 1)
            if len(faces):
                face = faces[0]
                x, y, size = get_boundingbox(face, width, height)
                cropped = image[y:y + size, x:x + size]
                if cropped.size > 0:
                    crops.append(cropped)
            if len(crops) >= n_frames:
                break

        frame_idx += 1

    reader.release()
    return crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset_manifest", required=True,
                         help="subset_manifest.csv from select_subset.py (the CLEAN subset)")
    parser.add_argument("--output_dir", required=True,
                         help="Where to save extracted face-crop JPEGs")
    parser.add_argument("--frames_per_video", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.subset_manifest):
        print(f"[ERROR] Manifest not found: {args.subset_manifest}", file=sys.stderr)
        sys.exit(1)

    with open(args.subset_manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[INFO] Loaded {len(rows)} source videos from subset manifest")

    # Video-level train/val split BEFORE any frame extraction, so frames
    # from the same source video never appear in both splits.
    random.seed(args.seed)
    random.shuffle(rows)
    n_val = int(len(rows) * args.val_fraction)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    print(f"[INFO] Split: {len(train_rows)} videos train, {len(val_rows)} videos val")

    face_detector = dlib.get_frontal_face_detector()

    real_dir = os.path.join(args.output_dir, "0")
    fake_dir = os.path.join(args.output_dir, "1")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    def process_split(split_rows, split_name):
        lines = []
        n_no_face = 0
        for row in tqdm(split_rows, desc=f"Extracting ({split_name})"):
            video_path = row["source_path"]
            label = 0 if row["label"] == "real" else 1
            class_dir = real_dir if label == 0 else fake_dir

            if not os.path.exists(video_path):
                continue

            crops = extract_frames_from_video(video_path, face_detector, args.frames_per_video)
            if not crops:
                n_no_face += 1
                continue

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            method = row["method"]
            for i, crop in enumerate(crops):
                out_name = f"{method}_{base_name}_{i:03d}.jpg"
                out_path = os.path.join(class_dir, out_name)
                cv2.imwrite(out_path, crop)
                lines.append(f"{os.path.abspath(out_path)} {label}")

        print(f"[INFO] {split_name}: {len(lines)} frames extracted, "
              f"{n_no_face} videos had no detectable face")
        return lines

    train_lines = process_split(train_rows, "train")
    val_lines = process_split(val_rows, "val")

    os.makedirs("data_list", exist_ok=True)
    train_list_path = os.path.join("data_list", "my_train.txt")
    val_list_path = os.path.join("data_list", "my_val.txt")

    with open(train_list_path, "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(val_list_path, "w") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"\n[DONE] Wrote {len(train_lines)} lines to {train_list_path}")
    print(f"       Wrote {len(val_lines)} lines to {val_list_path}")


if __name__ == "__main__":
    main()
