#!/usr/bin/env python3
"""
batch_infer.py

Runs the pretrained Xception FF++ detector (from HongguLiu/Deepfake-Detection)
across every video listed in degradation_manifest.csv (produced by degrade.py),
WITHOUT writing annotated output videos and WITHOUT reloading the model per
video — both of which make detect_from_video.py impractical for a 9,000-file
batch.

Must be placed inside the cloned HongguLiu/Deepfake-Detection repo folder
(same directory as detect_from_video.py, network/, dataset/) since it imports
directly from those modules.

For each video, samples up to --max_frames evenly-spaced frames, runs face
detection + the Xception classifier on each, and records the average fake
probability and the fraction of sampled frames classified as fake.

Usage:
    python batch_infer.py \
        --degradation_manifest "C:\\path\\to\\degraded\\degradation_manifest.csv" \
        --model_path ".\\pretrained_model\\ffpp_c23.pth" \
        --output_csv "C:\\path\\to\\inference_results.csv" \
        --max_frames 20 \
        --cuda
"""

import argparse
import csv
import os
import sys

import cv2
import dlib
import torch
from tqdm import tqdm

# Reuse the actual functions from detect_from_video.py rather than
# re-implementing face cropping / preprocessing / prediction logic.
from detect_from_video import get_boundingbox, predict_with_model
from network.models import model_selection


def run_inference_on_video(video_path, model, face_detector, cuda, max_frames):
    """Returns (avg_fake_prob, frac_frames_fake, n_frames_used) or None if
    no face could be detected in any sampled frame."""
    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        return None

    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, num_frames // max_frames) if num_frames > 0 else 1

    fake_probs = []
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
                cropped_face = image[y:y + size, x:x + size]

                if cropped_face.size > 0:
                    with torch.no_grad():
                        _, output = predict_with_model(cropped_face, model, cuda=cuda)
                    # output is post-softmax [1, 2]; index 1 = fake class
                    fake_prob = float(output.detach().cpu().numpy()[0][1])
                    fake_probs.append(fake_prob)

            if len(fake_probs) >= max_frames:
                break

        frame_idx += 1

    reader.release()

    if not fake_probs:
        return None

    avg_fake_prob = sum(fake_probs) / len(fake_probs)
    frac_frames_fake = sum(1 for p in fake_probs if p > 0.5) / len(fake_probs)
    return avg_fake_prob, frac_frames_fake, len(fake_probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degradation_manifest", required=True,
                         help="degradation_manifest.csv from degrade.py")
    parser.add_argument("--model_path", required=True,
                         help="Path to ffpp_c23.pth")
    parser.add_argument("--output_csv", required=True,
                         help="Where to write per-video inference results")
    parser.add_argument("--max_frames", type=int, default=20,
                         help="Max sampled frames per video (evenly spaced)")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.degradation_manifest):
        print(f"[ERROR] Manifest not found: {args.degradation_manifest}", file=sys.stderr)
        sys.exit(1)

    with open(args.degradation_manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"[INFO] Loaded {len(rows)} manifest rows")

    face_detector = dlib.get_frontal_face_detector()

    model = model_selection(modelname="xception", num_out_classes=2, dropout=0.5)
    state = torch.load(args.model_path, map_location="cuda" if args.cuda else "cpu")
    model.load_state_dict(state)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if args.cuda:
        model = model.cuda()
    model.eval()
    print("[INFO] Model loaded once, reused across all videos")

    results = []
    n_skipped_missing = 0
    n_skipped_status = 0
    n_no_face = 0

    for row in tqdm(rows, desc="Running inference"):
        video_path = row.get("output_path")
        status = row.get("status", "")

        if status not in ("ok", "skipped_exists"):
            n_skipped_status += 1
            continue
        if not video_path or not os.path.exists(video_path):
            n_skipped_missing += 1
            continue

        try:
            result = run_inference_on_video(
                video_path, model, face_detector, args.cuda, args.max_frames
            )
        except Exception as e:
            print(f"[ERROR] Inference failed on {video_path}: {e}", file=sys.stderr)
            result = None

        if result is None:
            n_no_face += 1
            avg_fake_prob, frac_frames_fake, n_frames_used = "", "", 0
        else:
            avg_fake_prob, frac_frames_fake, n_frames_used = result

        results.append({
            **row,
            "avg_fake_prob": avg_fake_prob,
            "frac_frames_fake": frac_frames_fake,
            "n_frames_used": n_frames_used,
        })

    if not results:
        print("[ERROR] No results produced — check manifest paths and status values.",
              file=sys.stderr)
        sys.exit(1)

    fieldnames = list(results[0].keys())
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[DONE] Wrote {len(results)} rows to {args.output_csv}")
    print(f"       Skipped (bad status): {n_skipped_status}")
    print(f"       Skipped (file missing): {n_skipped_missing}")
    print(f"       No face detected in any sampled frame: {n_no_face}")


if __name__ == "__main__":
    main()
