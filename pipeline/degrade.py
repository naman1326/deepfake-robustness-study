#!/usr/bin/env python3
"""
degrade.py

Applies a matrix of realistic video-call degradations to every video listed
in subset_manifest.csv (produced by select_subset.py) using ffmpeg, and
writes a new manifest that maps every degraded output file back to its
source video, class, method, and the exact degradation condition applied.

Degradation dimensions (edit the CONFIG section below to adjust):
    1. Bitrate sweep      (H.264, fixed resolution)   -- primary
    2. Resolution sweep   (fixed bitrate)              -- primary
    3. Blur               (Gaussian blur)               -- secondary
    4. Packet loss (sim.) (frame corruption/drop)        -- secondary
    A "none" / original condition is always included as the baseline.

Usage:
    python degrade.py \
        --manifest   /path/to/subset/subset_manifest.csv \
        --output_dir /path/to/degraded \
        --workers    4

Output:
    <output_dir>/<condition_name>/<original_filename>
    <output_dir>/degradation_manifest.csv
        columns: source_path, label, method, condition_type,
                 condition_value, output_path, status
"""

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------- CONFIG ------------------------------------

BITRATES_KBPS = [500, 1000, 2000, 5000]          # primary sweep
RESOLUTIONS = [(1280, 720), (854, 480), (640, 360), (426, 240)]  # primary sweep
BLUR_SIGMAS = [1.5, 3.0]                          # secondary
PACKET_LOSS_PCTS = [1]                     # secondary, simulated via noise

RUN_SECONDARY = False  # flip to True once the primary sweep results are in

# ---------------------------------------------------------------------------


def build_conditions():
    """Return a list of (condition_type, condition_value, ffmpeg_args) tuples."""
    conditions = [("baseline", "original", [])]

    for kbps in BITRATES_KBPS:
        conditions.append((
            "bitrate", f"{kbps}kbps",
            ["-c:v", "libx264", "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k",
             "-bufsize", f"{kbps*2}k", "-c:a", "aac", "-b:a", "64k"]
        ))

    for (w, h) in RESOLUTIONS:
        conditions.append((
            "resolution", f"{h}p",
            ["-vf", f"scale={w}:{h}", "-c:v", "libx264", "-crf", "23",
             "-c:a", "aac", "-b:a", "64k"]
        ))

    if RUN_SECONDARY:
        for sigma in BLUR_SIGMAS:
            conditions.append((
                "blur", f"sigma{sigma}",
                ["-vf", f"gblur=sigma={sigma}", "-c:v", "libx264", "-crf", "23",
                 "-c:a", "aac", "-b:a", "64k"]
            ))
        for pct in PACKET_LOSS_PCTS:
            # Simulated via a noise filter as a proxy for corrupted macroblocks;
            # true packet-loss simulation typically needs a network emulator
            # (e.g. tc/netem) in front of a live stream, which is out of scope
            # for offline file-based degradation.
            conditions.append((
                "packet_loss_sim", f"{pct}pct",
                ["-vf", f"noise=alls={pct*5}:allf=t", "-c:v", "libx264",
                 "-crf", "23", "-c:a", "aac", "-b:a", "64k"]
            ))

    return conditions


def run_ffmpeg(src_path, dst_path, ffmpeg_args):
    if os.path.exists(dst_path):
        return "skipped_exists"
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path] + ffmpeg_args + [dst_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return "ok"
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed on {src_path} ({dst_path}): "
              f"{e.stderr.decode(errors='ignore')[:300]}", file=sys.stderr)
        return "failed"
    except subprocess.TimeoutExpired:
        print(f"[ERROR] ffmpeg timed out on {src_path}", file=sys.stderr)
        return "timeout"


def process_one(row, condition, output_dir):
    condition_type, condition_value, ffmpeg_args = condition
    src_path = row["source_path"]
    filename = os.path.basename(src_path)
    condition_name = f"{condition_type}_{condition_value}"
    dst_path = os.path.join(output_dir, condition_name, filename)

    if condition_type == "baseline":
        # baseline = just re-encode losslessly / or copy, so it goes through
        # the same pipeline as everything else for a fair comparison
        status = run_ffmpeg(src_path, dst_path, ["-c", "copy"]) if not os.path.exists(dst_path) else "skipped_exists"
    else:
        status = run_ffmpeg(src_path, dst_path, ffmpeg_args)

    return {
        "source_path": src_path,
        "label": row["label"],
        "method": row["method"],
        "condition_type": condition_type,
        "condition_value": condition_value,
        "output_path": dst_path,
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                         help="subset_manifest.csv from select_subset.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=4,
                         help="Parallel ffmpeg processes (keep modest on shared/laptop CPUs)")
    parser.add_argument("--secondary", action="store_true",
                         help="Also run blur + packet-loss-sim conditions")
    args = parser.parse_args()

    global RUN_SECONDARY
    RUN_SECONDARY = args.secondary

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    conditions = build_conditions()
    print(f"[INFO] {len(rows)} source videos x {len(conditions)} conditions "
          f"= {len(rows) * len(conditions)} output files")

    tasks = [(row, cond) for row in rows for cond in conditions]
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_one, row, cond, args.output_dir)
            for row, cond in tasks
        ]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 25 == 0 or i == len(futures):
                print(f"[PROGRESS] {i}/{len(futures)}")

    out_manifest = os.path.join(args.output_dir, "degradation_manifest.csv")
    with open(out_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_path", "label", "method", "condition_type",
            "condition_value", "output_path", "status"
        ])
        writer.writeheader()
        writer.writerows(results)

    n_ok = sum(1 for r in results if r["status"] in ("ok", "skipped_exists"))
    n_fail = sum(1 for r in results if r["status"] in ("failed", "timeout"))
    print(f"\n[DONE] {n_ok} succeeded, {n_fail} failed. Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
