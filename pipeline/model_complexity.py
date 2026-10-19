#!/usr/bin/env python3
r"""
model_complexity.py

Measures what the paper reports for CallGuard-X:
  * number of parameters of the Xception classifier (paper: about 20.8 million)
  * forward-pass latency for one 299x299 face crop (not reported in the paper; fill in if you use it)

Run from the Deepfake-Detection folder (it imports network.models):
    python model_complexity.py --cuda
"""
import argparse
import statistics
import time

import torch
from network.models import model_selection

ap = argparse.ArgumentParser()
ap.add_argument("--cuda", action="store_true")
ap.add_argument("--runs", type=int, default=100)
a = ap.parse_args()

model = model_selection(modelname="xception", num_out_classes=2, dropout=0.5).eval()
n = sum(p.numel() for p in model.parameters())
print(f"parameters: {n:,}  ({n/1e6:.2f} M)")

dev = "cuda" if a.cuda and torch.cuda.is_available() else "cpu"
model = model.to(dev)
x = torch.randn(1, 3, 299, 299, device=dev)
with torch.no_grad():
    for _ in range(10):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(a.runs):
        t0 = time.perf_counter()
        model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
print(f"device: {dev}; forward pass, batch 1: mean {statistics.mean(ts):.1f} ms, std {statistics.pstdev(ts):.1f} ms over {a.runs} runs")
