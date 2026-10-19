#!/usr/bin/env python3
r"""
verify_checkpoint.py

train_CNN.py keeps its "best" weights with `best_model_wts = model.state_dict()`. In PyTorch
state_dict() returns references to the live parameters, not a copy, so after training the
exported best.pkl is expected to equal the LAST epoch's weights. This script checks that by
hashing the files.

Usage (from the Deepfake-Detection folder):
    python verify_checkpoint.py .\output\my_finetune
"""
import hashlib
import os
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "./output/my_finetune"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = sorted(f for f in os.listdir(folder) if f.endswith((".pkl", ".pth")))
hashes = {f: sha(os.path.join(folder, f)) for f in files}
for f, h in hashes.items():
    print(f"{f:32s} {h[:16]}...")
best = hashes.get("best.pkl")
same = [f for f, h in hashes.items() if h == best and f != "best.pkl"]
print()
if same:
    print("best.pkl is byte-identical to:", ", ".join(same))
    print("-> the evaluated weights are those of that epoch (the paper text assumes epoch 5, 4_xception_finetuned.pkl).")
else:
    print("best.pkl matches no per-epoch snapshot. Tell the authors of the paper to revise Section IX-B and Algorithm 2.")
