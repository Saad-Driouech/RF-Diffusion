"""Diagnostic: print statistics for the GNSS condition vector.

Run from the repo root:
    python check_cond_stats.py

The condition has 7 components:
  [0] x         (position, meters — could be large if ECEF)
  [1] y
  [2] z
  [3] sin(az)   (always in [-1, 1])
  [4] cos(az)
  [5] sin(el)
  [6] cos(el)

If x/y/z have std >> 1 while sin/cos have std ≈ 0.5–0.7, the model
receives very unbalanced inputs and needs per-feature normalisation.
"""

import torch
import numpy as np
from tfdiff.dataset import GNSSDataset

N_SAMPLES = 500
NAMES = ["x", "y", "z", "sin(az)", "cos(az)", "sin(el)", "cos(el)"]

print("Loading dataset …")
ds = GNSSDataset([], mode="train")
n = min(N_SAMPLES, len(ds))

conds = []
for i in range(n):
    _, cond = ds[i]
    conds.append(cond)

conds = torch.stack(conds).numpy()  # [N, 7]

print(f"\nCondition vector statistics over {n} samples\n")
print(f"{'Feature':<12} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
print("-" * 64)
for i, name in enumerate(NAMES):
    col = conds[:, i]
    print(f"{name:<12} {col.min():>12.4f} {col.max():>12.4f} {col.mean():>12.4f} {col.std():>12.4f}")

print()
print("-- Interpretation --")
print("sin/cos features should have |min|, |max| <= 1.0 and std ≈ 0.5–0.7.")
print("If x/y/z std >> 1, add per-feature normalisation in dataset.py Collator.")

# Check whether normalisation is needed
xyz_std = conds[:, :3].std(axis=0)
if xyz_std.max() > 10:
    print(f"\n[WARNING] x/y/z std = {xyz_std} — normalisation recommended.")
    print("Suggested fix in dataset.py Collator (task_id==4 block):")
    print()
    print("  xyz_mean = cond[:, :3].mean(dim=0)")
    print("  xyz_std  = cond[:, :3].std(dim=0).clamp(min=1e-6)")
    print("  cond[:, :3] = (cond[:, :3] - xyz_mean) / xyz_std")
else:
    print("\n[OK] x/y/z std looks reasonable — no normalisation needed.")
