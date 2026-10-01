"""Plot validation curves for the four SCNet DCU runs."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\A\ML\scnet_results\runs")
OUT = Path(r"D:\A\ML\scnet_results\val_curves.png")

RUNS = [
    ("r18_full", "resnet18 @224", "#4C78A8", "-"),
    ("r18_box", "resnet18 @224 +bbox", "#9ECAE9", "--"),
    ("r50_full", "resnet50 @224", "#E45756", "-"),
    ("r50_box", "resnet50 @320 +bbox", "#F2A29A", "--"),
]

fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))

for run, label, color, ls in RUNS:
    rows = list(csv.DictReader((ROOT / run / "history.csv").open(encoding="utf-8")))
    ep = [int(r["epoch"]) for r in rows]
    acc = [float(r["val_acc"]) for r in rows]
    loss = [float(r["val_loss"]) for r in rows]
    best_i = max(range(len(acc)), key=lambda i: acc[i])
    ax_acc.plot(ep, acc, color=color, ls=ls, lw=2, label=f"{label}  (best {acc[best_i]:.4f} @ep{ep[best_i]})")
    ax_acc.plot(ep[best_i], acc[best_i], "o", color=color, ms=7, mec="white", mew=1.2, zorder=5)
    ax_loss.plot(ep, loss, color=color, ls=ls, lw=2, label=label)

ax_acc.set_title("Validation accuracy (marker = selected checkpoint)")
ax_acc.set_xlabel("epoch")
ax_acc.set_ylabel("val acc")
ax_acc.grid(alpha=0.3)
ax_acc.legend(fontsize=8, loc="lower right")

ax_loss.set_title("Validation loss")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("val loss")
ax_loss.grid(alpha=0.3)
ax_loss.legend(fontsize=8, loc="upper left")

fig.suptitle("CUB-200-2011 · SCNet Zhengzhou · Hygon DCU (BW) · custom split, train 8487 / val 2122", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("wrote", OUT)

for run, label, _, _ in RUNS:
    rows = list(csv.DictReader((ROOT / run / "history.csv").open(encoding="utf-8")))
    acc = [float(r["val_acc"]) for r in rows]
    best_i = max(range(len(acc)), key=lambda i: acc[i])
    print(f"{run:10s} epochs={len(rows):2d}  best_epoch={best_i+1:2d}  best_val_acc={acc[best_i]:.5f}  final={acc[-1]:.5f}")
