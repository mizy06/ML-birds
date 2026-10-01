"""Interaction plot for the 2x2 (resolution x GT bbox) ablation, plus bn-train."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(r"D:\A\ML\scnet_results\runs")
OUT = Path(r"D:\A\ML\scnet_results\ablation_2x2.png")

CELLS = {
    (224, False): "r50_full",
    (320, False): "r50_full320",
    (224, True): "r50_box224",
    (320, True): "r50_box",
}


def best(run):
    rows = list(csv.DictReader((RUNS / run / "history.csv").open(encoding="utf-8")))
    acc = [float(r["val_acc"]) for r in rows]
    return max(acc), len(rows)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

sizes = [224, 320]
for bbox, color, marker in ((False, "#4C78A8", "o"), (True, "#E45756", "s")):
    ys = [best(CELLS[(s, bbox)])[0] for s in sizes]
    ax1.plot(sizes, ys, color=color, marker=marker, ms=9, lw=2.2,
             label=f"bbox={'yes' if bbox else 'no'}")
    for s, y in zip(sizes, ys):
        ax1.annotate(f"{y:.4f}", (s, y), textcoords="offset points",
                     xytext=(0, 10 if bbox else -16), ha="center", fontsize=9, color=color)

base = best("r50_full")[0]
ax1.axhline(base, color="#999", ls=":", lw=1.2)
ax1.annotate("baseline r50_full", (224, base), textcoords="offset points",
             xytext=(4, -14), fontsize=8, color="#777")

ax1.set_xticks(sizes)
ax1.set_xlabel("input resolution")
ax1.set_ylabel("best val acc")
ax1.set_title("resnet50 · 2x2 ablation: resolution × GT bbox")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9, loc="lower left")
ax1.set_ylim(0.71, 0.83)

names = ["r50_full\n224 no-bbox", "r50_full320\n320 no-bbox",
         "r50_box224\n224 bbox", "r50_box\n320 bbox", "r50_full_bntrain\n224 no-bbox +bn"]
vals = [best("r50_full")[0], best("r50_full320")[0],
        best("r50_box224")[0], best("r50_box")[0], best("r50_full_bntrain")[0]]
colors = ["#4C78A8", "#4C78A8", "#E45756", "#E45756", "#54A24B"]
bars = ax2.bar(range(len(vals)), vals, color=colors, width=0.62)
for i, (b, v) in enumerate(zip(bars, vals)):
    delta = v - base
    ax2.annotate(f"{v:.4f}\n{delta:+.4f}" if i else f"{v:.4f}\n(baseline)",
                 (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=8.5)
ax2.axhline(base, color="#999", ls=":", lw=1.2)
ax2.set_xticks(range(len(vals)))
ax2.set_xticklabels(names, fontsize=8.5)
ax2.set_ylabel("best val acc")
ax2.set_title("all runs vs baseline")
ax2.grid(alpha=0.3, axis="y")
ax2.set_ylim(0.71, 0.84)

fig.suptitle("CUB-200-2011 · SCNet Hygon DCU · custom split (train 8487 / val 2122)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
