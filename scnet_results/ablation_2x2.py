"""2x2 ablation over (resolution, GT bbox) for resnet50, plus the --bn-train variant.

All runs share: custom split, batch 16, lr_head 1e-4, lr_backbone 1e-5,
wd 1e-4, seed 42, 25 epochs, patience 6.
"""
import csv
import json
from pathlib import Path

RUNS = Path(r"D:\A\ML\scnet_results\runs")


def load(run):
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((RUNS / run / "history.csv").open(encoding="utf-8")))
    acc = [float(r["val_acc"]) for r in rows]
    loss = [float(r["val_loss"]) for r in rows]
    bi = max(range(len(acc)), key=lambda i: acc[i])
    return cfg, rows, acc, loss, bi


def best(run):
    return load(run)[2][load(run)[4]]


CELLS = {
    (224, False): "r50_full",
    (320, False): "r50_full320",
    (224, True): "r50_box224",
    (320, True): "r50_box",
}

print("=" * 74)
print("2x2 ablation - resnet50, bn_train=False (val acc, best epoch)")
print("=" * 74)
print(f"{'':12s}{'224':>18s}{'320':>18s}")
for bbox in (False, True):
    label = "bbox=yes" if bbox else "bbox=no "
    cells = []
    for size in (224, 320):
        run = CELLS[(size, bbox)]
        _, rows, acc, _, bi = load(run)
        cells.append(f"{acc[bi]:.5f}@ep{bi+1}(n={len(rows)})")
    print(f"{label:12s}{cells[0]:>18s}{cells[1]:>18s}")

base = best("r50_full")
b320 = best("r50_full320")
bb224 = best("r50_box224")
bb320 = best("r50_box")
bn = best("r50_full_bntrain")

print()
print("=" * 74)
print("Effect decomposition (vs r50_full baseline)")
print("=" * 74)
print(f"baseline  r50_full  (224, no bbox)        : {base:.5f}")
print(f"A) +resolution only  (320, no bbox)       : {b320:.5f}   {b320-base:+.5f}")
print(f"B) +bbox only        (224, bbox)          : {bb224:.5f}   {bb224-base:+.5f}")
print(f"A+B observed         (320, bbox)          : {bb320:.5f}   {bb320-base:+.5f}")
additive = (b320 - base) + (bb224 - base)
print(f"A+B if additive                           : {base+additive:.5f}   {additive:+.5f}")
print(f"   interaction (observed - additive)      : {bb320-base-additive:+.5f}")
print()
print(f"bbox effect @224                          : {bb224-base:+.5f}")
print(f"bbox effect @320                          : {bb320-b320:+.5f}")
print(f"resolution effect, no bbox                : {b320-base:+.5f}")
print(f"resolution effect, with bbox              : {bb320-bb224:+.5f}")
print()
print(f"--bn-train on r50_full (224, no bbox)     : {bn:.5f}   {bn-base:+.5f}")
print()
print("=" * 74)
print("Overfitting signature: epochs until early stop")
print("=" * 74)
for key in [(224, False), (320, False), (224, True), (320, True)]:
    run = CELLS[key]
    _, rows, acc, loss, bi = load(run)
    print(f"{run:18s} size={key[0]:3d} bbox={str(key[1]):5s} "
          f"best_ep={bi+1:2d} ran={len(rows):2d} "
          f"val_loss@best={loss[bi]:.4f} val_loss@final={loss[-1]:.4f}")
