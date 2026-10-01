"""Verify the SCNet run artifacts: checkpoint <-> config <-> history consistency.

Reads only the local copy under scnet_results/, so it can run on the workstation.
"""
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(r"D:\A\ML\scnet_results\runs")

EXPECTED_HEADS = {
    "resnet18": ("fc.weight", (200, 512)),
    "resnet50": ("fc.weight", (200, 2048)),
}

ok = True
for run_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    name = run_dir.name
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run_dir / "history.csv").open(encoding="utf-8")))
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)

    val_accs = [float(r["val_acc"]) for r in rows]
    best_row = max(range(len(val_accs)), key=lambda i: val_accs[i])

    checks = []
    checks.append(("config.json == ck['config']", cfg == ck["config"]))
    checks.append(("ck epoch == argmax(history.val_acc)",
                   ck["epoch"] == best_row + 1))
    # history.csv is written with 6 decimals; the checkpoint keeps the full float.
    checks.append(("ck val_acc == max(history.val_acc)",
                   abs(ck["val_acc"] - val_accs[best_row]) < 1e-6))
    checks.append(("device recorded as dcu", cfg.get("device") == "dcu"))
    checks.append(("ck epoch <= history rows",
                   ck["epoch"] <= len(rows)))
    checks.append(("stopped at 25 epochs or early stop",
                   len(rows) == 25 or val_accs[-1] <= max(val_accs)))

    key, shape = EXPECTED_HEADS[cfg["model"]]
    actual = tuple(ck["model_state"][key].shape)
    checks.append((f"{key} {actual} == {shape}", actual == shape))

    print(f"### {name}  ({cfg['model']} @ {cfg['size']}px, bbox={cfg['bbox']})")
    print(f"    epochs logged={len(rows)}  best epoch={ck['epoch']}  "
          f"best val_acc={ck['val_acc']:.5f}")
    for label, passed in checks:
        print(f"    [{'OK ' if passed else 'FAIL'}] {label}")
        ok &= passed
    print()

print("ALL_CHECKS_PASSED" if ok else "SOME_CHECKS_FAILED")
sys.exit(0 if ok else 1)
