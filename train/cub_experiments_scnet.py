r"""Controlled CUB-200-2011 fine-tuning experiments.

SCNet / Slurm edition of ``cub_experiments.py``.

Changes relative to the original desktop script:
  * ``device`` selection is accelerator-aware. On the SCNet Zhengzhou nodes the
    heterogeneous accelerator is a Hygon DCU, which a DTK/ROCm build of PyTorch
    exposes through ``torch.cuda`` with ``torch.version.hip`` set. That backend
    is detected and reported as ``dcu`` instead of being silently labelled
    ``cuda``. Use ``--device`` to force one, ``--device-info`` to inspect.
  * ``--data-root`` / ``--out-root`` default to the cluster layout
    (``$CUB_DATA_ROOT`` / ``$CUB_OUT_ROOT``, falling back to ``~/cub_job/...``).
  * ``pin_memory`` follows the device, and ``torch.cuda`` seeds are set.

Examples (PowerShell, original desktop script):
  python cub_experiments.py --data-root D:\A\ML\data --audit
  python cub_experiments.py --data-root D:\A\ML\data --model resnet18 --size 224 --run r18_full
  python cub_experiments.py --data-root D:\A\ML\data --model resnet18 --size 224 --bbox --run r18_box
  python cub_experiments.py --data-root D:\A\ML\data --model resnet50 --size 224 --run r50_full
  python cub_experiments.py --data-root D:\A\ML\data --model resnet50 --size 320 --bbox --run r50_box
  python cub_experiments.py --data-root D:\A\ML\data --model resnet50 --size 320 --bbox --eval-only --checkpoint runs\r50_box\best.pt --subset val --tta
  python cub_experiments.py --data-root D:\A\ML\data --eval-only --legacy-state-dict --checkpoint D:\A\ML\model\best_model.pth --tta

Examples (SCNet, run through the matching sbatch in ./sbatch/):
  sbatch sbatch/01_audit.sbatch
  sbatch sbatch/02_r18_full.sbatch
  sbatch sbatch/03_r18_box.sbatch
  sbatch sbatch/04_r50_full.sbatch
  sbatch sbatch/05_r50_box.sbatch
  sbatch sbatch/06_eval_r50_box.sbatch
  sbatch sbatch/07_eval_legacy.sbatch

The split file is the user's current custom split by default. Use --split-file
train_test_split.txt for the official CUB split (and a separate run directory).
Bounding-box experiments use ground-truth boxes for validation and testing.
"""

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    resnet18,
    resnet50,
)
from tqdm import tqdm


# --------------------------------------------------------------------------
# Heterogeneous accelerator handling
# --------------------------------------------------------------------------

def available_backends():
    """Return the accelerator backends this interpreter can actually use.

    A Hygon DCU is driven by a DTK (ROCm/HIP) build of PyTorch: it shows up as
    ``torch.cuda`` but ``torch.version.hip`` is non-empty. We name that backend
    ``dcu`` so the logs say what hardware really ran the job.
    """
    backends = []
    if torch.cuda.is_available():
        backends.append("dcu" if torch.version.hip else "cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        backends.append("xpu")
    backends.append("cpu")
    return backends


def select_device(requested=None):
    """Resolve --device into a torch.device, preferring the node's accelerator.

    Preference order: DCU (Hygon) / CUDA > XPU > CPU. ``requested`` may force a
    backend, but it must be one this node actually offers.
    """
    backends = available_backends()
    if requested in (None, "auto"):
        kind = backends[0]
    else:
        kind = requested.lower()
        # On a DCU node the HIP device is reached through the torch.cuda API,
        # so both spellings are accepted for the same piece of hardware.
        if kind == "cuda" and "dcu" in backends:
            kind = "dcu"
        if kind == "dcu" and "dcu" not in backends and "cuda" in backends:
            kind = "cuda"
        if kind not in backends:
            raise SystemExit(f"requested device {requested!r} unavailable; "
                             f"this node offers {backends}")
    if kind in ("dcu", "cuda"):
        torch.cuda.set_device(0)
        device = torch.device("cuda")
    else:
        device = torch.device(kind)
    return device, kind


def describe_device(device, kind):
    """One-line human-readable summary of the accelerator in use."""
    if kind in ("dcu", "cuda"):
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 2 ** 30
        stack = "DTK/HIP" if torch.version.hip else "CUDA"
        return (f"backend={kind} stack={stack} name={name} "
                f"mem={total:.1f}GiB count={torch.cuda.device_count()}")
    if kind == "xpu":
        return f"backend=xpu name={torch.xpu.get_device_name(0)}"
    return "backend=cpu (no accelerator visible)"


def read_pairs(path, value_type=str):
    with path.open(encoding="utf-8") as f:
        return {int(parts[0]): value_type(parts[1])
                for line in f if (parts := line.split())}


def read_boxes(path):
    with path.open(encoding="utf-8") as f:
        return {int(parts[0]): tuple(map(float, parts[1:5]))
                for line in f if (parts := line.split())}


def min_smoothed_loss(classes, epsilon):
    if epsilon == 0:
        return 0.0
    target = 1 - epsilon + epsilon / classes
    other = epsilon / classes
    return -target * math.log(target) - (classes - 1) * other * math.log(other)


def load_split(root, split_file, seed):
    paths = read_pairs(root / "images.txt")
    labels = read_pairs(root / "image_class_labels.txt", int)
    split_path = root / split_file
    split = read_pairs(split_path, int)
    if set(paths) != set(labels) or set(paths) != set(split):
        raise ValueError("images.txt, image_class_labels.txt and split file must have identical image IDs")
    if set(split.values()) != {0, 1}:
        raise ValueError("split labels must be 0 (holdout) or 1 (train+val)")
    if set(labels.values()) != set(range(1, 201)):
        raise ValueError("expected 200 classes numbered 1..200")
    train_val = sorted(i for i in paths if split[i] == 1)
    test = sorted(i for i in paths if split[i] == 0)
    train, val = train_test_split(train_val, test_size=0.2, random_state=seed,
                                  stratify=[labels[i] for i in train_val])
    train, val = sorted(train), sorted(val)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    train_counts = Counter(labels[i] for i in train)
    val_counts = Counter(labels[i] for i in val)
    if len(train_counts) != 200 or len(val_counts) != 200:
        raise ValueError(f"missing classes: train={len(train_counts)}, val={len(val_counts)}")
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    print(f"split={split_path} sha256={digest}")
    print(f"train={len(train)} val={len(val)} holdout={len(test)}")
    print(f"images per class: train={min(train_counts.values())}..{max(train_counts.values())}, "
          f"val={min(val_counts.values())}..{max(val_counts.values())}")
    official_path = root / "train_test_split.txt"
    if official_path.is_file():
        official = read_pairs(official_path, int)
        print(f"original official test images in training: "
              f"{sum(official.get(i) == 0 for i in train)}; "
              f"in validation: {sum(official.get(i) == 0 for i in val)}")
        if split_path.resolve() != official_path.resolve():
            print("This custom split is not the official CUB train/test protocol.")
    return paths, labels, train, val, test, digest


class Birds(Dataset):
    def __init__(self, ids, root, paths, labels, transform, boxes=None, margin=0.12):
        self.ids, self.root, self.paths, self.labels = ids, root, paths, labels
        self.transform, self.boxes, self.margin = transform, boxes, margin
        if boxes is not None and any(i not in boxes for i in ids):
            raise ValueError("bounding_boxes.txt does not contain every selected image")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        with Image.open(self.root / "images" / self.paths[image_id]) as source:
            image = source.convert("RGB")
        if self.boxes is not None:
            x, y, w, h = self.boxes[image_id]
            if w <= 0 or h <= 0:
                raise ValueError(f"invalid box for image {image_id}")
            left = max(0, int(math.floor(x - self.margin * w)))
            top = max(0, int(math.floor(y - self.margin * h)))
            right = min(image.width, int(math.ceil(x + w * (1 + self.margin))))
            bottom = min(image.height, int(math.ceil(y + h * (1 + self.margin))))
            if left >= right or top >= bottom:
                raise ValueError(f"empty box for image {image_id}")
            image = image.crop((left, top, right, bottom))
        return self.transform(image), self.labels[image_id] - 1


def transforms_for(size):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = round(size * 256 / 224)
    train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        transforms.ToTensor(), norm,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(resize), transforms.CenterCrop(size),
        transforms.ToTensor(), norm,
    ])
    return train, eval_transform


def build_model(name, pretrained):
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 200)
        head_name = "fc."
    elif name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 200)
        head_name = "fc."
    elif name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 200)
        head_name = "classifier.2."
    else:
        raise ValueError(name)
    return model, head_name


PRETRAINED_URLS = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1.url,
    "resnet50": ResNet50_Weights.IMAGENET1K_V2.url,
    "convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url,
}


def pretrained_cache_dir():
    return Path(torch.hub.get_dir()) / "checkpoints"


def build_pretrained_model(name):
    """Build a model with ImageNet weights, with an offline-friendly failure mode.

    SCNet compute nodes have no outbound network, so torchvision cannot fetch the
    weights at job time. If the file is not already in the torch hub cache we say
    exactly which file is missing and how to pre-fetch it on a login node, instead
    of letting urllib raise a bare DNS/SSL error.
    """
    url = PRETRAINED_URLS[name]
    target = pretrained_cache_dir() / url.rsplit("/", 1)[-1]
    if target.is_file():
        print(f"pretrained weights: {target}")
    else:
        print(f"WARNING: pretrained weights not cached at {target}")
        print("  Compute nodes are offline; pre-fetch on a login node with:")
        print(f"    mkdir -p {target.parent}")
        print(f"    curl -L -o {target} {url}")
    try:
        return build_model(name, pretrained=True)
    except Exception as exc:
        raise SystemExit(
            f"could not build a pretrained {name}: {type(exc).__name__}: {exc}\n"
            f"missing/expected weights file: {target}\n"
            f"pre-fetch it on a login node (which has network), then resubmit."
        ) from exc


def set_batchnorm_eval(model):
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()  # freeze running means/variances; affine parameters still train


def evaluate(model, loader, device, criterion, tta=False):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if tta:
                logits = (logits + model(torch.flip(images, dims=(-1,)))) / 2
            total_loss += criterion(logits, labels).item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, total_correct / total


def make_loader(ids, root, paths, labels, transform, boxes, args, shuffle):
    return DataLoader(Birds(ids, root, paths, labels, transform, boxes, args.margin),
                      batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=args.pin_memory,
                      persistent_workers=args.workers > 0)


def default_data_root():
    """Cluster layout: $CUB_DATA_ROOT, else ~/cub_job/data, else the old desktop path."""
    env = os.environ.get("CUB_DATA_ROOT")
    if env:
        return Path(env)
    cluster = Path.home() / "cub_job" / "data"
    if cluster.is_dir():
        return cluster
    return Path(r"D:\A\ML\data")


def default_out_root():
    env = os.environ.get("CUB_OUT_ROOT")
    if env:
        return Path(env)
    submit_dir = os.environ.get("SLURM_SUBMIT_DIR")
    if submit_dir:
        return Path(submit_dir) / "runs"
    return Path("runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--split-file", default="train_test_split_0to1_1to9.txt")
    parser.add_argument("--audit", action="store_true", help="only audit split, without loading PyTorch model")
    parser.add_argument("--model", choices=["resnet18", "resnet50", "convnext_tiny"], default="resnet18")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--bbox", action="store_true", help="GT bounding boxes on train/val/test")
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--run", default="r18_full")
    parser.add_argument("--out-root", type=Path, default=default_out_root())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--bn-train", action="store_true", help="update ResNet batch-norm running statistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "dcu", "cuda", "xpu", "cpu"], default="auto",
                        help="accelerator to use; 'auto' prefers DCU/CUDA, then XPU, then CPU")
    parser.add_argument("--device-info", action="store_true",
                        help="print the accelerator the node offers and exit")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--legacy-state-dict", action="store_true",
                        help="load old model.state_dict() checkpoint; user must supply correct architecture, size and split")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--subset", choices=["val", "test"], default="val")
    parser.add_argument("--tta", action="store_true", help="average original and horizontal-flip logits in eval-only mode")
    args = parser.parse_args()
    if not 0 <= args.smoothing < 1 or args.size < 128 or args.margin < 0:
        parser.error("require smoothing in [0,1), size >= 128 and margin >= 0")
    if args.eval_only and args.checkpoint is None:
        parser.error("--eval-only requires --checkpoint")
    if args.tta and not args.eval_only:
        parser.error("--tta is for --eval-only; checkpoint selection uses plain validation")

    if args.device_info:
        device, kind = select_device(args.device)
        print(f"torch={torch.__version__} hip={torch.version.hip} cuda={torch.version.cuda}")
        print(f"available={available_backends()}")
        print(describe_device(device, kind))
        return

    device, kind = select_device(args.device)
    args.pin_memory = kind in ("dcu", "cuda")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if kind in ("dcu", "cuda"):
        torch.cuda.manual_seed_all(args.seed)
    paths, labels, train_ids, val_ids, test_ids, split_hash = load_split(
        args.data_root, args.split_file, args.seed)
    if args.audit:
        return
    print(f"device={device} {describe_device(device, kind)}")
    print(f"torch={torch.__version__} hip={torch.version.hip} smoothing={args.smoothing} "
          f"minimum_possible_CE={min_smoothed_loss(200, args.smoothing):.6f}")
    train_tf, eval_tf = transforms_for(args.size)
    boxes = read_boxes(args.data_root / "bounding_boxes.txt") if args.bbox else None

    if args.eval_only:
        try:
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # for older installed PyTorch versions
            saved = torch.load(args.checkpoint, map_location="cpu")
        if args.legacy_state_dict:
            print("Legacy checkpoint: architecture, data split and preprocessing cannot be verified.")
            state = saved
        else:
            expected = dict(model=args.model, size=args.size, bbox=args.bbox,
                            margin=args.margin, split_hash=split_hash, seed=args.seed)
            if any(saved["config"][k] != v for k, v in expected.items()):
                raise ValueError(f"checkpoint configuration mismatch; expected {saved['config']}")
            state = saved["model_state"]
        model, _ = build_model(args.model, pretrained=False)
        model.load_state_dict(state)
        model.to(device)
        selected = val_ids if args.subset == "val" else test_ids
        loader = make_loader(selected, args.data_root, paths, labels, eval_tf, boxes, args, False)
        loss, acc = evaluate(model, loader, device, nn.CrossEntropyLoss(), args.tta)
        print(f"{args.subset} n={len(selected)} tta={args.tta} loss={loss:.4f} acc={acc:.5f}")
        return

    run_dir = args.out_root / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    if best_path.exists():
        raise FileExistsError(f"{best_path} exists; choose a new --run to avoid overwriting")
    config = dict(model=args.model, size=args.size, bbox=args.bbox, margin=args.margin,
                  split_hash=split_hash, seed=args.seed, smoothing=args.smoothing,
                  bn_train=args.bn_train, lr_backbone=args.lr_backbone, lr_head=args.lr_head,
                  weight_decay=args.weight_decay, batch_size=args.batch_size,
                  train_n=len(train_ids), val_n=len(val_ids), test_n=len(test_ids),
                  device=kind)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_loader = make_loader(train_ids, args.data_root, paths, labels, train_tf, boxes, args, True)
    val_loader = make_loader(val_ids, args.data_root, paths, labels, eval_tf, boxes, args, False)
    model, head_name = build_pretrained_model(args.model)
    model.to(device)
    head = [p for n, p in model.named_parameters() if n.startswith(head_name)]
    backbone = [p for n, p in model.named_parameters() if not n.startswith(head_name)]
    optimizer = torch.optim.AdamW([{"params": head, "lr": args.lr_head},
                                   {"params": backbone, "lr": args.lr_backbone}],
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    best_acc, best_epoch, stale = -1, 0, 0
    with (run_dir / "history.csv").open("w", encoding="utf-8") as history:
        history.write("epoch,train_loss,train_acc,val_loss,val_acc,head_lr,backbone_lr\n")
        for epoch in range(1, args.epochs + 1):
            model.train()
            if not args.bn_train:
                set_batchnorm_eval(model)
            loss_sum = correct = seen = 0
            for images, targets in tqdm(train_loader, desc=f"epoch {epoch}"):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * targets.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
                seen += targets.size(0)
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            head_lr, backbone_lr = (group["lr"] for group in optimizer.param_groups)
            history.write(f"{epoch},{loss_sum/seen:.6f},{correct/seen:.6f},"
                          f"{val_loss:.6f},{val_acc:.6f},{head_lr:.9g},{backbone_lr:.9g}\n")
            history.flush()
            print(f"epoch={epoch} train={correct/seen:.4f} val={val_acc:.4f} "
                  f"val_loss={val_loss:.4f} lr={head_lr:.2g}/{backbone_lr:.2g}")
            if val_acc > best_acc + 1e-4:
                best_acc, best_epoch, stale = val_acc, epoch, 0
                torch.save({"model_state": model.state_dict(), "config": config,
                            "epoch": epoch, "val_acc": val_acc}, best_path)
                print(f"saved {best_path}")
            else:
                stale += 1
            if stale >= args.patience:
                print(f"early stop; best epoch={best_epoch} val={best_acc:.5f}")
                break
            scheduler.step()
    print(f"best epoch={best_epoch} val={best_acc:.5f} path={best_path}")


if __name__ == "__main__":
    main()
