r"""Controlled CUB-200-2011 fine-tuning experiments.

Examples (PowerShell):
  python cub_experiments.py --data-root D:\A\ML\data --audit
  python cub_experiments.py --data-root D:\A\ML\data --model resnet18 --size 224 --run r18_full
  python cub_experiments.py --data-root D:\A\ML\data --model resnet18 --size 224 --bbox --run r18_box
  python cub_experiments.py --data-root D:\A\ML\data --model resnet50 --size 224 --run r50_full
  python cub_experiments.py --data-root D:\A\ML\data --model resnet50 --size 320 --bbox --run r50_box
  python cub_experiments.py --data-root D:\A\ML\data --model resnet50 --size 320 --bbox --eval-only --checkpoint runs\r50_box\best.pt --subset val --tta
  python cub_experiments.py --data-root D:\A\ML\data --eval-only --legacy-state-dict --checkpoint D:\A\ML\model\best_model.pth --tta

The split file is the user's current custom split by default. Use --split-file
train_test_split.txt for the official CUB split (and a separate run directory).
Bounding-box experiments use ground-truth boxes for validation and testing.
"""

import argparse
import hashlib
import json
import math
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
                      num_workers=args.workers, pin_memory=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\A\ML\data"))
    parser.add_argument("--split-file", default="train_test_split_0to1_1to9.txt")
    parser.add_argument("--audit", action="store_true", help="only audit split, without loading PyTorch model")
    parser.add_argument("--model", choices=["resnet18", "resnet50", "convnext_tiny"], default="resnet18")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--bbox", action="store_true", help="GT bounding boxes on train/val/test")
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--run", default="r18_full")
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths, labels, train_ids, val_ids, test_ids, split_hash = load_split(
        args.data_root, args.split_file, args.seed)
    if args.audit:
        return
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} smoothing={args.smoothing} "
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
                  train_n=len(train_ids), val_n=len(val_ids), test_n=len(test_ids))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_loader = make_loader(train_ids, args.data_root, paths, labels, train_tf, boxes, args, True)
    val_loader = make_loader(val_ids, args.data_root, paths, labels, eval_tf, boxes, args, False)
    model, head_name = build_model(args.model, pretrained=True)
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
