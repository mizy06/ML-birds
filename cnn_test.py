from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50
from train.train_cnn import test_ids, val_transform, image_classes, image_locations, BirdDataset


def main():
    images_root = Path(r"D:\A\ML\data\images")
    checkpoint = Path(r"D:\A\ML\model\best_model.pth")
    test_dataset = BirdDataset(test_ids, image_locations, image_classes, images_root, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    total = correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)
    print("Test Accuracy:", correct / total)


if __name__ == "__main__":
    main()