from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import ResNet50_Weights, resnet50
train_test = {}
with open("D:\\A\\ML\\data\\train_test_split_0to1_1to9.txt", "r") as f:
    for line in f:
        line = line.strip()
        image_id, split_label = line.split(maxsplit=1)
        image_id=int(image_id)
        split_label = int(split_label)
        train_test[image_id] = split_label
image_locations = {}
with open("D:\\A\\ML\\data\\images.txt", "r") as f:
    for line in f:
        line = line.strip()
        image_id, image_location = line.split(maxsplit=1)
        image_id=int(image_id)
        image_locations[image_id] = image_location
image_classes = {}
with open("D:\\A\\ML\\data\\image_class_labels.txt", "r") as f:
    for line in f:
        line = line.strip()
        image_id, class_id = line.split(maxsplit=1)
        image_id=int(image_id)
        class_id = int(class_id)
        image_classes[image_id] = class_id
class_names = {}
with open("D:\\A\\ML\\data\\classes.txt", "r") as f:
    for line in f:
        line = line.strip()
        class_id, class_name = line.split(maxsplit=1)
        class_id = int(class_id)
        class_names[class_id] = class_name
#train_test:key:image_id,value:split_label(int, 1 for train, 0 for test);
#image_locations:key:image_id,value:image_location(string);
#image_classes:key:image_id,value:class_id(int);
#class_names:key:class_id,value:class_name(string);
train_val_ids = []
test_ids = []
for image_id, split_label in train_test.items():
    if split_label == 1:
        train_val_ids.append(image_id)
    else:
        test_ids.append(image_id)
train_ids = []
val_ids = []
labels = [image_classes[image_id] for image_id in train_val_ids]
train_ids, val_ids = train_test_split(
    train_val_ids,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15,
                           saturation=0.15),
    transforms.ToTensor(), norm,
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), norm,
])
class BirdDataset(Dataset):
    def __init__(self, image_ids, image_locations, image_classes, images_root, transform):
        self.image_ids = image_ids
        self.image_locations = image_locations
        self.image_classes = image_classes
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_location = self.image_locations[image_id]
        image_class = self.image_classes[image_id]-1  # Convert to 0-based index
        image = Image.open(f"{self.images_root}/{image_location}").convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_class
if __name__ == "__main__":
    train_dataset = BirdDataset(train_ids, image_locations, image_classes, "D:\\A\\ML\\data\\images", train_transform)
    print(len(train_dataset))
    for i in range(5):
        image, label = train_dataset[i]
        print(i, image.shape, label)
    dataset = train_dataset
    batch_size = 32
    shuffle = True
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    num_workers = 0
    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)
    val_dataset = BirdDataset(val_ids,image_locations,image_classes,"D:\\A\\ML\\data\\images",val_transform)
    val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)
    device=torch.device(
        "xpu" 
    )
    print(device)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    model.fc = nn.Linear(
        model.fc.in_features,
        200
    )
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {"params": model.fc.parameters(), "lr": 1e-4},
        {"params": (p for n, p in model.named_parameters()
                    if not n.startswith("fc.")), "lr": 1e-5},
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    epochs=20
    best_acc=0
    for epoch in range(epochs):
        model.train()
        running_loss=0
        train_correct=0
        train_total=0
        for images,labels in tqdm(train_loader):
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _,predicted=torch.max(outputs,1)
            train_total += labels.size(0)
            train_correct += (predicted==labels).sum().item()
        print("Epoch:",epoch+1,"Loss:",running_loss/len(train_loader))
        print("Training Accuracy:",train_correct/train_total)
        model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for images,labels in val_loader:
                images=images.to(device)
                labels=labels.to(device)
                outputs=model(images)
                _,predicted=torch.max(outputs,1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        acc=correct/total
        print("Validation Accuracy:",acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),"D:\\A\\ML\\model\\best_model.pth")
            print("Model saved!")