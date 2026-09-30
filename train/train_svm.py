import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.svm import SVC
train_test = {}
with open("D:\\A\\ML\\data\\train_test_split.txt", "r") as f:
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
bounding_boxes = {}
with open("D:\\A\\ML\\data\\bounding_boxes.txt", "r") as f:
    for line in f:
        line = line.strip()
        image_id, x, y, width, height = line.split(maxsplit=4)
        image_id = int(image_id)
        x = float(x)
        y = float(y)
        width = float(width)
        height = float(height)
        bounding_boxes[image_id] = (x, y, width, height)
#train_test:key:image_id,value:split_label(int, 1 for train, 0 for test);
#image_locations:key:image_id,value:image_location(string);
#image_classes:key:image_id,value:class_id(int);
#class_names:key:class_id,value:class_name(string);
#bounding_boxes:key:image_id,value:(x, y, width, height)(float, float, float, float);
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

def extract_feature_from_image(image):
    hsv_image = image.convert("HSV")
    numpy_hsv = np.array(hsv_image)
    h_hist, _ = np.histogram(
    numpy_hsv[:, :, 0],
    bins=16,
    range=(0, 256)
    )
    s_hist, _ = np.histogram(
    numpy_hsv[:, :, 1],
    bins=16,
    range=(0, 256)
    )
    v_hist, _ = np.histogram(
    numpy_hsv[:, :, 2],
    bins=16,
    range=(0, 256)
    )
    hist = np.concatenate([h_hist, s_hist, v_hist])
    return hist / np.sum(hist)
def extract_feature(image_ids):
    all_features = []

    for image_id in image_ids:
        location = image_locations[image_id]

        image = Image.open(
            "D:\\A\\ML\\data\\images\\" + location
        ).convert("RGB")
        x, y, width, height = bounding_boxes[image_id]
        box = (
        int(x),
        int(y),
        int(x + width),
        int(y + height)
        )
        image = image.crop(box)
        image = image.resize((128, 128))
        histogram_feature = extract_feature_from_image(image)
        numpy_image = np.array(image)

        feature = hog(
            numpy_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            channel_axis=-1
        )
        all_features.append(np.concatenate([histogram_feature, feature]))
    return all_features
x_train_file = Path("X_train_hsv.npy")
y_train_file = Path("y_train_hsv.npy")
x_val_file = Path("X_val_hsv.npy")
y_val_file = Path("y_val_hsv.npy")

if (
    x_train_file.exists()
    and y_train_file.exists()
    and x_val_file.exists()
    and y_val_file.exists()
):
    print("发现已有特征，直接读取")

    X_train = np.load(x_train_file)
    y_train = np.load(y_train_file)
    X_val = np.load(x_val_file)
    y_val = np.load(y_val_file)

else:
    print("没有缓存，开始提取 HOG 特征")

    train_feature = extract_feature(train_ids)
    val_feature = extract_feature(val_ids)
    X_train = np.array(train_feature)
    X_val = np.array(val_feature)
    y_train = np.array(
        [image_classes[image_id] for image_id in train_ids]
    )
    y_val = np.array(
        [image_classes[image_id] for image_id in val_ids]
    )

    np.save(x_train_file, X_train)
    np.save(y_train_file, y_train)
    np.save(x_val_file, X_val)
    np.save(y_val_file, y_val)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

model = SVC(
    kernel="rbf",
    C=100,
    gamma="scale",
    cache_size=2048,
    verbose=False,
)


model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {accuracy}")