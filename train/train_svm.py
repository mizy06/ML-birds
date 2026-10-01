import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from PIL import Image
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
PART_SCALE = {
    1: 0.35,   # back
    2: 0.18,   # beak
    3: 0.35,   # belly
    4: 0.35,   # breast
    5: 0.20,   # crown
    6: 0.20,   # forehead
    7: 0.16,   # left eye
    8: 0.25,   # left leg
    9: 0.40,   # left wing
    10: 0.22,  # nape
    11: 0.16,  # right eye
    12: 0.25,  # right leg
    13: 0.40,  # right wing
    14: 0.35,  # tail
    15: 0.22,  # throat
}
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
parts_box = {}
with open("D:\\A\\ML\\data\\parts\\part_locs.txt", "r") as f:
    for line in f:
        line = line.strip()
        image_id, parts_id, x, y, visible = line.split(maxsplit=4)
        image_id = int(image_id)
        parts_id = int(parts_id)
        x = float(x)
        y = float(y)
        visible = int(visible)  # Assuming width is the first value in visible
        parts_box[image_id, parts_id] = (x, y, visible)
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

def hsv_histogram(region):
    hsv = np.asarray(region.convert("HSV"))

    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(
            hsv[:, :, channel],
            bins=16,
            range=(0, 256),
        )
        histograms.append(hist)

    result = np.concatenate(histograms).astype(np.float32)
    result /= max(result.sum(), 1.0)
    return result

def extract_feature_from_parts(image,image_id):
    all_features = []
    for part_id in range(1, 16):
        if (image_id, part_id) not in parts_box:
            continue
        x, y, visible = parts_box[image_id, part_id]
        if visible == 0:
            all_features.append(np.zeros(375))
            continue
        scale = PART_SCALE[part_id]
        width, height = bounding_boxes[image_id][2], bounding_boxes[image_id][3]    
        part_width = width*scale
        part_height = height*scale
        box = (
            int(max(0, x-part_width/2)),
            int(max(0, y-part_height/2)),
            int(min(image.size[0], x + part_width/2)),
            int(min(image.size[1], y + part_height/2))
        )
        part_image = image.crop(box)
        part_image = part_image.resize((32, 32))
        histogram_feature = hsv_histogram(part_image)
        numpy_image = np.array(part_image)
        feature = hog(
            numpy_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            channel_axis=-1
        )
        all_features.append(np.concatenate([histogram_feature, feature,[(x-bounding_boxes[image_id][0])/width, (y-bounding_boxes[image_id][1])/height,visible]]))
    return np.concatenate(all_features) if all_features else np.zeros(0)

def extract_feature_from_image(image):
    width, height = image.size
    mid_x, mid_y = width // 2, height // 2

    boxes = [
        (0, 0, width, height),       # 整鸟
        (0, 0, mid_x, mid_y),       # 左上
        (mid_x, 0, width, mid_y),   # 右上
        (0, mid_y, mid_x, height),  # 左下
        (mid_x, mid_y, width, height),  # 右下
    ]

    return np.concatenate([
        hsv_histogram(image.crop(box))
        for box in boxes
    ])
def extract_feature(image_ids):
    all_features = []

    for image_id in image_ids:
        location = image_locations[image_id]

        image = Image.open(
            "D:\\A\\ML\\data\\images\\" + location
        ).convert("RGB")
        parts_features = extract_feature_from_parts(image,image_id)
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
        global_hog = feature / (np.linalg.norm(feature) + 1e-8)
        histogram_feature = histogram_feature / (np.linalg.norm(histogram_feature) + 1e-8)
        parts_features = parts_features / (np.linalg.norm(parts_features) + 1e-8)
        final_feature = np.concatenate([
            global_hog,
            0.5 * histogram_feature,
            3.0 * parts_features
            ])
        all_features.append(final_feature)
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
# 完成第 2 步的空间颜色特征后，这里是 5 个区域 × 3 个通道 × 16 个 bin
COLOR_DIM = 5 * 3 * 16
for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
    base_model = LinearSVC(max_iter=10000,C=0.1)
    model = OneVsRestClassifier(
    base_model,
    n_jobs=8
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    print(C, acc)
