 # 基于 SVM 与 CNN 的鸟类图像分类器

本项目使用两种方法实现鸟类图片分类：支持向量机（SVM）和卷积神经网络（CNN），用于比较传统机器学习与深度学习方法在图像分类任务上的表现。

## 项目目标

- 对鸟类图像进行预处理与分类
- 使用 SVM 建立基线模型
- 使用 CNN 提取图像特征并完成分类
- 对比两种模型的准确率、训练时间和泛化能力

## 数据集目录结构

建议按照类别存放图片：

```text
dataset/
├── train/
│   ├── class_1/
│   └── class_2/
├── val/
│   ├── class_1/
│   └── class_2/
└── test/
	├── class_1/
	└── class_2/
```

每个子目录名称代表一个鸟类类别。请确保数据集已获得合法使用授权，并尽量保持各类别样本数量均衡。

## 方法说明

### SVM

1. 调整图片尺寸并进行归一化
2. 将图片转换为特征向量（可使用像素值、HOG 或预训练模型特征）
3. 使用支持向量机进行训练与分类

### CNN

1. 对图片进行缩放、归一化和数据增强
2. 通过卷积层和池化层提取图像特征
3. 通过全连接层输出各类别概率
4. 使用交叉熵损失和优化器训练模型

## 环境安装

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

Linux/macOS：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pillow torch torchvision jupyter
```

## 使用流程

1. 准备并划分鸟类图像数据集。
2. 完成图像预处理和标签编码。
3. 训练 SVM 模型，记录验证集和测试集结果。
4. 训练 CNN 模型，保存最佳权重。
5. 在测试集上评估并比较两个模型。

示例命令（根据实际代码调整）：

```bash
python train_svm.py --data_dir ./dataset
python train_cnn.py --data_dir ./dataset --epochs 30 --batch_size 32
python predict.py --image ./example.jpg --model ./checkpoints/best.pt
```

## 评估指标

- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-score
- 混淆矩阵

建议同时报告每个类别的指标，避免仅使用总体准确率掩盖类别不均衡问题。

## 实验记录

| 模型 | 输入特征 | 准确率 | F1-score | 训练时间 |
|---|---|---:|---:|---:|
| SVM | 待填写 | 待填写 | 待填写 | 待填写 |
| CNN | 待填写 | 待填写 | 待填写 | 待填写 |

## 注意事项

- 训练集、验证集和测试集应避免出现同一图片或高度相似图片。
- CNN 训练时可使用早停、学习率调整和数据增强降低过拟合风险。
- SVM 的核函数、C 值和 gamma 参数需要通过验证集调优。
- 固定随机种子，便于复现实验结果。

## 许可证

本项目许可证和数据集许可证待补充。
