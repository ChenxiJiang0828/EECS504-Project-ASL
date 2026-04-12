# EECS504 Project - ASL (Sign Language MNIST)

本项目基于 `model.ipynb`，使用 TensorFlow/Keras 对 Sign Language MNIST 数据集进行字母手势分类，并对多种模型进行对比与调参。

![Sign Language Sample](american_sign_1.png)

## 项目内容

- 数据：`data/sign_mnist_train.csv`、`data/sign_mnist_test.csv`
- 主实验：`model.ipynb`
- 训练后模型：`sign_language_resnet_small.keras`
- 可视化示例：`model_summary.png`、`prediction_output.png`

## 环境依赖

推荐 Python 3.9+，安装：

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python
```

## 训练配置（来自最新 `model.ipynb`）

- 输入尺寸：`28 x 28 x 1`
- Epochs：`8`
- Batch Size：`128`
- Learning Rate：`1e-3`
- 优化器：`Adam`
- 损失函数：`sparse_categorical_crossentropy`
- 早停：`EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)`

## 模型对比结果（`results_df`）

| model | train_acc | val_acc | val_loss | params |
|---|---:|---:|---:|---:|
| resnet_small | 1.0000 | 0.9972 | 0.0137 | 681,529 |
| mobilenet_small | 0.9648 | 0.9794 | 0.0762 | 2,290,009 |
| vgg_tiny | 0.9808 | 0.9678 | 0.0703 | 142,073 |
| basic_cnn | 0.9887 | 0.9426 | 0.2010 | 243,481 |

结论：最新 notebook 中选择 `resnet_small` 作为最佳模型并保存为：

```text
sign_language_resnet_small.keras
```

## ResNet 调参结果（`resnet_tune_df`）

调参维度：`learning rate`、`batch size`、`dropout`。

| setting | lr | batch_size | dropout | val_acc | best_val_acc |
|---|---:|---:|---:|---:|---:|
| small_batch | 1e-3 | 64 | 0.30 | 0.9985 | 1.0000 |
| base | 1e-3 | 128 | 0.30 | 0.9992 | 0.9992 |
| strong_dropout | 1e-3 | 128 | 0.45 | 0.9974 | 0.9974 |
| low_lr_small_batch | 3e-4 | 64 | 0.45 | 0.9968 | 0.9968 |
| low_lr | 3e-4 | 128 | 0.30 | 0.9877 | 0.9902 |

## 测试结论（Notebook 最后一部分）

- 在 validation set 上绘制了 confusion matrix。
- 在 `data/photo` 的真实图片小样本（10 张）上测试，记录结果：
  - `Dataset: 10 | Accuracy: 80.00%`

这说明模型在 Sign Language MNIST 分布内表现很强，但在真实图片域上仍存在泛化差距。

## 运行方式

1. 打开 `model.ipynb`。
2. 按顺序运行：数据读取 -> 模型对比训练 -> 保存最佳模型 -> 混淆矩阵 -> 真实图像测试。
3. 若只做推理，请确保 `sign_language_resnet_small.keras` 存在。

## 当前目录（已轻量整理）

- 已清理：`.idea/`、`.ipynb_checkpoints/`
- 新增：`.gitignore`（忽略 IDE 文件、notebook checkpoint、本地数据与新训练产物）

![Prediction Example](prediction_output.png)
