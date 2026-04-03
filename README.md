# EECS504-Project-ASL

## CNN Baseline

This repo now includes a simple PyTorch CNN baseline for ASL alphabet image classification.

### 1. Install dependencies

```bash
pip install torch torchvision pillow
```

### 2. Dataset structure (already in this repo)

- `asl_alphabet_train/asl_alphabet_train/<class_name>/*.jpg`
- `asl_alphabet_test/asl_alphabet_test/*_test.jpg`

### 3. Train and evaluate

```bash
python cnn_baseline.py
```

Optional example with custom settings:

```bash
python cnn_baseline.py --epochs 12 --batch-size 64 --image-size 64 --lr 0.001
```

### 4. Outputs

- Best checkpoint: `checkpoints/cnn_baseline_best.pt`
- Console logs: train/val loss+accuracy each epoch, plus final test accuracy
