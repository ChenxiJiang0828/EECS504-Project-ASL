import argparse
import copy
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ASLTestDataset(Dataset):
    """
    Test set loader for files named like: A_test.jpg, nothing_test.jpg, ...
    """

    def __init__(self, test_root: Path, class_to_idx: Dict[str, int], transform=None):
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples: List[Tuple[Path, int]] = []

        for image_path in sorted(test_root.glob("*.jpg")):
            label_name = image_path.stem.replace("_test", "")
            if label_name not in class_to_idx:
                continue
            self.samples.append((image_path, class_to_idx[label_name]))

        if not self.samples:
            raise RuntimeError(
                f"No valid test images found under {test_root}. "
                "Expected files like A_test.jpg or space_test.jpg."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    phase = "train" if train else "val"
    if tqdm is not None:
        iterator = tqdm(
            loader,
            desc=f"Epoch {epoch:02d}/{total_epochs} [{phase}]",
            leave=False,
        )
    else:
        iterator = loader

    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        if tqdm is not None:
            iterator.set_postfix(loss=f"{total_loss / num_batches:.4f}", acc=f"{total_acc / num_batches:.4f}")

    return total_loss / num_batches, total_acc / num_batches


def evaluate_test(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    if tqdm is not None:
        iterator = tqdm(loader, desc="Testing", leave=False)
    else:
        iterator = loader
    with torch.no_grad():
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="CNN baseline for ASL alphabet classification")
    parser.add_argument("--train-dir", type=str, default="asl_alphabet_train/asl_alphabet_train")
    parser.add_argument("--test-dir", type=str, default="asl_alphabet_test/asl_alphabet_test")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/cnn_baseline_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = Path(args.train_dir)
    test_root = Path(args.test_dir)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    base_dataset = datasets.ImageFolder(root=str(train_root))
    classes = base_dataset.classes
    class_to_idx = base_dataset.class_to_idx
    num_classes = len(classes)

    all_indices = list(range(len(base_dataset)))
    random.shuffle(all_indices)
    val_size = int(len(all_indices) * args.val_ratio)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    train_dataset_full = datasets.ImageFolder(root=str(train_root), transform=train_tf)
    val_dataset_full = datasets.ImageFolder(root=str(train_root), transform=eval_tf)
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    test_dataset = ASLTestDataset(test_root=test_root, class_to_idx=class_to_idx, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SmallCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    print(f"Device: {device}")
    print(f"Classes ({num_classes}): {classes}")
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")
    if tqdm is None:
        print("Tip: install tqdm for progress bars: pip install tqdm")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "val_acc": best_val_acc,
                    "args": vars(args),
                },
                save_path,
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    test_acc = evaluate_test(model, test_loader, device)
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
