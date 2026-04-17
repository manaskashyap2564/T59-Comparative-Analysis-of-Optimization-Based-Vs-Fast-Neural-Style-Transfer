"""
Training Script — Custom VGG-like Feature Extractor (FIXED + RAM LOADED)
"""

import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from vgg_like_cnn import VGGLikeExtractor


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val  ", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    DATA_DIR    = "../../data"
    CHECKPT_DIR = "../../checkpoints"
    NUM_CLASSES = 10
    BATCH_SIZE  = 256
    EPOCHS      = 25
    LR          = 0.001
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Augmentation ──────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])

    # ── RAM mein load karo ────────────────────────────────────────────────
    print("Loading CIFAR-10 into RAM...")
    train_raw = torchvision.datasets.CIFAR10(DATA_DIR, train=True,
                                             download=True, transform=train_transform)
    val_raw   = torchvision.datasets.CIFAR10(DATA_DIR, train=False,
                                             download=True, transform=val_transform)

    print("Stacking tensors into RAM (one-time cost ~2-3 min)...")
    train_data = torch.stack([train_raw[i][0] for i in tqdm(range(len(train_raw)), desc="Train RAM")])
    train_lbls = torch.tensor([train_raw[i][1] for i in range(len(train_raw))])
    val_data   = torch.stack([val_raw[i][0] for i in tqdm(range(len(val_raw)), desc="Val RAM")])
    val_lbls   = torch.tensor([val_raw[i][1] for i in range(len(val_raw))])

    print(f"Train: {train_data.shape} | Val: {val_data.shape}")
    print(f"RAM used: ~{(train_data.nbytes + val_data.nbytes) / 1e6:.0f} MB")

    # num_workers=0 — RAM se load ho raha hai, workers ki zaroorat nahi
    train_loader = DataLoader(TensorDataset(train_data, train_lbls),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(val_data, val_lbls),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = VGGLikeExtractor(num_classes=NUM_CLASSES, cifar_mode=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc     = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CHECKPT_DIR, "best_extractor.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  >> Best checkpoint saved: val_acc={val_acc:.2f}%")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {CHECKPT_DIR}/best_extractor.pth")


if __name__ == "__main__":
    main()
