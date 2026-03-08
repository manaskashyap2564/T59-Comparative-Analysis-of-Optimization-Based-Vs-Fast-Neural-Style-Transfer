"""
Training Script — Custom VGG-like Feature Extractor
Owner: Shubhansh Gupta
Goal : Train CNN as image classifier (>70% val accuracy target).
       This trained backbone will be used for NST perceptual losses.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from vgg_like_cnn import VGGLikeExtractor
from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    # ── Config ──────────────────────────────────────────────────────────────
    DATA_DIR      = "../../data"          # update to your dataset path
    CHECKPT_DIR   = "../../checkpoints"
    NUM_CLASSES   = 10                    # update based on your dataset
    BATCH_SIZE    = 32
    LR            = 0.001
    EPOCHS        = 2 #20
    IMG_SIZE      = 224
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Data ────────────────────────────────────────────────────────────────
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_data = torchvision.datasets.CIFAR10('../../data', train=True,
                                           download=True, transform=transform)
    val_data   = torchvision.datasets.CIFAR10('../../data', train=False,
                                           download=True, transform=transform)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=2)


    # ── Model ───────────────────────────────────────────────────────────────
    model     = VGGLikeExtractor(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    # ── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc   = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        # Save best checkpoint
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
