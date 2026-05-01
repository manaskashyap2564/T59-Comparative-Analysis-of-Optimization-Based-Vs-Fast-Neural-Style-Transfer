import os, sys, time, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from vgglikecnn import VGGLikeExtractor

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='Train', leave=False):
        images, labels = images.to(device), labels.to(device)
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
        for images, labels in tqdm(loader, desc='Val', leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
    return total_loss / total, 100.0 * correct / total

def main():
    DATADIR = '../../data'
    CHECKPTDIR = '../../checkpoints'
    EPOCHS = 3
    BATCHSIZE = 128
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CHECKPTDIR, exist_ok=True)
    print(f'Device: {DEVICE}')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])

    train_data = torchvision.datasets.CIFAR10(DATADIR, train=True, download=True, transform=train_transform)
    val_data   = torchvision.datasets.CIFAR10(DATADIR, train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCHSIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f'Train: {len(train_data)}  Val: {len(val_data)}')

    model     = VGGLikeExtractor(num_classes=10, cifarmode=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler('cuda')
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        vl_loss, vl_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0
        print(f'Epoch {epoch:02d}/{EPOCHS}  Train Acc: {tr_acc:.2f}  Val Acc: {vl_acc:.2f}  Time: {elapsed:.1f}s')
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt = os.path.join(CHECKPTDIR, 'best_extractor.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': vl_acc}, ckpt)
            print(f'  --> Best checkpoint saved val_acc={vl_acc:.2f}')

    print(f'Done! Best Val Acc: {best_val_acc:.2f}')
    print(f'Checkpoint: {CHECKPTDIR}/best_extractor.pth')

if __name__ == '__main__':
    main()
