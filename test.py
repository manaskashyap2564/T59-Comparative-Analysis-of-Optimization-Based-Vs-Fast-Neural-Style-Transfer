import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# CIFAR-10 auto download hoga (~170MB)
train_data = torchvision.datasets.CIFAR10(
    root='../../data', train=True, download=True, transform=transform)
val_data = torchvision.datasets.CIFAR10(
    root='../../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)

# Check one batch
images, labels = next(iter(train_loader))
print(f"Batch shape : {images.shape}")   # torch.Size([32, 3, 224, 224])
print(f"Labels shape: {labels.shape}")   # torch.Size([32])
print(f"Train size  : {len(train_data)}")
print(f"Val size    : {len(val_data)}")
print("Dataset loader working!")
