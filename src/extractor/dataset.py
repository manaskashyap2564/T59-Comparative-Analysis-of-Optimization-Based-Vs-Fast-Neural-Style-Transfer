from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_data   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Classes: {train_data.classes}")
    return train_loader, val_loader
