from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    Returns train and val DataLoaders from an ImageFolder structure:
        data_dir/
            train/  class1/  class2/ ...
            val/    class1/  class2/ ...
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root=f"{data_dir}/train",
                                      transform=transform)
    val_data   = datasets.ImageFolder(root=f"{data_dir}/val",
                                      transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples : {len(train_data)}")
    print(f"Val   samples : {len(val_data)}")
    print(f"Classes       : {train_data.classes}")
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick smoke-test — replace path with your actual dataset path
    # get_dataloaders("data/imagenet_subset")
    print("Dataset loader ready. Set data_dir and call get_dataloaders().")
