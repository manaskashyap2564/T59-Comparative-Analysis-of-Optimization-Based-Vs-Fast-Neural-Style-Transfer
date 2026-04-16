"""
Checkpoint Loader — VGG-like Extractor
Owner: Shubhansh Gupta
"""

import torch
from vgg_like_cnn import VGGLikeExtractor


def load_extractor(checkpoint_path: str, num_classes: int = 10,
                   device: str = "cpu", cifar_mode: bool = True):
    model = VGGLikeExtractor(num_classes=num_classes, cifar_mode=cifar_mode)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded extractor from: {checkpoint_path}")
    print(f"  Trained epoch : {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val accuracy  : {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  All params frozen for NST use.")

    return model.to(device), checkpoint


if __name__ == "__main__":
    model, info = load_extractor("../../checkpoints/best_extractor.pth")
    feats = model.get_feature_maps(torch.randn(1, 3, 32, 32))
    for k, v in feats.items():
        print(f"{k}: {v.shape}")
