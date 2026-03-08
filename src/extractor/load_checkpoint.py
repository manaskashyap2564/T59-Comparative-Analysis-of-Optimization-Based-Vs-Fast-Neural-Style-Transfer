"""
Checkpoint Loader — VGG-like Extractor
Use this to load trained backbone for NST pipelines.
Owner: Shubhansh Gupta
"""

import torch
from vgg_like_cnn import VGGLikeExtractor


def load_extractor(checkpoint_path: str, num_classes: int = 10, device: str = "cpu"):
    """
    Loads trained VGGLikeExtractor from checkpoint.
    Freezes all parameters (backbone used only for feature extraction in NST).

    Returns:
        model (VGGLikeExtractor) — eval mode, frozen
        checkpoint info dict
    """
    model = VGGLikeExtractor(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Freeze backbone for NST feature extraction
    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded extractor from: {checkpoint_path}")
    print(f"  Trained epoch : {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val accuracy  : {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  All params frozen for NST use.")

    return model.to(device), checkpoint


if __name__ == "__main__":
    # Smoke test (update path)
    # model, info = load_extractor("../../checkpoints/best_extractor.pth")
    # feats = model.get_feature_maps(torch.randn(1, 3, 224, 224))
    # for k, v in feats.items():
    #     print(f"{k}: {v.shape}")
    print("load_checkpoint.py ready. Provide checkpoint path to test.")
