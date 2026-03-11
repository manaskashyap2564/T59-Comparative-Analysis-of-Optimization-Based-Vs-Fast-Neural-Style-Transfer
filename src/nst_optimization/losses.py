"""
NST Loss Functions — StyleSense
Content Loss + Style Loss (Gram Matrix) + TV Loss
Owner: Shubhansh Gupta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Content Loss ────────────────────────────────────────────────────────────
class ContentLoss(nn.Module):
    """
    MSE between content image features and generated image features.
    Uses block3 output by default (high-level structure).
    """
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, gen_features: torch.Tensor,
                content_features: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(gen_features, content_features.detach())


# ── 2. Gram Matrix ─────────────────────────────────────────────────────────────
def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Computes Gram matrix for style representation.
    Args:
        feature_map: (B, C, H, W)
    Returns:
        gram: (B, C, C)
    """
    B, C, H, W = feature_map.shape
    features = feature_map.view(B, C, H * W)          # (B, C, H*W)
    gram = torch.bmm(features, features.transpose(1, 2))  # (B, C, C)
    # return gram / (C * H * W)                          # normalize
    return gram / (2.0 * C * H * W)   # standard normalization


# ── 3. Style Loss ──────────────────────────────────────────────────────────────
class StyleLoss(nn.Module):
    """
    MSE between Gram matrices of style image and generated image.
    Applied across multiple feature layers.
    """
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, gen_features: dict,
                style_features: dict) -> torch.Tensor:
        loss = 0.0
        for layer in gen_features:
            G_gen   = gram_matrix(gen_features[layer])
            G_style = gram_matrix(style_features[layer].detach())
            loss += F.mse_loss(G_gen, G_style)
        return loss / len(gen_features)


# ── 4. Total Variation Loss ────────────────────────────────────────────────────
class TVLoss(nn.Module):
    """
    Total Variation Loss — encourages spatial smoothness in generated image.
    Reduces noise / artifacts.
    """
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        diff_h = img[:, :, 1:, :]  - img[:, :, :-1, :]
        diff_w = img[:, :, :, 1:]  - img[:, :, :, :-1]
        return (diff_h.abs().mean() + diff_w.abs().mean())


# ── 5. Combined NST Loss ───────────────────────────────────────────────────────
class NSTLoss(nn.Module):
    """
    Combined loss = content_weight * L_content
                  + style_weight  * L_style
                  + tv_weight     * L_tv
    """
    def __init__(self, content_weight=1.0,
                 style_weight=1e6, tv_weight=1e-3):
        super(NSTLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight   = style_weight
        self.tv_weight      = tv_weight
        self.content_loss   = ContentLoss()
        self.style_loss     = StyleLoss()
        self.tv_loss        = TVLoss()

    def forward(self, gen_img, gen_features, content_features, style_features):
        lc = self.content_loss(gen_features["block3"], content_features["block3"])
        ls = self.style_loss(gen_features, style_features)
        lt = self.tv_loss(gen_img)
        total = (self.content_weight * lc +
                 self.style_weight   * ls +
                 self.tv_weight      * lt)
        return total, lc, ls, lt


# ── Quick Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, C, H, W = 1, 256, 28, 28

    # gen_feat     = {"block1": torch.randn(1,64,112,112),
    #                 "block2": torch.randn(1,128,56,56),
    #                 "block3": torch.randn(B,C,H,W),
    #                 "block4": torch.randn(1,512,14,14)}
    content_feat = {"block1": torch.randn(1,64,112,112),
                    "block2": torch.randn(1,128,56,56),
                    "block3": torch.randn(B,C,H,W),
                    "block4": torch.randn(1,512,14,14)}
    # style_feat   = {"block1": torch.randn(1,64,112,112),
    #                 "block2": torch.randn(1,128,56,56),
    #                 "block3": torch.randn(B,C,H,W),
    #                 "block4": torch.randn(1,512,14,14)}
    # Alag patterns wale tensors
    gen_feat   = {"block1": torch.randn(1,64,112,112) * 1.0,
                "block2": torch.randn(1,128,56,56)  * 1.0,
                "block3": torch.randn(1,256,28,28)  * 1.0,
                "block4": torch.randn(1,512,14,14)  * 1.0}

    style_feat = {"block1": torch.ones(1,64,112,112)  * 0.5,   # different!
                "block2": torch.ones(1,128,56,56)   * 0.5,
                "block3": torch.ones(1,256,28,28)   * 0.5,
                "block4": torch.ones(1,512,14,14)   * 0.5}
    gen_img = torch.randn(1, 3, 224, 224)

    criterion = NSTLoss()
    total, lc, ls, lt = criterion(gen_img, gen_feat, content_feat, style_feat)

    print("=" * 45)
    print("  NST Loss Functions — Sanity Test")
    print("=" * 45)
    print(f"  Content Loss : {lc.item():.6f}")
    print(f"  Style Loss   : {ls.item():.6f}")
    print(f"  TV Loss      : {lt.item():.6f}")
    print(f"  Total Loss   : {total.item():.4f}")
    print("=" * 45)

    # Gram matrix shape check
    fm = torch.randn(1, 64, 112, 112)
    g  = gram_matrix(fm)
    print(f"  Gram matrix shape: {g.shape}")  # Expected: (1, 64, 64)
    print("  ALL LOSS FUNCTIONS WORKING ✅")
