"""
NST Loss Functions — StyleSense (Fixed v2)
Owner: Shubhansh Gupta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def forward(self, gen_feat, content_feat):
        return F.mse_loss(gen_feat, content_feat.detach())


def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Gram matrix — NO over-normalization.
    Just divide by C to keep scale reasonable.
    """
    B, C, H, W = feature_map.shape
    features = feature_map.view(B, C, H * W)          # (B, C, H*W)
    gram     = torch.bmm(features, features.transpose(1, 2))  # (B, C, C)
    return gram / C                                    # only /C — not /C*H*W


class StyleLoss(nn.Module):
    def forward(self, gen_features: dict, style_features: dict) -> torch.Tensor:
        loss = 0.0
        for layer in gen_features:
            G_gen   = gram_matrix(gen_features[layer])
            G_style = gram_matrix(style_features[layer].detach())
            loss   += F.mse_loss(G_gen, G_style)
        return loss / len(gen_features)


class TVLoss(nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        dh = img[:, :, 1:, :]  - img[:, :, :-1, :]
        dw = img[:, :, :, 1:]  - img[:, :, :, :-1]
        return dh.abs().mean() + dw.abs().mean()


class NSTLoss(nn.Module):
    def __init__(self, content_weight=1.0,
                 style_weight=50.0,    # tuned for gram/C normalization
                 tv_weight=1e-4):
        super().__init__()
        self.cw = content_weight
        self.sw = style_weight
        self.tw = tv_weight
        self.content_loss = ContentLoss()
        self.style_loss   = StyleLoss()
        self.tv_loss      = TVLoss()

    def forward(self, gen_img, gen_features, content_features, style_features):
        lc    = self.content_loss(gen_features["block3"],
                                  content_features["block3"])
        ls    = self.style_loss(gen_features, style_features)
        lt    = self.tv_loss(gen_img)
        total = self.cw * lc + self.sw * ls + self.tw * lt
        return total, lc, ls, lt


if __name__ == "__main__":
    # Test with clearly different tensors
    gen_feat = {
        "block1": torch.randn(1, 64,  32, 32),
        "block2": torch.randn(1, 128, 16, 16),
        "block3": torch.randn(1, 256,  8,  8),
        "block4": torch.randn(1, 512,  4,  4),
    }
    # Style = all ones (very different from random)
    style_feat = {
        "block1": torch.ones(1, 64,  32, 32),
        "block2": torch.ones(1, 128, 16, 16),
        "block3": torch.ones(1, 256,  8,  8),
        "block4": torch.ones(1, 512,  4,  4),
    }
    gen_img = torch.randn(1, 3, 64, 64)

    criterion = NSTLoss()
    total, lc, ls, lt = criterion(gen_img, gen_feat, style_feat, style_feat)

    print("=" * 50)
    print("  NST Loss Functions v2 — Sanity Test")
    print("=" * 50)
    print(f"  Content Loss : {lc.item():.6f}")
    print(f"  Style Loss   : {ls.item():.6f}  ← must be NON-ZERO")
    print(f"  TV Loss      : {lt.item():.6f}")
    print(f"  Total Loss   : {total.item():.4f}")

    g = gram_matrix(torch.randn(1, 64, 32, 32))
    print(f"  Gram matrix  : {g.shape}  mean={g.abs().mean().item():.4f}")

    assert ls.item() > 0, "Style Loss is ZERO — fix gram_matrix!"
    print("  ALL CHECKS PASSED ✅")
