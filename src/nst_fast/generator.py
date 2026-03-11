"""
Fast NST Feed-Forward Generator — StyleSense
ResNet-style architecture for real-time style transfer.
Owner: Shubhansh Gupta
"""

import torch
import torch.nn as nn


# ── Building Blocks ────────────────────────────────────────────────────────────

class ConvNormReLU(nn.Module):
    """Conv2d + InstanceNorm + ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1,
                 padding=1, relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with 2x Conv + InstanceNorm."""
    def __init__(self, channels=128):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormReLU(channels, channels, 3, 1, 1, relu=True),
            ConvNormReLU(channels, channels, 3, 1, 1, relu=False),
        )

    def forward(self, x):
        return x + self.block(x)   # residual connection


class UpsampleConv(nn.Module):
    """Upsample + Conv (better than ConvTranspose2d — no checkerboard)."""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv = ConvNormReLU(in_ch, out_ch, 3, 1, 1, relu=True)

    def forward(self, x):
        return self.conv(self.up(x))


# ── Fast NST Generator ─────────────────────────────────────────────────────────

class FastNSTGenerator(nn.Module):
    """
    Feed-forward generator for Fast NST.
    Architecture: Encoder → 5x Residual Blocks → Decoder
    Input  : (B, 3, H, W) — content image
    Output : (B, 3, H, W) — stylized image
    """
    def __init__(self, n_residual=5):
        super().__init__()

        # ── Encoder (Downsampling) ─────────────────────────────────
        self.encoder = nn.Sequential(
            ConvNormReLU(3,   32,  9, 1, 4),   # 9x9 conv, large receptive field
            ConvNormReLU(32,  64,  3, 2, 1),   # stride 2 → H/2
            ConvNormReLU(64, 128,  3, 2, 1),   # stride 2 → H/4
        )

        # ── Residual Blocks (Style Transformation) ────────────────
        self.residuals = nn.Sequential(
            *[ResidualBlock(128) for _ in range(n_residual)]
        )

        # ── Decoder (Upsampling) ───────────────────────────────────
        self.decoder = nn.Sequential(
            UpsampleConv(128, 64, scale=2),    # H/4 → H/2
            UpsampleConv(64,  32, scale=2),    # H/2 → H
            nn.Conv2d(32, 3, 9, 1, 4),         # final 9x9 conv
            nn.Tanh()                          # output in [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        return x


# ── Sanity Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model  = FastNSTGenerator(n_residual=5)
    dummy  = torch.randn(1, 3, 256, 256)
    output = model(dummy)

    total_params = sum(p.numel() for p in model.parameters())

    print("=" * 50)
    print("  Fast NST Generator — Sanity Test")
    print("=" * 50)
    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Total params : {total_params:,}")
    print(f"  Output range : [{output.min():.3f}, {output.max():.3f}]")
    assert dummy.shape == output.shape, "Shape mismatch!"
    assert output.min() >= -1.0 and output.max() <= 1.0, "Tanh range error!"
    print("  ALL CHECKS PASSED ✅")
