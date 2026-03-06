import torch
import torch.nn as nn


class VGGLikeExtractor(nn.Module):
    """
    Custom VGG-like CNN trained from scratch.
    NO pretrained VGG16/VGG19 used.
    """

    def __init__(self, num_classes=10):
        super(VGGLikeExtractor, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 112x112
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 56x56
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 28x28
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 14x14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """Returns intermediate feature maps for NST loss computation."""
        features = {}
        features['block1'] = self.block1(x)
        features['block2'] = self.block2(features['block1'])
        features['block3'] = self.block3(features['block2'])
        features['block4'] = self.block4(features['block3'])
        return features


if __name__ == "__main__":
    model = VGGLikeExtractor(num_classes=10)
    dummy = torch.randn(1, 3, 224, 224)

    out = model(dummy)
    print("Output shape:", out.shape)   # Expected: torch.Size([1, 10])

    feats = model.get_feature_maps(dummy)
    for k, v in feats.items():
        print(f"{k}: {v.shape}")
