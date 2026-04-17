import torch
import torch.nn as nn


class VGGLikeExtractor(nn.Module):
    """
    Custom VGG-like CNN.
    cifar_mode=True  → AdaptiveAvgPool2d(1,1)  for 32x32 input
    cifar_mode=False → AdaptiveAvgPool2d(7,7)  for 224x224 input
    """

    def __init__(self, num_classes=10, cifar_mode=True):
        super(VGGLikeExtractor, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        pool_size = 1 if cifar_mode else 7
        fc_input  = 512 * pool_size * pool_size

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
            nn.Flatten(),
            nn.Linear(fc_input, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        features = {}
        features['block1'] = self.block1(x)
        features['block2'] = self.block2(features['block1'])
        features['block3'] = self.block3(features['block2'])
        features['block4'] = self.block4(features['block3'])
        return features
