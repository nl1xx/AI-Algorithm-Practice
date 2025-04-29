import torch
import torch.nn as nn


class DeepLabv1(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv1, self).__init__()

        # Input: 224x224x3

        # Conv2d+ReLU x2 (k3x3, s1, p1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # Maxpool (k3, s2, p1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2d+ReLU x2 (k3x3, s1, p1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # Maxpool (k3, s2, p1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2d+ReLU x3 (k3x3, s1, p1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # Maxpool (k3, s2, p1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2d+ReLU x3 (k3x3, s1, p1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # Maxpool (k3, s1, p1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Conv2d+ReLU x3 (k3x3, s1, r2, p2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        # Maxpool (k3, s1, p1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Avgpool (k3, s1, p1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Conv2d (FC1)+ReLU (k3x3, s1, r12, p12)
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.ReLU(inplace=True)
        )
        # Dropout
        self.dropout1 = nn.Dropout2d(p=0.5)

        # Conv2d (FC2)+ReLU (k1x1, s1)
        self.fc7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        # Dropout
        self.dropout2 = nn.Dropout2d(p=0.5)

        # Conv2d (k1x1, s1)
        self.fc8 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1)

        # Upsample x8
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.pool5(x)

        x = self.avgpool(x)

        x = self.fc6(x)
        x = self.dropout1(x)

        x = self.fc7(x)
        x = self.dropout2(x)

        x = self.fc8(x)

        x = self.upsample(x)

        return x


num_classes = 21
model = DeepLabv1(num_classes)
print(model)
