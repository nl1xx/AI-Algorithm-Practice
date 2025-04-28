import torch
import torch.nn as nn


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.stage_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.stage_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.stage_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1, padding=1),
        )

        self.stage_5 = nn.Sequential(
            # 空洞卷积
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
        )

    def forward(self, x):
        x = x.float()
        x1 = self.stage_1(x)
        x2 = self.stage_2(x1)
        x3 = self.stage_3(x2)
        x4 = self.stage_4(x3)
        x5 = self.stage_5(x4)
        return [x1, x2, x3, x4, x5]


class DeepLabV1(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV1, self).__init__()
        # 前13层是VGG16的前13层,分为5个stage
        self.num_classes = num_classes
        self.backbone = VGG13()

        self.stage_1 = nn.Sequential(
            # 空洞卷积
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 调用VGG16的前13层 VGG13
        x = self.backbone(x)[-1]
        x = self.stage_1(x)
        x = nn.functional.interpolate(input=x, scale_factor=8, mode='bilinear')
        x = self.final(x)
        return x
    