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


class ASPP_module(nn.ModuleList):
    def __init__(self, in_channels, out_channels, dilation_list=[1, 6, 12, 18]):
        super(ASPP_module, self).__init__()
        self.dilation_list = dilation_list
        for dia_rate in self.dilation_list:
            self.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1 if dia_rate == 1 else 3, dilation=dia_rate,
                              padding=0 if dia_rate == 1 else dia_rate),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )

    def forward(self, x):
        outputs = []
        for aspp_module in self:
            outputs.append(aspp_module(x))
        return torch.cat(outputs, 1)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.ASPP_module = ASPP_module(512, 256, dilation_list=[1, 6, 12, 18])
        self.backbone = VGG13()
        self.final = nn.Sequential(
            nn.Conv2d(256 * 4 + 256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.num_classes, kernel_size=1)
        )
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1)),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.backbone(x)[-1]
        x_1 = self.ASPP_module(x)
        x_2 = nn.functional.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                        align_corners=True)
        x = torch.cat([x_1, x_2], 1)
        x = nn.functional.interpolate(input=x, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.final(x)
        return x
