import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module
    """
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPPModule, self).__init__()
        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        # image-level pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(dilation_rates) + 1) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[2:]
        res = []
        for branch in self.branches:
            res.append(branch(x))
        img_feat = self.image_pool(x)
        img_feat = F.interpolate(img_feat, size=size, mode='bilinear', align_corners=False)
        res.append(img_feat)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV2(nn.Module):
    """
    DeepLab v2 with ResNet-101 backbone
    """
    def __init__(self, num_classes, pretrained_backbone=True):
        super(DeepLabV2, self).__init__()
        # Load pre-trained ResNet-101
        resnet = models.resnet101(pretrained=pretrained_backbone)
        # Remove last classifier and pooling
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        # Modify layer3 and layer4 to use atrous convolutions
        self.layer1 = resnet.layer1  # output stride 4
        self.layer2 = resnet.layer2  # output stride 8
        self.layer3 = self._make_dilated(resnet.layer3, dilation=2)  # output stride 16
        self.layer4 = self._make_dilated(resnet.layer4, dilation=4)  # output stride 16

        # ASPP with dilation rates [6, 12, 18, 24]
        self.aspp = ASPPModule(in_channels=2048, out_channels=256, dilation_rates=[6, 12, 18, 24])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def _make_dilated(self, layer, dilation):
        # Modify Bottleneck blocks to use dilation and remove downsampling stride
        for block in layer:
            # adjust conv2 (3x3) dilation and padding
            block.conv2.dilation = (dilation, dilation)
            block.conv2.padding = (dilation, dilation)
            # ensure conv2 stride is 1
            block.conv2.stride = (1, 1)
            # if downsample exists, set its stride to 1
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        return layer

    def forward(self, x):
        # Backbone
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Classifier
        x = self.classifier(x)
        # Upsample to input size
        # x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x


if __name__ == "__main__":
    model = DeepLabV2(num_classes=21, pretrained_backbone=True)
    model.eval()
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)
    print("Output shape:", output.shape)
