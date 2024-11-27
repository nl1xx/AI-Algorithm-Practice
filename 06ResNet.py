from torch import nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 3, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(3)  # 数据归一化处理
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(3)

    def forward(self, x):
        x_ = F.relu(self.conv1(x))
        x_ = self.bn1(x_)
        x_ = self.bn2(self.conv2(x_))
        return x + x_


class ResNet(nn.Module):
    def __init__(self, num_block):
        super().__init__()
        self.num_block = num_block
        # self.ResNetBlock = nn.ModuleList([ResNetBlock() for _ in range(self.num_block)])
        self.ResNet = ResNetBlock()

    def forward(self, inputs):
        for i in range(self.num_block):
            x = self.ResNetBlock(inputs)
            inputs = x
        return inputs
