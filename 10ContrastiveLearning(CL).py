import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128)
        )

    def forward(self, x1, x2):
        out1 = self.conv(x1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc(out1)

        out2 = self.conv(x2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc(out2)

        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# 重写CIFAR10类
class PairedDataset(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(PairedDataset, self).__init__(root, train, transform, target_transform)
        self.samples = list(range(len(self)))

    def __getitem__(self, index):
        img1, target = self.data[index], self.targets[index]
        img1 = Image.fromarray(img1)

        # 确保img2与img1不同
        index2 = index
        while index2 == index:
            index2 = np.random.randint(0, len(self))
        img2, _ = self.data[index2], self.targets[index2]
        img2 = Image.fromarray(img2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # 生成标签，如果两个图像属于同一类别，则标签为0，否则为1
        label = 0 if target == _ else 1

        return img1, img2, label

    def __len__(self):
        return len(self.data)


# 加载数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = PairedDataset(root='./xiaotudui/torchvisionDataset', train=True, transform=transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)

# 构建网络
model = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 100

for epoch in range(num_epochs):
    for batch_idx, (img1, img2, label) in enumerate(train_dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, num_epochs, batch_idx + 1, len(train_dataloader), loss.item()))
