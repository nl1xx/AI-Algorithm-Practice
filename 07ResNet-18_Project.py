# 使用ResNet-18实现CIFAR10数据集的区分

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class ResNetBlock1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, self.kernel_size, self.stride, self.padding)
        self.batchNorm1 = nn.BatchNorm2d(self.out_channel)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding)
        self.batchNorm2 = nn.BatchNorm2d(self.out_channel)

    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(self.batchNorm1(output))
        output = self.conv2(output)
        output = self.batchNorm2(output)
        return F.relu(input + output)


class ResNetBlock2(nn.Module):
    def __init__(self, input_channel, output_channel, stride, kernel_size=3, padding=1):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, self.kernel_size, self.stride[0], self.padding)
        self.batchNorm1 = nn.BatchNorm2d(self.output_channel)
        self.conv2 = nn.Conv2d(self.output_channel, self.output_channel, self.kernel_size, self.stride[1], self.padding)
        self.batchNorm2 = nn.BatchNorm2d(self.output_channel)
        self.extra = nn.Sequential(
            nn.Conv2d(self.input_channel, self.output_channel, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(self.output_channel)
        )

    def forward(self, input):
        x = self.extra(input)
        output = self.conv1(input)
        output = F.relu(self.batchNorm1(output))
        output = self.conv2(output)
        output = self.batchNorm2(output)
        return F.relu(x + output)


class ResNet_18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.maxPooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(ResNetBlock1(64, 64),
                                    ResNetBlock1(64, 64)
                                    )
        self.layer2 = nn.Sequential(ResNetBlock2(64, 128, [2, 1]),
                                    ResNetBlock1(128, 128)
                                    )
        self.layer3 = nn.Sequential(ResNetBlock2(128, 256, [2, 1]),
                                    ResNetBlock1(256, 256)
                                    )
        self.layer4 = nn.Sequential(ResNetBlock2(256, 512, [2, 1]),
                                    ResNetBlock1(512, 512)
                                    )
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.FC = nn.Linear(2048, 1000)

    def forward(self, input):
        output = self.conv1(input)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = output.reshape(input.shape[0], -1)
        return self.FC(output)


data_train = datasets.CIFAR10(root="./xiaotudui/torchvisionDataset", train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                              ]))
data_test = datasets.CIFAR10(root="./xiaotudui/torchvisionDataset", train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize((32, 32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))

data_train = DataLoader(batch_size=128, dataset=data_train)
data_test = DataLoader(batch_size=128, dataset=data_test)

model = ResNet_18().to('cuda')
lr = 1e-3
loss_function = nn.CrossEntropyLoss().to('cuda')
loss_list = []
optimizer = optim.Adam(model.parameters(), lr=lr)
epoch = 10


def train():
    model.train()
    for i in range(epoch):
        train_loss = 0
        for data in data_train:
            img, label = data
            img = img.to('cuda')
            label = label.to('cuda')
            output_img = model(img)
            loss = loss_function(output_img, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if (i + 1) % 2 == 0:
            print("current epoch: {}, loss: {}".format(i + 1, loss.item()))
        loss_list.append(train_loss/len(data_train))

    plt.plot(loss_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def test():
    total_loss = 0
    total_correct_item = 0
    model.eval()
    with torch.no_grad():
        for data in data_test:
            img, label = data
            img = img.to('cuda')
            label = label.to('cuda')
            predict = model(img)
            loss = loss_function(predict, label)
            total_loss += loss.item()
            correct_item = (predict.argmax(1) == label).sum()
            total_correct_item += correct_item
        print("Accuracy: {}".format(total_correct_item/len(data_test)))


if __name__ == '__main__':
    train()
    test()
