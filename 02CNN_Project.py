# https://github.com/ki-ljl/cnn-dogs-vs-cats
# 使用CNN实现猫狗区分

import torch
from PIL import Image
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


# 定义CNN网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)
        self.out = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.size())
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        # x = F.log_softmax(x, dim=1)
        return x


# 准备数据集
def Myloader(path):
    return Image.open(path).convert('RGB')


def init_process(path, lens):
    data = []
    name = find_label(path)  # 1(dog)还是0(cat)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])  # path % i: 将i的值插入到path字符串中的%d占位符处

    return data


class MyDataset(Dataset):  # 重写Dataset类
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


# 看是dog还是cat这两个单词中的哪一个
def find_label(str):  # 根据文件路径查找label, 参数代表路径
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):  # 末尾开始向前查找
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last]
    if name == 'dog':
        return 1
    else:
        return 0


def load_data():
    print('data processing...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])
    path1 = './datasets/DATA_DOG&CAT/training_data/cats/cat.%d.jpg'
    data1 = init_process(path1, [0, 500])
    path2 = './datasets/DATA_DOG&CAT/training_data/dogs/dog.%d.jpg'
    data2 = init_process(path2, [0, 500])
    path3 = './datasets/DATA_DOG&CAT/testing_data/cats/cat.%d.jpg'
    data3 = init_process(path3, [1000, 1200])
    path4 = './datasets/DATA_DOG&CAT/testing_data/dogs/dog.%d.jpg'
    data4 = init_process(path4, [1000, 1200])
    data = data1 + data2 + data3 + data4   # 1400
    # shuffle
    np.random.shuffle(data)
    # train, val, test = 900 + 200 + 300
    train_data, val_data, test_data = data[:900], data[900:1100], data[1100:]

    train_data = MyDataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=0)

    val_data = MyDataset(val_data, transform=transform, loader=Myloader)
    Val = DataLoader(dataset=val_data, batch_size=50, shuffle=True, num_workers=0)

    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=50, shuffle=True, num_workers=0)

    return Dtr, Val, Dte


dtr, val, dte = load_data()
model = CNN().to('cuda')
loss_function = nn.CrossEntropyLoss().to('cuda')


# 开始训练
def train():
    epoch = 100
    record_every = 10

    loss_list = []
    lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for i in range(epoch):
        for data in dtr:
            img, label = data
            img = img.to('cuda')
            label = label.to('cuda')
            img_hat = model(img)
            loss = loss_function(img_hat, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % record_every == 0:
                loss_list.append(loss.item())
        if (i+1) % 10 == 0:
            print("current epoch: {}, current loss: {}".format(i + 1, loss.item()))

    # plt.plot(loss_list)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()


# 开始测试
def test():
    model.eval()
    total_test_loss = 0
    total_accuracy_item = 0
    with torch.no_grad():
        for data in dte:
            img, label = data
            img = img.to('cuda')
            label = label.to('cuda')
            output_img = model(img)
            loss = loss_function(output_img, label)
            total_test_loss += loss.item()
            # argmax(1): 按行遍历找最大值, 返回第几位(从0开始)
            accuracy = (output_img.argmax(1) == label).sum()  # 计算预测正确的样本数量
            total_accuracy_item += accuracy
    print("Accuracy: {}".format(total_accuracy_item/len(dte)))


if __name__ == '__main__':
    train()
    test()
