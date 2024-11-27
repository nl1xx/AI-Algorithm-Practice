# DCGAN + CelebA数据集
import os

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


noise_size = 100


file_src = './images/GANGeneration'
os.makedirs(file_src, exist_ok=True)


# 生成器G (CBR×4+CT)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_size, out_channels=64 * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            # 输出特征图尺寸[64*8, 4, 4]

            nn.ConvTranspose2d(in_channels=64 * 8, out_channels=64 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            # 输出特征图尺寸[64*4, 8, 8]

            nn.ConvTranspose2d(in_channels=64 * 4, out_channels=64 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            # 输出特征图尺寸[64*2, 16, 16]

            nn.ConvTranspose2d(in_channels=64 * 2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 输出特征图尺寸[64, 32, 32]

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 输出特征图尺寸[3, 64, 64]
            )

    def forward(self, x):
        return self.net(x)


# 判别器D
# CL+CBL×3+CS
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64 * 2, out_channels=64 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64 * 4, out_channels=64 * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64 * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 参数初始化
def parameter_init(m):
    # 获取当前层的类名
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0)


GModel = Generator().to('cuda')
DModel = Discriminator().to('cuda')
GModel.apply(parameter_init)
DModel.apply(parameter_init)


# 数据集
data_train = datasets.CelebA(root="./datasets", split="train", download=False, transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

data_test = datasets.CelebA(root="./datasets", split="test", download=False, transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

dataLoader_train = DataLoader(dataset=data_train, shuffle=True, batch_size=128)
dataLoader_test = DataLoader(dataset=data_test, shuffle=True, batch_size=128)


# 损失函数和优化器
loss_function = nn.BCELoss().to('cuda')
# lr=1e-3在这10轮始终是loss_G=0, loss_D=100
lr = 0.5
optimizer_G = optim.Adam(GModel.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(DModel.parameters(), lr=lr, betas=(0.5, 0.999))


# 训练
true_label = torch.ones(128).to('cuda')
false_label = torch.zeros(128).to('cuda')


def train(epoch):
    GModel.train()
    DModel.train()
    for epoch_item in range(epoch):
        for i, (image, label) in enumerate(dataLoader_train):
            # 训练生成器G
            noise = torch.rand(128, noise_size, 1, 1, device='cuda')
            output = GModel(noise)
            fake_labels = DModel(output).view(-1)
            loss_G = loss_function(fake_labels, true_label)
            loss_G_mean = loss_G.mean().item()
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # 训练判别器D
            if i % 2 == 0:
                optimizer_D.zero_grad()
                image = image.to('cuda')
                output_true = DModel(image).view(-1)
                loss_t = loss_function(output_true, true_label)
                output_fake = DModel(output.detach()).view(-1)
                loss_f = loss_function(output_fake, false_label)
                loss_D = loss_t + loss_f
                loss_D_mean = loss_D.mean().item()
                loss_D.backward()
                optimizer_D.step()

            if i % 10 == 0:
                # print("epoch:%i/[%i] iter:%i/[1550]" % (epoch_item, epoch, i), '|', "loss_D:%f" % loss_D_mean, '|',
                #       "loss_G:%f" % loss_G_mean)
                iter = epoch_item * 1550 + i
                plt.scatter(iter, loss_G_mean, c='r', s=5)
                plt.scatter(iter, loss_D_mean, c='g', s=5)

        if (epoch_item + 1) % 2 == 0:
            print("Epoch: {}, Loss_D: {}, Loss_G: {}".format(epoch_item + 1, loss_D.item(), loss_G.item()))
    plt.show()


# 测试
def test(epoch):
    noise = torch.randn(128, noise_size, 1, 1, device='cuda')
    GModel.eval()
    DModel.eval()
    with torch.no_grad():
        for epoch_item in range(epoch):
            output = GModel(noise)
            output = output.detach().cpu()

            # ???
            concate1 = torch.concat((output[0].permute(1, 2, 0), output[2].permute(1, 2, 0)), dim=1)
            concate2 = torch.concat((output[7].permute(1, 2, 0), output[6].permute(1, 2, 0)), dim=1)
            concate3 = torch.concat((concate1, concate2), dim=0)
            plt.imshow(concate3)
            plt.savefig('./images/GANGeneration/composed_%i.png' % epoch_item)


if __name__ == '__main__':
    epoch = 10
    train(epoch)
    test(epoch)
