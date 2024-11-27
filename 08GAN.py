import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

# 创建文件夹
os.makedirs("./images/ganGeneration/", exist_ok=True)

# 超参数配置
n_epochs = 50
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 2
latent_dim = 100
img_size = 28
channels = 1
sample_interval = 500

img_shape = (channels, img_size, img_size)
img_area = np.prod(img_shape)

# 设置cuda
cuda = True if torch.cuda.is_available() else False

# mnist数据集下载
mnist = datasets.MNIST(
    root='./datasets/', train=True, download=True, transform=transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ),
)

# 配置数据到加载器
dataloader = DataLoader(
    mnist,
    batch_size=batch_size,
    shuffle=True,
)


# 定义判别器 Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_area, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# 定义生成器 Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, img_area),
            nn.Tanh()
        )

    def forward(self, z):
        imgs = self.model(z)
        imgs = imgs.view(imgs.size(0), *img_shape)
        return imgs


generator = Generator()
discriminator = Discriminator()

loss_function = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_function = loss_function.cuda()

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.view(imgs.size(0), -1)
        real_img = Variable(imgs).cuda()
        real_label = Variable(torch.ones(imgs.size(0), 1)).cuda()
        fake_label = Variable(torch.zeros(imgs.size(0), 1)).cuda()

        # 训练判别器D
        real_out = discriminator(real_img)
        loss_real_D = loss_function(real_out, real_label)
        real_scores = real_out
        z = Variable(torch.randn(imgs.size(0), latent_dim)).cuda()
        fake_img = generator(z).detach()
        fake_out = discriminator(fake_img)
        loss_fake_D = loss_function(fake_out, fake_label)
        fake_scores = fake_out
        loss_D = loss_real_D + loss_fake_D
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # 训练生成器G
        z = Variable(torch.randn(imgs.size(0), latent_dim)).cuda()
        fake_img = generator(z)
        output = discriminator(fake_img)
        loss_G = loss_function(output, real_label)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # 打印训练过程中的日志
        if (i + 1) % 10 == 0:
            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D.item():.4f}] "
                  f"[G loss: {loss_G.item():.4f}] [D real: {real_scores.data.mean():.2f}] "
                  f"[D fake: {fake_scores.data.mean():.2f}]")
        # 保存训练过程中的图像
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(fake_img.data[:25], f"./images/ganGeneration/{batches_done}.png", nrow=5, normalize=True)

