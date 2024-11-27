# https://github.com/yzwxx/vae-celebA
# 使用VAE和CelebA数据集实现

import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image

IMAGE_SIZE = 150
LATENT_DIM = 128


directory = 'images/vaeGeneration'
os.makedirs(directory, exist_ok=True)


# VAE模型
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dims = [32, 64, 128, 256, 512]
        self.final_dim = hidden_dims[-1]
        in_channels = 3
        modules = []
        # Encoder
        for i in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(i),
                    nn.LeakyReLU()
                )
            )
            in_channels = i

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, LATENT_DIM)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, LATENT_DIM)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(LATENT_DIM, hidden_dims[-1] * self.size * self.size)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        esp = torch.randn_like(std)
        return mu + esp * std

    def decode(self, z):
        inputs = self.decoder_input(z)
        inputs = inputs.view(-1, self.final_dim, self.size, self.size)
        outputs = self.decoder(inputs)
        outputs = self.final_layer(outputs)
        outputs = transforms.Compose([
            transforms.Resize(size=IMAGE_SIZE, antialias=True),
            transforms.CenterCrop(size=IMAGE_SIZE)
        ])(outputs)
        outputs = outputs.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        outputs = torch.nan_to_num(outputs)
        return outputs

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


model = VAE().to('cuda')


# 数据集
data_train = datasets.CelebA(root="./datasets", download=False, split="train", transform=transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()]
))

data_test = datasets.CelebA(root="./datasets", download=False, split="test", transform=transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()]
))

dataloader_train = DataLoader(dataset=data_train, batch_size=16, shuffle=True)
dataLoader_test = DataLoader(dataset=data_test, batch_size=16 ,shuffle=True)


# 损失函数
def loss_function(x_hat, x, mu, log_var):
    loss1 = F.binary_cross_entropy(x_hat, x, reduction='sum')
    loss2 = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return loss1 + loss2


loss_list = []

# 优化器
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)


# 训练
def train():
    print("----------start training----------")
    epoch = 20
    total_loss = 0
    model.train()
    for i in range(epoch):
        for data in dataloader_train:
            images, _ = data  # We only use the images, not the labels
            images = images.to('cuda')
            output_images, mu, log_var = model(images)  # Get reconstructed images
            loss = loss_function(output_images, images, mu, log_var).to('cuda')  # Use images for loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (i + 1) % 2 == 0:
            print(f"Epoch: {i + 1}, Loss: {loss.item()}")
        average_loss = total_loss / len(dataloader_train)
        loss_list.append(average_loss)

    # Plot the training loss
    plt.plot(loss_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# 测试
def test(epoch):
    print("----------start testing----------")
    model.eval()
    total_loss = 0
    average_test_loss = 0
    with torch.no_grad():
        for epoch_item in range(epoch):
            for i, (data, _) in enumerate(dataLoader_test):
                data = data.to('cuda')
                recon_batch, mu, log_var = model(data)
                total_loss += loss_function(recon_batch, data, mu, log_var).item()
                # ???
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(16, 3, IMAGE_SIZE, IMAGE_SIZE)[:n]])
                    save_image(comparison.cpu(),
                               f'{directory}/reconstruction_{str(epoch_item + 1)}.png', nrow=n)

    average_test_loss = total_loss / len(dataLoader_test)


if __name__ == '__main__':
    epoch = 20
    train()
    test(epoch)
    for epoch_item in range(epoch):
        # 为什么还要再生成图片
        with torch.no_grad():
            sample = torch.randn(64, LATENT_DIM).to('cuda')
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
                       f'{directory}/sample_{str(epoch_item + 1)}.png')
