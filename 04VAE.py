import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        # Encoder部分：从输入到隐变量的均值(mu)和方差(log_var)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入 -> 隐层
        self.fc_mu = nn.Linear(hidden_dim, z_dim)    # 隐层 -> 均值
        self.fc_log_var = nn.Linear(hidden_dim, z_dim)  # 隐层 -> 方差

        # Decoder部分：从隐变量还原到输入空间
        self.fc_z = nn.Linear(z_dim, hidden_dim)  # 隐变量 -> 隐层
        self.fc_out = nn.Linear(hidden_dim, input_dim)  # 隐层 -> 输出

    # 编码器：输入->隐变量的均值和方差
    def encoder(self, x):
        h = F.relu(self.fc1(x))  # 经过隐藏层并使用ReLU激活函数
        mu = self.fc_mu(h)       # 计算均值
        log_var = self.fc_log_var(h)  # 计算方差
        return mu, log_var

    # 重参数化技巧：通过均值和方差来采样隐变量
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)     # 标准正态分布噪声
        return mu + eps * std           # 使用 reparameterization trick

    # 解码器：隐变量 -> 重建的输入
    def decoder(self, z):
        h = F.relu(self.fc_z(z))       # 隐变量输入到隐藏层并使用ReLU激活
        x_hat = torch.sigmoid(self.fc_out(h))  # 使用Sigmoid生成输出
        return x_hat

    # 前向传播：从输入到输出
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 将输入展平为 [batch_size, 784]
        mu, log_var = self.encoder(x)  # 编码输入
        z = self.reparameterize(mu, log_var)  # 通过重参数化采样隐变量
        x_hat = self.decoder(z)  # 解码生成输出
        return x_hat, mu, log_var  # 返回生成的x、均值、方差


def loss_function(x_hat, x, mu, log_var):
    # 计算重建损失
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # 计算KL散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD, BCE, KLD


transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


epochs = 10
loss_list = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(data.size(0), -1)  # 将数据展开为 [batch_size, 784]
        optimizer.zero_grad()

        x_hat, mu, log_var = model(data)  # 前向传播
        loss, BCE, KLD = loss_function(x_hat, data, mu, log_var)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader.dataset)
    loss_list.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")

# 绘制损失曲线
plt.plot(loss_list, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
