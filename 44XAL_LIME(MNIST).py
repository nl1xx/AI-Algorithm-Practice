# LIME
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, 320)
        x = self.fc_layers(x)
        return x

model = Net().to('cuda')
train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = datasets.MNIST('./datasets', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
batch_size = 256
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
num_epochs = 5
device = torch.device('cuda')

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)


def predict(images):
    model.eval()
    # LIME 生成的图像是 (batch_size, height, width, channels)，需要转换为 (batch_size, channels, height, width)
    images = images.astype(np.float32)  # 确保数据类型为 float32
    images = np.transpose(images, (0, 3, 1, 2))  # 转换为 (batch_size, channels, height, width)
    images = images[:, :1, :, :]  # 选择单通道 (batch_size, 1, height, width)
    images = torch.tensor(images).to(device)  # 转换为 Tensor 并移动到设备上

    with torch.no_grad():
        outputs = model(images)  # 传递给模型
    return outputs.cpu().numpy()  # 返回CPU上的numpy数组

# 获取一个测试样本
test_images, test_labels = next(iter(test_loader))
image = test_images[0].cpu().numpy().squeeze()  # 获取第一个测试图像，形状为 (28, 28)
label = test_labels[0].item()  # 获取第一个图像的标签

# 将单通道图像扩展为三通道，因为 LIME 需要 RGB 图像
image_rgb = np.stack((image,) * 3, axis=-1)  # 形状变为 (28, 28, 3)

explainer = lime_image.LimeImageExplainer()

# 生成解释
explanation = explainer.explain_instance(
    image_rgb,  # 输入图像
    predict,  # 预测函数
    labels=(label,),  # 指定感兴趣的类别
    top_labels=1,  # 选择最重要的类别
    hide_color=0,  # 背景颜色
    num_samples=1000  # 采样的扰动样本数量
)

# 获取解释结果和掩码
temp, mask = explanation.get_image_and_mask(
    label,  # 指定类别
    positive_only=True,  # 仅显示正贡献区域
    num_features=5,  # 显示的最重要特征数量
    hide_rest=False  # 是否隐藏其余部分
)

# 显示图像和解释
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')  # 原始图像
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("LIME Explanation")
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))  # 解释叠加图像
plt.axis('off')

plt.show()