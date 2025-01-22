import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, 500)
        x = self.fc_layers(x)
        return x


# 模型及数据加载
model = Net().to('cuda')
train_dataset = datasets.CIFAR10('./xiaotudui/torchvisionDataset', train=True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))

test_dataset = datasets.CIFAR10('./xiaotudui/torchvisionDataset', train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

batch_size = 100
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 优化器和训练参数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
num_epochs = 5
device = torch.device('cuda')


# 训练函数
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
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# 训练模型
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)


# 预测函数（支持 LIME）
def predict(images):
    model.eval()
    if images.ndim == 3:  # 单张图片
        images = images.transpose(2, 0, 1)  # 转换为 (channels, height, width)
        images = np.expand_dims(images, axis=0)  # 添加批次维度
    elif images.ndim == 4:  # 批次图片
        images = images.transpose(0, 3, 1, 2)  # 转换为 (batch_size, channels, height, width)

    images = torch.tensor(images, dtype=torch.float32).to(device)  # 转为 Tensor 并移动到设备上
    with torch.no_grad():
        outputs = model(images)  # 前向传播
    return outputs.cpu().numpy()  # 返回 numpy 格式的结果


# 获取一张测试图片
test_images, test_labels = next(iter(test_loader))
image = test_images[0].cpu().numpy().transpose(1, 2, 0)  # 转换为 (height, width, channels)

# 获取模型的预测结果，获取预测的最大类别
model.eval()
image_tensor = torch.tensor(test_images[0].unsqueeze(0).numpy(), dtype=torch.float32).to(device)
output = model(image_tensor)
predicted_label = output.argmax(dim=1).item()

# 使用 LIME 生成解释，动态传递模型预测的标签
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    image,  # 输入图片
    predict,  # 预测函数
    labels=(predicted_label,),  # 使用模型的预测标签
    top_labels=1,  # 选择最重要的类别
    hide_color=0,  # 背景颜色设置为黑色
    num_samples=2000  # 采样的扰动样本数量
)

# 获取解释结果和掩码
temp, mask = explanation.get_image_and_mask(
    predicted_label,  # 使用预测的标签
    positive_only=False,  # 显示正负贡献区域
    num_features=10,  # 显示的最重要特征数量
    hide_rest=False  # 不隐藏其余部分
)

# 显示图片和解释
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)  # 原始图片
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("LIME Explanation")
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask, color=[1, 0, 0], mode='thick'))  # 叠加解释图
plt.axis('off')

plt.show()
