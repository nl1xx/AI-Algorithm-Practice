import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import shap

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
        x = x.view(-1, 320)
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

model.eval()
background = next(iter(train_loader))[0][:100].to(device)  # 使用训练集的一部分作为背景数据
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:10].to(device)  # 选择一些测试图像进行解释

# 定义SHAP的DeepExplainer
explainer = shap.DeepExplainer(model, background)
# 对测试图像进行解释
shap_values = explainer.shap_values(test_images)

# 可视化结果
for i in range(len(test_images)):
    # 选择第i个图像，去掉通道维度，得到形状(28, 28)
    single_image = test_images.cpu().numpy()[i, 0]  # 获取单通道图像
    # 选择第一个类别的SHAP值
    # 可以选择特定类别的SHAP值，例如shap_values[class_index][i]
    single_shap_values = shap_values[0][i]  # 获取第0类别的SHAP值(对于分类问题)

    # 显示图像和SHAP值
    shap.image_plot([single_shap_values], single_image[np.newaxis, :, :])
