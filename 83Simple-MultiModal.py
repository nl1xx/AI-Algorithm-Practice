import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 10  # CIFAR-10
embedding_dim = 128
hidden_dim = 256
sequence_length = 10  # 文本序列长度
vocab_size = 100  # 词汇表大小（这里简单起见, 使用随机生成的字符）


# 文本数据集类
class TextDataset(Dataset):
    def __init__(self, size, vocab_size, sequence_length):
        self.size = size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 随机生成一个文本序列（这里只是随机生成索引, 实际应用中应使用真实文本数据）
        text = torch.randint(0, self.vocab_size, (self.sequence_length,))
        return text


# 多模态数据集类
class MultimodalDataset(Dataset):
    def __init__(self, image_dataset, text_dataset):
        self.image_dataset = image_dataset
        self.text_dataset = text_dataset

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        text = self.text_dataset[idx]
        return image, text, label


# 加载CIFAR-10图像数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_image_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_image_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 创建文本数据集
train_text_dataset = TextDataset(len(train_image_dataset), vocab_size, sequence_length)
test_text_dataset = TextDataset(len(test_image_dataset), vocab_size, sequence_length)

# 创建多模态数据集
train_dataset = MultimodalDataset(train_image_dataset, train_text_dataset)
test_dataset = MultimodalDataset(test_image_dataset, test_text_dataset)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 多模态模型
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, hidden_dim):
        super(MultimodalModel, self).__init__()
        # 图像处理部分
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 文本处理部分
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 融合和分类部分
        self.fc2 = nn.Linear(120 + hidden_dim, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, images, texts):
        # 图像特征提取
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # 文本特征提取
        embedded = self.embedding(texts)
        _, (hidden, _) = self.rnn(embedded)
        text_features = hidden[-1]  # 取最后一层隐藏状态
        # 特征融合
        combined_features = torch.cat((x, text_features), dim=1)
        x = F.relu(self.fc2(combined_features))
        x = self.fc3(x)
        return x


# 初始化模型、损失函数和优化器
model = MultimodalModel(num_classes, vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, texts, labels in train_loader:
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images, texts)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(images, texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'multimodal_model.pth')
