import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import string

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

# 创建类别相关的文本模板
class_templates = {
    0: [  # airplane
        "an airplane flying in the sky",
        "a jet soaring through the clouds",
        "passenger plane at the airport",
        "military aircraft in flight",
        "propeller plane over mountains"
    ],
    1: [  # automobile
        "a car driving on the highway",
        "red sports car on city street",
        "vintage automobile from the 1950s",
        "electric vehicle charging station",
        "police car with flashing lights"
    ],
    2: [  # bird
        "a small bird perched on a branch",
        "flock of birds migrating south",
        "colorful parrot in the rainforest",
        "eagle soaring high above mountains",
        "hummingbird hovering near flowers"
    ],
    3: [  # cat
        "a cat sleeping on the sofa",
        "kitten playing with yarn ball",
        "tabby cat hunting in the garden",
        "persian cat with fluffy fur",
        "cat sitting on a windowsill"
    ],
    4: [  # deer
        "a deer in the forest clearing",
        "buck with large antlers grazing",
        "doe and fawn in the meadow",
        "deer crossing a country road",
        "stag standing in the snow"
    ],
    5: [  # dog
        "a dog playing fetch in the park",
        "puppy learning new tricks",
        "golden retriever swimming in lake",
        "guard dog protecting the house",
        "small dog wearing a sweater"
    ],
    6: [  # frog
        "a frog sitting on a lily pad",
        "tree frog in the rainforest",
        "frog jumping into the pond",
        "bullfrog with puffed throat",
        "tiny frog on a mushroom"
    ],
    7: [  # horse
        "a horse running in the field",
        "mare with her newborn foal",
        "black stallion galloping freely",
        "horse pulling a carriage",
        "rider jumping over obstacle"
    ],
    8: [  # ship
        "a cargo ship at sea",
        "cruise ship in tropical waters",
        "sailboat gliding on calm lake",
        "fishing boat returning to harbor",
        "navy vessel on military exercise"
    ],
    9: [  # truck
        "a delivery truck on highway",
        "dump truck at construction site",
        "semi-trailer truck long haul",
        "fire truck responding to emergency",
        "pickup truck on dirt road"
    ]
}

# 构建词汇表
# 收集所有单词
all_words = set()
for templates in class_templates.values():
    for sentence in templates:
        # 简单分词, 转换为小写并移除标点
        words = sentence.translate(str.maketrans('', '', string.punctuation)).lower().split()
        all_words.update(words)

all_words.add('<pad>')  # 填充标记

# 创建词汇表映射
word_to_idx = {word: idx for idx, word in enumerate(sorted(all_words))}
vocab_size = len(word_to_idx)

print(f"Vocabulary size: {vocab_size}")
print(f"Sample words: {list(word_to_idx.keys())[:10]}")


# 文本数据集类
class TextDataset(Dataset):
    def __init__(self, labels, templates, word_to_idx, sequence_length):
        """
        labels: 图像对应的标签列表
        templates: 类别到模板的映射字典
        word_to_idx: 单词到索引的映射
        sequence_length: 固定序列长度
        """
        self.labels = labels
        self.templates = templates
        self.word_to_idx = word_to_idx
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        # 随机选择该类别的模板
        sentence = random.choice(self.templates[label])

        # 移除标点、转换为小写
        cleaned_sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
        words = cleaned_sentence.split()

        # 将单词转换为索引
        indices = [self.word_to_idx.get(word, self.word_to_idx['<pad>']) for word in words]

        # 填充或截断序列
        if len(indices) < self.sequence_length:
            # 填充
            indices += [self.word_to_idx['<pad>']] * (self.sequence_length - len(indices))
        else:
            # 截断
            indices = indices[:self.sequence_length]

        return torch.tensor(indices, dtype=torch.long)


# 加载图像数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_image_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_image_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 创建文本数据集
train_text_dataset = TextDataset(
    train_image_dataset.targets,
    class_templates,
    word_to_idx,
    sequence_length
)

test_text_dataset = TextDataset(
    test_image_dataset.targets,
    class_templates,
    word_to_idx,
    sequence_length
)

# 验证文本生成
sample_idx = random.randint(0, len(train_text_dataset) - 1)
sample_text = train_text_dataset[sample_idx]
sample_label = train_image_dataset.targets[sample_idx]
print(f"Sample text tensor (label={sample_label}): {sample_text}")
print("Decoded text:", ' '.join([list(word_to_idx.keys())[idx] for idx in sample_text.numpy() if idx != word_to_idx['<pad>']]))


# 创建多模态数据集
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


train_dataset = MultimodalDataset(train_image_dataset, train_text_dataset)
test_dataset = MultimodalDataset(test_image_dataset, test_text_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = MultimodalModel(num_classes, vocab_size, embedding_dim, hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("\nModel architecture:")
print(model)


def train_model():
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, texts, labels) in enumerate(train_loader):
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

            running_loss += loss.item()

            # 每100个batch打印一次状态
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')

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
        print(f'Test Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%')

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_multimodal_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%')

    print(f'Training completed. Best accuracy: {best_accuracy:.2f}%')


train_model()
