import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


class ExpertModel(nn.Module):
    def __init__(self, input_dim):
        super(ExpertModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 13)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 门控网络
class GatingNetwork(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(GatingNetwork, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_output is the output tensor from multi-head self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


# 混合专家模型
class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, num_experts, top_k):
        super(MixtureOfExperts, self).__init__()
        # 专家列表，根据num_experts生成对应个数的专家模型
        self.experts = nn.ModuleList([ExpertModel(input_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts, top_k)

    def forward(self, x):
        gating_output, indices = self.gating_network(x)
        final_output = torch.zeros_like(x)

        flatten_x = x.view(-1, x.size(-1))
        flatten_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flatten_mask = expert_mask.view(-1)
            if flatten_mask.any():
                input = flatten_x[flatten_mask]
                output = expert(input)

                gating_score = flatten_gating_output[flatten_mask, i].unsqueeze(1)
                weighted_output = gating_score * output

                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output


# 加载红酒数据集
data = load_wine()
X, y = data.data, data.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 初始化模型、损失函数和优化器
input_dim = X_train.shape[1]
num_experts = 5
top_k = 2
model = MixtureOfExperts(input_dim, num_experts, top_k)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 评估模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

# 将PyTorch张量转换为NumPy数组，以便使用sklearn的函数
predicted_numpy = predicted.cpu().numpy()
y_test_numpy = y_test.cpu().numpy()

# 计算精确度、召回率和F1分数
precision = precision_score(y_test_numpy, predicted_numpy, average='macro')
recall = recall_score(y_test_numpy, predicted_numpy, average='macro')
f1 = f1_score(y_test_numpy, predicted_numpy, average='macro')

# 打印结果
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')
