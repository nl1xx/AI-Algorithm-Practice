# GNN使用torch_geometric包, 一般不超过三层神经网络
# GCN多用于分类问题
import torch
import torch_geometric.nn
from torch import nn, optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

dataset = Planetoid(root='./datasets/Planetoid', name='Cora', transform=NormalizeFeatures())

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')  # 1433
print(f'Number of classes: {dataset.num_classes}')  # 7

data = dataset[0]  # Get the first graph object.
print(data)
# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])


class GCN(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = torch_geometric.nn.GCNConv(dataset.num_features, self.hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(self.hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 可视化
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


model1 = GCN(hidden_channels=16)
out1 = model1(data.x, data.edge_index)
visualize(out1, data.y)


model = GCN(hidden_channels=16)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
model.eval()
out = model(data.x, data.edge_index)
visualize(out, data.y)


# 使用GATConv实现(多头)
# class GAT(nn.Module):
#     def __init__(self, hidden_channels, head):
#         super().__init__()
#         self.hidden_channels = hidden_channels
#         self.head = head
#         self.conv1 = torch_geometric.nn.GATConv(dataset.num_features, self.hidden_layer, self.head)
#         self.conv2 = torch_geometric.nn.GATConv(self.head * self.hidden_channels, dataset.num_classes)
#
#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x
#
#
# model2 = GAT(hidden_channels=8, head=8)
# optimizer2 = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
# criterion2 = torch.nn.CrossEntropyLoss()
#
#
# def trainAndTestGAT():
#     model.train()
#     for i in range(1, 101):
#         optimizer2.zero_grad()
#         out = model2(data.x, data.edge_index)
#         loss = criterion2(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer2.step()
#         print(f"Epoch: {i:03d}, Loss: {loss:.4f}")
#
#     model.eval()
#     with torch.no_grad():
#         out_ = model2(data.x, data.edge_index)
#         pred_ = out_.argmax(dim=1)
#         total_acc_ = pred_[data.test_mask] == data.y[data.test_mask]
#         accuracy_rate = int(total_acc_.sum())/int(data.test_mask.sum())
#         print(f"Accuracy Rate: {accuracy_rate:.4f}")
