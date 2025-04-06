import random
import torch
from torchvision import datasets
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import warnings

warnings.filterwarnings("ignore")
# Set seeds
random.seed(2024)
torch.manual_seed(2024)
np.random.seed(2024)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PermutedMNIST(datasets.MNIST):
    def __init__(self, root="./my_datasets", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28
        if self.train:
            # 拼接
            self.data = torch.stack([img.float().view(-1)[permute_idx] / 255 for img in self.data])
        else:
            self.data = torch.stack([img.float().view(-1)[permute_idx] / 255 for img in self.data])

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.train_labels[index]
        else:
            img, target = self.data[index], self.test_labels[index]
        return img.view(1, 28, 28), target

    def get_sample(self, sample_size):
        random.seed(2024)
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img.view(1, 28, 28) for img in self.data[sample_idx]]


def worker_init_fn(worker_id):
    # 确保每个worker的随机种子一致
    random.seed(2024 + worker_id)
    np.random.seed(2024 + worker_id)


def get_permute_mnist(num_task, batch_size):
    random.seed(2024)
    train_loader = {}
    test_loader = {}
    root_dir = './my_datasets/PermutedMNIST'
    os.makedirs(root_dir, exist_ok=True)

    for i in range(num_task):
        permute_idx = list(range(28 * 28))
        random.shuffle(permute_idx)

        train_dataset_path = os.path.join(root_dir, f'train_dataset_{i}.pt')
        test_dataset_path = os.path.join(root_dir, f'test_dataset_{i}.pt')

        if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
            train_dataset = torch.load(train_dataset_path)
            test_dataset = torch.load(test_dataset_path)
        else:
            train_dataset = PermutedMNIST(train=True, permute_idx=permute_idx)
            test_dataset = PermutedMNIST(train=False, permute_idx=permute_idx)
            torch.save(train_dataset, train_dataset_path)
            torch.save(test_dataset, test_dataset_path)

        train_loader[i] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, pin_memory=True)
        test_loader[i] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, pin_memory=True)

    return train_loader, test_loader


class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, num_classes_per_task=10, hidden_size=[400, 400, 400]):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # 初始化类别计数器
        self.total_classes = num_classes_per_task
        self.num_classes_per_task = num_classes_per_task

        # 定义网络结构
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc_before_last = nn.Linear(hidden_size[1], hidden_size[2])

        self.fc_out = nn.Linear(hidden_size[2], self.total_classes)

    def forward(self, input, task_id=-1):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_before_last(x))
        x = self.fc_out(x)
        return x


class Baseline:
    def __init__(self, num_classes_per_task=10, num_tasks=10, batch_size=256, epochs=2, neurons=0):
        self.num_classes_per_task = num_classes_per_task
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.epochs = epochs
        self.neurons = neurons
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 28 * 28

        # Initialize model
        self.model = MLP(num_classes_per_task=self.num_classes_per_task).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Get dataset
        self.train_loaders, self.test_loaders = get_permute_mnist(self.num_tasks, self.batch_size)

    def evaluate(self, test_loader, task_id):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                # Move data to GPU in batches
                images = images.view(-1, self.input_size)
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images, task_id)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total

    def train_task(self, train_loader, optimizer, task_id):
        self.model.train()
        for images, labels in train_loader:
            images = images.view(-1, self.input_size)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            outputs = self.model(images, task_id)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def run(self):
        all_avg_acc = []

        for task_id in range(self.num_tasks):
            train_loader = self.train_loaders[task_id]
            self.model = self.model.to(self.device)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
            for epoch in range(self.epochs):
                self.train_task(train_loader, optimizer, task_id)
            task_acc = []
            for eval_task_id in range(task_id + 1):
                accuracy = self.evaluate(self.test_loaders[eval_task_id], eval_task_id)
                task_acc.append(accuracy)
            mean_avg = np.round(np.mean(task_acc), 2)

            print(f"Task {task_id}: Task Acc = {task_acc},AVG={mean_avg}")
            all_avg_acc.append(mean_avg)
        avg_acc = np.mean(all_avg_acc)
        print(f"Task AVG Acc: {all_avg_acc},AVG = {avg_acc}")


if __name__ == '__main__':
    print('Baseline' + "=" * 50)
    random.seed(2024)
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    baseline = Baseline(num_classes_per_task=10, num_tasks=3, batch_size=256, epochs=2)
    baseline.run()


# EWC
class EWC:
    def __init__(self, num_classes_per_task=10, num_tasks=10, batch_size=256, epochs=2):
        self.num_classes_per_task = num_classes_per_task
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 28 * 28

        # Initialize model
        self.model = MLP(num_classes_per_task=self.num_classes_per_task).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision
        self.importance_dict = {}
        self.previous_params = {}
        self.lambda_ = 10000

        self.train_loaders, self.test_loaders = get_permute_mnist(self.num_tasks, self.batch_size)

    def evaluate(self, test_loader, task_id):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, self.input_size)
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images, task_id)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total

    def train_task(self, train_loader, optimizer, task_id):
        self.model.train()
        for images, labels in train_loader:
            images = images.view(-1, self.input_size)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            outputs = self.model(images, task_id)
            if task_id > 0:
                loss = self.ewc_multi_objective_loss(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def ewc_compute_importance(self, data_loader, task_id):
        importance_dict = {name: torch.zeros_like(param, device=self.device) for name, param in
                           self.model.named_parameters() if 'task' not in name}
        self.model.eval()
        for images, labels in data_loader:
            images = images.view(-1, self.input_size)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.model.zero_grad()

            outputs = self.model(images, task_id=task_id)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            for name, param in self.model.named_parameters():
                if name in importance_dict and param.requires_grad:
                    importance_dict[name] += param.grad ** 2 / len(data_loader)
        return importance_dict

    def update(self, dataset, task_id):
        importance_dict = self.ewc_compute_importance(dataset, task_id)

        for name in importance_dict:
            if name in self.importance_dict:
                self.importance_dict[name] += importance_dict[name]
            else:
                self.importance_dict[name] = importance_dict[name]

        for name, param in self.model.named_parameters():
            self.previous_params[name] = param.clone().detach()

    # 多目标优化中的正则化损失计算
    def ewc_multi_objective_loss(self, outputs, labels):
        regularization_loss = 0.0
        # 遍历模型的所有参数, 返回一个生成器, 包含每个参数的名称和值
        for name, param in self.model.named_parameters():
            if 'task' not in name and name in self.importance_dict and name in self.previous_params:
                importance = self.importance_dict[name]
                previous_param = self.previous_params[name]
                regularization_loss += (importance * (param - previous_param).pow(2)).sum()

        loss = self.criterion(outputs, labels)
        total_loss = loss + self.lambda_ * regularization_loss
        return total_loss

    def run(self):
        all_avg_acc = []
        for task_id in range(self.num_tasks):
            train_loader = self.train_loaders[task_id]
            self.model = self.model.to(self.device)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
            for epoch in range(self.epochs):
                self.train_task(train_loader, optimizer, task_id)
            self.update(train_loader, task_id)

            task_acc = []
            for eval_task_id in range(task_id + 1):
                accuracy = self.evaluate(self.test_loaders[eval_task_id], eval_task_id)
                task_acc.append(accuracy)
            mean_avg = np.round(np.mean(task_acc), 2)
            print(f"Task {task_id}: Task Acc = {task_acc},AVG={mean_avg},")
            all_avg_acc.append(mean_avg)
        avg_acc = np.mean(all_avg_acc)
        print(f"Task AVG Acc: {all_avg_acc},AVG = {avg_acc}")


if __name__ == '__main__':
    print('EWC' + "=" * 50)
    # 每次循环前重置随机种子
    random.seed(2024)
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ewc = EWC(num_classes_per_task=10, num_tasks=5, batch_size=256, epochs=2)
    ewc.run()
