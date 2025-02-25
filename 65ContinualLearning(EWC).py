import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_accuracy(model, dataloader):
    model = model.eval()
    acc = 0
    for input, target in dataloader:
        o = model(input.to(device))
        acc += (o.argmax(dim=1).long() == target.to(device)).float().mean()
    return acc / len(dataloader)


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu', use_bn=False):
        super(LinearLayer, self).__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU() if act == 'relu' else act
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        if self.use_bn:
            return self.bn(self.act(self.lin(x)))
        return self.act(self.lin(x))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Model(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(Model, self).__init__()
        self.f1 = Flatten()
        self.lin1 = LinearLayer(num_inputs, num_hidden, use_bn=True)
        self.lin2 = LinearLayer(num_hidden, num_hidden, use_bn=True)
        self.lin3 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(self.f1(x))))


mnist_train = datasets.MNIST("./my_datasets", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./my_datasets", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

f_mnist_train = datasets.FashionMNIST("./my_datasets", train=True, download=True, transform=transforms.ToTensor())
f_mnist_test = datasets.FashionMNIST("./my_datasets", train=False, download=True, transform=transforms.ToTensor())
f_train_loader = DataLoader(f_mnist_train, batch_size=100, shuffle=True)
f_test_loader = DataLoader(f_mnist_test, batch_size=100, shuffle=False)


EPOCHS = 4
lr = 0.001
weight = 100000
accuracies = {}
criterion = nn.CrossEntropyLoss()

# 在A上训练模型
model = Model(28 * 28, 100, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr)
model.train()
for epoch in range(EPOCHS):
    for input, target in tqdm(train_loader):
        output = model(input.to(device))
        loss = criterion(output, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

accuracies['mnist_initial'] = get_accuracy(model, test_loader)


# EWC
def estimate_ewc_params(model, train_ds, batch_size=100, num_batch=300, estimate_type='true'):
    # 记录模型参数在完成第一个任务训练后的值
    estimated_mean = {}

    for param_name, param in model.named_parameters():
        estimated_mean[param_name] = param.data.clone()

    estimated_fisher = {}
    dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

    for n, p in model.named_parameters():
        estimated_fisher[n] = torch.zeros_like(p)

    model.eval()
    for i, (input, target) in enumerate(dl):
        if i > num_batch:
            break
        model.zero_grad()
        output = model(input.to(device))
        # https://www.inference.vc/on-empirical-fisher-information/
        if estimate_type == 'empirical':
            # Empirical Fisher
            label = target.to(device)
        else:
            # True Fisher
            label = output.max(1)[1]

        loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        loss.backward()

        # 累积所有的梯度
        for n, p in model.named_parameters():
            estimated_fisher[n].data += p.grad.data ** 2 / len(dl)

    estimated_fisher = {n: p for n, p in estimated_fisher.items()}
    return estimated_mean, estimated_fisher


def ewc_loss(model, weight, estimated_fishers, estimated_means):
    losses = []
    for param_name, param in model.named_parameters():
        estimated_mean = estimated_means[param_name]
        estimated_fisher = estimated_fishers[param_name]
        losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())

    return (weight / 2) * sum(losses)


# Compute Fisher and mean parameters for EWC loss
estimated_mean, estimated_fisher = estimate_ewc_params(model, mnist_train)

# 在B上训练
# 不使用EWC
for epoch in range(EPOCHS):
    for input, target in tqdm(f_train_loader):
        optimizer.zero_grad()
        output = model(input.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
accuracies['mnist_no_EWC'] = get_accuracy(model, test_loader)
accuracies['f_mnist_no_EWC'] = get_accuracy(model, f_test_loader)
print(accuracies)
print("--"*20)

for epoch in range(EPOCHS):
    for input, target in tqdm(f_train_loader):
        output = model(input.to(device))
        ewc_penalty = ewc_loss(model, weight, estimated_fisher, estimated_mean)
        task_loss = criterion(output, target.to(device))
        loss = ewc_penalty + task_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

accuracies['mnist_EWC'] = get_accuracy(model, test_loader)
accuracies['f_mnist_EWC'] = get_accuracy(model, f_test_loader)
print(accuracies['mnist_EWC'], accuracies['f_mnist_EWC'])
