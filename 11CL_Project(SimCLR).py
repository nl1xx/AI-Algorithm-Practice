# https://blog.csdn.net/qq_43027065/article/details/118657728
# SimCLR + CIFAR10
# 包含无监督学习和有监督学习两个部分
# 在第一阶段先进行无监督学习, 对输入图像进行两次随机图像增强, 即由一幅图像得到两个随机处理过后的图像, 依次放入网络进行训练, 计算损失并更新梯度
# 第二阶段, 加载第一阶段的特征提取层训练参数, 用少量带标签样本进行有监督学习(只训练全连接层), 损失函数为交叉熵损失函数

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import argparse, os
from torch.utils.data import DataLoader
import numpy as np
import visdom


# stage one, unsupervised learning
class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLRStage1, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# stage two ,supervised learning
class SimCLRStage2(torch.nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.f = SimCLRStage1().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# stage one, Loss function(infoNCE)
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, out_1, out_2, batch_size, temperature=0.5):
        # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        # 和sim_matrix形状一样的全1矩阵, 对角线全为0, 转为布尔张量
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B]
        # 选择为True的数据, 转为2*batch_size行列数自动计算的矩阵
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


# if __name__=="__main__":
#     for name, module in resnet50().named_children():
#         print(name,module)

pre_model = os.path.join('./document', 'model.pth')
save_path = "./document"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


# 重写了__getitem__方法(增加了imgR)
class PreDataset(CIFAR10):
    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]
        img = Image.fromarray(img)

        if self.transform is not None:
            imgL = self.transform(img)
            imgR = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL, imgR, target

# if __name__=="__main__":
#
#     train_data = PreDataset(root='dataset', train=True, transform=train_transform, download=True)
#     print(train_data[0])


# train stage one
def trainOne(args):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    # print("current deveice:", DEVICE)

    train_dataset1 = PreDataset(root='./xiaotudui/torchvisionDataset', train=True, transform=train_transform, download=True)
    train_data1 = DataLoader(dataset=train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True)

    model = SimCLRStage1().to(DEVICE)
    lossLR = Loss().to(DEVICE)
    optimizer1 = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(1, args.max_epoch+1):
        model.train()
        total_loss = 0
        for batch, (imgL, imgR, labels) in enumerate(train_data1):
            imgL, imgR, labels = imgL.to(DEVICE), imgR.to(DEVICE), labels.to(DEVICE)

            _, pre_L = model(imgL)
            _, pre_R = model(imgR)

            loss1 = lossLR(pre_L, pre_R, args.batch_size)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            # if epoch % 20 == 0 and batch % 10 == 0:
            print("epoch", epoch, "batch", batch, "loss:", loss1.detach().item())
            total_loss += loss1.detach().item()

        print("epoch loss:", total_loss/len(train_dataset1)*args.batch_size)

        with open(os.path.join(save_path, "stage1_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset1) * args.batch_size) + " ")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_stage1_epoch' + str(epoch) + '.pth'))


# train stage two
def trainTwo(args):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    # print("current device:", DEVICE)

    # load dataset for train and eval
    train_dataset2 = CIFAR10(root='./xiaotudui/torchvisionDataset', train=True, transform=train_transform, download=True)
    train_data2 = DataLoader(dataset=train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    eval_dataset2 = CIFAR10(root='./xiaotudui/torchvisionDataset', train=False, transform=test_transform, download=True)
    eval_data2 = DataLoader(dataset=eval_dataset2, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    model = SimCLRStage2(num_class=len(train_dataset2.classes)).to(DEVICE)
    model.load_state_dict(torch.load(args.pre_model, map_location='cpu'), strict=False)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(1, args.max_epoch+1):
        model.train()
        total_loss = 0
        for batch, (data, target) in enumerate(train_data2):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data)

            loss2 = loss_criterion(pred, target)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            total_loss += loss2.item()

        print("epoch", epoch, "loss:", total_loss / len(train_dataset2)*args.batch_size)
        with open(os.path.join(save_path, "stage2_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset2)*args.batch_size) + " ")

        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_stage2_epoch' + str(epoch) + '.pth'))

            model.eval()
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top5 acc")
                total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0

                for batch, (data, target) in enumerate(train_data2):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    pred = model(data)

                    total_num += data.size(0)
                    prediction = torch.argsort(pred, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_5 += top5_acc

                    print("  {:02}  ".format(batch + 1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100),
                          "{:02.3f}%  ".format(top5_acc / data.size(0) * 100))

                print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100),
                          "top5 acc:{:02.3f}%".format(total_correct_5 / total_num * 100))

                with open(os.path.join(save_path, "stage2_top1_acc.txt"), "a") as f:
                    f.write(str(total_correct_1 / total_num * 100) + " ")
                with open(os.path.join(save_path, "stage2_top5_acc.txt"), "a") as f:
                    f.write(str(total_correct_5 / total_num * 100) + " ")


# 可视化部分
def show_loss(path, name, step=1):
    with open(path, "r") as f:
        data = f.read()
    data = data.split(" ")[:-1]
    x = np.linspace(1, len(data) + 1, len(data)) * step
    y = []
    for i in range(len(data)):
        y.append(float(data[i]))

    vis = visdom.Visdom(env='loss')
    vis.line(X=x, Y=y, win=name, opts={'title': name, "xlabel": "epoch", "ylabel": name})


def compare2(path_1, path_2, title="xxx", legends=["a", "b"], x="epoch", step=20):
    with open(path_1, "r") as f:
        data_1 = f.read()
    data_1 = data_1.split(" ")[:-1]

    with open(path_2, "r") as f:
        data_2 = f.read()
    data_2 = data_2.split(" ")[:-1]

    x = np.linspace(1, len(data_1) + 1, len(data_1)) * step
    y = []
    for i in range(len(data_1)):
        y.append([float(data_1[i]), float(data_2[i])])

    vis = visdom.Visdom(env='loss')
    vis.line(X=x, Y=y, win="compare",
             opts={"title": "compare " + title, "legend": legends, "xlabel": "epoch", "ylabel": title})


def eval(args):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    eval_dataset = CIFAR10(root='./xiaotudui/torchvisionDataset', train=False, transform=test_transform, download=True)
    eval_data = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

    model = SimCLRStage2(num_class=len(eval_dataset.classes)).to(DEVICE)
    model.load_state_dict(torch.load(pre_model, map_location='cpu'), strict=False)

    # total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(eval_data)
    total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0

    model.eval()
    with torch.no_grad():
        print("batch", " "*1, "top1 acc", " "*1,"top5 acc")
        for batch, (data, target) in enumerate(eval_data):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data)

            total_num += data.size(0)
            # 对模型的预测结果进行降序排序, 返回一个索引数组
            prediction = torch.argsort(pred, dim=-1, descending=True)
            top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_1 += top1_acc
            total_correct_5 += top5_acc

            print("  {:02}  ".format(batch+1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100), "{:02.3f}%  "
                  .format(top5_acc / data.size(0) * 100))

        print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100), "top5 acc:{:02.3f}%"
              .format(total_correct_5 / total_num * 100))


if __name__ == '__main__':
    parser1 = argparse.ArgumentParser(description='Train SimCLR')
    parser1.add_argument('--batch_size', default=200, type=int, help='')
    parser1.add_argument('--max_epoch', default=20, type=int, help='')
    args1 = parser1.parse_args()
    trainOne(args1)

    parser2 = argparse.ArgumentParser(description='Train SimCLR')
    parser2.add_argument('--batch_size', default=200, type=int, help='')
    parser2.add_argument('--max_epoch', default=10, type=int, help='')
    parser2.add_argument('--pre_model', default=pre_model, type=str, help='')
    args2 = parser2.parse_args()
    trainTwo(args2)

    show_loss("stage1_loss.txt", "loss1")
    show_loss("stage2_loss.txt", "loss2")
    show_loss("stage2_top1_acc.txt", "acc1")
    show_loss("stage2_top5_acc.txt", "acc1")

    parserEval = argparse.ArgumentParser(description='test SimCLR')
    parserEval.add_argument('--batch_size', default=512, type=int, help='')
    argsEval = parserEval.parse_args()
    eval(argsEval)
