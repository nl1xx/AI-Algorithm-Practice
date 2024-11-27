import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 模型输出(logits)与真实标签(target)之间的差异

def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class NPairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)  # target.size(0)行1列

        target = (target == torch.transpose(target, 0, 1)).float()  # 16行16列的矩阵
        target = target / torch.sum(target, dim=1, keepdim=True).float()  # 将每一行归一化，使得每行的和为1

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))  # 计算anchor和positive的点积
        loss_ce = cross_entropy(logit, target)  # 交叉熵损失函数计算损失

        # 计算anchor和positive的L2范数并除以批次大小(L2范数可理解为空间中两个点的距离)
        l2_loss = torch.sum(anchor ** 2) / batch_size + torch.sum(positive ** 2) / batch_size

        loss = loss_ce + self.l2_reg * l2_loss * 0.25
        return loss


# Test the implementation
def test_npairs_loss():
    num_data = 16
    feat_dim = 5
    num_classes = 3
    reg_lambda = 0.02

    embeddings_anchor = np.random.rand(num_data, feat_dim).astype(np.float32)  # 16行5列
    embeddings_positive = np.random.rand(num_data, feat_dim).astype(np.float32)  # 16行5列

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)  # 16个0/1/2的随机整数

    npairloss = NPairLoss(l2_reg=reg_lambda)
    loss_tc = npairloss(
        anchor=torch.tensor(embeddings_anchor, dtype=torch.float32),
        positive=torch.tensor(embeddings_positive, dtype=torch.float32),
        target=torch.tensor(labels, dtype=torch.float32)
    )

    print('N-Pair Loss', loss_tc.item())


if __name__ == '__main__':
    test_npairs_loss()
