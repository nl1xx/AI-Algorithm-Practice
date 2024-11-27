import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):    # x特征矩阵,adj邻接矩阵
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 没使用度矩阵
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 使用度矩阵
    # def forward(self, input, adj):
    #     # 计算度矩阵的逆平方根
    #     deg = adj.sum(1)
    #     deg_inv_sqrt = torch.pow(deg, -0.5)
    #     deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    #
    #     # 归一化邻接矩阵
    #     adj_normalized = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    #
    #     # 计算图卷积
    #     support = torch.mm(input, self.weight)
    #     output = torch.spmm(adj_normalized, support)
    #     if self.bias is not None:
    #         return output + self.bias
    #     else:
    #         return output
