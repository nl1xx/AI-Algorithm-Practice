import torch
from torch import nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        input = torch.mm(self.weight, input)
        output = torch.spmm(adj, input)
        output = self.act(output)
        return output


class VGAE(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/pdf/1611.07308
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, log_var = self.encode(x, adj)
        z = self.reparameterize(mu, log_var)
        return self.dc(z), mu, log_var


class InnerProductDecoder(nn.Module):
    def __init(self, dropout, act=torch.sigmoid):
        super().__init()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = torch.mm(z, z.t())
        return self.act(z)
