import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GRU(nn.Module):
    def __init__(self, indim, hidim, outdim):
        super(GRU, self).__init__()
        self.indim = indim
        self.hidim = hidim
        self.outdim = outdim
        self.W_zh, self.W_zx, self.b_z = self.get_three_parameters()
        self.W_rh, self.W_rx, self.b_r = self.get_three_parameters()
        self.W_hh, self.W_hx, self.b_h = self.get_three_parameters()
        self.Linear = nn.Linear(hidim, outdim)  # 全连接层做输出
        self.reset()

    def forward(self, input, state):
        input = input.type(torch.float32)
        if torch.cuda.is_available():
            input = input.cuda()
        Y = []
        h = state
        h = h.cuda()
        for x in input:
            z = F.sigmoid(h @ self.W_zh + x @ self.W_zx + self.b_z)
            r = F.sigmoid(h @ self.W_rh + x @ self.W_rx + self.b_r)
            ht = F.tanh((h * r) @ self.W_hh + x @ self.W_hx + self.b_h)
            h = (1 - z) * h + z * ht
            y = self.Linear(h)
            Y.append(y)
        return torch.cat(Y, dim=0), h

    def get_three_parameters(self):
        indim, hidim, outdim = self.indim, self.hidim, self.outdim
        return nn.Parameter(torch.FloatTensor(hidim, hidim)), \
            nn.Parameter(torch.FloatTensor(indim, hidim)), \
            nn.Parameter(torch.FloatTensor(hidim))

    def reset(self):
        stdv = 1.0 / math.sqrt(self.hidim)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)
