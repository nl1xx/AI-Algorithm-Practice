# 1.创建原始数据
import torch
from torch import nn

x = torch.rand((3, 4))
y = torch.tensor([3, 0, 2])

# 2.计算x_sfm=softmax(x)，求出归一化后的每个类别概率值
softmax_func = nn.Softmax()
x_sfm = softmax_func(x)

# 3.计算log(x_sfm)，由于原来的概率值位于0-1，取对数后一定是负值
# 概率值越大，取对数后的绝对值越小，符合我们的损失目标
x_log = torch.log(x_sfm)

# ls = nn.LogSoftmax(dim=1)# 也可以使用nn.LogSoftmax()进行测试，二者结果一致
# print(ls(x))

# 4.最后使用nn.NLLLoss求损失
# 思路，按照交叉熵的计算过程，将真值与经过LogSoftmax后的预测值求和取平均
index = range(len(x))
loss = x_log[index, y]
print(abs(sum(loss) / len(x)))
