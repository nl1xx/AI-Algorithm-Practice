# 对各个python文件代码中不理解的部分的简易实现, 便于理解

import numpy as np
import torch

# 1.
# N-PairLoss.py的部分代码
# labels = np.random.randint(0, 3, size=(16)).astype(np.float32)  # 16个0/1/2的随机整数
# target = torch.tensor(labels, dtype=torch.float32)
# target = target.view(target.size(0), 1)
# target = (target == torch.transpose(target, 0, 1)).float()
# target = target / torch.sum(target, dim=1, keepdim=True).float()
# print(target)
# ----------------------------------------------------------------------------------------------------------------------

# 2.
# 11CL_Project(SimCLR).py自定义的Loss函数
# 假设每个特征向量的维度为 D，批次大小为 B
# D = 10  # 特征向量的维度
# B = 5   # 批次大小
# out_1 = np.random.rand(B, D)
# out_2 = np.random.rand(B, D)
#
# # 将 out_1 和 out_2 沿着第一个维度（批次维度）拼接起来
# out = np.vstack((out_1, out_2))
#
# # 计算相似性矩阵，使用点积并应用指数和温度参数
# temperature = 0.5
# sim_matrix = np.exp(np.dot(out, out.T) / temperature)
#
# # 创建掩码以去除对角线元素
# mask = np.ones_like(sim_matrix, dtype=bool)
# np.fill_diagonal(mask, False)
#
# # 应用掩码并重塑相似性矩阵
# sim_matrix = sim_matrix[mask].reshape(2 * B, -1)
#
# # 计算分子，即 out_1 和 out_2 之间的点积，并应用指数和温度参数
# pos_sim = np.exp(np.sum(out_1 * out_2, axis=-1) / temperature)
#
# # 将 pos_sim 复制一份并拼接，以匹配分母的形状
# pos_sim = np.concatenate([pos_sim, pos_sim])
#
# # 计算最终的损失值
# loss = (-np.log(pos_sim / np.sum(sim_matrix, axis=-1))).mean()
# print("Loss:", loss)
# ----------------------------------------------------------------------------------------------------------------------

# 3.
# 11CL_Project(SimCLR).py的top-1准确率
# pred = torch.tensor([[0.1, 0.2, 0.05, 0.3, 0.35],  # 第一个样本的预测概率
#                      [0.4, 0.2, 0.1, 0.15, 0.15],  # 第二个样本的预测概率
#                      [0.25, 0.25, 0.15, 0.1, 0.25]])  # 第三个样本的预测概率
#
# target = torch.tensor([2, 0, 4])  # 真实标签，分别对应类别2，0，4
#
# prediction = torch.argsort(pred, dim=-1, descending=True)  # Top-K预测的索引
# top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(1)).any(dim=-1).float()).item()
# print(f"Top-1 Accuracy: {top1_acc / len(target):.4f}")
# ----------------------------------------------------------------------------------------------------------------------

# 4.
# 19RNN_Project.py的.topK()方法
# output = torch.tensor([1, 2, 3, 4])
# values, indices = output.topk(1)  # 返回最大数值和该数值的索引
# print(values, indices)
# ----------------------------------------------------------------------------------------------------------------------

