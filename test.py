# 对各个python文件代码中不理解的部分的简易实现, 便于理解

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

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

# 5.
# 34Transformer.py的unsqueeze(): unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度
# input = torch.arange(0, 6)
# print(input)  # tensor([0, 1, 2, 3, 4, 5])
# print(input.shape)  # torch.Size([6])
# print(input.unsqueeze(0))  # tensor([[0, 1, 2, 3, 4, 5]])
# print(input.unsqueeze(0).shape)  # torch.Size([1, 6])
# print(input.unsqueeze(1))
# # tensor([[0],
# #         [1],
# #         [2],
# #         [3],
# #         [4],
# #         [5]])
# print(input.unsqueeze(1).shape)  # torch.Size([6, 1])
# ----------------------------------------------------------------------------------------------------------------------

# 6.
# 35Transformer.py的torchtext模块
# 6.1 get_tokenizer()
# 创建一个分词器，将语料喂给相应的分词器，分词器支持’basic_english’，‘spacy’，‘moses’，‘toktok’，‘revtok’，'subword’等规则
# message = "I love China."
# tokenizer = get_tokenizer('basic_english')
# print(tokenizer(message))  # ['i', 'love', 'china', '.']
# ----------------------------------------------------------------------------------------------------------------------
# 6.2 build_vocab_from_iterator()
# 从一个可迭代对象中统计token的频次，并返回一个vocab(词汇字典)
# vocab = build_vocab_from_iterator(tokenizer(message), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
# print(vocab)  # Vocab()
# vocab.set_default_index(vocab['<unk>'])  # 设置默认索引为 <unk>
# sentences = [["The", "cat", "sat", "on", "the", "mat"], ["The", "dog", "played", "with", "cat", "ball"], ['cat', 'like',
#              'dog', 'kidding']]
# # min_feq设置最小频率为1，即只要出现过的都不会在这里被筛掉
# # max_tokens设置为10，表示词典的长度为10，但是因为有了specials，所以真正的词典中有效token为9个
# vocab = build_vocab_from_iterator(sentences, min_freq=1, max_tokens=10, specials=['<unk>'])
# # 设置默认索引，若是索引的单词不在词典内，则返回0，此例中0与<unk>对应
# vocab.set_default_index(0)
# # 查看词典(字典形式)
# print(vocab.get_stoi())
# # {'dog': 3,'<unk>': 0, 'kidding': 5, 'cat': 1, 'ball': 4, 'The': 2, 'like': 6, 'mat': 7, 'on': 8, 'played': 9}
# # 查看字典(列表形式)
# print(vocab.get_itos())
# # ['<unk>', 'cat', 'The', 'dog', 'ball', 'kidding', 'like', 'mat', 'on', 'played']
# ----------------------------------------------------------------------------------------------------------------------
# 6.3 将句子转换为索引序列
# vocab = build_vocab_from_iterator(tokenizer(message), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
# vocab.set_default_index(vocab['<unk>'])  # 设置默认索引为 <unk>
# def process_sentence(sentence, tokenizer, vocab):
#     sentence_list = tokenizer(sentence)
#     tokens = ['<bos>'] + sentence_list + ['<eos>']
#     indices = [vocab[token] for token in tokens]
#     return indices
# print(process_sentence(message, tokenizer, vocab))  # [2, 4, 0, 0, 5, 3] ?????
# ----------------------------------------------------------------------------------------------------------------------

# 7.
# 35Transformer.py的pad_sequence()
# 填充句子到相同长度
# a = torch.ones(25, 300)
# b = torch.ones(22, 300)
# c = torch.ones(15, 300)
# print(pad_sequence([a, b, c]).size())  # torch.Size([25, 3, 300])
# ----------------------------------------------------------------------------------------------------------------------

# 8.
# 36ViT.py的transpose()
# 转置函数, 交换维度dim0和dim1(交换行列索引值)
# num = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(num.transpose(-1, -2))  # 等价于print(num.transpose(0, 1))、print(num.transpose(-2, -1))
# 输出：tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
# ----------------------------------------------------------------------------------------------------------------------

# 9.
# RelativePositionEmbedding.py的.contiguous()
# https://zhuanlan.zhihu.com/p/64551412
# ----------------------------------------------------------------------------------------------------------------------

# 10.
# RelativePositionEmbedding.py的.permute()
# 转置函数
# coords_h = torch.arange(2)
# coords_w = torch.arange(2)
# mesh = torch.meshgrid([coords_h, coords_w])  # 2*2*2
# mesh = torch.stack(mesh)  # 2*4*2
# print(mesh.permute(0, 2, 1))  # 2*2*4
# ----------------------------------------------------------------------------------------------------------------------

# 11.
#
# ----------------------------------------------------------------------------------------------------------------------
