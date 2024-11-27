# https://blog.csdn.net/m0_53328738/article/details/128344749
# 根据人名判断属于哪个国家
import glob
import math
import os
import random
import string
import unicodedata
import torch
import time

from matplotlib import ticker
from torch import nn, optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取文件, 处理字符
def findFiles(path):
    return glob.glob(path)


name_files = findFiles('./datasets/names/*.txt')
class_num = len(name_files)
# print(class_num)  # 18

all_letters = string.ascii_letters + " .,;'"  # 记录了所有字符 (大小写英文字符和, . ; '字符)
n_letters = len(all_letters)
# print(n_letters)  # 57


# 将带有重音标记的英文字母的字符串转为英文字母的字符串
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# print(unicodeToAscii('Ślusàrski'))  # Slusarski


category_lines = {}  # 创建一个用于存储各国语言名字的字典
all_categories = []  # 存取每个类别的名称


# 读取txt名字列表里面的每个名字，并存为列表
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 非英文字母的转换处理
    return [unicodeToAscii(line) for line in lines]


# 构造用于存储各国语言名字的字典
for filename in name_files:
    category = os.path.splitext(os.path.basename(filename))[0]  # 对应字典的键
    all_categories.append(category)  # 添加到字典里面去
    lines = readLines(filename)  # 获取这个键的值（名字列表）
    category_lines[category] = lines

n_categories = len(all_categories)
# print(n_categories)  # 18
# print(category_lines['Chinese'][:10], all_categories)

# 创造字典存储每个类的样本数量
category_sample_num = {}
# 遍历上面的得到的category_lines字典， 统计每个类别的样本数量
for key, value in category_lines.items():
    category_sample_num[key] = len(value)
print(category_sample_num)


# 返回字符所对应的标签数字
def letterToIndex(letter):
    return all_letters.find(letter)


# 将单个字符转为一维的Tensor张量
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters).to(device)
    tensor[0][letterToIndex(letter)] = 1
    return tensor
# a_tensor = letterToTensor('a')
# print(letterToIndex("a"), a_tensor, a_tensor.shape)


# 将单词转为Tensor张量
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters).to(device)  # 首先初始化一个全零的张量，形状是(len(line),1,n_letters)
    # 遍历每个人名中的每个字符, 并搜索其对应的索引, 将该索引位置置1
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
# test = lineToTensor('abc')
# print(test.size(), test)


# 定义模型
# RNN
class RNN(nn.Module):
    """
    input_size:代表RNN输入的最后一个维度
    hidden_size:代表RNN隐藏层的最后一个维度
    output_size:代表RNN网络最后线性层的输出维度
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # 下一个隐藏层
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # 输出
        self.softmax = nn.LogSoftmax(dim=1)

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden


# LSTM
class LSTM(nn.Module):
    """
    input_size:代表输入张量x中最后一个维度
    hidden_size:代表隐藏层张量的最后一个维度
    output_size:代表线性层最后的输出维度
    num_layers:代表LSTM网络的层数
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)  # 实例化全连接线性层，将RNN的输出维度转换成指定的输出维度
        self.softmax = nn.LogSoftmax(dim=-1)  # 实例化nn中预定义的softmax层，用于从输出层中获得类别的结果

    def initHiddenAndC(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)  # 初始化隐藏层和细胞状态并移动到GPU
        c = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        return hidden, c

    def forward(self, input1, hidden, c):
        input1 = input1.unsqueeze(0)  # 注意LSTM网络的输入有3个张量，因为还有一个细胞状态c
        rr, (hn, cn) = self.lstm(input1, (hidden, c))
        return self.softmax(self.linear(rr)), hn, cn  # 最后将3个张量结果全部返回，同时rr要经过线性层和softmax的处理


# GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)  # 全连接线性层，将GRU的输出维度转换成指定的输出维度
        self.softmax = nn.LogSoftmax(dim=-1)  # softmax层，获得类别的概率结果

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).to(device)

    def forward(self, input1, hidden):
        input1 = input1.unsqueeze(0)  # 输入到GRU中的张量要求是三维，所以需要扩充维度
        rr, hn = self.gru(input1, hidden)
        return self.softmax(self.linear(rr)), hn


# 根据输出，匹配最大概率对应的类别名称
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


# 随机选择
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# 随机选择一个名称数据，返回语言类别，名字，语言类别对应的标签，名字张量数据
def randomTrainingExample():
    # 随机选择一个类
    category = randomChoice(all_categories)
    # 随机选择该类中的一个单词
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long).to(device)  # 匹配这个类的数字标签
    line_tensor = lineToTensor(line).to(device)
    return category, line, category_tensor, line_tensor
# for i in range(1):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ line =', line, '/category_tensor = ', category_tensor, 'line_tensor = ', line_tensor)
# print("--" * 20)


criterion = nn.NLLLoss()
learning_rate = 5e-3


# 训练
# 对一个数据进行训练，传入名字对应的标签，名字的张量数据
def trainRnn(net, category_tensor, line_tensor):
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    hidden = net.initHidden().to(device)
    net.zero_grad()

    for i in range(line_tensor.size(0)):
        output, hidden = net(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.step()
    return output, loss.item()


def trainLstm(net, category_tensor, line_tensor):
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    hidden, c = net.initHiddenAndC()
    hidden, c = hidden.to(device), c.to(device)
    net.zero_grad()

    for i in range(line_tensor.size(0)):  # 遍历序列的每个时间步
        # 获取当前时间步的输入，形状为 (1, 特征数量)
        input = line_tensor[i].to(device)
        output, hidden, c = net(input, hidden, c)

    loss = criterion(output.squeeze(0), category_tensor)  # 压缩输出张量的额外维度
    loss.backward()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.step()
    return output, loss.item()


def trainGru(net, category_tensor, line_tensor):
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    hidden = net.initHidden().to(device)
    net.zero_grad()

    for i in range(line_tensor.size(0)):  # 遍历序列的每个时间步
        # 获取当前时间步的输入，形状为 (1, 特征数量)
        input = line_tensor[i]
        output, hidden = net(input, hidden)

    loss = criterion(output.squeeze(0), category_tensor)  # 压缩输出张量的额外维度
    loss.backward()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.step()
    return output, loss.item()


# 全局训练函数
def train(name, net, trainNet):
    print("========================={}Start Training===========================".format(name))
    n_iters = 10000
    print_every = 100
    plot_every = 50

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()  # 随机生成一个名字张量数据
        output, loss = trainNet(net, category_tensor, line_tensor)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)  # 根据输出概率匹配预测标签
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    print('Finish!!!')
    return all_losses


n_hidden = 128
# 创建模型
rnn = RNN(n_letters, n_hidden, n_categories).to(device)
lstm = LSTM(n_letters, n_hidden, n_categories).to(device)  # 默认隐藏层是1层
gru = GRU(n_letters, n_hidden, n_categories).to(device)

# 模型名称, 模型, 模型训练函数列表
nets_name_list = ['RNN', 'LSTM', 'GRU']
nets_list = [rnn, lstm, gru]
trainNets_list = [trainRnn, trainLstm, trainGru]

# 记录每个模型的损失值列表
nets_loss = []
# 训练模型
for i in range(len(nets_list)):
    net_loss = train(nets_name_list[i], nets_list[i], trainNets_list[i])
    nets_loss.append(net_loss)

# 损失值折线图
plt.figure()
for i in range(len(nets_list)):
    plt.plot(nets_loss[i])
plt.legend(nets_name_list)
plt.show()


# 传入名字标张量数据进行预测
def evaluateRNN(line_tensor):
    line_tensor = line_tensor.to(device)
    hidden = rnn.initHidden().to(device)
    for i in range(line_tensor.size(0)):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def evaluateLSTM(line_tensor):
    line_tensor = line_tensor.to(device)
    hidden, c = lstm.initHiddenAndC()
    hidden, c = hidden.to(device), c.to(device)
    for i in range(line_tensor.size(0)):
        output, hidden, c = lstm(line_tensor[i].to(device), hidden, c)
    return output.squeeze(0)


def evaluateGRU(line_tensor):
    line_tensor = line_tensor.to(device)
    hidden = gru.initHidden().to(device)
    for i in range(line_tensor.size(0)):
        output, hidden = gru(line_tensor[i], hidden)
    return output.squeeze(0)


# 对模型进行迭代评估并生成混淆矩阵
def evaluate_confusion(net='RNN'):
    # 用混淆矩阵记录正确的猜测
    confusion = torch.zeros(n_categories, n_categories)

    # 通过一系列的例子，记录是正确的猜测，形成混淆矩阵 ---> Polish Kozlow 17 17、Chinese Foong 13 4
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()  # 随机选择名字

        if net == 'RNN':
            output = evaluateRNN(line_tensor)  # 获取预测输出
        elif net == 'LSTM':
            output = evaluateLSTM(line_tensor)  # 获取预测输出
        else:
            output = evaluateGRU(line_tensor)  # 获取预测输出

        guess, guess_index = categoryFromOutput(output)  # 根据预测输出匹配预测的类别和对应标签
        category_i = all_categories.index(category)  # 匹配对应的真实标签
        confusion[category_i][guess_index] += 1

    # 统计概率 正确预测的样本数/该样本的总数 （利用广播机制，计算每一类的概率）
    '''
    tensor([ 0.,  1.,  2.,  0.,  5., 50.,  1.,  1.,  5.,  0.,  2.,  1.,  0.,  0., 1.,  0.,  1.,  5.])
    tensor(75.)
    tensor([0.0000, 0.0133, 0.0267, 0.0000, 0.0667, 0.6667, 0.0133, 0.0133, 0.0667, 0.0000, 0.0267, 0.0133, 0.0000, 
            0.0000, 0.0133, 0.0000, 0.0133, 0.0667])
    '''
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    return confusion


# 预测迭代参数
n_confusion = 100

for i in range(len(nets_name_list)):
    confusion = evaluate_confusion(nets_name_list[i])

    print('Accuracy：', end='')
    for j in range(n_categories):
        print('%.3f ' % confusion[j][j].item(), end='')

    # 绘制混淆矩阵
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    plt.title(nets_name_list[i], fontsize=20)

    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def predict(input_line, evaluate_fn, n_predictions=3):
    """
      input_line:代表输入字符串名
      evaluate_fn:代表评估的模型函数 [RNN、LSTM、GRU]
      n_predictions:代表需要取得最有可能的n_predictions个结果
    """
    print('\n>%s' % input_line)  # 将输入的名字打印出来
    # 测试
    with torch.no_grad():
        output = evaluate_fn(lineToTensor(input_line))  # 将人名转换成张量，然后调用评估函数得到预测的结果
        # 从预测的结果中取出top3个最大值及其索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 初始化结果的列表
        predictions = []
        # 遍历3个最可能的结果
        for i in range(n_predictions):
            value = topv[0][i].item()  # 首先从topv中取出概率值
            category_index = topi[0][i].item()  # 然后从topi中取出索引值
            # print('(%.2f)%s' % (value, all_categories[category_index]))  # 打印概率值及其对应的真实国家名称
            predictions.append([value, all_categories[category_index]])  # 将结果封装成列表格式，添加到最终的结果列表中
        return predictions


for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]:
    # print('-' * 20)
    print('Predict')
    predict('Dovesky', evaluate_fn)
    predict('Jackson', evaluate_fn)
    predict('Satoshi', evaluate_fn)
    predict('Zeng', evaluate_fn)
