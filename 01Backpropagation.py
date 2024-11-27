# 反向传播算法(梯度下降+链式法则)

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (np.ones(s.shape) - s)


def mean_square_loss(s, y):
    return np.sum(np.square(s - y) / 2)


def d_mean_square_loss(s, y):
    return s - y


def forward(W1, W2, b1, b2, X, y):
    # 输入层到隐藏层
    y1 = np.matmul(X, W1) + b1  # [2, 3]
    z1 = sigmoid(y1)  # [2, 3]
    # 隐藏层到输出层
    y2 = np.matmul(z1, W2) + b2  # [2, 2]
    z2 = sigmoid(y2)  # [2, 2]
    # 求均方差损失
    loss = mean_square_loss(z2, y)
    return y1, z1, y2, z2, loss


def backward_update(epochs, lr=0.01):
    # 随机创建数据和权重，偏置值
    X = np.array([[0.6, 0.1], [0.3, 0.6]])
    y = np.array([0, 1])
    W1 = np.array([[0.4, 0.3, 0.6], [0.3, 0.4, 0.2]])
    b1 = np.array([0.4, 0.1, 0.2])
    W2 = np.array([[0.2, 0.3], [0.3, 0.4], [0.5, 0.3]])
    b2 = np.array([0.1, 0.2])
    # 先进行一次前向传播
    y1, z1, y2, z2, loss = forward(W1, W2, b1, b2, X, y)
    for i in range(epochs):
        # 求得隐藏层的学习信号(损失函数导数乘激活函数导数)
        ds2 = d_mean_square_loss(z2, y) * d_sigmoid(y2)
        # 根据上面推导结果式子--->(学习信号乘隐藏层z1的输出结果),注意形状需要转置
        dW2 = np.matmul(z1.T, ds2)
        # 对隐藏层的偏置梯度求和,注意是对列求和
        db2 = np.sum(ds2, axis=0)
        # 式子前两个元素相乘
        dx = np.matmul(ds2, W2.T)

        # 对照式子
        ds1 = d_sigmoid(y1) * dx
        # 式子
        dW1 = np.matmul(X.T, ds1)
        # 对隐藏层的偏置梯度求和,注意是对列求和
        db1 = np.sum(ds1, axis=0)
        # 参数更新
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2

        y1, z1, y2, z2, loss = forward(W1, W2, b1, b2, X, y)
        # 每隔100次打印一次损失
        if i % 100 == 0:
            print('第%d批次' % (i / 100))
            print('当前损失为：{:.4f}'.format(loss))
            print(z2)
            # sigmoid激活函数将结果大于0.5的值分为正类，小于0.5的值分为负类
            z2[z2 > 0.5] = 1
            z2[z2 < 0.5] = 0
            print(z2)


if __name__ == '__main__':
    backward_update(epochs=50001, lr=0.01)
