# 梯度下降

import numpy as np
import matplotlib.pyplot as plt


def fn(x):
    return 0.5 * (x - 0.25) ** 2


def d_fn(x):
    return x - 0.25


echo = 100
alpha = 0.1
GD_X = []
GD_Y = []
X = np.arange(-4, 4, 0.05)
Y = fn(X)
x = 3

for i in range(echo):
    # 梯度下降
    x = x - alpha * d_fn(x)
    y = fn(x)
    GD_X.append(x)
    GD_Y.append(y)


plt.plot(X, Y)
plt.scatter(GD_X, GD_Y)
plt.title("$y = 0.5(x - 0.25)^2$")
plt.show()
