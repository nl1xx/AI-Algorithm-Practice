# GBDT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import plot_tree

# 数据准备
X = np.arange(1, 11).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

# 训练梯度提升回归模型
gbdt = GradientBoostingRegressor(max_depth=4, criterion='squared_error', n_estimators=10).fit(X, y)

# 提取其中的一棵树（例如第6棵树）
# 注意：索引从0开始，第6棵树的索引是5
sub_tree = gbdt.estimators_[5, 0]

# 使用 plot_tree 绘制这棵树
plt.figure(figsize=(12, 8))  # 设置图形大小
plot_tree(sub_tree, filled=True, rounded=True, precision=2)
plt.show()
