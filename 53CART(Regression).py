# CART回归树

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor

X = np.arange(1, 11).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

tree = DecisionTreeRegressor(max_depth=4).fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, rounded=True, feature_names=["X"], precision=2)
plt.show()
