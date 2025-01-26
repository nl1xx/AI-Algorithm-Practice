# CART分类树

import pandas as pd
import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 加载数据
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 构建决策树模型
dtree = DecisionTreeClassifier(criterion='gini').fit(df, y)

# 使用 plot_tree 绘制决策树
plt.figure(figsize=(20, 10))  # 设置图形大小
plot_tree(dtree, filled=True, rounded=True, feature_names=iris.feature_names, class_names=iris.target_names, precision=2)
plt.show()
