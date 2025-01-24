# 随机森林回归

import numpy as np
from sklearn.tree import DecisionTreeRegressor

class rfr:
    """
    随机森林回归器
    """

    def __init__(self, n_estimators=100, random_state=0):
        # 随机森林的大小
        self.n_estimators = n_estimators
        # 随机森林的随机种子
        self.random_state = random_state

    def fit(self, X, y):
        """
        随机森林回归器拟合
        """
        # 决策树数组
        dts = []
        n = X.shape[0]
        rs = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            # 创建决策树回归器
            dt = DecisionTreeRegressor(random_state=rs.randint(np.iinfo(np.int32).max), max_features="auto")
            # 根据随机生成的权重，拟合数据集
            dt.fit(X, y, sample_weight=np.bincount(rs.randint(0, n, n), minlength=n))
            dts.append(dt)
        self.trees = dts

    def predict(self, X):
        """
        随机森林回归器预测
        """
        # 预测结果
        ys = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            # 决策树回归器
            dt = self.trees[i]
            # 依次预测结果
            ys += dt.predict(X)
        # 预测结果取平均
        ys /= self.n_estimators
        return ys
