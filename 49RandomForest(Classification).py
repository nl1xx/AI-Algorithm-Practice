# 随机森林分类

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class rfc:
    """
    随机森林分类器
    """

    def __init__(self, n_estimators=100, random_state=0):
        # 随机森林的大小
        self.n_estimators = n_estimators
        # 随机森林的随机种子
        self.random_state = random_state

    def fit(self, X, y):
        """
        随机森林分类器拟合
        """
        self.y_classes = np.unique(y)
        # 决策树数组
        dts = []
        n = X.shape[0]
        rs = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            # 创建决策树分类器
            dt = DecisionTreeClassifier(random_state=rs.randint(np.iinfo(np.int32).max), max_features="auto")
            # 根据随机生成的权重，拟合数据集
            dt.fit(X, y, sample_weight=np.bincount(rs.randint(0, n, n), minlength=n))
            dts.append(dt)
        self.trees = dts

    def predict(self, X):
        """
        随机森林分类器预测
        """
        # 预测结果数组
        probas = np.zeros((X.shape[0], len(self.y_classes)))
        for i in range(self.n_estimators):
            # 决策树分类器
            dt = self.trees[i]
            # 依次预测结果可能性
            probas += dt.predict_proba(X)
        # 预测结果可能性取平均
        probas /= self.n_estimators
        # 返回预测结果
        return self.y_classes.take(np.argmax(probas, axis=1), axis=0)
