# 决策树(均方误差)
import numpy as np


class RegressorNode:
    """
    回归决策树中的结点
    """

    def __init__(self, feature=None, threshold=None, mse=None, left=None, right=None):
        # 结点划分的特征下标
        self.feature = feature
        # 结点划分的临界值，当结点为叶子结点时为分类值
        self.threshold = threshold
        # 结点的均方误差值
        self.mse = mse
        # 左结点
        self.left = left
        # 右结点
        self.right = right


class RegressorTree:
    """
    回归决策树
    """

    def __init__(self, max_depth=None, min_samples_leaf=None):
        # 决策树最大深度
        self.max_depth = max_depth
        # 决策树叶结点最小样本数
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """
        回归决策树拟合
        """
        self.root = self.buildNode(X, y, 0)
        return self

    def buildNode(self, X, y, depth):
        """
        构建回归决策树结点
        """
        node = RegressorNode()
        # 当没有样本时直接返回
        if len(y) == 0:
            return node
        y_classes = np.unique(y)
        # 当样本中只存在一种分类时直接返回该分类
        if len(y_classes) == 1:
            node.threshold = y_classes[0]
            return node
        # 当决策树深度达到最大深度限制时返回样本中分类占比最大的分类
        if self.max_depth is not None and depth >= self.max_depth:
            node.threshold = np.average(y)
            return node
        # 当决策树叶结点样本数达到最小样本数限制时返回样本中分类占比最大的分类
        if self.min_samples_leaf is not None and len(y) <= self.min_samples_leaf:
            node.threshold = np.average(y)
            return node
        min_mse = np.inf
        min_middle = None
        min_feature = None
        # 遍历所有特征，获取均方误差最小的特征
        for i in range(X.shape[1]):
            # 计算特征的均方误差
            mse, middle = self.calcMse(X[:, i], y)
            if min_mse > mse:
                min_mse = mse
                min_middle = middle
                min_feature = i
        # 均方误差最小的特征
        node.feature = min_feature
        # 临界值
        node.threshold = min_middle
        # 均方误差
        node.mse = min_mse
        X_lt = X[:, min_feature] < min_middle
        X_gt = X[:, min_feature] > min_middle
        # 递归处理左集合
        node.left = self.buildNode(X[X_lt, :], y[X_lt], depth + 1)
        # 递归处理右集合
        node.right = self.buildNode(X[X_gt, :], y[X_gt], depth + 1)
        return node

    def calcMiddle(self, x):
        """
        计算连续型特征的俩俩平均值
        """
        middle = []
        if len(x) == 0:
            return np.array(middle)
        start = x[0]
        for i in range(len(x) - 1):
            if x[i] == x[i + 1]:
                continue
            middle.append((start + x[i + 1]) / 2)
            start = x[i + 1]
        return np.array(middle)

    def calcMse(self, x, y):
        """
        计算均方误差
        """
        x_sort = np.sort(x)
        middle = self.calcMiddle(x_sort)
        min_middle = np.inf
        min_mse = np.inf
        for i in range(len(middle)):
            y_gt = y[x > middle[i]]
            y_lt = y[x < middle[i]]
            avg_gt = np.average(y_gt)
            avg_lt = np.average(y_lt)
            mse = np.sum((y_lt - avg_lt) ** 2) + np.sum((y_gt - avg_gt) ** 2)
            if min_mse > mse:
                min_mse = mse
                min_middle = middle[i]
        return min_mse, min_middle

    def predict(self, X):
        """
        回归决策树预测
        """
        y = np.zeros(X.shape[0])
        self.checkNode(X, y, self.root)
        return y

    def checkNode(self, X, y, node, cond=None):
        """
        通过回归决策树结点判断分类
        """
        if node.left is None and node.right is None:
            return node.threshold
        X_lt = X[:, node.feature] < node.threshold
        if cond is not None:
            X_lt = X_lt & cond
        lt = self.checkNode(X, y, node.left, X_lt)
        if lt is not None:
            y[X_lt] = lt
        X_gt = X[:, node.feature] > node.threshold
        if cond is not None:
            X_gt = X_gt & cond
        gt = self.checkNode(X, y, node.right, X_gt)
        if gt is not None:
            y[X_gt] = gt
