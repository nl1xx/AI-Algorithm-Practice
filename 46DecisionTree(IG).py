# 决策树(信息增益)
import numpy as np


class GainNode:
    """
    分类决策树中的结点
    基于信息增益-Information Gain
    """

    def __init__(self, feature=None, threshold=None, gain=None, left=None, right=None):
        # 结点划分的特征下标
        self.feature = feature
        # 结点划分的临界值，当结点为叶子结点时为分类值
        self.threshold = threshold
        # 结点的信息增益值
        self.gain = gain
        # 左结点
        self.left = left
        # 右结点
        self.right = right


class GainTree:
    """
    分类决策树
    基于信息增益-Information Gain
    """

    def __init__(self, max_depth=None, min_samples_leaf=None):
        # 决策树最大深度
        self.max_depth = max_depth
        # 决策树叶结点最小样本数
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """
        分类决策树拟合
        基于信息增益-Information Gain
        """
        y = np.array(y)
        self.root = self.buildNode(X, y, 0)
        return self

    def buildNode(self, X, y, depth):
        """
        构建分类决策树结点
        基于信息增益-Information Gain
        """
        node = GainNode()
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
            node.threshold = max(y_classes, key=y.tolist().count)
            return node
        # 当决策树叶结点样本数达到最小样本数限制时返回样本中分类占比最大的分类
        if self.min_samples_leaf is not None and len(y) <= self.min_samples_leaf:
            node.threshold = max(y_classes, key=y.tolist().count)
            return node
        max_gain = -np.inf
        max_middle = None
        max_feature = None
        # 遍历所有特征，获取信息增益最大的特征
        for i in range(X.shape[1]):
            # 计算特征的信息增益
            gain, middle = self.calcGain(X[:, i], y, y_classes)
            if max_gain < gain:
                max_gain = gain
                max_middle = middle
                max_feature = i
        # 信息增益最大的特征
        node.feature = max_feature
        # 临界值
        node.threshold = max_middle
        # 信息增益
        node.gain = max_gain
        X_lt = X[:, max_feature] < max_middle
        X_gt = X[:, max_feature] > max_middle
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

    def calcEnt(self, y, y_classes):
        """
        计算信息熵
        """
        ent = 0
        for j in range(len(y_classes)):
            p = len(y[y == y_classes[j]]) / len(y)
            if p != 0:
                ent = ent + p * np.log2(p)
        return -ent

    def calcGain(self, x, y, y_classes):
        """
        计算信息增益
        """
        x_sort = np.sort(x)
        middle = self.calcMiddle(x_sort)
        max_middle = -np.inf
        max_gain = -np.inf
        ent = self.calcEnt(y, y_classes)
        # 遍历每个平均值
        for i in range(len(middle)):
            y_gt = y[x > middle[i]]
            y_lt = y[x < middle[i]]
            ent_gt = self.calcEnt(y_gt, y_classes)
            ent_lt = self.calcEnt(y_lt, y_classes)
            # 计算信息增益
            gain = ent - (ent_gt * len(y_gt) / len(x) + ent_lt * len(y_lt) / len(x))
            if max_gain < gain:
                max_gain = gain
                max_middle = middle[i]
        return max_gain, max_middle

    def predict(self, X):
        """
        分类决策树预测
        """
        y = np.zeros(X.shape[0])
        self.checkNode(X, y, self.root)
        return y

    def checkNode(self, X, y, node, cond=None):
        """
        通过分类决策树结点判断分类
        """
        # 当没有子结点时，直接返回当前临界值
        if node.left is None and node.right is None:
            return node.threshold
        X_lt = X[:, node.feature] < node.threshold
        if cond is not None:
            X_lt = X_lt & cond
        # 递归判断左结点
        lt = self.checkNode(X, y, node.left, X_lt)
        if lt is not None:
            y[X_lt] = lt
        X_gt = X[:, node.feature] > node.threshold
        if cond is not None:
            X_gt = X_gt & cond
        # 递归判断右结点
        gt = self.checkNode(X, y, node.right, X_gt)
        if gt is not None:
            y[X_gt] = gt
