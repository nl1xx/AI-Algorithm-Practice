import numpy as np
import matplotlib.pyplot as plt


class HMM(object):
    def __init__(self, N, M, pi=None, A=None, B=None):
        self.N = N
        self.M = M
        self.pi = pi
        self.A = A
        self.B = B

    def get_data_with_distribute(self, dist):  # 根据给定的概率分布随机返回数据（索引）
        r = np.random.rand()
        for i, p in enumerate(dist):
            if r < p:
                return i
            r -= p

    def generate(self, T: int):
        """
        根据给定的参数生成观测序列
        T: 指定要生成数据的数量
        """
        z = self.get_data_with_distribute(self.pi)    # 根据初始概率分布生成第一个状态
        x = self.get_data_with_distribute(self.B[z])  # 生成第一个观测数据
        result = [x]
        for _ in range(T-1):        # 依次生成余下的状态和观测数据
            z = self.get_data_with_distribute(self.A[z])
            x = self.get_data_with_distribute(self.B[z])
            result.append(x)
        return result

    # Evaluation
    def forward_evaluate(self, X):
        """
        根据给定的参数计算条件概率
        X: 观测数据
        """
        # 所有状态下观测到X[0]的概率
        alpha = self.pi * self.B[:, X[0]]
        for x in X[1:]:
            alpha_next = np.empty(self.N)
            for j in range(self.N):
                # 转移到状态j的概率分布, 在状态j下观测到当前值x的概率
                alpha_next[j] = np.sum(self.A[:,j] * alpha * self.B[j,x])
            alpha = alpha_next
            # alpha = np.sum(self.A * alpha.reshape(-1, 1) * self.B[:, x].reshape(1, -1), axis=0)
        return alpha.sum()

    def backward_evaluate(self, X):
        beta = np.ones(self.N)
        for x in X[:0:-1]:
            beta_next = np.empty(self.N)
            for i in range(self.N):
                # 从状态i转移到其他状态的概率分布, 所有状态观测x的概率
                beta_next[i] = np.sum(self.A[i, :] * self.B[:, x] * beta)
            beta = beta_next
        return np.sum(beta * self.pi * self.B[:, X[0]])

    # Learning
    def fit(self, X):
        """
        根据给定观测序列反推参数
        """
        # 初始化参数 pi, A, B
        self.pi = np.random.sample(self.N)
        self.A = np.ones((self.N, self.N)) / self.N
        self.B = np.ones((self.N, self.M)) / self.M
        self.pi = self.pi / self.pi.sum()
        for _ in range(50):
            # 按公式计算下一时刻的参数
            alpha, beta = self.get_something(X)
            gamma = alpha * beta

            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = np.sum(alpha[:-1, i] * beta[1:, j] * self.A[i, j] * self.B[j, X[1:]]) / gamma[:-1,i].sum()

            for j in range(self.N):
                for k in range(self.M):
                    self.B[j, k] = np.sum(gamma[:, j] * (X == k)) / gamma[:, j].sum()

            self.pi = gamma[0] / gamma[-1].sum()

    def get_something(self, X):
        """
        根据给定数据与参数，计算所有时刻前向概率和后向概率
        """
        T = len(X)
        alpha = np.zeros((T, self.N))
        alpha[0, :] = self.pi * self.B[:, X[0]]
        for i in range(T - 1):
            x = X[i + 1]
            alpha[i + 1, :] = np.sum(self.A * alpha[i].reshape(-1, 1) * self.B[:, x].reshape(1, -1), axis=0)

        beta = np.ones((T, self.N))
        for j in range(T - 1, 0, -1):
            for i in range(self.N):
                beta[j - 1, i] = np.sum(self.A[i, :] * self.B[:, X[j]] * beta[j])

        return alpha, beta

    # Decoding
    def decode(self, X):
        T = len(X)
        x = X[0]
        delta = self.pi * self.B[:, x]
        # varphi[i, j]存储了在时间点i达到状态j的最大概率路径中, 前一个时间点的状态
        varphi = np.zeros((T, self.N), dtype=int)
        path = [0] * T
        for i in range(1, T):
            delta = delta.reshape(-1, 1)  # 转成一列方便广播
            tmp = delta * self.A
            varphi[i, :] = np.argmax(tmp, axis=0)  # 在tmp矩阵的每一列中找到最大值的索引
            delta = np.max(tmp, axis=0) * self.B[:, X[i]]
        path[-1] = np.argmax(delta)
        # 回溯最优路径
        for i in range(T - 1, 0, -1):
            path[i-1] = varphi[i, path[i]]
        return path


if __name__ == "__main__":
    pi = np.array([.25, .25, .25, .25])
    A = np.array([
        [0,  1,  0, 0],
        [.4, 0, .6, 0],
        [0, .4, 0, .6],
        [0, 0, .5, .5]])
    B = np.array([
        [.5, .5],
        [.3, .7],
        [.6, .4],
        [.8, .2]])
    hmm = HMM(4, 2, pi, A, B)
    X = hmm.generate(10)
    print(X)   # 生成10个数据

    print(hmm.forward_evaluate(X))
    print(hmm.backward_evaluate(X))
    print(hmm.decode(X))

    def triangle_data(T):  # 生成三角波形状的序列
        data = []
        for x in range(T):
            x = x % 6
            data.append(x if x <= 3 else 6 - x)
        return data

    data = np.array(triangle_data(30))
    hmm = HMM(10, 4)
    hmm.fit(data)  # 先根据给定数据反推参数
    gen_obs = hmm.generate(30)  # 再根据学习的参数生成数据
    x = np.arange(30)
    plt.scatter(x, gen_obs, marker='*', color='r')
    plt.plot(x, data, color='g')
    plt.show()
