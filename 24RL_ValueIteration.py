import numpy as np
import matplotlib.pyplot as plt

# 0代表可行走的路径，-1代表障碍物或墙壁，1代表迷宫的终点位置
maze = np.array([
    [0, 0, 0, 0],
    [0, -1, 0, -1],
    [0, 0, 0, 0],
    [-1, 0, -1, 1]
])

gamma = 0.9  # 折扣因子
epsilon = 1e-6  # 收敛阈值

# 初始化值函数和策略
V = np.zeros(maze.shape)
policy = np.zeros(maze.shape, dtype=object)

# 进行值迭代
while True:
    delta = 0
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == -1 or maze[i, j] == 1:
                continue
            # 计算当前状态的最大价值
            max_value = float("-inf")
            for action in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + action[0], j + action[1]
                if ni >= 0 and ni < maze.shape[0] and nj >= 0 and nj < maze.shape[1] and maze[ni, nj] != -1:
                    max_value = max(max_value, gamma * V[ni, nj])
            # 更新值函数
            new_value = maze[i, j] + max_value
            delta = max(delta, abs(new_value - V[i, j]))
            V[i, j] = new_value
    if delta < epsilon:
        break


print("最优值函数：")
print(V)
