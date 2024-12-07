# 使用值迭代解决冰湖问题

import gym
import numpy as np

env = gym.make('FrozenLake-v1')
# 4*4的网格，有16个格子（状态），分别用0-15表示
grid_state = env.observation_space.n
# 4个动作——上下左右，分别用0-3表示
grid_action = env.action_space.n


# 值迭代
def value_iteration(env, gamma=1.0):
    value_table = np.zeros(grid_state)
    no_of_iterations = 100000
    threshold = 1e-20

    for i in range(no_of_iterations):
        delta = 0
        for state in range(grid_state):
            v = value_table[state]
            # 计算当前状态下所有动作的最大Q值
            action_values = []
            for action in range(grid_action):
                q = 0
                # P[][]是环境定义的变量,存储状态s下采取动作a得到的元组数据（转移概率，下一步状态，奖励，完成标志）
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward, done = next_sr
                    q += trans_prob * (reward + gamma * value_table[next_state])
                action_values.append(q)
            value_table[state] = max(action_values)
            delta = max(delta, abs(v - value_table[state]))

        if delta < threshold:
            print("Value-iteration converged at iteration %d" % (i + 1))
            break

    return value_table


# 策略选取
def extract_policy(value_table, gamma=1.0):
    # 初始化存储策略的数组
    policy = np.zeros(grid_state)
    # 对每个状态构建Q表，并在该状态下对每个行为计算Q值，
    for state in range(grid_state):
        # 初始化Q表
        Q_table = np.zeros(grid_action)
        # 对每个动作计算
        for action in range(grid_action):
            # 同上
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward, done = next_sr
                # 更新Q表，即更新动作对应的Q值（4个动作分别由0-3表示）
                Q_table[action] += (trans_prob * (reward + gamma * value_table[next_state]))
        # 当前状态下，选取使Q值最大的那个动作
        policy[state] = np.argmax(Q_table)
    # 返回动作
    return policy


# 最优值函数
optimal_value_function = value_iteration(env=env, gamma=1.0)
# 最优策略
optimal_policy = extract_policy(optimal_value_function, gamma=1.0)

# 输出最优策略
print(optimal_policy)
