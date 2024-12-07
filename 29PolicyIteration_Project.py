# 使用策略迭代解决冰湖问题

import gym
import numpy as np

env = gym.make('FrozenLake-v1')
# 4*4的网格，有16个格子（状态），分别用0-15表示
grid_state = env.observation_space.n
# 4个动作——上下左右，分别用0-3表示
grid_action = env.action_space.n


# 策略评估
def policy_evaluation(policy, env, gamma=1.0, theta=1e-10):
    value_table = np.zeros(grid_state)
    while True:
        delta = 0
        for state in range(grid_state):
            v = value_table[state]
            # 根据当前策略计算状态的值
            action = policy[state]
            q = 0
            for next_sr in env.P[state][action]:
                # P[][]是环境定义的变量,存储状态s下采取动作a得到的元组数据（转移概率，下一步状态，奖励，完成标志）
                trans_prob, next_state, reward, done = next_sr
                q += trans_prob * (reward + gamma * value_table[next_state])
            value_table[state] = q
            delta = max(delta, abs(v - value_table[state]))
        if delta < theta:
            break
    return value_table


# 策略改进
def policy_improvement(value_table, env, gamma=1.0):
    policy = np.zeros(grid_state)
    for state in range(grid_state):
        # 初始化Q表
        Q_table = np.zeros(grid_action)
        # 对每个动作计算
        for action in range(grid_action):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward, done = next_sr
                Q_table[action] += trans_prob * (reward + gamma * value_table[next_state])
        # 当前状态下，选取使Q值最大的那个动作
        policy[state] = np.argmax(Q_table)
    return policy


# 策略迭代
def policy_iteration(env, gamma=1.0, theta=1e-10):
    policy = np.ones(grid_state) * 0  # 初始策略，可以是任意动作
    while True:
        value_table = policy_evaluation(policy, env, gamma, theta)
        new_policy = policy_improvement(value_table, env, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy
    return policy, value_table


# 最优策略和值函数
optimal_policy, optimal_value_function = policy_iteration(env, gamma=1.0)

# 输出最优策略
print(optimal_policy)
