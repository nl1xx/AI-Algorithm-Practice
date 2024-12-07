import numpy as np

num_states = 9
num_actions = 4  # 上下左右

# 迷宫奖励矩阵
rewards = np.array([
    [-1, -1, -1, -1],
    [-1, 0, -1, 1],
    [-1, -1, -1, -1],
    [1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, 0, -1, 1],
    [-1, -1, -1, -1],
    [-1, 0, -1, 1],
    [-1, 1, -1, 1],
])

policy = np.ones((num_states, num_actions)) / num_actions
values = np.zeros(num_states)


def policy_evaluation():
    global values
    delta = 1e-6
    max_iterations = 1000
    for _ in range(max_iterations):
        new_values = np.zeros(num_states)
        for s in range(num_states):
            v = 0
            for a in range(num_actions):
                next_state = get_next_state(s, a)  # 获取下一个状态
                v += policy[s][a] * (rewards[s][a] + values[next_state])
            new_values[s] = v
        if np.max(np.abs(new_values - values)) < delta:
            print(f"Policy Evaluation: Values converged after {_+1} iterations.")
            break
        values = new_values


def get_next_state(state, action):
    if state == 1 and action == 3:
        return 3
    elif state == 3 and action == 0:
        return 1
    else:
        return state


# 定义策略改进函数
def policy_improvement():
    global policy
    for s in range(num_states):
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            next_state = get_next_state(s, a)  # 获取下一个状态
            q_values[a] = rewards[s][a] + values[next_state]
        best_action = np.argmax(q_values)
        new_policy = np.zeros(num_actions)
        new_policy[best_action] = 1
        policy[s] = new_policy


# 策略迭代算法
def policy_iteration():
    max_iterations = 100  # 最大迭代次数
    for i in range(max_iterations):
        policy_evaluation()  # 策略评估
        policy_improvement()  # 策略改进
        if i == max_iterations - 1:
            print(f"Final policy {policy}.")
        if np.all(policy == np.ones((num_states, num_actions)) / num_actions):
            print(f"Policy Iteration: Policy converged after {i+1} iterations.")
            break


policy_iteration()
