# 使用Q-Learning解决冰湖问题

import gym
import numpy as np


class QLearningAgent:
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.obs_n = obs_n
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))

    # 根据当前的Q值表和ε-greedy策略选择一个行动。以1-ε的概率选择当前最优动作，以ε的概率随机选择一个动作
    # obs -> state
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = np.argmax(self.Q[obs, :])
        else:
            action = np.random.choice(self.act_n)
        return action

    #  根据当前的Q值表选择一个最优动作
    # obs -> state
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        # 如果有多个最优选择则随机选取一个
        action = np.random.choice(action_list)
        return action

    # 更新公式
    def learn(self, obs, action, reward, next_obs, done):
        pre_Q = self.Q[obs, action]
        if done:
            tar_Q = reward
        else:
            tar_Q = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (tar_Q - pre_Q)


# 定义运行单集episode的函数
def run_episode(env, agent):
    total_steps = 0
    total_reward = 0
    obs = env.reset()
    obs = obs[0]
    action = agent.sample(obs)
    while True:
        next_obs, reward, done, truncated, info = env.step(action)
        next_action = agent.sample(next_obs)
        agent.learn(obs, action, reward, next_obs, done)
        action = next_action
        obs = next_obs
        total_reward += reward
        total_steps += 1
        if done:
            break
    return total_reward, total_steps


# 定义测试单集episode的函数
def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    obs = obs[0]
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        if done:
            print('test reward = %.1f' % (total_reward))
            break


# 创建环境
env = gym.make("FrozenLake-v1", is_slippery=False)
obs_n = env.observation_space.n  # 16
act_n = env.action_space.n  # 4


agent = QLearningAgent(obs_n=obs_n, act_n=act_n, learning_rate=0.2, gamma=0.9, e_greed=0.2)

# 训练智能体
for episode in range(10000):
    ep_reward, ep_steps = run_episode(env, agent)
    print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# 测试智能体
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
test_episode(env, agent)
