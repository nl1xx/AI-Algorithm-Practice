# AC
# s使用TD-Error

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        probs = self.layer(state)
        return probs


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        value = self.layer(state)
        return value


# 定义AC算法
class ACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.01)

    def select_action(self, state):
        # 通过Actor网络计算出动作的概率分布，然后根据这个分布随机选择一个动作，这个方法返回选择的动作的索引
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        action = torch.distributions.Categorical(probs).sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])  # 将reward放入列表中
        done = torch.FloatTensor([done])  # 将done放入列表中

        # 计算Critic的值
        current_value = self.critic(state)
        next_value = self.critic(next_state)

        # TD目标
        td_target = reward + (1 - done) * 0.99 * next_value

        td_error = td_target - current_value

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss = (td_error ** 2).mean()
        critic_loss.backward()
        self.critic_optimizer.step()
        print(f"Critic Loss: {critic_loss.item():.4f}")

        # 计算Actor的梯度
        probs = self.actor(state)
        selected_action_prob = probs[action]

        # 计算Actor的损失
        actor_loss = -torch.log(selected_action_prob) * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        print(f"Actor Loss: {actor_loss.mean().item():.4f}")


# 假设环境和交互
state_dim = 4  # 状态空间维度
action_dim = 2  # 动作空间维度

agent = ACAgent(state_dim, action_dim)

# 模拟环境交互
state = np.random.rand(state_dim)
action = agent.select_action(state)
next_state = np.random.rand(state_dim)  # 假设的下一个状态
reward = np.random.rand()  # 假设的奖励
done = np.random.choice([0, 1])  # 假设的结束信号

# 更新网络
agent.update(state, action, reward, next_state, done)
