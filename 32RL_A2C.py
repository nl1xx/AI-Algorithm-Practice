# A2C
# A(s, a) = Q(s, a) - V(s)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


# 定义A2C算法
class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.01)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def advantage_fn(self, rewards, values, dones, gamma=0.99):
        # 初始化优势列表
        advantages = []
        # 计算回报
        returns = []
        G = 0
        for i in reversed(range(len(rewards))):
            # 确保dones[i]是标量
            done = dones[i] if i < len(dones) else 0
            G = rewards[i] + gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).view(-1, 1)  # 确保returns是二维张量

        for i in range(len(values)):
            advantages.append(returns[i] - values[i].item())
        advantages = torch.tensor(advantages, dtype=torch.float32).view(-1, 1)  # 确保advantages是二维张量
        return advantages, returns

    def update(self, states, actions, rewards, next_states, dones, gamma=0.99):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 计算Critic的值
        current_values = self.critic(states)
        next_values = self.critic(next_states)

        # 计算Critic的损失
        advantages, returns = self.advantage_fn(rewards, current_values, dones, gamma)
        critic_loss = F.mse_loss(current_values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算Actor的梯度
        probs = self.actor(states)
        selected_action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算Actor的损失
        actor_loss = -torch.log(selected_action_probs) * advantages.detach()
        actor_loss = actor_loss.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        print(f"Critic Loss: {critic_loss.item():.4f}")
        print(f"Actor Loss: {actor_loss.item():.4f}")


# 假设环境和交互
state_dim = 4  # 状态空间维度
action_dim = 2  # 动作空间维度

agent = A2CAgent(state_dim, action_dim)

# 模拟环境交互
states = [np.random.rand(state_dim) for _ in range(10)]  # 假设的状态列表
actions = [agent.select_action(state) for state in states]
next_states = [np.random.rand(state_dim) for _ in range(10)]  # 假设的下一个状态列表
rewards = [np.random.rand() for _ in range(10)]  # 假设的奖励列表
dones = [np.random.choice([0, 1]) for _ in range(10)]  # 假设的结束信号列表

# 更新网络
agent.update(states, actions, rewards, next_states, dones)
