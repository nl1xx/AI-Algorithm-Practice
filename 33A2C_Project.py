# A2C
# A(s, a) = Q(s, a) - V(s)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym


class Actor2Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor2Critic, self).__init__()
        # Shared layers
        self.shared = nn.Linear(state_dim, 128)
        # Actor-specific layers
        self.actor = nn.Linear(128, action_dim)
        # Critic-specific layers
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        shared_out = F.relu(self.shared(state))
        # Actor output (policy distribution over actions)
        policy_dist = F.softmax(self.actor(shared_out), dim=-1)
        # Critic output (state value estimation)
        value = self.critic(shared_out)
        return policy_dist, value


env_name = "CartPole-v1"
learning_rate = 0.01
gamma = 0.99


env = gym.make(env_name, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = Actor2Critic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def compute_advantages(rewards, values, gamma):
    advantages = []
    returns = []
    G = 0
    for reward, value in zip(reversed(rewards), reversed(values)):
        G = reward + gamma * G
        returns.insert(0, G)
        advantages.insert(0, G - value.item())
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)


num_episodes = 500
for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)

    log_probs = []
    values = []
    rewards = []

    done = False
    while not done:
        # Get policy distribution and value
        policy_dist, value = model(state)
        action = torch.multinomial(policy_dist, 1).item()

        next_state, reward, done, _, _ = env.step(action)

        log_prob = torch.log(policy_dist.squeeze(0)[action])
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        state = torch.FloatTensor(next_state).unsqueeze(0)

    # Compute advantages and returns
    advantages, returns = compute_advantages(rewards, values, gamma)

    log_probs = torch.stack(log_probs)
    values = torch.cat(values).squeeze(-1)

    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = F.mse_loss(values, returns)
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}, Loss: {loss.item():.4f}")

env.close()
