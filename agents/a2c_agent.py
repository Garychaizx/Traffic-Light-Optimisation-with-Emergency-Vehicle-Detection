import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

class A2CAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.gamma = gamma

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def choose_action(self, state, valid_phases):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_tensor).squeeze()
        probs_valid = probs[valid_phases]
        action_index = torch.multinomial(probs_valid, 1).item()
        return valid_phases[action_index]

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)

        target = reward + self.gamma * next_value * (1 - done)
        advantage = (target - value).detach()

        # Update Critic
        critic_loss = nn.functional.mse_loss(value, target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update Actor
        probs = self.actor(state_tensor)
        log_prob = torch.log(probs[0, action])
        actor_loss = -log_prob * advantage
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
