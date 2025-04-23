import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F

class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic networks (Q1 and Q2 for double Q-learning)
        self.critic_1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr
        )

        # Target critic networks
        self.target_critic_1 = nn.Sequential(*[layer for layer in self.critic_1])
        self.target_critic_2 = nn.Sequential(*[layer for layer in self.critic_2])
        self.update_target_networks(1.0)

        # Replay buffer
        self.replay_buffer = []

    def choose_action(self, state, valid_actions=None):
        # Get raw action probabilities
        with torch.no_grad():
            action_probs = self.actor(torch.FloatTensor(state).unsqueeze(0))
            probs = torch.softmax(action_probs, dim=-1).cpu().numpy().flatten()

        # Mask invalid actions
        if valid_actions is not None:
            mask = np.zeros_like(probs)
            mask[valid_actions] = probs[valid_actions]
            probs = mask / mask.sum()  # normalize

        return np.random.choice(len(probs), p=probs)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > 100000:  # Limit buffer size
            self.replay_buffer.pop(0)

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # One-hot encode actions
        next_probs = self.actor(next_states)
        next_dist = Categorical(next_probs)
        next_actions = next_dist.sample().unsqueeze(1)
        next_actions_one_hot = F.one_hot(next_actions.squeeze(-1), num_classes=self.action_dim).float()

        # Compute target Q-values
        with torch.no_grad():
            target_q1 = self.target_critic_1(torch.cat([next_states, next_actions_one_hot], dim=1))
            target_q2 = self.target_critic_2(torch.cat([next_states, next_actions_one_hot], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_dist.entropy().unsqueeze(1)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # Update critics
        current_q1 = self.critic_1(torch.cat([states, F.one_hot(actions.squeeze(-1), num_classes=self.action_dim).float()], dim=1))
        current_q2 = self.critic_2(torch.cat([states, F.one_hot(actions.squeeze(-1), num_classes=self.action_dim).float()], dim=1))
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        probs = self.actor(states)
        dist = Categorical(probs)
        sampled_actions = dist.sample().unsqueeze(1)
        sampled_actions_one_hot = F.one_hot(sampled_actions.squeeze(-1), num_classes=self.action_dim).float()

        q1 = self.critic_1(torch.cat([states, sampled_actions_one_hot], dim=1))
        q2 = self.critic_2(torch.cat([states, sampled_actions_one_hot], dim=1))
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * dist.entropy().unsqueeze(1) - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks(self.tau)
    def update_target_networks(self, tau):
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)