import random

import torch
import torch.nn as nn


class QlAgent:
    def __init__(self, input_shape=165, output_shape=8, loss_fn=nn.MSELoss, learning_rate=1e-03, epsilon=1):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def predict_rewards(self, observations):
        if observations.shape != (self.input_shape, ):
            raise Exception(f'Invalid input shape. Expected {self.input_shape}, got {observations.shape}!')
        return self.model(observations)

    def learn(self, pred_reward, actual_reward):
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred_reward, actual_reward)
        loss.backward()
        self.optimizer.step()

    def get_params(self):
        return self.model.parameters()

    def set_params(self, params):
        for i, param in enumerate(params):
            self.model[i].weight.data = param.clone().detach()

    def store_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
        return zip(*batch)

    def train_from_batch(self, gamma):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states = self.sample_batch()
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards)

        with torch.no_grad():
            q_next = self.model(next_states).max(1)[0]
            q_targets = rewards + gamma * q_next

        q_preds = self.model(states)
        q_vals = q_preds[range(len(actions)), actions]

        loss = self.loss_fn(q_vals, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Define input and output sizes
input_Size = 33*5
output_Size = 2**15

# Initialize Q-network
q_network = QlAgent(input_Size, output_Size)

# Print the network architecture
print(q_network)
