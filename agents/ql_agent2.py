import random
import torch
import torch.nn as nn

class QlAgent2:
    def __init__(
        self,
        input_shape,
        output_shape,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-4,
        gamma: float = 0.9,
        batch_size: int = 64
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size

        # simple feed‑forward Q‑network
        self.model = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # replay buffer
        self.replay_buffer = []

    def select_action(self, state: torch.Tensor, valid_actions: list) -> int:
        # ε‑greedy over only valid phases
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            qvals = self.model(state)
            # mask out invalid actions
            masked = torch.full((self.output_shape,), float('-inf'))
            for a in valid_actions:
                masked[a] = qvals[a]
            return torch.argmax(masked).item()

    def store_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
        states, actions, rewards, next_states = zip(*batch)
        return states, actions, rewards, next_states

    def train_from_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states = self.sample_batch()
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards)
        actions = torch.tensor(actions)

        # compute targets
        with torch.no_grad():
            q_next = self.model(next_states).max(1)[0]
            q_targets = rewards + self.gamma * q_next

        q_preds = self.model(states)
        q_vals = q_preds[range(len(actions)), actions]

        loss = self.loss_fn(q_vals, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay ε
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
