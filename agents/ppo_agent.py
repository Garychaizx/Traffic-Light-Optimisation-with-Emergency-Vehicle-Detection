import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sumo_rl


# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


# Proximal Policy Optimization
class PPO:
    def __init__(self, state_dim, action_dim, hidden_size, lr, gamma, clip_ratio, K_epoch):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.K_epoch = K_epoch
        self.continuous_loss = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.actor_critic(state)
        action_probs = F.softmax(action_probs, dim=-1)
        action = torch.multinomial(action_probs, 1)
        return action.item()

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def update(self, states, actions, old_probs, rewards):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_probs = torch.FloatTensor(old_probs)
        returns = self.compute_returns(rewards)

        for _ in range(self.K_epoch):
            new_probs, values = self.actor_critic(states)
            new_probs = F.softmax(new_probs, dim=-1)
            critic_loss = F.mse_loss(values.squeeze(), returns)

            probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze()
            ratio = probs / old_probs

            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * returns
            actor_loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss = actor_loss + critic_loss
            self.continuous_loss.append(loss)
            loss.backward()
            self.optimizer.step()


# Training
def train(hidden_size=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, K_epoch=10, max_steps=100,
          max_episodes=100):
    base_env = sumo_rl.SumoEnvironment(net_file='../nets/grid4x4/grid4x4.net.xml',
                                       route_file='../nets/grid4x4/grid4x4_1.rou.xml',
                                       use_gui=False,
                                       num_seconds=1000,
                                       single_agent=True)

    state_dim = base_env.observation_space.shape[0]
    print("State Dim: ", state_dim)
    # Since our action space is gym.Discrete
    action_dim = base_env.action_space.n
    print("Action Dim: ", action_dim)

    agent = PPO(state_dim, action_dim, hidden_size, lr, gamma, clip_ratio, K_epoch)

    for episode in range(max_episodes):
        state = base_env.reset()[0]
        done = False
        total_reward = 0

        states = []
        actions = []
        old_probs = []
        rewards = []

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = base_env.step(base_env.action_space.sample())

            states.append(state)
            actions.append(action)
            old_probs.append(agent.actor_critic.actor(torch.FloatTensor(state)).squeeze()[action])
            rewards.append(reward)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.update(states, actions, old_probs, rewards)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    print(agent.continuous_loss)
    base_env.close()


if __name__ == "__main__":
    train()
