# train_sac_double.py

import os
import random
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
from agents.sac_agent import SACAgent

def build_neighbours(env):
    """After env.reset(), map each TL id to its 2‑char neighbours (edges like "AB", "BC", etc.)."""
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """Concatenate own obs + 0.2× neighbours, pad or truncate to pad_len."""
    x = obs[tl].copy()
    for n in neighbours.get(tl, []):
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return x

def train_sac_double(
    net_file="nets/double/network.net.xml",
    route_file="nets/double/doubleRoutes.rou.xml",
    num_seconds=20000,
    max_steps=500,
    max_episodes=200,
    batch_size=64,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    replay_size=100000
):
    # 1) Create SUMO env (no GUI for faster training)
    env = sumo_rl.SumoEnvironment(
        net_file    = net_file,
        route_file  = route_file,
        use_gui     = False,
        num_seconds = num_seconds,
        single_agent=False
    )

    # initialize traci and build neighbour map
    obs0 = env.reset()
    tls  = env.ts_ids
    neighbours = build_neighbours(env)

    # 2) Instantiate one SACAgent per TL
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = SACAgent(
            state_dim  = 165,
            action_dim = n_phases,
            gamma      = gamma,
            tau        = tau,
            alpha      = alpha,
            lr         = lr
        )
        # share a single replay buffer across all TLs (optional)
        agent.replay_buffer = []  # reset
        agents[tl] = agent

    avg_rewards = []
    # 3) Main training loop
    for ep in range(1, max_episodes+1):
        obs   = env.reset()
        state = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}
        done  = {"__all__": False}
        step  = 0
        ep_rewards = []

        while not done["__all__"] and step < max_steps:
            step += 1
            actions = {}

            # 3a) select and mask legal actions for each TL
            for tl, agent in agents.items():
                curr   = env.traffic_signals[tl].green_phase
                valid  = [
                    p for p in range(agent.action_dim)
                    if (curr, p) in env.traffic_signals[tl].yellow_dict
                ]
                a = agent.choose_action(state[tl], valid_actions=valid)
                actions[tl] = a

            # 3b) step environment
            obs2, reward_dict, done, _ = env.step(actions)

            # 3c) store & learn
            next_state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}
            for tl, agent in agents.items():
                r    = reward_dict[tl]
                s    = state[tl]
                a    = actions[tl]
                s2   = next_state[tl]
                done_flag = float(done["__all__"])
                agent.store_experience(s, a, r, s2, done_flag)
                agent.update(batch_size=batch_size)
                ep_rewards.append(r)

            state = next_state

        avg = np.mean(ep_rewards) if ep_rewards else 0.0
        avg_rewards.append(avg)
        print(f"Episode {ep:3d}/{max_episodes:3d} | Steps {step:4d} | Avg Reward {avg:.3f}")

    # 4) Save each agent’s networks
    os.makedirs("trained_models", exist_ok=True)
    for tl, agent in agents.items():
        torch.save({
            "actor":      agent.actor.state_dict(),
            "critic_1":   agent.critic_1.state_dict(),
            "critic_2":   agent.critic_2.state_dict(),
            "target_1":   agent.target_critic_1.state_dict(),
            "target_2":   agent.target_critic_2.state_dict()
        }, f"trained_models/sac_double_{tl}.pth")
        print(f"→ Saved SAC for TL {tl}")

    # 5) Plot learning curve
    plt.figure(figsize=(8,4))
    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Double‐Intersection SAC Training")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    env.close()

if __name__ == "__main__":
    train_sac_double()
