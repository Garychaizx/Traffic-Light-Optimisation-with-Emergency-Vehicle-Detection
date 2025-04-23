# train_a2c_double.py

import os
import re
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
from agents.a2c_agent import A2CAgent

def build_neighbours(env):
    """
    After env.reset(), build a map from each TL id to its 2‑char neighbours
    (edges like "AB", "BC", etc.).
    """
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    # dedupe
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """
    Stack own observation + 0.2× each neighbour’s, then pad (with -1)
    or truncate to pad_len.
    Returns a numpy array.
    """
    x = obs[tl].copy()
    for n in neighbours.get(tl, []):
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return x

def main():
    # 1) Create SUMO env
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = False,
        num_seconds = 20000,
        single_agent=False
    )
    # init TraCI / SUMO
    obs0      = env.reset()
    tls       = env.ts_ids
    neighbours = build_neighbours(env)

    # 2) One A2C agent per traffic light
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agents[tl] = A2CAgent(
            state_dim  = 165,
            action_dim = n_phases,
            gamma      = 0.99,
            lr         = 1e-3
        )

    # 3) Training loop
    N_EPISODES = 200
    MAX_STEPS  = 500
    avg_rewards = []

    for ep in range(1, N_EPISODES+1):
        obs   = env.reset()
        state = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}
        done  = {"__all__": False}
        ep_reward = 0.0
        step = 0

        while not done["__all__"] and step < MAX_STEPS:
            step += 1
            # pick actions
            actions = {}
            for tl, agent in agents.items():
                curr = env.traffic_signals[tl].green_phase
                valid = [p for p in range(agent.actor.net[-2].out_features)
                         if (curr, p) in env.traffic_signals[tl].yellow_dict]
                actions[tl] = agent.choose_action(state[tl], valid)

            # advance
            obs2, reward_dict, done, _ = env.step(actions)
            next_state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

            # update each agent immediately
            for tl, agent in agents.items():
                r = reward_dict[tl]
                d = done["__all__"]
                agent.update(
                    state      = state[tl],
                    action     = actions[tl],
                    reward     = r,
                    next_state = next_state[tl],
                    done       = d
                )
                ep_reward += r

            state = next_state

        avg = ep_reward / step
        avg_rewards.append(avg)
        print(f"Episode {ep:3d}/{N_EPISODES:3d}  ⎸  AvgReward: {avg:.3f}")

    # 4) Save all agents
    os.makedirs("trained_models", exist_ok=True)
    for tl, agent in agents.items():
        path = f"trained_models/a2c_double_{tl}.pth"
        torch.save({
            'actor':  agent.actor.state_dict(),
            'critic': agent.critic.state_dict()
        }, path)
        print(f"→ Saved {tl} → {path}")

    # 5) Plot learning curve
    plt.figure(figsize=(8,4))
    plt.plot(avg_rewards, label="Avg Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Double Intersection A2C Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
