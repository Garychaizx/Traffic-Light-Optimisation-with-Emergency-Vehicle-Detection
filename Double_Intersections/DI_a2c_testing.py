# test_a2c_double.py

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
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """
    Stack own observation + 0.2× each neighbour’s, then pad (with -1)
    or truncate to pad_len.
    Returns a torch.FloatTensor.
    """
    x = obs[tl].copy()
    for n in neighbours.get(tl, []):
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return torch.FloatTensor(x)

def main():
    # 1) Create SUMO env in GUI mode for visualization
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    # initialize and build neighbours
    obs = env.reset()
    tls  = env.ts_ids
    neighbours = build_neighbours(env)

    # 2) Load each TL’s trained A2C agent
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = A2CAgent(state_dim=165, action_dim=n_phases)
        ckpt = torch.load(f"trained_models/a2c_double_{tl}.pth")
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
        agent.actor.eval()
        agent.critic.eval()
        agents[tl] = agent

    # 3) Run one test episode
    obs   = env.reset()
    state = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}

    done = {"__all__": False}
    rewards = []
    queues  = []
    phase_counts = {tl: {p: 0 for p in range(len(env.traffic_signals[tl].all_phases))}
                    for tl in tls}
    step = 0
    MAX_STEPS = 5000

    while not done["__all__"] and step < MAX_STEPS:
        step += 1
        actions = {}

        # for each TL, pick the highest‑prob valid phase
        for tl, agent in agents.items():
            logits = agent.actor(state[tl].unsqueeze(0)).squeeze(0)
            probs  = logits.softmax(dim=-1)

            # mask illegal transitions
            curr  = env.traffic_signals[tl].green_phase
            valid = [p for p in range(probs.size(0))
                     if (curr, p) in env.traffic_signals[tl].yellow_dict]
            mask  = torch.zeros_like(probs)
            mask[valid] = 1.0
            masked = probs * mask
            if masked.sum() == 0:
                # fallback uniform on valid
                masked[valid] = 1.0 / len(valid)
            else:
                masked /= masked.sum()

            # deterministic: pick argmax
            a = int(masked.argmax().item())
            phase_counts[tl][a] += 1
            actions[tl] = a

        # step environment
        obs2, reward_dict, done, _ = env.step(actions)

        # metrics
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # average queue length across all TLs
        total_q = 0
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(total_q / len(tls))

        print(f"Step {step:4d} | AvgReward {avg_r: .3f} | AvgQueue {queues[-1]: .1f}")

        # prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

    # 4) Summary
    print("\n✅ Testing complete")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Mean queue:  {np.mean(queues):.1f}\n")
    for tl in tls:
        print(f"Phase counts for TL {tl}:")
        for p, cnt in phase_counts[tl].items():
            print(f"  Phase {p}: {cnt} times")

    # 5) Plots
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("A2C: Reward per Step")
    plt.grid(); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("A2C: Queue Length per Step")
    plt.grid(); plt.legend()

    plt.show()

    env.close()

if __name__ == "__main__":
    main()
