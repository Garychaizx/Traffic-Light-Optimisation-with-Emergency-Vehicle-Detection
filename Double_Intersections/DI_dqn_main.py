# test_dqn_double.py

import os
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent

def build_neighbours(env):
    """
    After env.reset(), build a map from each TL id to the
    list of its two‑char neighbours (edges like "AB", "BC", etc.).
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
    Stack own observation + 0.2× each neighbour’s,
    then pad (with -1) or truncate to pad_len.
    Returns a flat numpy array.
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
    # 1) Create the SUMO environment with GUI for visualization
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    # 2) Reset and build neighbour map
    obs0 = env.reset()
    tls  = env.ts_ids
    neighbours = build_neighbours(env)

    # 3) Instantiate and load one DQNAgent per TL
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = DQNAgent(
            state_dim   = 165,
            action_dim  = n_phases,
            gamma       = 0.99,
            lr          = 1e-3,
            epsilon     = 0.0,      # greedy
            min_epsilon = 0.0,
            epsilon_decay = 0.0,
            batch_size  = 64,
            memory_size = 1        # dummy
        )
        ckpt = f"trained_models/dqn_double_{tl}.pth"
        agent.load(ckpt)
        agents[tl] = agent

    # 4) Run one evaluation episode
    obs    = env.reset()
    state  = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}
    done   = {"__all__": False}
    rewards = []
    queues   = []
    phase_counts = {tl: {p:0 for p in range(len(env.traffic_signals[tl].all_phases))} for tl in tls}
    step = 0
    MAX_STEPS = 5000

    while not done["__all__"] and step < MAX_STEPS:
        step += 1
        actions = {}

        # select greedy actions
        for tl, agent in agents.items():
            curr = env.traffic_signals[tl].green_phase
            valid = [p for p in range(agent.action_dim)
                     if (curr, p) in env.traffic_signals[tl].yellow_dict]
            a = agent.choose_action(state[tl], valid)
            phase_counts[tl][a] += 1
            actions[tl] = a

        # step environment
        obs2, reward_dict, done, _ = env.step(actions)

        # record avg reward
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # record avg queue length
        total_q = 0
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(total_q / len(tls))

        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {queues[-1]: .1f}")

        # prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

    # 5) Summary
    print("\n✅ Evaluation complete.")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue   over episode: {np.mean(queues):.1f}")
    for tl in tls:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # 6) Plotting
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("DQN Double Intersection: Reward")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue Length"); plt.title("DQN Double Intersection: Queue")
    plt.grid(True); plt.legend()

    plt.show()
    env.close()

if __name__ == "__main__":
    main()