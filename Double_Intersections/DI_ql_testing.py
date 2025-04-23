# test_double_intersection.py

import os
import re
import numpy as np
import torch
import sumo_rl
from agents.ql_agent2 import QlAgent2
from matplotlib import pyplot as plt

def build_neighbours(env):
    """Build a map of neighbouring traffic lights by looking
    for two‐letter edge IDs linking TLs."""
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for edge_id in env.sumo.edge.getIDList():
        # look for exactly two‐char edge IDs where both chars are TL IDs
        if len(edge_id) == 2 and edge_id[0] in ts and edge_id[1] in ts:
            a, b = edge_id[0], edge_id[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: np.array(v) for tl, v in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """Concatenate own obs plus 0.2× neighbours’ obs, then pad/truncate."""
    x = obs[tl].copy()
    for n in neighbours[tl]:
        x = x + 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return torch.FloatTensor(x)

def main():
    # 1) create the SUMO env
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    obs = env.reset()

    # 2) build neighbour map
    neighbours = build_neighbours(env)

    # 3) load each TL’s trained Q‑agent
    agents = {}
    for tl in env.ts_ids:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = QlAgent2(input_shape=165, output_shape=n_phases)
        ckpt = f"trained_models/ql_double_{tl}.pth"
        agent.model.load_state_dict(torch.load(ckpt))
        agent.model.eval()
        agent.epsilon = 0.0    # fully greedy
        agents[tl] = agent

    # 4) run one test episode
    obs = env.reset()
    state = {tl: prepare_obs(obs, tl, neighbours) for tl in env.ts_ids}

    done = {"__all__": False}
    rewards = []
    queues  = []
    phase_counts = {
        tl: {p: 0 for p in range(len(env.traffic_signals[tl].all_phases))}
        for tl in env.ts_ids
    }

    step = 0
    while not done["__all__"] and step < 5000:
        step += 1

        # choose a greedy, valid phase for each TL
        actions = {}
        for tl, agent in agents.items():
            curr = env.traffic_signals[tl].green_phase
            # valid next phases according to yellow_dict
            valid = [
                p for p in range(agent.output_shape)
                if (curr, p) in env.traffic_signals[tl].yellow_dict
            ]
            q_vals = agent.model(state[tl])
            a = int(torch.argmax(q_vals))
            if a not in valid:
                a = curr
            phase_counts[tl][a] += 1
            actions[tl] = a

        # step
        obs2, reward_dict, done, _ = env.step(actions)

        # record metrics
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # average queue over all TLs
        total_q = 0
        for tl in env.ts_ids:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(total_q / len(env.ts_ids))

        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {queues[-1]: .1f}")

        # prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in env.ts_ids}

    # summary
    print("\n✅ Testing done")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue   over episode: {np.mean(queues):.1f}")
    for tl in env.ts_ids:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # plots
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("QL Testing: Rewards")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("QL Testing: Queue Lengths")
    plt.grid(True); plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
