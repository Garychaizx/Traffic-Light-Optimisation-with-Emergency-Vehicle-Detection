import os
import re
import random
import numpy as np
import torch
import sumo_rl
from matplotlib import pyplot as plt
from agents.ql_agent2 import QlAgent2

def build_neighbours(env):
    """
    Map each TL id to its neighbouring TLs based on 4‑char internal edges.
    """
    pattern = re.compile(r"[A-Z]\d[A-Z]\d")
    edges = [e for e in env.sumo.edge.getIDList() if pattern.fullmatch(e)]
    nbr = {}
    for e in edges:
        a, b = e[:-2], e[-2:]
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    # dedupe
    return {tl: list(set(lst)) for tl, lst in nbr.items()}

def prepare_obs(obs, tl_id, neighbours):
    # start with own obs, then each neighbour * 0.2
    parts = [obs[tl_id]]
    for n in neighbours.get(tl_id, []):
        parts.append(obs[n] * 0.2)
    arr = np.concatenate(parts)
    # pad / truncate to 165
    if len(arr) < 165:
        arr = np.concatenate([arr, np.full(165 - len(arr), -1.)])
    else:
        arr = arr[:165]
    return torch.FloatTensor(arr)

def main():
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 20000,
        single_agent=False
    )

    obs = env.reset()
    tls = env.ts_ids  # e.g. ['A','B','C','D']
    neighbours = build_neighbours(env)
    # ensure every TL has at least an empty list
    for tl in tls:
        neighbours.setdefault(tl, [])

    # create one agent per TL
    agents = {
        tl: QlAgent2(
            input_shape=165,
            output_shape=len(env.traffic_signals[tl].all_phases)
        )
        for tl in tls
    }

    # initial state dict
    state = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}
    done = {"__all__": False}
    avg_rewards = []
    step = 0

    while not done["__all__"] and step < 10000:
        step += 1

        # pick actions for each TL
        actions = {}
        for tl in tls:
            curr = env.traffic_signals[tl].green_phase
            valid = [
                p for p in range(agents[tl].output_shape)
                if (curr, p) in env.traffic_signals[tl].yellow_dict
            ]
            actions[tl] = agents[tl].select_action(state[tl], valid)

        obs2, rewards, done, _ = env.step(actions)

        # store and train
        for tl in tls:
            s2 = prepare_obs(obs2, tl, neighbours)
            agents[tl].store_experience(state[tl], actions[tl], rewards[tl], s2)
            agents[tl].train_from_batch()

        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}
        avg_rewards.append(np.mean(list(rewards.values())))

        if step % 500 == 0:
            print(f"Step {step:5d}  AvgRwd {avg_rewards[-1]:.3f}  Eps {agents[tls[0]].epsilon:.2f}")

    # save models
    os.makedirs("trained_models", exist_ok=True)
    for tl in tls:
        torch.save(
            agents[tl].model.state_dict(),
            f"trained_models/ql_double_{tl}.pth"
        )

    # plot performance
    plt.figure(figsize=(8,4))
    plt.plot(avg_rewards)
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Double Intersection Q‑Learning")
    plt.grid(True)
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
