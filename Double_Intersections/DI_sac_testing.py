# test_sac_double.py

import os
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
from agents.sac_agent import SACAgent

def build_neighbours(env):
    """Map each TL to its neighbours via two‐char edge IDs."""
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """Concatenate own obs + 0.2× each neighbour’s, pad/truncate to pad_len."""
    x = obs[tl].copy()
    for n in neighbours.get(tl, []):
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return x

def main():
    # 1) create SUMO env
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    # 2) init and build neighbour map
    obs0       = env.reset()
    tls        = env.ts_ids
    neighbours = build_neighbours(env)

    # 3) create and load SACAgent for each TL
    agents = {}
    for tl in tls:
        n_ph = len(env.traffic_signals[tl].all_phases)
        ag = SACAgent(
            state_dim  = 165,
            action_dim = n_ph,
            gamma      = 0.99,
            tau        = 0.005,
            alpha      = 0.2,
            lr         = 3e-4
        )
        ckpt = torch.load(f"trained_models/sac_double_{tl}.pth", map_location="cpu")
        ag.actor.load_state_dict(ckpt["actor"])
        ag.critic_1.load_state_dict(ckpt["critic_1"])
        ag.critic_2.load_state_dict(ckpt["critic_2"])
        ag.target_critic_1.load_state_dict(ckpt["target_1"])
        ag.target_critic_2.load_state_dict(ckpt["target_2"])
        ag.actor.eval()
        agents[tl] = ag

    # 4) run one test episode
    state = {tl: prepare_obs(obs0, tl, neighbours) for tl in tls}
    done  = {"__all__": False}
    rewards = []
    queues  = []
    step = 0
    MAX_STEPS = 5000

    while not done["__all__"] and step < MAX_STEPS:
        step += 1
        actions = {}

        # choose greedy action for each TL
        for tl, ag in agents.items():
            curr = env.traffic_signals[tl].green_phase
            valid = [p for p in range(ag.action_dim)
                     if (curr, p) in env.traffic_signals[tl].yellow_dict]
            if not valid:
                valid = [curr]

            # get softmax probabilities
            with torch.no_grad():
                probs = ag.actor(torch.FloatTensor(state[tl]).unsqueeze(0)).squeeze().numpy()
            # mask invalid & pick argmax
            mask = np.zeros_like(probs)
            mask[valid] = probs[valid]
            if mask.sum() > 0:
                mask /= mask.sum()
            else:
                mask[:] = 1.0 / len(mask)
            action = int(np.argmax(mask))
            actions[tl] = action

        # step
        obs2, rdict, done, _ = env.step(actions)

        # record avg reward
        avg_r = np.mean(list(rdict.values()))
        rewards.append(avg_r)

        # record avg queue length
        tot_q = 0
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            tot_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(tot_q / len(tls))

        print(f"Step {step:4d} | Avg R {avg_r: .3f} | Avg Q {queues[-1]:.1f}")

        # next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

    # 5) summary
    print("\n✅ Testing complete")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Mean queue : {np.mean(queues):.1f}")

    # 6) plots
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("SAC Test: Rewards")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("SAC Test: Queue Lengths")
    plt.grid(True); plt.legend()

    plt.show()

    env.close()

if __name__ == "__main__":
    main()
