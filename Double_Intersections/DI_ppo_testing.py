# test_double_intersection_ppo.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import sumo_rl
from agents.ppo_agent import PPO
from matplotlib import pyplot as plt

def build_neighbours(env):
    """
    Build a map of neighbouring traffic lights by looking
    for two-letter edge IDs linking TLs.
    """
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    # must call env.reset() first so env.sumo is initialized
    for edge_id in env.sumo.edge.getIDList():
        if len(edge_id) == 2 and edge_id[0] in ts and edge_id[1] in ts:
            a, b = edge_id[0], edge_id[1]
            neigh[a].append(b)
            neigh[b].append(a)
    # convert lists to arrays (optional)
    return {tl: np.array(v) for tl, v in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """
    Concatenate own obs plus 0.2× neighbours’ obs, then pad/truncate.
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
    # 1) create the SUMO env (GUI on for visualization)
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    # 2) initialize and build neighbour map
    obs = env.reset()
    neighbours = build_neighbours(env)

    # 3) load each TL’s trained PPO agent
    agents = {}
    for tl in env.ts_ids:
        n_phases = len(env.traffic_signals[tl].all_phases)
        # instantiate with the same hyperparams used during training:
        agent = PPO(
            state_dim   = 165,
            action_dim  = n_phases,
            hidden_size = 64,
            lr          = 3e-4,
            gamma       = 0.99,
            clip_ratio  = 0.2,
            K_epoch     = 4
        )
        ckpt = f"trained_models/ppo_double_{tl}.pth"
        # load into actor_critic
        agent.actor_critic.load_state_dict(torch.load(ckpt))
        agent.actor_critic.eval()
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
        actions = {}

        # choose a greedy, valid phase for each TL
        for tl, agent in agents.items():
            s = state[tl].unsqueeze(0)           # [1,165]
            with torch.no_grad():
                logits, _ = agent.actor_critic(s)
                probs = F.softmax(logits, dim=-1).squeeze(0)  # [n_phases]
            # mask invalid transitions
            curr = env.traffic_signals[tl].green_phase
            valid = [p for p in range(probs.size(0))
                     if (curr, p) in env.traffic_signals[tl].yellow_dict]
            # pick highest‐prob among valid
            best = torch.argmax(probs).item()
            action = best if best in valid else curr
            phase_counts[tl][action] += 1
            actions[tl] = action

        # step env
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

    # 5) summary
    print("\n✅ Testing done")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue   over episode: {np.mean(queues):.1f}")
    for tl in env.ts_ids:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # 6) plots
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("PPO Testing: Rewards")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("PPO Testing: Queue Lengths")
    plt.grid(True); plt.legend()

    plt.show()

    env.close()

if __name__ == "__main__":
    main()
